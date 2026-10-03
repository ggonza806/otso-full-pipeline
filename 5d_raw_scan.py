#!/usr/bin/env python3
"""
5d_raw_scan.py — Stage 1 of the penumbra barcode pipeline.

Runs OTSO.cone with default magnetic field (IGRF + Tsyganenko 89c + Kobel),
saves raw barcodes, case/plot plans, and supports resume after interruption.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
import OTSO

from pipeline_paths import activate_run, runs_base

from rigidity_window_grid import (
    MAT_PATH as MATLAB_BOUNDARY_PATH,
    add_scan_bounds,
    clean_windows,
    collapse_vertical_azimuths,
    densify_windows,
    load_boundary_file,
)

# ---------------------------------------------------------------------------
# Test / resume controls
# ---------------------------------------------------------------------------
MAX_CASES = None  # optional execution limit after the plan has been created
STRATIFIED_TEST_CASES = None  # None = keep the complete interpolated plan
RESUME_RUN_ID = None  # e.g. "run_20250617_120000_default_TSY89c_Kobel_raw"

# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------
print("Python executable:", sys.executable)
print("OTSO path:", OTSO.__file__)
EXPECTED_OTSO_VERSION = "1.2.19"
INSTALLED_OTSO_VERSION = package_version("OTSO")
print("OTSO version:", INSTALLED_OTSO_VERSION)
print()

if INSTALLED_OTSO_VERSION != EXPECTED_OTSO_VERSION:
    raise RuntimeError(
        f"Wrong OTSO version: expected {EXPECTED_OTSO_VERSION}, "
        f"found {INSTALLED_OTSO_VERSION}"
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
RUNS_BASE = runs_base()

# ---------------------------------------------------------------------------
# Scan grids
# ---------------------------------------------------------------------------
# "regular" preserves the original Stage-1 plan. "matlab_interpolated" uses
# penumbra_boundaries.mat to create a dense 4D plan.
GRID_SOURCE = "matlab_interpolated"

LAT_VALUES = list(range(-60, 61, 10))
LON_VALUES = list(range(0, 360, 30))  # 0..330, no 360

GEO_H0 = 400.0
ALT_H0_VALUES = [20.0, 400.0]

GEO_DIRECTIONS = [
    (0.0, 0.0),
    (60.0, 0.0),
    (60.0, 180.0),
]

ASUNCION_LAT = -25.2637
ASUNCION_LON = -57.5759
ASUNCION_PLOT_LON_0_360 = 302.4241

ZENITH_PLOT_VALUES = list(range(0, 181, 20))  # 0,20,...,180
AZIMUTH_PLOT_VALUES = list(range(0, 360, 15))  # 24 values

DR = 0.01

# The MATLAB boundary file has no altitude axis. For now the researcher fixes
# h0 here. Adding values creates another copy of the interpolated 4D grid.
INTERPOLATED_H0_VALUES = (400.0,)
WINDOW_MARGIN_LOW_GV = 0.20
WINDOW_MARGIN_HIGH_GV = 0.20
TOTAL_TOLERANCE_LOW_GV = 0.25
TOTAL_TOLERANCE_HIGH_GV = 3.00
RIGIDITY_FLOOR_GV = 0.01

# Generic tolerance limits for the legacy regular-window mode.
REGULAR_TOLERANCE_MIN_GV = 0.10
REGULAR_TOLERANCE_MAX_GV = 50.0

# ---------------------------------------------------------------------------
# OTSO fixed parameters
# ---------------------------------------------------------------------------
DATETIME_PARAMS = {
    "year": 2020,
    "month": 3,
    "day": 20,
    "hour": 12,
    "minute": 0,
    "second": 0,
}

COORDINATE_PARAMS = {
    "inputcoord": "GDZ",
    "coordsystem": "GEO",
}

COMPUTATION_PARAMS = {"corenum": 1}

LOCATION_NAME = "X"


def make_case_key(lat, lon, h0, zenith, azimuth):
    return (
        f"{round(float(lat), 6):.6f}|"
        f"{round(float(lon), 6):.6f}|"
        f"{round(float(h0), 6):.6f}|"
        f"{round(float(zenith), 6):.6f}|"
        f"{round(float(azimuth), 6):.6f}"
    )


def rigidity_window(zenith, scan_type):
    if scan_type == "angular":
        return 30.0, 0.5
    if float(zenith) == 0.0:
        return 20.0, 0.5
    return 30.0, 0.5


def integration_params(h0):
    return {
        "intmodel": "Boris",
        "gyropercent": 15,
        "startaltitude": float(h0),
        "minaltitude": 1,
        "maxdistance": 100,
        "maxtime": 0,
    }


def particle_params(zenith, azimuth):
    return {
        "Anum": 1,
        "anti": "YES",
        "zenith": float(zenith),
        "azimuth": float(azimuth),
    }


def add_case(
    cases,
    lat,
    lon,
    h0,
    zenith,
    azimuth,
    scan_type,
    r_start=None,
    r_end=None,
    window_metadata=None,
):
    key = make_case_key(lat, lon, h0, zenith, azimuth)
    if r_start is None or r_end is None:
        r_start, r_end = rigidity_window(zenith, scan_type)

    if key not in cases:
        cases[key] = {
            "lat": float(lat),
            "lon": float(lon),
            "h0": float(h0),
            "zenith": float(zenith),
            "azimuth": float(azimuth),
            "R_start": r_start,
            "R_end": r_end,
            "dR": DR,
            "rigidityscan": "OFF",
            "case_key": key,
            "membership_scans": set(),
            "R_tolerance_min": float(REGULAR_TOLERANCE_MIN_GV),
            "R_tolerance_max": float(REGULAR_TOLERANCE_MAX_GV),
        }
        if window_metadata:
            cases[key].update(window_metadata)

    cases[key]["membership_scans"].add(scan_type)
    return key


def build_regular_cases_and_plot_cells():
    cases = {}
    plot_cells = []
    plot_cell_counter = 0

    def append_plot_cell(scan_type, case_key, lat, lon, h0, zenith, azimuth,
                         plot_lat, plot_lon, plot_h0, plot_zenith, plot_azimuth,
                         figure_group):
        nonlocal plot_cell_counter
        plot_cell_counter += 1
        plot_cells.append({
            "plot_cell_id": f"pc_{plot_cell_counter:04d}",
            "scan_type": scan_type,
            "case_key": case_key,
            "lat": cases[case_key]["lat"],
            "lon": cases[case_key]["lon"],
            "h0": cases[case_key]["h0"],
            "zenith": cases[case_key]["zenith"],
            "azimuth": cases[case_key]["azimuth"],
            "plot_lat": float(plot_lat),
            "plot_lon": float(plot_lon),
            "plot_h0": float(plot_h0),
            "plot_zenith": float(plot_zenith),
            "plot_azimuth": float(plot_azimuth),
            "figure_group": figure_group,
        })

    # A. geographic scan
    for lat in LAT_VALUES:
        for lon in LON_VALUES:
            for zenith, azimuth in GEO_DIRECTIONS:
                case_key = add_case(
                    cases, lat, lon, GEO_H0, zenith, azimuth, "geographic"
                )
                append_plot_cell(
                    scan_type="geographic",
                    case_key=case_key,
                    lat=lat,
                    lon=lon,
                    h0=GEO_H0,
                    zenith=zenith,
                    azimuth=azimuth,
                    plot_lat=lat,
                    plot_lon=lon,
                    plot_h0=GEO_H0,
                    plot_zenith=zenith,
                    plot_azimuth=azimuth,
                    figure_group=f"geo_z{int(zenith)}_az{int(azimuth)}",
                )

    # B. angular scan (Asunción)
    for plot_zenith in ZENITH_PLOT_VALUES:
        run_azimuth = 0.0 if plot_zenith in (0.0, 180.0) else None
        for plot_azimuth in AZIMUTH_PLOT_VALUES:
            zenith = plot_zenith
            azimuth = run_azimuth if run_azimuth is not None else plot_azimuth
            case_key = add_case(
                cases,
                ASUNCION_LAT,
                ASUNCION_LON,
                GEO_H0,
                zenith,
                azimuth,
                "angular",
            )
            append_plot_cell(
                scan_type="angular",
                case_key=case_key,
                lat=ASUNCION_LAT,
                lon=ASUNCION_LON,
                h0=GEO_H0,
                zenith=zenith,
                azimuth=azimuth,
                plot_lat=ASUNCION_LAT,
                plot_lon=ASUNCION_PLOT_LON_0_360,
                plot_h0=GEO_H0,
                plot_zenith=plot_zenith,
                plot_azimuth=plot_azimuth,
                figure_group="angular_asuncion",
            )

    # C. altitude scan
    for lat in LAT_VALUES:
        for lon in LON_VALUES:
            for h0 in ALT_H0_VALUES:
                case_key = add_case(
                    cases, lat, lon, h0, 0.0, 0.0, "altitude"
                )
                append_plot_cell(
                    scan_type="altitude",
                    case_key=case_key,
                    lat=lat,
                    lon=lon,
                    h0=cases[case_key]["h0"],
                    zenith=0.0,
                    azimuth=0.0,
                    plot_lat=lat,
                    plot_lon=lon,
                    plot_h0=h0,
                    plot_zenith=0.0,
                    plot_azimuth=0.0,
                    figure_group=f"alt_h{int(h0)}",
                )

    return cases, plot_cells, None


def build_interpolated_cases_and_plot_cells():
    """Build one case for every valid interpolated coordinate combination."""
    grid = load_boundary_file(MATLAB_BOUNDARY_PATH)
    r_low, r_high, valid, _ = clean_windows(grid["Rl"], grid["Ru"])
    grid_plot_windows = densify_windows(grid, r_low, r_high, valid)
    physical_windows = collapse_vertical_azimuths(grid_plot_windows)
    physical_windows = add_scan_bounds(
        physical_windows,
        margin_low_gv=WINDOW_MARGIN_LOW_GV,
        margin_high_gv=WINDOW_MARGIN_HIGH_GV,
        rigidity_floor_gv=RIGIDITY_FLOOR_GV,
        rigidity_step_gv=DR,
    )
    physical_windows["R_tolerance_min"] = np.maximum(
        RIGIDITY_FLOOR_GV,
        np.floor(
            (physical_windows["R_low"] - TOTAL_TOLERANCE_LOW_GV) / DR + 1e-12
        ) * DR,
    ).round(12)
    physical_windows["R_tolerance_max"] = (
        np.ceil(
            (physical_windows["R_high"] + TOTAL_TOLERANCE_HIGH_GV) / DR - 1e-12
        ) * DR
    ).round(12)
    if (physical_windows["R_tolerance_min"] > physical_windows["R_end"]).any():
        raise ValueError("Lower tolerance limit is inside the initial scan window")
    if (physical_windows["R_tolerance_max"] < physical_windows["R_start"]).any():
        raise ValueError("Upper tolerance limit is inside the initial scan window")

    cases = {}
    for h0 in INTERPOLATED_H0_VALUES:
        for row in physical_windows.itertuples(index=False):
            add_case(
                cases,
                row.lat,
                row.lon,
                h0,
                row.zenith,
                row.azimuth,
                "interpolated_grid",
                r_start=row.R_start,
                r_end=row.R_end,
                window_metadata={
                    "R_low_estimated": float(row.R_low),
                    "R_high_estimated": float(row.R_high),
                    "R_tolerance_min": float(row.R_tolerance_min),
                    "R_tolerance_max": float(row.R_tolerance_max),
                    "window_source": row.window_source,
                    "window_status": row.window_status,
                    "scan_eligible": True,
                },
            )

    plot_cells = []
    counter = 0
    def append_cells(windows, h0, scan_type, figure_group):
        nonlocal counter
        for row in windows.itertuples(index=False):
            counter += 1
            run_azimuth = 0.0 if row.zenith in (0.0, 180.0) else row.azimuth
            case_key = make_case_key(row.lat, row.lon, h0, row.zenith, run_azimuth)
            if case_key not in cases:
                raise KeyError(f"No physical case for interpolated plot cell: {case_key}")
            plot_cells.append({
                "plot_cell_id": f"pc_{counter:06d}",
                "scan_type": scan_type,
                "case_key": case_key,
                "lat": float(row.lat),
                "lon": float(row.lon),
                "h0": float(h0),
                "zenith": float(row.zenith),
                "azimuth": float(run_azimuth),
                "plot_lat": float(row.lat),
                "plot_lon": float(row.lon),
                "plot_h0": float(h0),
                "plot_zenith": float(row.zenith),
                "plot_azimuth": float(row.azimuth),
                "figure_group": figure_group,
            })

    for h0 in INTERPOLATED_H0_VALUES:
        append_cells(
            grid_plot_windows,
            float(h0),
            "interpolated_grid",
            "interpolated_5d_grid",
        )
    audit = pd.concat(
        [physical_windows.assign(h0=float(h0)) for h0 in INTERPOLATED_H0_VALUES],
        ignore_index=True,
    )
    audit["scan_eligible"] = True
    return cases, plot_cells, audit


def build_cases_and_plot_cells():
    if GRID_SOURCE == "regular":
        return build_regular_cases_and_plot_cells()
    if GRID_SOURCE == "matlab_interpolated":
        return build_interpolated_cases_and_plot_cells()
    raise ValueError(f"Unknown GRID_SOURCE: {GRID_SOURCE!r}")


def cases_to_dataframe(cases):
    rows = []
    for idx, (key, case) in enumerate(sorted(cases.items()), start=1):
        row = {
            "case_id": f"case_{idx:04d}",
            "lat": case["lat"],
            "lon": case["lon"],
            "h0": case["h0"],
            "zenith": case["zenith"],
            "azimuth": case["azimuth"],
            "R_start": case["R_start"],
            "R_end": case["R_end"],
            "dR": case["dR"],
            "rigidityscan": case["rigidityscan"],
            "case_key": key,
            "membership_scans": ";".join(sorted(case["membership_scans"])),
        }
        for name in (
            "R_low_estimated",
            "R_high_estimated",
            "R_tolerance_min",
            "R_tolerance_max",
            "window_source",
            "window_status",
            "scan_eligible",
        ):
            if name in case:
                row[name] = case[name]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_cells_to_dataframe(plot_cells, case_plan_df):
    key_to_case_id = dict(zip(case_plan_df["case_key"], case_plan_df["case_id"]))
    rows = []
    for cell in plot_cells:
        row = dict(cell)
        row["case_id"] = key_to_case_id[row.pop("case_key")]
        rows.append(row)
    return pd.DataFrame(rows)


def select_stratified_test_plan(case_plan_df, plot_plan_df, n_cases):
    """Select a deterministic, distributed plan for an end-to-end test."""
    n_cases = int(n_cases)
    if n_cases <= 0:
        raise ValueError("STRATIFIED_TEST_CASES must be positive or None")
    if n_cases >= len(case_plan_df):
        return case_plan_df.copy(), plot_plan_df.copy()

    ordered = case_plan_df.sort_values(
        ["lat", "lon", "zenith", "azimuth"], kind="stable"
    ).reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, n_cases, dtype=int)
    selected = ordered.iloc[np.unique(indices)].copy()
    selected_ids = set(selected["case_id"])
    selected_plots = plot_plan_df[plot_plan_df["case_id"].isin(selected_ids)].copy()
    if set(selected_plots["case_id"]) != selected_ids:
        missing = selected_ids - set(selected_plots["case_id"])
        raise ValueError(f"Selected cases without plot cells: {sorted(missing)[:5]}")
    return selected, selected_plots


def parse_otso_cone(cone_df):
    df = cone_df.copy()
    df["R [GV]"] = pd.to_numeric(df["R [GV]"], errors="coerce")
    df = df[df["R [GV]"].notna()].copy()

    station_col = None
    for col in df.columns:
        if col == "R [GV]":
            continue
        if df[col].astype(str).str.contains(";", regex=False).any():
            station_col = col
            break

    if station_col is None:
        raise ValueError("Could not find station column with 'filter;lat;lon' format.")

    parts = df[station_col].astype(str).str.split(";", expand=True)
    df["raw_filter"] = pd.to_numeric(parts[0], errors="coerce")
    df = df[df["raw_filter"].notna()].copy()
    df = df.sort_values("R [GV]")
    return df[["R [GV]", "raw_filter"]].rename(columns={"R [GV]": "R_GV"})


def run_otso_case(case_row):
    payload = {
        "lat": float(case_row["lat"]),
        "lon": float(case_row["lon"]),
        "h0": float(case_row["h0"]),
        "zenith": float(case_row["zenith"]),
        "azimuth": float(case_row["azimuth"]),
        "R_start": float(case_row["R_start"]),
        "R_end": float(case_row["R_end"]),
        "dR": float(case_row["dR"]),
    }
    with tempfile.TemporaryDirectory(prefix="penumbra_stage1_") as temp_dir:
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--otso-worker",
                json.dumps(payload),
                temp_dir,
            ],
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()[-2000:]
            raise RuntimeError(
                f"OTSO worker failed with exit code {proc.returncode}: {detail}"
            )
        temp_path = Path(temp_dir)
        parsed = pd.read_csv(temp_path / "parsed.csv")
        summary_path = temp_path / "summary.txt"
        summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    return parsed, summary_text


def otso_worker(payload_text: str, output_dir: str) -> None:
    payload = json.loads(payload_text)
    rigidity_params = {
        "startrigidity": payload["R_start"],
        "endrigidity": payload["R_end"],
        "rigiditystep": payload["dR"],
        "rigidityscan": "OFF",
    }

    cone_df, cutoff_df, summary_text = OTSO.cone(
        Stations=[],
        customlocations=[[LOCATION_NAME, payload["lat"], payload["lon"]]],
        datetime_params=DATETIME_PARAMS,
        rigidity_params=rigidity_params,
        integration_params=integration_params(payload["h0"]),
        particle_params=particle_params(payload["zenith"], payload["azimuth"]),
        coordinate_params=COORDINATE_PARAMS,
        computation_params=COMPUTATION_PARAMS,
    )
    parsed = parse_otso_cone(cone_df)
    out = Path(output_dir)
    parsed.to_csv(out / "parsed.csv", index=False)
    if summary_text:
        (out / "summary.txt").write_text(summary_text, encoding="utf-8")


def format_filter_counts(series):
    counts = series.value_counts(dropna=False).sort_index()
    return ",".join(f"{int(k) if float(k).is_integer() else k}:{int(v)}" for k, v in counts.items())


def build_status_row(case_row, parsed_df, status, error_message=""):
    if parsed_df.empty:
        return {
            "case_id": case_row["case_id"],
            "lat": case_row["lat"],
            "lon": case_row["lon"],
            "h0": case_row["h0"],
            "zenith": case_row["zenith"],
            "azimuth": case_row["azimuth"],
            "R_start": case_row["R_start"],
            "R_end": case_row["R_end"],
            "n_points": 0,
            "raw_filter_counts": "",
            "lowest_R": float("nan"),
            "highest_R": float("nan"),
            "raw_filter_at_lowest_R": float("nan"),
            "raw_filter_at_highest_R": float("nan"),
            "starts_black_single": float("nan"),
            "ends_white_single": float("nan"),
            "status": status,
            "error_message": error_message,
        }

    lowest_idx = parsed_df["R_GV"].idxmin()
    highest_idx = parsed_df["R_GV"].idxmax()
    rf_low = parsed_df.loc[lowest_idx, "raw_filter"]
    rf_high = parsed_df.loc[highest_idx, "raw_filter"]

    return {
        "case_id": case_row["case_id"],
        "lat": case_row["lat"],
        "lon": case_row["lon"],
        "h0": case_row["h0"],
        "zenith": case_row["zenith"],
        "azimuth": case_row["azimuth"],
        "R_start": case_row["R_start"],
        "R_end": case_row["R_end"],
        "n_points": int(len(parsed_df)),
        "raw_filter_counts": format_filter_counts(parsed_df["raw_filter"]),
        "lowest_R": float(parsed_df.loc[lowest_idx, "R_GV"]),
        "highest_R": float(parsed_df.loc[highest_idx, "R_GV"]),
        "raw_filter_at_lowest_R": float(rf_low),
        "raw_filter_at_highest_R": float(rf_high),
        "starts_black_single": bool(rf_low != 1),
        "ends_white_single": bool(rf_high == 1),
        "status": status,
        "error_message": error_message,
    }


def load_completed_case_ids(output_dir):
    status_path = output_dir / "case_status_initial.csv"
    raw_path = output_dir / "raw_barcodes_initial.csv"
    completed = set()

    if status_path.exists():
        status_df = pd.read_csv(status_path)
        completed.update(
            status_df.loc[
                status_df["status"].isin(["completed", "not_scanned"]), "case_id"
            ].astype(str)
        )

    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        completed.update(raw_df["case_id"].astype(str).unique())

    return completed


def append_csv(df, path):
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def update_case_status_file(output_dir, status_row):
    path = output_dir / "case_status_initial.csv"
    if path.exists():
        status_df = pd.read_csv(path)
        status_df = status_df[status_df["case_id"] != status_row["case_id"]]
        status_df = pd.concat([status_df, pd.DataFrame([status_row])], ignore_index=True)
    else:
        status_df = pd.DataFrame([status_row])
    status_df.to_csv(path, index=False)


def write_run_parameters(output_dir, case_plan_df, plot_plan_df, first_summary_text):
    counts_by_scan = defaultdict(int)
    for membership in case_plan_df["membership_scans"]:
        for scan in membership.split(";"):
            counts_by_scan[scan] += 1

    actual_lat = sorted(case_plan_df["lat"].dropna().unique().tolist())
    actual_lon = sorted(case_plan_df["lon"].dropna().unique().tolist())
    actual_h0 = sorted(case_plan_df["h0"].dropna().unique().tolist())
    actual_zenith = sorted(case_plan_df["zenith"].dropna().unique().tolist())
    actual_azimuth = sorted(case_plan_df["azimuth"].dropna().unique().tolist())

    lines = [
        f"creation_time: {datetime.now().isoformat()}",
        f"grid_source: {GRID_SOURCE}",
        f"python_executable: {sys.executable}",
        f"otso_path: {OTSO.__file__}",
        f"otso_version: {INSTALLED_OTSO_VERSION}",
        "magfield_params: NOT PASSED (OTSO default)",
        "geomagnetic: NOT PASSED",
        "expected_models_from_first_successful_case:",
    ]

    if first_summary_text:
        for line in first_summary_text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("Internal Model", "External Model", "Magnetopause Model")):
                lines.append(f"  {stripped}")

    lines.extend([
        "",
        "raw_filter convention:",
        "  white / allowed / escaped: raw_filter == 1",
        "  black / forbidden: raw_filter != 1",
        "",
        f"actual_lat_values: {actual_lat}",
        f"actual_lon_values: {actual_lon}",
        f"actual_h0_values: {actual_h0}",
        f"actual_zenith_values: {actual_zenith}",
        f"actual_azimuth_values: {actual_azimuth}",
        "",
        "rigidity windows:",
        "  regular mode geographic zenith=0: 20 -> 0.5 GV",
        "  regular mode geographic zenith>0: 30 -> 0.5 GV",
        "  regular mode angular (all): 30 -> 0.5 GV",
        "  regular mode altitude vertical: 20 -> 0.5 GV",
        "  matlab_interpolated mode: R_high+margin_high -> "
        "max(R_low-margin_low, rigidity_floor)",
        f"rigidity step: {DR} GV",
        "rigidityscan: OFF",
        "",
        f"asuncion_physical_coords: lat={ASUNCION_LAT}, lon={ASUNCION_LON}",
        f"asuncion_plot_lon_0_360: {ASUNCION_PLOT_LON_0_360}",
        "",
        f"total_plot_cells: {len(plot_plan_df)}",
        f"total_unique_physical_cases: {len(case_plan_df)}",
        f"cases_by_scan_type: {dict(counts_by_scan)}",
        "",
        "deduplication:",
        "  unique physical key = lat, lon, h0, zenith, azimuth (6 decimals)",
        "  at zenith=0/180, all plot azimuths reuse the azimuth=0 physical case",
        "",
        f"MAX_CASES: {MAX_CASES}",
        f"STRATIFIED_TEST_CASES: {STRATIFIED_TEST_CASES}",
        f"RESUME_RUN_ID: {RESUME_RUN_ID}",
        f"matlab_boundary_path: {MATLAB_BOUNDARY_PATH}",
        "interpolated_grid_has_altitude_axis: False",
        f"interpolated_h0_values: {INTERPOLATED_H0_VALUES}",
        f"window_margin_low_gv: {WINDOW_MARGIN_LOW_GV}",
        f"window_margin_high_gv: {WINDOW_MARGIN_HIGH_GV}",
        f"total_tolerance_low_gv: {TOTAL_TOLERANCE_LOW_GV}",
        f"total_tolerance_high_gv: {TOTAL_TOLERANCE_HIGH_GV}",
        f"rigidity_floor_gv: {RIGIDITY_FLOOR_GV}",
    ])

    (output_dir / "run_parameters.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    start_time = datetime.now()
    t0 = time.time()

    RUNS_BASE.mkdir(parents=True, exist_ok=True)

    if RESUME_RUN_ID:
        output_dir = RUNS_BASE / RESUME_RUN_ID
        if not output_dir.exists():
            raise FileNotFoundError(f"Resume run directory not found: {output_dir}")
        run_id = RESUME_RUN_ID
        print(f"Resuming run: {run_id}")
        case_plan_df = pd.read_csv(output_dir / "case_plan.csv")
        plot_plan_df = pd.read_csv(output_dir / "plot_cell_plan.csv")
    else:
        run_id = f"run_{start_time.strftime('%Y%m%d_%H%M%S')}_default_TSY89c_Kobel_raw"
        output_dir = RUNS_BASE / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        cases, plot_cells, interpolated_windows = build_cases_and_plot_cells()
        case_plan_df = cases_to_dataframe(cases)
        plot_plan_df = plot_cells_to_dataframe(plot_cells, case_plan_df)

        if STRATIFIED_TEST_CASES is not None:
            case_plan_df.to_csv(output_dir / "case_plan_full.csv", index=False)
            plot_plan_df.to_csv(output_dir / "plot_cell_plan_full.csv", index=False)
            case_plan_df, plot_plan_df = select_stratified_test_plan(
                case_plan_df, plot_plan_df, STRATIFIED_TEST_CASES
            )

        case_plan_df.to_csv(output_dir / "case_plan.csv", index=False)
        plot_plan_df.to_csv(output_dir / "plot_cell_plan.csv", index=False)
        if interpolated_windows is not None:
            interpolated_windows.to_csv(
                output_dir / "interpolated_rigidity_windows.csv", index=False
            )
        print(f"Created run directory: {output_dir}")
        print(f"Total plot cells: {len(plot_plan_df)}")
        print(f"Total unique cases: {len(case_plan_df)}")

    activate_run(output_dir)

    cases_to_run = case_plan_df.copy()
    if MAX_CASES is not None:
        cases_to_run = cases_to_run.head(int(MAX_CASES))

    completed_case_ids = load_completed_case_ids(output_dir)

    raw_path = output_dir / "raw_barcodes_initial.csv"
    errors_path = output_dir / "errors.csv"

    first_summary_saved = (output_dir / "otso_summary_first_case.txt").exists()
    first_summary_text = None

    completed_count = 0
    error_count = 0
    skipped_count = 0
    not_scanned_count = 0

    total_to_run = len(cases_to_run)

    for i, case_row in cases_to_run.iterrows():
        case_id = str(case_row["case_id"])

        if case_id in completed_case_ids:
            skipped_count += 1
            print(f"[skip] {case_id} already has raw data")
            continue

        scan_eligible = case_row.get("scan_eligible", True)
        if pd.isna(scan_eligible):
            scan_eligible = True
        if not bool(scan_eligible):
            not_scanned_count += 1
            reason = str(case_row.get("window_status", "interpolation_unavailable"))
            print(f"[not scanned] {case_id}: {reason}")
            status_row = build_status_row(
                case_row,
                pd.DataFrame(),
                status="not_scanned",
                error_message=reason,
            )
            update_case_status_file(output_dir, status_row)
            continue

        print(
            f"[run {completed_count + error_count + skipped_count + 1}/{total_to_run}] "
            f"{case_id} lat={case_row['lat']} lon={case_row['lon']} "
            f"h0={case_row['h0']} zen={case_row['zenith']} az={case_row['azimuth']} "
            f"R={case_row['R_start']}->{case_row['R_end']}"
        )

        try:
            parsed_df, summary_text = run_otso_case(case_row)

            if not first_summary_saved and summary_text:
                (output_dir / "otso_summary_first_case.txt").write_text(
                    summary_text, encoding="utf-8"
                )
                first_summary_text = summary_text
                first_summary_saved = True

            raw_out = parsed_df.copy()
            raw_out.insert(0, "case_id", case_id)
            raw_out.insert(1, "scan_membership", case_row["membership_scans"])
            raw_out.insert(2, "lat", case_row["lat"])
            raw_out.insert(3, "lon", case_row["lon"])
            raw_out.insert(4, "h0", case_row["h0"])
            raw_out.insert(5, "zenith", case_row["zenith"])
            raw_out.insert(6, "azimuth", case_row["azimuth"])

            append_csv(raw_out, raw_path)

            status_row = build_status_row(case_row, parsed_df, status="completed")
            update_case_status_file(output_dir, status_row)
            completed_count += 1

        except Exception as exc:
            error_count += 1
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"[error] {case_id}: {err_msg}")
            traceback.print_exc()

            status_row = build_status_row(case_row, pd.DataFrame(), status="error", error_message=err_msg)
            update_case_status_file(output_dir, status_row)

            err_row = {
                "case_id": case_id,
                "lat": case_row["lat"],
                "lon": case_row["lon"],
                "h0": case_row["h0"],
                "zenith": case_row["zenith"],
                "azimuth": case_row["azimuth"],
                "error_message": err_msg,
                "traceback": traceback.format_exc(),
            }
            append_csv(pd.DataFrame([err_row]), errors_path)

    end_time = datetime.now()
    elapsed = time.time() - t0

    if first_summary_text is None and (output_dir / "otso_summary_first_case.txt").exists():
        first_summary_text = (output_dir / "otso_summary_first_case.txt").read_text(encoding="utf-8")

    if not RESUME_RUN_ID or not (output_dir / "run_parameters.txt").exists():
        write_run_parameters(output_dir, case_plan_df, plot_plan_df, first_summary_text)

    runtime = {
        "run_id": run_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "total_cases": int(len(cases_to_run)),
        "completed_cases": int(completed_count),
        "skipped_cases": int(skipped_count),
        "error_cases": int(error_count),
        "not_scanned_cases": int(not_scanned_count),
        "max_cases_limit": MAX_CASES,
    }
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")

    print()
    print("=" * 70)
    print("RUN FINISHED")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Total plot cells: {len(plot_plan_df)}")
    print(f"Total unique cases in plan: {len(case_plan_df)}")
    print(f"Cases attempted this run: {total_to_run}")
    print(f"Completed this run: {completed_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Not scanned (no valid interpolated window): {not_scanned_count}")
    print(f"Errors this run: {error_count}")
    print(f"raw_barcodes_initial.csv: {raw_path}")
    print(f"case_status_initial.csv: {output_dir / 'case_status_initial.csv'}")
    print(f"Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--otso-worker":
        otso_worker(sys.argv[2], sys.argv[3])
    else:
        main()
