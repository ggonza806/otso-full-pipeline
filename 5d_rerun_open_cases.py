#!/usr/bin/env python3
"""
5d_rerun_open_cases.py — Stage 3 of the penumbra barcode pipeline.

Selective OTSO reruns for open cases (tolerance-limited rigidity extensions),
then merge with initial raw. Extension limits are calculated by Stage 2.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
import OTSO

from pipeline_paths import resolve_run_dir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_OPEN_CASES = "open_cases_for_rerun.csv"
INPUT_RAW_INITIAL = "raw_barcodes_initial.csv"
INPUT_CASE_PLAN = "case_plan.csv"

OUTPUT_EXTENSION_PLAN = "extension_plan.csv"
OUTPUT_RAW_EXTENSIONS = "raw_barcodes_extensions.csv"
OUTPUT_CASE_STATUS_EXTENSIONS = "case_status_extensions.csv"
OUTPUT_RAW_MERGED = "raw_barcodes_merged.csv"
OUTPUT_MERGE_SUMMARY = "merge_summary.txt"
OUTPUT_EXTENSION_RUNTIME = "extension_runtime.json"
OUTPUT_EXTENSION_ERRORS = "extension_errors.csv"
OUTPUT_DUPLICATE_CONFLICTS = "duplicate_conflicts.csv"

TEST_MAX_EXTENSIONS = None  # set to int for testing; None = run all

DR = 0.01
RIGIDITYSCAN = "OFF"
LOCATION_NAME = "X"
MAX_OTSO_CHUNK_WIDTH_GV = 1.5
MIN_OTSO_CHUNK_SPAN_GV = 0.10  # 11 points at dR=0.01; shorter calls can segfault

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


def run_dir() -> Path:
    return resolve_run_dir()


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
    return df.sort_values("R [GV]")[["R [GV]", "raw_filter"]].rename(columns={"R [GV]": "R_GV"})


def build_extension_plan(open_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in open_df.iterrows():
        base = {
            "case_id": row["case_id"],
            "membership_scans": row["membership_scans"],
            "lat": row["lat"],
            "lon": row["lon"],
            "h0": row["h0"],
            "zenith": row["zenith"],
            "azimuth": row["azimuth"],
            "R_original_start": row["R_start"],
            "R_original_end": row["R_end"],
            "R_low_estimated": row.get("R_low_estimated", np.nan),
            "R_high_estimated": row.get("R_high_estimated", np.nan),
            "dR": row["dR"],
            "rigidityscan": RIGIDITYSCAN,
            "status_from_stage2": row["status"],
            "needs_low_extension": row["needs_low_extension"],
            "needs_high_extension": row["needs_high_extension"],
        }

        if bool(row["needs_high_extension"]):
            ext = dict(base)
            ext.update({
                "extension_id": f"{row['case_id']}__high",
                "extension_type": "high",
                "extension_R_start": row["high_extension_R_start"],
                "extension_R_end": row["high_extension_R_end"],
            })
            rows.append(ext)

        if bool(row["needs_low_extension"]):
            ext = dict(base)
            ext.update({
                "extension_id": f"{row['case_id']}__low",
                "extension_type": "low",
                "extension_R_start": row["low_extension_R_start"],
                "extension_R_end": row["low_extension_R_end"],
            })
            rows.append(ext)

    columns = [
        "case_id", "membership_scans", "lat", "lon", "h0", "zenith",
        "azimuth", "R_original_start", "R_original_end", "R_low_estimated",
        "R_high_estimated", "dR", "rigidityscan", "status_from_stage2",
        "needs_low_extension", "needs_high_extension", "extension_id",
        "extension_type", "extension_R_start", "extension_R_end",
    ]
    plan = pd.DataFrame(rows, columns=columns)
    if not plan.empty:
        for col in ("extension_R_start", "extension_R_end", "dR"):
            plan[col] = pd.to_numeric(plan[col], errors="coerce")
    return plan


def validate_inputs(out: Path, open_df: pd.DataFrame, plan_df: pd.DataFrame, ext_plan: pd.DataFrame):
    required_paths = [
        out / INPUT_OPEN_CASES,
        out / INPUT_RAW_INITIAL,
        out / INPUT_CASE_PLAN,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    open_ids = set(open_df["case_id"].astype(str))
    plan_ids = set(plan_df["case_id"].astype(str))
    missing = open_ids - plan_ids
    if missing:
        raise ValueError(f"case_id in open_cases not found in case_plan: {sorted(missing)[:5]}")

    for _, row in ext_plan.iterrows():
        if pd.isna(row["extension_R_start"]) or pd.isna(row["extension_R_end"]):
            raise ValueError(f"Empty extension rigidity for {row['extension_id']}")
        if float(row["dR"]) != DR:
            raise ValueError(f"dR must be {DR} for {row['extension_id']}, got {row['dR']}")
        if not float(row["extension_R_start"]) > float(row["extension_R_end"]):
            raise ValueError(
                f"extension_R_start must be > extension_R_end for {row['extension_id']}: "
                f"{row['extension_R_start']} -> {row['extension_R_end']}"
            )


def load_completed_extensions(status_path: Path) -> set[str]:
    if not status_path.exists():
        return set()
    status_df = pd.read_csv(status_path, dtype={"extension_id": str})
    return set(status_df.loc[status_df["status"] == "completed", "extension_id"].astype(str))


def append_csv(df: pd.DataFrame, path: Path):
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def update_extension_status(path: Path, status_row: dict):
    if path.exists():
        status_df = pd.read_csv(path, dtype={"extension_id": str})
        status_df = status_df[status_df["extension_id"] != status_row["extension_id"]]
        status_df = pd.concat([status_df, pd.DataFrame([status_row])], ignore_index=True)
    else:
        status_df = pd.DataFrame([status_row])
    status_df.to_csv(path, index=False)


def remove_extension_raw_rows(raw_path: Path, extension_id: str):
    if not raw_path.exists():
        return
    raw_df = pd.read_csv(raw_path, dtype={"extension_id": str, "case_id": str})
    filtered = raw_df[raw_df["extension_id"] != extension_id]
    if len(filtered) == len(raw_df):
        return
    filtered.to_csv(raw_path, index=False)


def build_extension_status_row(ext_row: pd.Series, parsed_df: pd.DataFrame, status: str, error_message=""):
    if parsed_df.empty:
        return {
            "extension_id": ext_row["extension_id"],
            "case_id": ext_row["case_id"],
            "extension_type": ext_row["extension_type"],
            "membership_scans": ext_row["membership_scans"],
            "lat": ext_row["lat"],
            "lon": ext_row["lon"],
            "h0": ext_row["h0"],
            "zenith": ext_row["zenith"],
            "azimuth": ext_row["azimuth"],
            "extension_R_start": ext_row["extension_R_start"],
            "extension_R_end": ext_row["extension_R_end"],
            "dR": ext_row["dR"],
            "n_points": 0,
            "lowest_R": np.nan,
            "highest_R": np.nan,
            "raw_filter_at_lowest_R": np.nan,
            "raw_filter_at_highest_R": np.nan,
            "starts_black_single": np.nan,
            "ends_white_single": np.nan,
            "status": status,
            "error_message": error_message,
        }

    lowest_idx = parsed_df["R_GV"].idxmin()
    highest_idx = parsed_df["R_GV"].idxmax()
    rf_low = parsed_df.loc[lowest_idx, "raw_filter"]
    rf_high = parsed_df.loc[highest_idx, "raw_filter"]

    return {
        "extension_id": ext_row["extension_id"],
        "case_id": ext_row["case_id"],
        "extension_type": ext_row["extension_type"],
        "membership_scans": ext_row["membership_scans"],
        "lat": ext_row["lat"],
        "lon": ext_row["lon"],
        "h0": ext_row["h0"],
        "zenith": ext_row["zenith"],
        "azimuth": ext_row["azimuth"],
        "extension_R_start": ext_row["extension_R_start"],
        "extension_R_end": ext_row["extension_R_end"],
        "dR": ext_row["dR"],
        "n_points": int(len(parsed_df)),
        "lowest_R": float(parsed_df.loc[lowest_idx, "R_GV"]),
        "highest_R": float(parsed_df.loc[highest_idx, "R_GV"]),
        "raw_filter_at_lowest_R": float(rf_low),
        "raw_filter_at_highest_R": float(rf_high),
        "starts_black_single": bool(rf_low != 1),
        "ends_white_single": bool(rf_high == 1),
        "status": status,
        "error_message": error_message,
    }


def run_otso_extension(ext_row: pd.Series):
    start = float(ext_row["extension_R_start"])
    end = float(ext_row["extension_R_end"])
    step = float(ext_row["dR"])
    if start - end < MIN_OTSO_CHUNK_SPAN_GV - step / 2:
        if str(ext_row["extension_type"]) == "low":
            # Overlap the already-scanned initial window; do not extend beyond
            # the configured lower tolerance.
            start = round(end + MIN_OTSO_CHUNK_SPAN_GV, 12)
        else:
            end = round(start - MIN_OTSO_CHUNK_SPAN_GV, 12)
    chunks = []
    chunk_start = start

    while chunk_start > end + step / 2:
        chunk_end = max(end, chunk_start - MAX_OTSO_CHUNK_WIDTH_GV)
        chunk_end = round(np.ceil(chunk_end / step - 1e-12) * step, 12)
        if chunk_end >= chunk_start:
            chunk_end = round(chunk_start - step, 12)

        chunks.append(run_otso_chunk_isolated(ext_row, chunk_start, chunk_end, step))
        if chunk_end <= end + step / 2:
            break
        chunk_start = chunk_end

    if not chunks:
        return pd.DataFrame(columns=["R_GV", "raw_filter"])
    combined = pd.concat(chunks, ignore_index=True)
    combined["R_key"] = combined["R_GV"].round(6)
    combined = combined.drop_duplicates("R_key", keep="first")
    return combined.drop(columns="R_key").sort_values("R_GV")


def run_otso_chunk_isolated(ext_row, chunk_start, chunk_end, step):
    """Run one native OTSO call in a fresh process to isolate memory faults."""
    payload = {
        "lat": float(ext_row["lat"]),
        "lon": float(ext_row["lon"]),
        "h0": float(ext_row["h0"]),
        "zenith": float(ext_row["zenith"]),
        "azimuth": float(ext_row["azimuth"]),
        "start": float(chunk_start),
        "end": float(chunk_end),
        "step": float(step),
    }
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        output_path = Path(handle.name)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--otso-worker",
                json.dumps(payload),
                str(output_path),
            ],
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()[-2000:]
            raise RuntimeError(
                f"OTSO worker failed with exit code {proc.returncode}: {detail}"
            )
        return pd.read_csv(output_path)
    finally:
        output_path.unlink(missing_ok=True)


def otso_worker(payload_text: str, output_path: str) -> None:
    payload = json.loads(payload_text)
    cone_df, _cutoff_df, _summary = OTSO.cone(
        Stations=[],
        customlocations=[[LOCATION_NAME, payload["lat"], payload["lon"]]],
        datetime_params=DATETIME_PARAMS,
        rigidity_params={
            "startrigidity": payload["start"],
            "endrigidity": payload["end"],
            "rigiditystep": payload["step"],
            "rigidityscan": RIGIDITYSCAN,
        },
        integration_params=integration_params(payload["h0"]),
        particle_params=particle_params(payload["zenith"], payload["azimuth"]),
        coordinate_params=COORDINATE_PARAMS,
        computation_params=COMPUTATION_PARAMS,
    )
    parsed = parse_otso_cone(cone_df)
    parsed.to_csv(output_path, index=False)


def normalize_initial_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source"] = "initial"
    out["extension_type"] = ""
    out["scan_membership"] = out["scan_membership"].astype(str)
    return out[
        [
            "case_id", "scan_membership", "lat", "lon", "h0", "zenith", "azimuth",
            "R_GV", "raw_filter", "source", "extension_type",
        ]
    ]


def normalize_extension_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source"] = "extension"
    out["scan_membership"] = out["membership_scans"].astype(str)
    return out[
        [
            "case_id", "scan_membership", "lat", "lon", "h0", "zenith", "azimuth",
            "R_GV", "raw_filter", "source", "extension_type",
        ]
    ]


def merge_raw_barcodes(out: Path) -> dict:
    initial = pd.read_csv(out / INPUT_RAW_INITIAL, dtype={"case_id": str})
    initial_norm = normalize_initial_raw(initial)

    ext_path = out / OUTPUT_RAW_EXTENSIONS
    if ext_path.exists():
        extensions = pd.read_csv(ext_path, dtype={"extension_id": str, "case_id": str})
        ext_norm = normalize_extension_raw(extensions) if not extensions.empty else pd.DataFrame()
    else:
        ext_norm = pd.DataFrame()

    rows_initial = len(initial_norm)
    rows_extensions = len(ext_norm)
    combined = pd.concat([initial_norm, ext_norm], ignore_index=True)
    rows_combined = len(combined)

    combined["R_GV"] = pd.to_numeric(combined["R_GV"], errors="coerce")
    combined["raw_filter"] = pd.to_numeric(combined["raw_filter"], errors="coerce")
    combined["R_key"] = combined["R_GV"].round(6)

    source_priority = {"initial": 0, "extension": 1}
    combined["_source_rank"] = combined["source"].map(source_priority).fillna(9)
    combined = combined.sort_values(["case_id", "R_key", "_source_rank"], kind="stable")

    dup_group_sizes = combined.groupby(["case_id", "R_key"], sort=False).size()
    duplicate_count = int((dup_group_sizes > 1).sum())

    filter_counts = combined.groupby(["case_id", "R_key"], sort=False)["raw_filter"].nunique()
    conflict_keys = set(filter_counts[filter_counts > 1].index)

    if conflict_keys:
        conflict_mask = combined.set_index(["case_id", "R_key"]).index.isin(conflict_keys)
        conflicts_df = combined.loc[conflict_mask].copy()
        conflicts_df = conflicts_df.drop(columns=["_source_rank"], errors="ignore")
        conflicts_df.to_csv(out / OUTPUT_DUPLICATE_CONFLICTS, index=False)
    elif (out / OUTPUT_DUPLICATE_CONFLICTS).exists():
        (out / OUTPUT_DUPLICATE_CONFLICTS).unlink()
        conflicts_df = pd.DataFrame()
    else:
        conflicts_df = pd.DataFrame()

    merged = combined.drop_duplicates(subset=["case_id", "R_key"], keep="first")
    merged = merged.drop(columns=["R_key", "_source_rank"], errors="ignore")
    merged = merged.sort_values(["case_id", "R_GV"], kind="stable")
    merged.to_csv(out / OUTPUT_RAW_MERGED, index=False)

    return {
        "rows_initial": rows_initial,
        "rows_extensions": rows_extensions,
        "rows_combined_before_dedup": rows_combined,
        "duplicate_case_id_r_key_count": duplicate_count,
        "duplicate_conflicts_count": len(conflicts_df),
        "rows_final_merged": len(merged),
    }


def write_merge_summary(
    out: Path,
    ext_plan: pd.DataFrame,
    open_cases_df: pd.DataFrame,
    merge_stats: dict,
    exec_stats: dict,
    status_ext_path: Path,
):
    high_n = int((ext_plan["extension_type"] == "high").sum())
    low_n = int((ext_plan["extension_type"] == "low").sum())
    open_both_cases = int(
        open_cases_df.loc[
            open_cases_df["needs_low_extension"] & open_cases_df["needs_high_extension"],
            "case_id",
        ].nunique()
    )

    lines = [
        f"creation_time: {datetime.now().isoformat()}",
        f"RUN_ID: {out.name}",
        f"run_directory: {out}",
        f"python_executable: {sys.executable}",
        f"otso_path: {OTSO.__file__}",
        f"otso_version: {getattr(OTSO, '__version__', 'no __version__ found')}",
        "magfield_params: NOT PASSED",
        "geomagnetic: NOT PASSED",
        "",
        "input_files:",
        f"  {INPUT_RAW_INITIAL}",
        f"  {INPUT_OPEN_CASES}",
        "",
        "extension_plan:",
        "  policy: incremental segments within the Stage-2 tolerance limits",
        f"  total_extensions: {len(ext_plan)}",
        f"  high_extensions: {high_n}",
        f"  low_extensions: {low_n}",
        f"  open_both_cases_with_two_extensions: {open_both_cases}",
        "",
        "execution:",
        f"  completed_extensions: {exec_stats['completed']}",
        f"  failed_extensions: {exec_stats['failed']}",
        f"  skipped_extensions: {exec_stats['skipped']}",
        f"  total_rows_raw_barcodes_extensions: {exec_stats['raw_ext_rows']}",
        "",
        "merge:",
        f"  rows_initial: {merge_stats['rows_initial']}",
        f"  rows_extensions: {merge_stats['rows_extensions']}",
        f"  rows_combined_before_dedup: {merge_stats['rows_combined_before_dedup']}",
        f"  duplicate_case_id_plus_R_key_groups: {merge_stats['duplicate_case_id_r_key_count']}",
        f"  duplicate_conflicts_count: {merge_stats['duplicate_conflicts_count']}",
        f"  rows_final_merged: {merge_stats['rows_final_merged']}",
        f"  output: {OUTPUT_RAW_MERGED}",
        "",
    ]

    if merge_stats["duplicate_conflicts_count"] > 0:
        lines.append("WARNING: duplicate_conflicts detected — see duplicate_conflicts.csv")
        lines.append("")

    if status_ext_path.exists():
        st = pd.read_csv(status_ext_path, dtype={"extension_id": str})
        high_ok = st[(st["extension_type"] == "high") & (st["status"] == "completed")]
        low_ok = st[(st["extension_type"] == "low") & (st["status"] == "completed")]
        lines.append("extension_diagnostics:")
        lines.append(
            f"  high_extensions_with_raw_filter_at_highest_R_eq_1: "
            f"{int((high_ok['raw_filter_at_highest_R'] == 1).sum())} / {len(high_ok)}"
        )
        lines.append(
            f"  low_extensions_with_raw_filter_at_lowest_R_ne_1: "
            f"{int((low_ok['raw_filter_at_lowest_R'] != 1).sum())} / {len(low_ok)}"
        )

    (out / OUTPUT_MERGE_SUMMARY).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    start_time = datetime.now()
    t0 = time.time()

    out = run_dir()
    print(f"Run directory: {out}")
    print()

    open_df = pd.read_csv(out / INPUT_OPEN_CASES, dtype={"case_id": str})
    plan_df = pd.read_csv(out / INPUT_CASE_PLAN, dtype={"case_id": str})

    ext_plan = build_extension_plan(open_df)
    validate_inputs(out, open_df, plan_df, ext_plan)

    ext_plan_path = out / OUTPUT_EXTENSION_PLAN
    ext_plan.to_csv(ext_plan_path, index=False)

    high_total = int((ext_plan["extension_type"] == "high").sum())
    low_total = int((ext_plan["extension_type"] == "low").sum())
    print("Extension plan:")
    print(f"  total high: {high_total}")
    print(f"  total low:  {low_total}")
    print(f"  total planned: {len(ext_plan)}")
    print(f"  saved: {ext_plan_path}")
    print()

    to_run = ext_plan.copy()
    if TEST_MAX_EXTENSIONS is not None:
        to_run = to_run.head(int(TEST_MAX_EXTENSIONS))
        print(f"TEST_MAX_EXTENSIONS={TEST_MAX_EXTENSIONS} — running first {len(to_run)} only")
        print()

    raw_ext_path = out / OUTPUT_RAW_EXTENSIONS
    status_ext_path = out / OUTPUT_CASE_STATUS_EXTENSIONS
    errors_path = out / OUTPUT_EXTENSION_ERRORS

    completed_ids = load_completed_extensions(status_ext_path)
    completed_count = 0
    failed_count = 0
    skipped_count = 0

    for _, ext_row in to_run.iterrows():
        ext_id = str(ext_row["extension_id"])

        if ext_id in completed_ids:
            skipped_count += 1
            print(f"[skip] {ext_id} already completed")
            continue

        prev_status = None
        if status_ext_path.exists():
            prev = pd.read_csv(status_ext_path, dtype={"extension_id": str})
            prev_match = prev[prev["extension_id"] == ext_id]
            if not prev_match.empty:
                prev_status = prev_match.iloc[-1]["status"]

        if prev_status == "error":
            remove_extension_raw_rows(raw_ext_path, ext_id)

        print(
            f"[run] {ext_id} type={ext_row['extension_type']} "
            f"R={ext_row['extension_R_start']}->{ext_row['extension_R_end']} "
            f"lat={ext_row['lat']} lon={ext_row['lon']} "
            f"h0={ext_row['h0']} zen={ext_row['zenith']} az={ext_row['azimuth']}"
        )

        try:
            parsed = run_otso_extension(ext_row)
            if parsed.empty:
                raise ValueError("OTSO returned no valid rigidity points")

            raw_out = parsed.copy()
            raw_out.insert(0, "extension_id", ext_id)
            raw_out.insert(1, "case_id", ext_row["case_id"])
            raw_out.insert(2, "extension_type", ext_row["extension_type"])
            raw_out.insert(3, "membership_scans", ext_row["membership_scans"])
            raw_out.insert(4, "lat", ext_row["lat"])
            raw_out.insert(5, "lon", ext_row["lon"])
            raw_out.insert(6, "h0", ext_row["h0"])
            raw_out.insert(7, "zenith", ext_row["zenith"])
            raw_out.insert(8, "azimuth", ext_row["azimuth"])

            append_csv(raw_out, raw_ext_path)

            status_row = build_extension_status_row(ext_row, parsed, status="completed")
            update_extension_status(status_ext_path, status_row)
            completed_count += 1

        except Exception as exc:
            failed_count += 1
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"[error] {ext_id}: {err_msg}")
            traceback.print_exc()

            status_row = build_extension_status_row(
                ext_row, pd.DataFrame(), status="error", error_message=err_msg
            )
            update_extension_status(status_ext_path, status_row)

            append_csv(
                pd.DataFrame([{
                    "extension_id": ext_id,
                    "case_id": ext_row["case_id"],
                    "extension_type": ext_row["extension_type"],
                    "error_message": err_msg,
                }]),
                errors_path,
            )

    raw_ext_rows = 0
    if raw_ext_path.exists():
        raw_ext_rows = len(pd.read_csv(raw_ext_path))

    exec_stats = {
        "completed": completed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "raw_ext_rows": raw_ext_rows,
    }

    print()
    print("Merging raw barcodes ...")
    merge_stats = merge_raw_barcodes(out)

    end_time = datetime.now()
    elapsed = time.time() - t0

    write_merge_summary(out, ext_plan, open_df, merge_stats, exec_stats, status_ext_path)

    runtime = {
        "run_id": out.name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "total_extensions_planned": int(len(ext_plan)),
        "extensions_attempted_this_run": int(len(to_run)),
        "completed_extensions": completed_count,
        "failed_extensions": failed_count,
        "skipped_extensions": skipped_count,
        "test_max_extensions": TEST_MAX_EXTENSIONS,
    }
    (out / OUTPUT_EXTENSION_RUNTIME).write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")

    conflicts_count = merge_stats["duplicate_conflicts_count"]
    errors_count = failed_count
    if status_ext_path.exists():
        current_status = pd.read_csv(status_ext_path)
        errors_count = int((current_status["status"] == "error").sum())

    print("=" * 70)
    print("STAGE 3 FINISHED")
    print("=" * 70)
    print(f"Output directory: {out}")
    print(f"extension_plan.csv: {ext_plan_path}")
    print(f"raw_barcodes_extensions.csv: {raw_ext_path}")
    print(f"case_status_extensions.csv: {status_ext_path}")
    print(f"raw_barcodes_merged.csv: {out / OUTPUT_RAW_MERGED}")
    print(f"merge_summary.txt: {out / OUTPUT_MERGE_SUMMARY}")
    print(f"unresolved extension errors count: {errors_count}")
    print(f"duplicate conflicts count: {conflicts_count}")
    print(f"Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--otso-worker":
        otso_worker(sys.argv[2], sys.argv[3])
    else:
        main()
