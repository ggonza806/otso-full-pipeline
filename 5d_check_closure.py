#!/usr/bin/env python3
"""
5d_check_closure.py — Stage 2 of the penumbra barcode pipeline.

Robust closure diagnosis: checks whether each barcode starts black and ends white.
Does NOT run OTSO, modify raw data, compute physical metrics, or plot.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_paths import resolve_run_dir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EDGE_N = 10

INPUT_RAW = "raw_barcodes_initial.csv"
INPUT_CASE_PLAN = "case_plan.csv"
INPUT_PLOT_CELL_PLAN = "plot_cell_plan.csv"
INPUT_CASE_STATUS = "case_status_initial.csv"

OUTPUT_CLOSURE_REPORT = "closure_report_initial.csv"
OUTPUT_OPEN_CASES = "open_cases_for_rerun.csv"
OUTPUT_SUMMARY = "closure_summary.txt"
OUTPUT_UNEXPECTED = "unexpected_case_ids.csv"

# raw_filter == 1 → white; raw_filter != 1 → black
WHITE_VALUE = 1


def run_dir() -> Path:
    return resolve_run_dir()


def is_black(raw_filter) -> bool:
    return raw_filter != WHITE_VALUE


def is_white(raw_filter) -> bool:
    return raw_filter == WHITE_VALUE


def filters_to_string(values) -> str:
    return ";".join(str(int(v) if float(v).is_integer() else v) for v in values)


def empty_extension_fields() -> dict:
    return {
        "needs_low_extension": False,
        "needs_high_extension": False,
        "low_extension_R_start": np.nan,
        "low_extension_R_end": np.nan,
        "high_extension_R_start": np.nan,
        "high_extension_R_end": np.nan,
    }


def _snap_up(value: float, step: float) -> float:
    return round(float(np.ceil(value / step - 1e-12) * step), 12)


def _snap_down(value: float, step: float) -> float:
    return round(float(np.floor(value / step + 1e-12) * step), 12)


def apply_extension_rules(status: str, plan_row: pd.Series) -> dict:
    out = empty_extension_fields()
    r_start = float(plan_row["R_start"])
    r_end = float(plan_row["R_end"])
    step = float(plan_row["dR"])
    tolerance_min = pd.to_numeric(
        plan_row.get("R_tolerance_min", np.nan), errors="coerce"
    )
    tolerance_max = pd.to_numeric(
        plan_row.get("R_tolerance_max", np.nan), errors="coerce"
    )
    has_tolerance_limits = np.isfinite(tolerance_min) and np.isfinite(tolerance_max)

    if status in ("open_low", "open_both"):
        if has_tolerance_limits:
            target_end = _snap_down(float(tolerance_min), step)
            if target_end < r_end - step / 2:
                out["needs_low_extension"] = True
                out["low_extension_R_start"] = r_end
                out["low_extension_R_end"] = target_end
        else:
            # Preserve the original policy for regular, non-interpolated runs.
            out["needs_low_extension"] = True
            out["low_extension_R_start"] = 0.5
            out["low_extension_R_end"] = 0.1

    if status in ("open_high", "open_both"):
        if has_tolerance_limits:
            target_start = _snap_up(float(tolerance_max), step)
            if target_start > r_start + step / 2:
                out["needs_high_extension"] = True
                out["high_extension_R_start"] = target_start
                out["high_extension_R_end"] = r_start
        else:
            # Preserve the original policy for regular, non-interpolated runs.
            out["needs_high_extension"] = True
            out["high_extension_R_start"] = 50.0
            out["high_extension_R_end"] = r_start

    return out


def base_row_from_plan(plan_row: pd.Series) -> dict:
    return {
        "case_id": plan_row["case_id"],
        "membership_scans": plan_row["membership_scans"],
        "lat": plan_row["lat"],
        "lon": plan_row["lon"],
        "h0": plan_row["h0"],
        "zenith": plan_row["zenith"],
        "azimuth": plan_row["azimuth"],
        "R_start": plan_row["R_start"],
        "R_end": plan_row["R_end"],
        "dR": plan_row["dR"],
        "R_low_estimated": plan_row.get("R_low_estimated", np.nan),
        "R_high_estimated": plan_row.get("R_high_estimated", np.nan),
        "R_tolerance_min": plan_row.get("R_tolerance_min", np.nan),
        "R_tolerance_max": plan_row.get("R_tolerance_max", np.nan),
    }


def empty_report_row(plan_row: pd.Series, notes: str) -> dict:
    row = base_row_from_plan(plan_row)
    row.update({
        "n_points": 0,
        "lowest_R": np.nan,
        "highest_R": np.nan,
        "raw_filter_at_lowest_R": np.nan,
        "raw_filter_at_highest_R": np.nan,
        "starts_black_single": np.nan,
        "ends_white_single": np.nan,
        "edge_n": EDGE_N,
        "low_edge_n": 0,
        "high_edge_n": 0,
        "low_edge_black_count": 0,
        "low_edge_white_count": 0,
        "high_edge_black_count": 0,
        "high_edge_white_count": 0,
        "low_edge_raw_filters": "",
        "high_edge_raw_filters": "",
        "starts_black_robust": np.nan,
        "ends_white_robust": np.nan,
        "status": "failed_or_empty",
        "notes": notes,
    })
    row.update(empty_extension_fields())
    return row


def diagnose_case(plan_row: pd.Series, case_df: pd.DataFrame | None) -> dict:
    if case_df is None or case_df.empty:
        return empty_report_row(plan_row, "case_id_missing_in_raw")

    df = case_df.copy()
    df["R_GV"] = pd.to_numeric(df["R_GV"], errors="coerce")
    df["raw_filter"] = pd.to_numeric(df["raw_filter"], errors="coerce")
    df = df[df["R_GV"].notna() & df["raw_filter"].notna()].copy()
    df = df.drop_duplicates(subset=["case_id", "R_GV"], keep="first")
    df = df.sort_values("R_GV")

    row = base_row_from_plan(plan_row)
    n_points = len(df)

    row.update({
        "n_points": n_points,
        "edge_n": EDGE_N,
        "low_edge_raw_filters": "",
        "high_edge_raw_filters": "",
        "notes": "",
    })
    row.update(empty_extension_fields())

    if n_points == 0:
        row["status"] = "failed_or_empty"
        row["notes"] = "manual_check_required"
        row.update({
            "lowest_R": np.nan,
            "highest_R": np.nan,
            "raw_filter_at_lowest_R": np.nan,
            "raw_filter_at_highest_R": np.nan,
            "starts_black_single": np.nan,
            "ends_white_single": np.nan,
            "low_edge_n": 0,
            "high_edge_n": 0,
            "low_edge_black_count": 0,
            "low_edge_white_count": 0,
            "high_edge_black_count": 0,
            "high_edge_white_count": 0,
            "starts_black_robust": np.nan,
            "ends_white_robust": np.nan,
        })
        return row

    lowest_idx = df["R_GV"].idxmin()
    highest_idx = df["R_GV"].idxmax()
    rf_low = df.loc[lowest_idx, "raw_filter"]
    rf_high = df.loc[highest_idx, "raw_filter"]

    low_edge = df.head(EDGE_N)
    high_edge = df.tail(EDGE_N)

    row.update({
        "lowest_R": float(df.loc[lowest_idx, "R_GV"]),
        "highest_R": float(df.loc[highest_idx, "R_GV"]),
        "raw_filter_at_lowest_R": float(rf_low),
        "raw_filter_at_highest_R": float(rf_high),
        "starts_black_single": bool(is_black(rf_low)),
        "ends_white_single": bool(is_white(rf_high)),
        "low_edge_n": len(low_edge),
        "high_edge_n": len(high_edge),
        "low_edge_black_count": int(low_edge["raw_filter"].apply(is_black).sum()),
        "low_edge_white_count": int(low_edge["raw_filter"].apply(is_white).sum()),
        "high_edge_black_count": int(high_edge["raw_filter"].apply(is_black).sum()),
        "high_edge_white_count": int(high_edge["raw_filter"].apply(is_white).sum()),
        "low_edge_raw_filters": filters_to_string(low_edge["raw_filter"].tolist()),
        "high_edge_raw_filters": filters_to_string(high_edge["raw_filter"].tolist()),
    })

    if n_points < EDGE_N:
        row.update({
            "starts_black_robust": np.nan,
            "ends_white_robust": np.nan,
            "status": "too_few_points",
            "notes": "manual_check_required",
        })
        return row

    starts_black_robust = bool(low_edge["raw_filter"].apply(is_black).all())
    ends_white_robust = bool(high_edge["raw_filter"].apply(is_white).all())

    row["starts_black_robust"] = starts_black_robust
    row["ends_white_robust"] = ends_white_robust

    if starts_black_robust and ends_white_robust:
        status = "closed"
    elif not starts_black_robust and ends_white_robust:
        status = "open_low"
    elif starts_black_robust and not ends_white_robust:
        status = "open_high"
    else:
        status = "open_both"

    row["status"] = status
    row.update(apply_extension_rules(status, plan_row))
    return row


def load_raw_grouped(raw_path: Path) -> tuple[dict[str, pd.DataFrame], set[str]]:
    print(f"Loading {raw_path.name} ...")
    raw = pd.read_csv(
        raw_path,
        usecols=["case_id", "R_GV", "raw_filter"],
        dtype={"case_id": str},
    )
    raw_ids = set(raw["case_id"].unique())
    print(f"  rows: {len(raw):,}  unique case_id: {len(raw_ids):,}")

    grouped = {case_id: group for case_id, group in raw.groupby("case_id", sort=False)}
    return grouped, raw_ids


def write_summary(
    output_path: Path,
    report_df: pd.DataFrame,
    plan_df: pd.DataFrame,
    raw_ids: set[str],
    unexpected_ids: set[str],
) -> None:
    status_counts = Counter(report_df["status"])
    lines = [
        f"creation_time: {datetime.now().isoformat()}",
        f"RUN_ID: {output_path.parent.name}",
        f"run_directory: {run_dir()}",
        f"EDGE_N: {EDGE_N}",
        "",
        "counts:",
        f"  total_cases_in_case_plan: {len(plan_df)}",
        f"  total_case_id_in_raw_barcodes: {len(raw_ids)}",
        f"  total_cases_diagnosed: {len(report_df)}",
        f"  closed: {status_counts.get('closed', 0)}",
        f"  open_low: {status_counts.get('open_low', 0)}",
        f"  open_high: {status_counts.get('open_high', 0)}",
        f"  open_both: {status_counts.get('open_both', 0)}",
        f"  too_few_points: {status_counts.get('too_few_points', 0)}",
        f"  failed_or_empty: {status_counts.get('failed_or_empty', 0)}",
        f"  needs_low_extension: {int(report_df['needs_low_extension'].sum())}",
        f"  needs_high_extension: {int(report_df['needs_high_extension'].sum())}",
        "",
    ]

    if unexpected_ids:
        lines.append(f"unexpected_case_ids_in_raw_not_in_plan: {len(unexpected_ids)}")
        lines.append("")

    lines.append("summary_by_membership_scans:")
    for membership, count in sorted(report_df["membership_scans"].value_counts().items()):
        lines.append(f"  {membership}: {count}")
    lines.append("")

    lines.append("summary_by_membership_scans_and_status:")
    cross = (
        report_df.groupby(["membership_scans", "status"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["membership_scans", "status"])
    )
    for _, r in cross.iterrows():
        lines.append(f"  {r['membership_scans']} | {r['status']}: {r['count']}")
    lines.append("")

    lines.append("summary_by_h0_zenith_azimuth_status:")
    combo = (
        report_df.groupby(["h0", "zenith", "azimuth", "status"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["h0", "zenith", "azimuth", "status"])
    )
    for _, r in combo.iterrows():
        lines.append(
            f"  h0={r['h0']} zen={r['zenith']} az={r['azimuth']} | {r['status']}: {r['count']}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    out = run_dir()
    if not out.is_dir():
        raise FileNotFoundError(f"Run directory not found: {out}")

    plan_path = out / INPUT_CASE_PLAN
    raw_path = out / INPUT_RAW

    for path in (plan_path, raw_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    print(f"Run directory: {out}")
    print(f"EDGE_N: {EDGE_N}")
    print()

    plan_df = pd.read_csv(plan_path, dtype={"case_id": str})
    plan_ids = set(plan_df["case_id"])

    grouped, raw_ids = load_raw_grouped(raw_path)

    missing_in_raw = plan_ids - raw_ids
    unexpected_ids = raw_ids - plan_ids

    if missing_in_raw:
        print(f"WARNING: {len(missing_in_raw)} case_id(s) in case_plan missing from raw:")
        for cid in sorted(missing_in_raw)[:10]:
            print(f"  {cid}")
        if len(missing_in_raw) > 10:
            print(f"  ... and {len(missing_in_raw) - 10} more")
        print()

    if unexpected_ids:
        print(f"WARNING: {len(unexpected_ids)} case_id(s) in raw not present in case_plan:")
        for cid in sorted(unexpected_ids)[:10]:
            print(f"  {cid}")
        if len(unexpected_ids) > 10:
            print(f"  ... and {len(unexpected_ids) - 10} more")
        print()
        pd.DataFrame({"case_id": sorted(unexpected_ids)}).to_csv(
            out / OUTPUT_UNEXPECTED, index=False
        )

    reports = []
    for _, plan_row in plan_df.iterrows():
        case_id = plan_row["case_id"]
        case_df = grouped.get(case_id)
        reports.append(diagnose_case(plan_row, case_df))

    report_df = pd.DataFrame(reports)
    report_df = report_df.sort_values(
        ["membership_scans", "h0", "zenith", "azimuth", "lat", "lon"],
        kind="stable",
    )

    open_df = report_df[
        report_df["needs_low_extension"] | report_df["needs_high_extension"]
    ].copy()
    open_cols = [
        "case_id",
        "membership_scans",
        "lat",
        "lon",
        "h0",
        "zenith",
        "azimuth",
        "R_start",
        "R_end",
        "dR",
        "R_low_estimated",
        "R_high_estimated",
        "R_tolerance_min",
        "R_tolerance_max",
        "status",
        "needs_low_extension",
        "needs_high_extension",
        "low_extension_R_start",
        "low_extension_R_end",
        "high_extension_R_start",
        "high_extension_R_end",
    ]
    open_df = open_df[open_cols].sort_values(
        ["status", "h0", "zenith", "azimuth", "lat", "lon"],
        kind="stable",
    )

    closure_path = out / OUTPUT_CLOSURE_REPORT
    open_path = out / OUTPUT_OPEN_CASES
    summary_path = out / OUTPUT_SUMMARY

    report_df.to_csv(closure_path, index=False)
    open_df.to_csv(open_path, index=False)
    write_summary(summary_path, report_df, plan_df, raw_ids, unexpected_ids)

    print("=" * 70)
    print("CLOSURE CHECK FINISHED")
    print("=" * 70)
    print(f"Output directory: {out}")
    print(f"closure_report_initial.csv: {closure_path}")
    print(f"open_cases_for_rerun.csv: {open_path}")
    print(f"closure_summary.txt: {summary_path}")
    if unexpected_ids:
        print(f"unexpected_case_ids.csv: {out / OUTPUT_UNEXPECTED}")
    print()
    print(report_df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
