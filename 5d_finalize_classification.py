#!/usr/bin/env python3
"""
5d_finalize_classification.py — Final closure + classification after Stage 3c (low2).

Reads raw_barcodes_merged_final.csv; assigns final_status and plot policies per case/plot cell.
Does NOT run OTSO, compute penumbra metrics, plot, or modify prior pipeline files.
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

INPUT_RAW_FINAL = "raw_barcodes_merged_final.csv"
INPUT_CASE_PLAN = "case_plan.csv"
INPUT_PLOT_CELL_PLAN = "plot_cell_plan.csv"
INPUT_INITIAL_CLOSURE = "closure_report_initial.csv"
INPUT_CLOSURE_BEFORE_LOW2 = "closure_report_final.csv"
INPUT_REMAINING_DIAGNOSIS = "remaining_open_diagnosis.csv"
INPUT_LOW2_STATUS = "case_status_second_extensions.csv"

OUTPUT_CLOSURE_AFTER_LOW2 = "closure_report_after_low2.csv"
OUTPUT_CASE_CLASSIFICATION = "final_case_classification.csv"
OUTPUT_PLOT_CLASSIFICATION = "final_plot_cell_classification.csv"
OUTPUT_SUMMARY = "final_classification_summary.txt"
OUTPUT_REMAINING_UNRESOLVED = "remaining_unresolved_cases_final.csv"

WHITE_VALUE = 1
EXPECTED_CASES = 818
EXPECTED_PLOT_CELLS = 1020


def run_dir() -> Path:
    return resolve_run_dir()


def resolve_raw_final(out: Path) -> Path:
    """Use the low2 merge when present, otherwise the Stage 3 merge."""
    preferred = out / INPUT_RAW_FINAL
    fallback = out / "raw_barcodes_merged.csv"
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"No final raw barcode file found: tried {preferred.name} and {fallback.name}"
    )


def is_black(raw_filter) -> bool:
    return raw_filter != WHITE_VALUE


def is_white(raw_filter) -> bool:
    return raw_filter == WHITE_VALUE


def filters_to_string(values) -> str:
    return ";".join(str(int(v) if float(v).is_integer() else v) for v in values)


def compute_closure_status(plan_row: pd.Series, case_df: pd.DataFrame | None) -> dict:
    base = {
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
        "edge_n": EDGE_N,
    }

    if case_df is None or case_df.empty:
        base.update({
            "n_points": 0,
            "lowest_R": np.nan,
            "highest_R": np.nan,
            "raw_filter_at_lowest_R": np.nan,
            "raw_filter_at_highest_R": np.nan,
            "starts_black_single": np.nan,
            "ends_white_single": np.nan,
            "low_edge_raw_filters": "",
            "high_edge_raw_filters": "",
            "starts_black_robust": np.nan,
            "ends_white_robust": np.nan,
            "status_after_low2": "failed_or_empty",
            "notes": "case_id_missing_in_merged_final",
        })
        return base

    df = case_df.copy()
    df["R_GV"] = pd.to_numeric(df["R_GV"], errors="coerce")
    df["raw_filter"] = pd.to_numeric(df["raw_filter"], errors="coerce")
    df = df[df["R_GV"].notna() & df["raw_filter"].notna()].copy()
    df = df.drop_duplicates(subset=["case_id", "R_GV"], keep="first")
    df = df.sort_values("R_GV")

    n_points = len(df)
    if n_points == 0:
        base.update({
            "n_points": 0,
            "lowest_R": np.nan,
            "highest_R": np.nan,
            "raw_filter_at_lowest_R": np.nan,
            "raw_filter_at_highest_R": np.nan,
            "starts_black_single": np.nan,
            "ends_white_single": np.nan,
            "low_edge_raw_filters": "",
            "high_edge_raw_filters": "",
            "starts_black_robust": np.nan,
            "ends_white_robust": np.nan,
            "status_after_low2": "failed_or_empty",
            "notes": "no_valid_points",
        })
        return base

    lowest_idx = df["R_GV"].idxmin()
    highest_idx = df["R_GV"].idxmax()
    rf_low = df.loc[lowest_idx, "raw_filter"]
    rf_high = df.loc[highest_idx, "raw_filter"]
    low_edge = df.head(EDGE_N)
    high_edge = df.tail(EDGE_N)

    base.update({
        "n_points": n_points,
        "lowest_R": float(df.loc[lowest_idx, "R_GV"]),
        "highest_R": float(df.loc[highest_idx, "R_GV"]),
        "raw_filter_at_lowest_R": float(rf_low),
        "raw_filter_at_highest_R": float(rf_high),
        "starts_black_single": bool(is_black(rf_low)),
        "ends_white_single": bool(is_white(rf_high)),
        "low_edge_raw_filters": filters_to_string(low_edge["raw_filter"].tolist()),
        "high_edge_raw_filters": filters_to_string(high_edge["raw_filter"].tolist()),
        "notes": "",
    })

    if n_points < EDGE_N:
        base.update({
            "starts_black_robust": np.nan,
            "ends_white_robust": np.nan,
            "status_after_low2": "too_few_points",
        })
        return base

    starts_black_robust = bool(low_edge["raw_filter"].apply(is_black).all())
    ends_white_robust = bool(high_edge["raw_filter"].apply(is_white).all())
    base["starts_black_robust"] = starts_black_robust
    base["ends_white_robust"] = ends_white_robust

    if starts_black_robust and ends_white_robust:
        status = "closed"
    elif not starts_black_robust and ends_white_robust:
        status = "open_low"
    elif starts_black_robust and not ends_white_robust:
        status = "open_high"
    else:
        status = "open_both"

    base["status_after_low2"] = status
    return base


def assign_final_classification(
    status_after_low2: str,
    preliminary_diagnosis: str,
    discard_after_tolerance: bool = False,
) -> tuple[str, str, float, bool, str]:
    if status_after_low2 == "closed":
        return "closed", "compute_metrics", np.nan, True, ""

    if discard_after_tolerance:
        return (
            "discarded_after_tolerance",
            "exclude",
            np.nan,
            False,
            f"closure_status={status_after_low2}",
        )

    if status_after_low2 in ("failed_or_empty", "too_few_points"):
        return "failed", "exclude_or_manual_review", np.nan, False, status_after_low2

    prelim = (preliminary_diagnosis or "").strip()

    if prelim == "all_forbidden_candidate":
        return "cone_all_forbidden", "fill_zero", 0.0, True, ""
    if prelim == "all_allowed_candidate":
        return "cone_all_allowed", "fill_zero", 0.0, True, ""
    if prelim == "earth_intersection_candidate":
        return "earth_intersection", "fill_zero", 0.0, True, ""
    if prelim == "low_extension_candidate" and status_after_low2 == "open_low":
        return "unresolved_low", "fill_zero", 0.0, True, "low2_did_not_close"
    if prelim == "nonstandard_low_mixed":
        return "unresolved_low", "fill_zero", 0.0, True, ""
    if prelim == "nonstandard_high_mixed":
        return "unresolved_high", "fill_zero", 0.0, True, ""
    if prelim == "high_extension_candidate":
        return "unresolved_high", "fill_zero", 0.0, True, ""

    return (
        "failed",
        "exclude_or_manual_review",
        np.nan,
        False,
        f"unmapped_open_status={status_after_low2};prelim={prelim or 'missing'}",
    )


def load_raw_grouped(raw_path: Path) -> dict[str, pd.DataFrame]:
    print(f"Loading {raw_path.name} ...")
    raw = pd.read_csv(
        raw_path,
        usecols=["case_id", "R_GV", "raw_filter"],
        dtype={"case_id": str},
        low_memory=False,
    )
    print(f"  rows: {len(raw):,}  unique case_id: {raw['case_id'].nunique():,}")
    return {cid: grp for cid, grp in raw.groupby("case_id", sort=False)}


def write_summary(
    out: Path,
    case_df: pd.DataFrame,
    plot_df: pd.DataFrame,
    low2_ids: set[str],
) -> None:
    low2_cases = case_df[case_df["case_id"].isin(low2_ids)]
    low2_closed = int((low2_cases["status_after_low2"] == "closed").sum())
    low2_unresolved = int((low2_cases["final_status"] == "unresolved_low").sum())

    lines = [
        f"creation_time: {datetime.now().isoformat()}",
        f"RUN_ID: {out.name}",
        f"EDGE_N: {EDGE_N}",
        f"input_raw_final_file: {INPUT_RAW_FINAL}",
        "",
        f"total_cases: {len(case_df)}",
        f"total_plot_cells: {len(plot_df)}",
        "",
        "status_after_low2:",
    ]
    for k, v in sorted(Counter(case_df["status_after_low2"]).items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("final_status:")
    for k, v in sorted(Counter(case_df["final_status"]).items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("plot_metric_policy:")
    for k, v in sorted(Counter(case_df["plot_metric_policy"]).items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("include_in_plots:")
    for k, v in sorted(Counter(case_df["include_in_plots"]).items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("final_status_by_membership_scans:")
    cross = (
        case_df.groupby(["membership_scans", "final_status"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["membership_scans", "final_status"])
    )
    for _, r in cross.iterrows():
        lines.append(f"  {r['membership_scans']} | {r['final_status']}: {r['count']}")
    lines.append("")
    lines.append("final_status_by_scan_type (plot cells):")
    scan_cross = (
        plot_df.groupby(["scan_type", "final_status"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["scan_type", "final_status"])
    )
    for _, r in scan_cross.iterrows():
        lines.append(f"  {r['scan_type']} | {r['final_status']}: {r['count']}")
    lines.append("")
    lines.append("low2_outcomes:")
    lines.append(f"  low2_cases: {len(low2_cases)}")
    lines.append(f"  low2_passed_to_closed: {low2_closed}")
    lines.append(f"  low2_remaining_unresolved_low: {low2_unresolved}")
    lines.append("")
    lines.append("plot_fill_zero_policy:")
    lines.append(
        "  fill_zero applies to final_status in: "
        "cone_all_forbidden, cone_all_allowed, earth_intersection, "
        "unresolved_low, unresolved_high"
    )
    lines.append(
        f"  cases_with_fill_zero: "
        f"{int((case_df['plot_metric_policy'] == 'fill_zero').sum())}"
    )
    lines.append(
        "  These use penumbra_metric_fill_value=0 in plots; "
        "final_status preserves physical distinction."
    )

    failed_n = int((case_df["final_status"] == "failed").sum())
    if failed_n:
        lines.append("")
        lines.append(f"WARNING: {failed_n} case(s) with final_status=failed")

    (out / OUTPUT_SUMMARY).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    out = run_dir()
    raw_final_path = resolve_raw_final(out)
    print(f"Run directory: {out}")
    print(f"EDGE_N: {EDGE_N}")
    print()

    required = [
        out / INPUT_CASE_PLAN,
        out / INPUT_PLOT_CELL_PLAN,
        out / INPUT_INITIAL_CLOSURE,
        out / INPUT_CLOSURE_BEFORE_LOW2,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    plan_df = pd.read_csv(out / INPUT_CASE_PLAN, dtype={"case_id": str})
    plot_plan = pd.read_csv(out / INPUT_PLOT_CELL_PLAN, dtype={"case_id": str, "plot_cell_id": str})
    initial_closure = pd.read_csv(out / INPUT_INITIAL_CLOSURE, dtype={"case_id": str})
    before_low2 = pd.read_csv(out / INPUT_CLOSURE_BEFORE_LOW2, dtype={"case_id": str})
    diagnosis_path = out / INPUT_REMAINING_DIAGNOSIS
    if diagnosis_path.is_file():
        diagnosis = pd.read_csv(diagnosis_path, dtype={"case_id": str})
    else:
        diagnosis = pd.DataFrame(
            columns=["case_id", "preliminary_diagnosis", "suggested_next_action"]
        )

    discard_after_tolerance = {
        "R_tolerance_min", "R_tolerance_max"
    }.issubset(plan_df.columns)
    if discard_after_tolerance:
        print("Policy: discard every case still open after configured tolerance")

    low2_ids: set[str] = set()
    if (out / INPUT_LOW2_STATUS).is_file():
        low2_status = pd.read_csv(out / INPUT_LOW2_STATUS, dtype={"case_id": str})
        low2_ids = set(low2_status.loc[low2_status["status"] == "completed", "case_id"])

    status_initial_map = dict(zip(initial_closure["case_id"], initial_closure["status"]))
    status_before_map = dict(zip(before_low2["case_id"], before_low2["status_final"]))
    diag_map = diagnosis.set_index("case_id")

    grouped = load_raw_grouped(raw_final_path)

    closure_rows = []
    for _, plan_row in plan_df.iterrows():
        closure_rows.append(compute_closure_status(plan_row, grouped.get(plan_row["case_id"])))

    closure_df = pd.DataFrame(closure_rows)
    closure_df.to_csv(out / OUTPUT_CLOSURE_AFTER_LOW2, index=False)

    case_rows = []
    for _, closure_row in closure_df.iterrows():
        case_id = closure_row["case_id"]
        prelim = ""
        suggested = ""
        if case_id in diag_map.index:
            prelim = str(diag_map.loc[case_id, "preliminary_diagnosis"])
            suggested = str(diag_map.loc[case_id, "suggested_next_action"])

        final_status, policy, fill_val, include, notes = assign_final_classification(
            closure_row["status_after_low2"],
            prelim,
            discard_after_tolerance=discard_after_tolerance,
        )
        extra_notes = closure_row.get("notes", "") or ""
        if notes and extra_notes:
            combined_notes = f"{extra_notes}; {notes}"
        else:
            combined_notes = notes or extra_notes

        case_rows.append({
            "case_id": case_id,
            "membership_scans": closure_row["membership_scans"],
            "lat": closure_row["lat"],
            "lon": closure_row["lon"],
            "h0": closure_row["h0"],
            "zenith": closure_row["zenith"],
            "azimuth": closure_row["azimuth"],
            "R_start": closure_row["R_start"],
            "R_end": closure_row["R_end"],
            "dR": closure_row["dR"],
            "status_initial": status_initial_map.get(case_id, "unknown"),
            "status_before_low2": status_before_map.get(case_id, "unknown"),
            "status_after_low2": closure_row["status_after_low2"],
            "preliminary_diagnosis": prelim,
            "suggested_next_action": suggested,
            "final_status": final_status,
            "plot_metric_policy": policy,
            "penumbra_metric_fill_value": fill_val,
            "include_in_plots": include,
            "n_points": closure_row["n_points"],
            "lowest_R": closure_row["lowest_R"],
            "highest_R": closure_row["highest_R"],
            "raw_filter_at_lowest_R": closure_row["raw_filter_at_lowest_R"],
            "raw_filter_at_highest_R": closure_row["raw_filter_at_highest_R"],
            "low_edge_raw_filters": closure_row["low_edge_raw_filters"],
            "high_edge_raw_filters": closure_row["high_edge_raw_filters"],
            "notes": combined_notes,
        })

    case_class = pd.DataFrame(case_rows)
    case_class = case_class.sort_values(
        ["membership_scans", "h0", "zenith", "azimuth", "lat", "lon"],
        kind="stable",
    )

    unresolved_statuses = {
        "unresolved_low", "unresolved_high", "failed",
        "open_low", "open_high", "open_both",
    }
    remaining_unresolved = case_class[
        case_class["final_status"].isin(unresolved_statuses)
        | case_class["status_after_low2"].isin({"open_low", "open_high", "open_both"})
    ].copy()
    if remaining_unresolved.empty:
        remaining_unresolved = case_class[case_class["final_status"] != "closed"].copy()

    plot_class = plot_plan.merge(
        case_class[[
            "case_id", "lat", "lon", "h0", "zenith", "azimuth",
            "final_status", "plot_metric_policy", "penumbra_metric_fill_value", "include_in_plots",
        ]],
        on="case_id",
        how="left",
        suffixes=("", "_case"),
    )

    plot_cols = [
        "plot_cell_id", "scan_type", "figure_group", "case_id",
        "plot_lat", "plot_lon", "plot_h0", "plot_zenith", "plot_azimuth",
        "lat", "lon", "h0", "zenith", "azimuth",
        "final_status", "plot_metric_policy", "penumbra_metric_fill_value", "include_in_plots",
    ]
    plot_class = plot_class[plot_cols].sort_values(
        ["scan_type", "figure_group", "plot_h0", "plot_zenith", "plot_azimuth", "plot_lat", "plot_lon"],
        kind="stable",
    )

    # Validations
    if not discard_after_tolerance and len(case_class) != EXPECTED_CASES:
        print(f"WARNING: expected {EXPECTED_CASES} cases, got {len(case_class)}")
    if not discard_after_tolerance and len(plot_class) != EXPECTED_PLOT_CELLS:
        print(f"WARNING: expected {EXPECTED_PLOT_CELLS} plot cells, got {len(plot_class)}")

    if case_class["final_status"].isna().any() or (case_class["final_status"] == "").any():
        raise ValueError("Empty final_status detected in case classification")
    if case_class["plot_metric_policy"].isna().any() or (case_class["plot_metric_policy"] == "").any():
        raise ValueError("Empty plot_metric_policy detected")

    missing_cases = set(plan_df["case_id"]) - set(case_class["case_id"])
    if missing_cases:
        raise ValueError(f"Missing case_id in classification: {sorted(missing_cases)[:5]}")

    missing_plots = set(plot_plan["plot_cell_id"]) - set(plot_class["plot_cell_id"])
    if missing_plots:
        raise ValueError(f"Missing plot_cell_id: {len(missing_plots)}")

    failed_n = int((case_class["final_status"] == "failed").sum())
    if failed_n:
        print(f"WARNING: {failed_n} case(s) with final_status=failed")

    case_path = out / OUTPUT_CASE_CLASSIFICATION
    plot_path = out / OUTPUT_PLOT_CLASSIFICATION
    summary_path = out / OUTPUT_SUMMARY
    unresolved_path = out / OUTPUT_REMAINING_UNRESOLVED

    case_class.to_csv(case_path, index=False)
    plot_class.to_csv(plot_path, index=False)
    remaining_unresolved.to_csv(unresolved_path, index=False)
    write_summary(out, case_class, plot_class, low2_ids)

    print("=" * 70)
    print("FINAL CLASSIFICATION FINISHED")
    print("=" * 70)
    print(f"closure_report_after_low2.csv: {out / OUTPUT_CLOSURE_AFTER_LOW2}")
    print(f"final_case_classification.csv: {case_path}")
    print(f"final_plot_cell_classification.csv: {plot_path}")
    print(f"final_classification_summary.txt: {summary_path}")
    print(f"remaining_unresolved_cases_final.csv: {unresolved_path}")
    print()
    print("status_after_low2:")
    print(case_class["status_after_low2"].value_counts().to_string())
    print()
    print("final_status:")
    print(case_class["final_status"].value_counts().to_string())
    print()
    print("plot_metric_policy:")
    print(case_class["plot_metric_policy"].value_counts().to_string())


if __name__ == "__main__":
    main()
