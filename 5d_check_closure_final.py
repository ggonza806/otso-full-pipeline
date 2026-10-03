#!/usr/bin/env python3
"""
5d_check_closure_final.py — Final robust closure check after Stage 3 merge.

Reads raw_barcodes_merged.csv; compares with Stage 2 closure_report_initial.csv.
Does NOT run OTSO, compute physical metrics, plot, or modify existing raw/closure files.
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

INPUT_RAW = "raw_barcodes_merged.csv"
INPUT_CASE_PLAN = "case_plan.csv"
INPUT_INITIAL_CLOSURE = "closure_report_initial.csv"
INPUT_OPEN_CASES = "open_cases_for_rerun.csv"
INPUT_EXTENSION_STATUS = "case_status_extensions.csv"
INPUT_PLOT_CELL_PLAN = "plot_cell_plan.csv"
INPUT_MERGE_SUMMARY = "merge_summary.txt"

OUTPUT_FINAL_CLOSURE = "closure_report_final.csv"
OUTPUT_REMAINING_OPEN = "remaining_open_cases_after_extensions.csv"
OUTPUT_DISCARDED = "discarded_cases_after_tolerance.csv"
OUTPUT_FINAL_SUMMARY = "closure_summary_final.txt"
OUTPUT_STATUS_TRANSITIONS = "closure_status_transitions.csv"
OUTPUT_UNEXPECTED_CASE_IDS = "unexpected_case_ids_final.csv"

OPEN_INITIAL_STATUSES = {"open_low", "open_high", "open_both"}
WHITE_VALUE = 1


def run_dir() -> Path:
    return resolve_run_dir()


def is_black(raw_filter) -> bool:
    return raw_filter != WHITE_VALUE


def is_white(raw_filter) -> bool:
    return raw_filter == WHITE_VALUE


def filters_to_string(values) -> str:
    return ";".join(str(int(v) if float(v).is_integer() else v) for v in values)


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
    }


def build_extension_lookup(ext_status_path: Path) -> tuple[dict[str, str], dict[str, bool]]:
    """case_id -> extension_type string; case_id -> was_extended."""
    if not ext_status_path.is_file():
        return {}, {}

    ext_df = pd.read_csv(ext_status_path, dtype={"case_id": str, "extension_id": str})
    ext_df = ext_df[ext_df["status"] == "completed"]

    lookup: dict[str, set[str]] = {}
    for _, row in ext_df.iterrows():
        lookup.setdefault(row["case_id"], set()).add(str(row["extension_type"]))

    type_map = {}
    extended_map = {}
    for case_id, types in lookup.items():
        type_map[case_id] = ";".join(sorted(types))
        extended_map[case_id] = True

    return type_map, extended_map


def diagnose_case_final(
    plan_row: pd.Series,
    case_df: pd.DataFrame | None,
    status_initial: str,
    extension_type: str,
    was_extended: bool,
) -> dict:
    was_open_initially = status_initial in OPEN_INITIAL_STATUSES

    if case_df is None or case_df.empty:
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
            "status_initial": status_initial,
            "status_final": "failed_or_empty",
            "was_open_initially": was_open_initially,
            "was_extended": was_extended,
            "extension_type": extension_type,
            "notes": "case_id_missing_in_merged_raw",
        })
        return row

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
        "status_initial": status_initial,
        "was_open_initially": was_open_initially,
        "was_extended": was_extended,
        "extension_type": extension_type,
        "notes": "",
    })

    if n_points == 0:
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
            "status_final": "failed_or_empty",
            "notes": "manual_check_required",
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
            "status_final": "too_few_points",
            "notes": "manual_check_required",
        })
        return row

    starts_black_robust = bool(low_edge["raw_filter"].apply(is_black).all())
    ends_white_robust = bool(high_edge["raw_filter"].apply(is_white).all())
    row["starts_black_robust"] = starts_black_robust
    row["ends_white_robust"] = ends_white_robust

    if starts_black_robust and ends_white_robust:
        status_final = "closed"
    elif not starts_black_robust and ends_white_robust:
        status_final = "open_low"
    elif starts_black_robust and not ends_white_robust:
        status_final = "open_high"
    else:
        status_final = "open_both"

    row["status_final"] = status_final
    row["pipeline_action"] = "keep" if status_final == "closed" else "discard"

    if not was_extended and status_initial != status_final:
        row["notes"] = "unexpected_status_change_without_extension"

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


def build_transition_tables(report_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    overall = (
        report_df.groupby(["status_initial", "status_final"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    overall.insert(0, "group_type", "overall")
    overall.insert(1, "membership_scans", "")
    overall.insert(2, "h0", np.nan)
    overall.insert(3, "zenith", np.nan)
    overall.insert(4, "azimuth", np.nan)
    rows.append(overall)

    by_membership = (
        report_df.groupby(["membership_scans", "status_initial", "status_final"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    by_membership.insert(0, "group_type", "membership_scans")
    by_membership.insert(2, "h0", np.nan)
    by_membership.insert(3, "zenith", np.nan)
    by_membership.insert(4, "azimuth", np.nan)
    rows.append(by_membership)

    by_geom = (
        report_df.groupby(
            ["h0", "zenith", "azimuth", "status_initial", "status_final"], dropna=False
        )
        .size()
        .reset_index(name="count")
    )
    by_geom.insert(0, "group_type", "geometry")
    by_geom.insert(1, "membership_scans", "")
    rows.append(by_geom)

    return pd.concat(rows, ignore_index=True)


def extension_stats(ext_status_path: Path) -> dict:
    if not ext_status_path.is_file():
        return {"total": 0, "high": 0, "low": 0, "failed": 0}

    ext_df = pd.read_csv(ext_status_path, dtype={"extension_id": str})
    return {
        "total": int((ext_df["status"] == "completed").sum()),
        "high": int(((ext_df["extension_type"] == "high") & (ext_df["status"] == "completed")).sum()),
        "low": int(((ext_df["extension_type"] == "low") & (ext_df["status"] == "completed")).sum()),
        "failed": int((ext_df["status"] == "error").sum()),
    }


def write_final_summary(
    out: Path,
    report_df: pd.DataFrame,
    plan_df: pd.DataFrame,
    raw_ids: set[str],
    unexpected_ids: set[str],
    ext_stats: dict,
    plot_summary: pd.DataFrame | None,
) -> None:
    final_counts = Counter(report_df["status_final"])
    initial_counts = Counter(report_df["status_initial"])

    open_high_to_closed = int(
        ((report_df["status_initial"] == "open_high") & (report_df["status_final"] == "closed")).sum()
    )
    open_low_to_closed = int(
        ((report_df["status_initial"] == "open_low") & (report_df["status_final"] == "closed")).sum()
    )
    still_open = int((report_df["status_final"] != "closed").sum())
    unexpected_non_extended = report_df[
        (~report_df["was_extended"]) & (report_df["status_initial"] != report_df["status_final"])
    ]

    lines = [
        f"creation_time: {datetime.now().isoformat()}",
        f"RUN_ID: {out.name}",
        f"run_directory: {out}",
        f"EDGE_N: {EDGE_N}",
        f"input_raw_file: {INPUT_RAW}",
        "",
        "general:",
        f"  total_cases_in_case_plan: {len(plan_df)}",
        f"  total_case_id_in_merged_raw: {len(raw_ids)}",
        f"  total_cases_diagnosed: {len(report_df)}",
        f"  closed_final: {final_counts.get('closed', 0)}",
        f"  open_low_final: {final_counts.get('open_low', 0)}",
        f"  open_high_final: {final_counts.get('open_high', 0)}",
        f"  open_both_final: {final_counts.get('open_both', 0)}",
        f"  too_few_points_final: {final_counts.get('too_few_points', 0)}",
        f"  failed_or_empty_final: {final_counts.get('failed_or_empty', 0)}",
        "",
        "stage2_comparison:",
        f"  closed_initial: {initial_counts.get('closed', 0)}",
        f"  open_high_initial: {initial_counts.get('open_high', 0)}",
        f"  open_low_initial: {initial_counts.get('open_low', 0)}",
        f"  open_both_initial: {initial_counts.get('open_both', 0)}",
        f"  open_high_initial_to_closed_final: {open_high_to_closed}",
        f"  open_low_initial_to_closed_final: {open_low_to_closed}",
        f"  remaining_open_final: {still_open}",
        f"  unexpected_changes_without_extension: {len(unexpected_non_extended)}",
        "",
        "extensions:",
        f"  total_extensions_completed: {ext_stats['total']}",
        f"  high_extensions_completed: {ext_stats['high']}",
        f"  low_extensions_completed: {ext_stats['low']}",
        f"  failed_extensions: {ext_stats['failed']}",
        "",
    ]

    if unexpected_ids:
        lines.append(f"unexpected_case_ids_in_merged_not_in_plan: {len(unexpected_ids)}")
        lines.append("")

    lines.append("summary_by_membership_scans_and_status_final:")
    cross = (
        report_df.groupby(["membership_scans", "status_final"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["membership_scans", "status_final"])
    )
    for _, r in cross.iterrows():
        lines.append(f"  {r['membership_scans']} | {r['status_final']}: {r['count']}")
    lines.append("")

    if plot_summary is not None and not plot_summary.empty:
        lines.append("summary_by_scan_type_and_status_final:")
        for _, r in plot_summary.iterrows():
            lines.append(
                f"  {r['scan_type']} | {r['status_final']}: "
                f"unique_cases={r['unique_case_count']} plot_cells={r['plot_cell_count']}"
            )
        lines.append("")

    lines.append("status_transitions_overall:")
    trans = (
        report_df.groupby(["status_initial", "status_final"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["status_initial", "status_final"])
    )
    for _, r in trans.iterrows():
        lines.append(f"  {r['status_initial']} -> {r['status_final']}: {r['count']}")

    (out / OUTPUT_FINAL_SUMMARY).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    out = run_dir()
    if not out.is_dir():
        raise FileNotFoundError(f"Run directory not found: {out}")

    required = [
        out / INPUT_RAW,
        out / INPUT_CASE_PLAN,
        out / INPUT_INITIAL_CLOSURE,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    print(f"Run directory: {out}")
    print(f"EDGE_N: {EDGE_N}")
    print(f"Input raw: {INPUT_RAW}")
    print()

    plan_df = pd.read_csv(out / INPUT_CASE_PLAN, dtype={"case_id": str})
    initial_closure = pd.read_csv(out / INPUT_INITIAL_CLOSURE, dtype={"case_id": str})
    status_initial_map = dict(zip(initial_closure["case_id"], initial_closure["status"]))

    ext_type_map, ext_extended_map = build_extension_lookup(out / INPUT_EXTENSION_STATUS)
    ext_stats = extension_stats(out / INPUT_EXTENSION_STATUS)

    grouped, raw_ids = load_raw_grouped(out / INPUT_RAW)
    plan_ids = set(plan_df["case_id"])

    missing_in_raw = plan_ids - raw_ids
    unexpected_ids = raw_ids - plan_ids

    if missing_in_raw:
        print(f"WARNING: {len(missing_in_raw)} case_id(s) in case_plan missing from merged raw")
        for cid in sorted(missing_in_raw)[:10]:
            print(f"  {cid}")
        if len(missing_in_raw) > 10:
            print(f"  ... and {len(missing_in_raw) - 10} more")
        print()

    if unexpected_ids:
        print(f"WARNING: {len(unexpected_ids)} case_id(s) in merged raw not in case_plan")
        pd.DataFrame({"case_id": sorted(unexpected_ids)}).to_csv(
            out / OUTPUT_UNEXPECTED_CASE_IDS, index=False
        )
        print()

    reports = []
    for _, plan_row in plan_df.iterrows():
        case_id = plan_row["case_id"]
        reports.append(
            diagnose_case_final(
                plan_row,
                grouped.get(case_id),
                status_initial_map.get(case_id, "unknown"),
                ext_type_map.get(case_id, ""),
                ext_extended_map.get(case_id, False),
            )
        )

    report_df = pd.DataFrame(reports)
    # The interpolated-grid policy has one bounded extension pass. Anything
    # that remains non-closed is retained for audit but excluded downstream.
    report_df["pipeline_action"] = np.where(
        report_df["status_final"] == "closed", "keep", "discard"
    )
    report_df = report_df.sort_values(
        ["membership_scans", "h0", "zenith", "azimuth", "lat", "lon"],
        kind="stable",
    )

    remaining_cols = [
        "case_id", "membership_scans", "lat", "lon", "h0", "zenith", "azimuth",
        "R_start", "R_end", "dR", "status_initial", "status_final",
        "lowest_R", "highest_R", "raw_filter_at_lowest_R", "raw_filter_at_highest_R",
        "low_edge_raw_filters", "high_edge_raw_filters",
        "starts_black_robust", "ends_white_robust", "notes",
        "pipeline_action",
    ]
    remaining_df = report_df[report_df["status_final"] != "closed"][remaining_cols].copy()
    remaining_df = remaining_df.sort_values(
        ["status_final", "h0", "zenith", "azimuth", "lat", "lon"],
        kind="stable",
    )
    discarded_df = report_df[report_df["pipeline_action"] == "discard"].copy()

    transitions_df = build_transition_tables(report_df)

    plot_summary = None
    plot_path = out / INPUT_PLOT_CELL_PLAN
    if plot_path.is_file():
        plot = pd.read_csv(plot_path, dtype={"case_id": str})
        plot_join = plot.merge(
            report_df[["case_id", "status_final"]], on="case_id", how="left"
        )
        plot_summary = (
            plot_join.groupby(["scan_type", "status_final"])
            .agg(unique_case_count=("case_id", "nunique"), plot_cell_count=("plot_cell_id", "count"))
            .reset_index()
            .sort_values(["scan_type", "status_final"])
        )

    closure_final_path = out / OUTPUT_FINAL_CLOSURE
    remaining_path = out / OUTPUT_REMAINING_OPEN
    summary_path = out / OUTPUT_FINAL_SUMMARY
    transitions_path = out / OUTPUT_STATUS_TRANSITIONS

    report_df.to_csv(closure_final_path, index=False)
    remaining_df.to_csv(remaining_path, index=False)
    discarded_df.to_csv(out / OUTPUT_DISCARDED, index=False)
    transitions_df.to_csv(transitions_path, index=False)
    write_final_summary(out, report_df, plan_df, raw_ids, unexpected_ids, ext_stats, plot_summary)

    print("=" * 70)
    print("FINAL CLOSURE CHECK FINISHED")
    print("=" * 70)
    print(f"closure_report_final.csv: {closure_final_path}")
    print(f"remaining_open_cases_after_extensions.csv: {remaining_path}")
    print(f"discarded_cases_after_tolerance.csv: {out / OUTPUT_DISCARDED}")
    print(f"closure_summary_final.txt: {summary_path}")
    print(f"closure_status_transitions.csv: {transitions_path}")
    if unexpected_ids:
        print(f"unexpected_case_ids_final.csv: {out / OUTPUT_UNEXPECTED_CASE_IDS}")
    print()
    print(f"total cases: {len(report_df)}")
    print(f"closed final: {(report_df['status_final'] == 'closed').sum()}")
    print(f"remaining open final: {(report_df['status_final'] != 'closed').sum()}")
    print()
    print("transition table (status_initial -> status_final):")
    overall = transitions_df[transitions_df["group_type"] == "overall"][
        ["status_initial", "status_final", "count"]
    ]
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
