#!/usr/bin/env python3
"""
5d_compute_penumbra_metrics.py — Penumbra metric computation after final classification.

Reads raw barcodes and final classification; writes penumbra metric tables.
Does NOT run OTSO, plot, or modify prior pipeline files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_paths import resolve_run_dir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_RAW_FINAL = "raw_barcodes_merged_final.csv"
INPUT_CASE_CLASSIFICATION = "final_case_classification.csv"
INPUT_PLOT_CELL_CLASSIFICATION = "final_plot_cell_classification.csv"
INPUT_CASE_PLAN = "case_plan.csv"
INPUT_PLOT_CELL_PLAN = "plot_cell_plan.csv"

OUTPUT_METRICS_BY_CASE = "penumbra_metrics_by_case.csv"
OUTPUT_METRICS_BY_PLOT_CELL = "penumbra_metrics_by_plot_cell.csv"
OUTPUT_SUMMARY = "penumbra_metrics_summary.txt"
OUTPUT_INTERVAL_DETAILS = "penumbra_interval_details.csv"

GRAY_LOOKAHEAD_K = 10
DR_STEP_EXPECTED = 0.01

WHITE_VALUE = 1
THRESH_100_GV = 0.1
THRESH_400_GV = 0.4

EXPECTED_CASES = 818
EXPECTED_PLOT_CELLS = 1020

METRIC_COLS = [
    "dR_total",
    "N_100",
    "N_400",
    "dR_white",
    "dR_black",
    "dR_gray",
    "f_gray",
]

DIAGNOSTIC_COLS = ["dR_eta_old"]

COUNT_COLS = [
    "n_raw_points",
    "n_penumbra_points",
    "n_intervals",
    "n_strips_total",
    "n_white_strips",
    "n_black_strips",
    "n_gray_strips",
]


def run_dir() -> Path:
    return resolve_run_dir()


def resolve_raw_final(out: Path) -> Path:
    preferred = out / INPUT_RAW_FINAL
    fallback = out / "raw_barcodes_merged.csv"
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"No final raw barcode file found: tried {preferred.name} and {fallback.name}"
    )


def base_label_from_filter(raw_filter: float) -> str:
    return "white" if raw_filter == WHITE_VALUE else "black"


def apply_gray_lookahead(labels: list[str], k: int) -> list[str]:
    n = len(labels)
    final = labels.copy()
    for i in range(n):
        lookahead = labels[i + 1 : i + k + 1]
        if any(lbl != labels[i] for lbl in lookahead):
            final[i] = "gray"
    return final


def intervals_to_strips(r_values: np.ndarray, final_labels: list[str]) -> list[dict]:
    n = len(r_values)
    if n < 2:
        return []

    strips: list[dict] = []
    strip_start = 0
    current_label = final_labels[0]

    for i in range(1, n - 1):
        if final_labels[i] != current_label:
            strips.append(
                {
                    "strip_label": current_label,
                    "strip_start_R": float(r_values[strip_start]),
                    "strip_end_R": float(r_values[i]),
                    "strip_width_GV": float(r_values[i] - r_values[strip_start]),
                }
            )
            strip_start = i
            current_label = final_labels[i]

    strips.append(
        {
            "strip_label": current_label,
            "strip_start_R": float(r_values[strip_start]),
            "strip_end_R": float(r_values[n - 1]),
            "strip_width_GV": float(r_values[n - 1] - r_values[strip_start]),
        }
    )
    return strips


def empty_metrics_row() -> dict:
    return {
        "metric_status": "",
        "R_L": np.nan,
        "R_U": np.nan,
        **{col: np.nan for col in METRIC_COLS},
        **{col: np.nan for col in DIAGNOSTIC_COLS},
        **{col: np.nan for col in COUNT_COLS},
        "notes": "",
    }


def fill_zero_metrics_row(n_raw_points: float | int | None = np.nan) -> dict:
    row = empty_metrics_row()
    row["metric_status"] = "fill_zero"
    row["R_L"] = np.nan
    row["R_U"] = np.nan
    for col in METRIC_COLS:
        row[col] = 0
    row["dR_eta_old"] = 0
    row["n_raw_points"] = n_raw_points
    for col in COUNT_COLS[1:]:
        row[col] = np.nan
    return row


def failed_metrics_row() -> dict:
    row = empty_metrics_row()
    row["metric_status"] = "failed"
    return row


def preprocess_case_raw(case_df: pd.DataFrame) -> pd.DataFrame:
    df = case_df.copy()
    df["R_GV"] = pd.to_numeric(df["R_GV"], errors="coerce")
    df["raw_filter"] = pd.to_numeric(df["raw_filter"], errors="coerce")
    df = df[df["R_GV"].notna() & df["raw_filter"].notna()].copy()
    df["R_GV_round"] = df["R_GV"].round(6)
    df = df.drop_duplicates(subset=["case_id", "R_GV_round"], keep="first")
    df = df.sort_values("R_GV").reset_index(drop=True)
    return df


def compute_closed_metrics(case_df: pd.DataFrame) -> tuple[dict, list[dict]]:
    df = preprocess_case_raw(case_df)
    n_raw = len(df)
    if n_raw == 0:
        row = empty_metrics_row()
        row["metric_status"] = "missing_raw"
        row["notes"] = "no_valid_points_after_preprocess"
        return row, []

    base_labels = [base_label_from_filter(v) for v in df["raw_filter"].tolist()]
    white_idx = [i for i, lbl in enumerate(base_labels) if lbl == "white"]
    black_idx = [i for i, lbl in enumerate(base_labels) if lbl == "black"]

    if not white_idx or not black_idx:
        row = empty_metrics_row()
        row["metric_status"] = "invalid_closed_no_boundary"
        row["n_raw_points"] = n_raw
        row["notes"] = "missing_white_or_black_boundary"
        return row, []

    r_first_white = float(df.loc[white_idx[0], "R_GV"])
    r_last_black = float(df.loc[black_idx[-1], "R_GV"])
    # Penumbra zone [R_L, R_U] uses rigidity-ordered bounds. When first white R
    # exceeds last black R (single-step transition), bounds swap automatically.
    r_l = min(r_first_white, r_last_black)
    r_u = max(r_first_white, r_last_black)

    if r_u <= r_l:
        row = empty_metrics_row()
        row["metric_status"] = "invalid_closed_bad_bounds"
        row["R_L"] = r_l
        row["R_U"] = r_u
        row["n_raw_points"] = n_raw
        row["notes"] = "R_U <= R_L"
        return row, []

    pen = df[(df["R_GV"] >= r_l) & (df["R_GV"] <= r_u)].copy().reset_index(drop=True)
    n_pen = len(pen)
    if n_pen < 2:
        row = empty_metrics_row()
        row["metric_status"] = "invalid_closed_bad_bounds"
        row["R_L"] = r_l
        row["R_U"] = r_u
        row["n_raw_points"] = n_raw
        row["n_penumbra_points"] = n_pen
        row["notes"] = "too_few_penumbra_points_for_intervals"
        return row, []

    pen_base = [base_label_from_filter(v) for v in pen["raw_filter"].tolist()]
    pen_final = apply_gray_lookahead(pen_base, GRAY_LOOKAHEAD_K)
    r_pen = pen["R_GV"].to_numpy(dtype=float)

    strips = intervals_to_strips(r_pen, pen_final)
    n_intervals = n_pen - 1

    dr_white = sum(s["strip_width_GV"] for s in strips if s["strip_label"] == "white")
    dr_black = sum(s["strip_width_GV"] for s in strips if s["strip_label"] == "black")
    dr_gray = sum(s["strip_width_GV"] for s in strips if s["strip_label"] == "gray")
    dr_total = r_u - r_l

    n_100 = sum(1 for s in strips if s["strip_width_GV"] >= THRESH_100_GV)
    n_400 = sum(1 for s in strips if s["strip_width_GV"] >= THRESH_400_GV)

    denom = dr_white + dr_black
    f_gray = dr_gray / dr_total if dr_total != 0 else 0.0
    dr_eta_old = dr_gray / denom if denom != 0 else 0.0

    n_white = sum(1 for s in strips if s["strip_label"] == "white")
    n_black = sum(1 for s in strips if s["strip_label"] == "black")
    n_gray = sum(1 for s in strips if s["strip_label"] == "gray")

    row = {
        "metric_status": "computed",
        "R_L": r_l,
        "R_U": r_u,
        "dR_total": dr_total,
        "N_100": n_100,
        "N_400": n_400,
        "dR_white": dr_white,
        "dR_black": dr_black,
        "dR_gray": dr_gray,
        "f_gray": f_gray,
        "dR_eta_old": dr_eta_old,
        "n_raw_points": n_raw,
        "n_penumbra_points": n_pen,
        "n_intervals": n_intervals,
        "n_strips_total": len(strips),
        "n_white_strips": n_white,
        "n_black_strips": n_black,
        "n_gray_strips": n_gray,
        "notes": "",
    }

    strip_rows = []
    for idx, strip in enumerate(strips):
        strip_rows.append(
            {
                "strip_index": idx,
                "strip_label": strip["strip_label"],
                "strip_start_R": strip["strip_start_R"],
                "strip_end_R": strip["strip_end_R"],
                "strip_width_GV": strip["strip_width_GV"],
                "is_ge_100MV": strip["strip_width_GV"] >= THRESH_100_GV,
                "is_ge_400MV": strip["strip_width_GV"] >= THRESH_400_GV,
            }
        )

    return row, strip_rows


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


def stats_block(values: pd.Series) -> str:
    if values.empty:
        return "  (no data)"
    return (
        f"  mean={values.mean():.6g}  min={values.min():.6g}  max={values.max():.6g}"
    )


def stats_block_with_median(values: pd.Series) -> str:
    if values.empty:
        return "  (no data)"
    return (
        f"  mean={values.mean():.6g}  median={values.median():.6g}  "
        f"min={values.min():.6g}  max={values.max():.6g}"
    )


def write_summary(
    out: Path,
    case_df: pd.DataFrame,
    plot_df: pd.DataFrame,
) -> None:
    computed = case_df[case_df["metric_status"] == "computed"]
    in_plots = case_df[case_df["include_in_plots"] == True]  # noqa: E712

    lines = [
        f"creation_time: {datetime.now().isoformat()}",
        f"RUN_ID: {out.name}",
        f"input raw file: {INPUT_RAW_FINAL}",
        f"GRAY_LOOKAHEAD_K: {GRAY_LOOKAHEAD_K}",
        f"DR_STEP_EXPECTED: {DR_STEP_EXPECTED}",
        "",
        "Counts:",
        f"  total cases: {len(case_df)}",
        f"  total plot cells: {len(plot_df)}",
        "",
        "metric_status counts:",
    ]
    for status, count in case_df["metric_status"].value_counts().sort_index().items():
        lines.append(f"  {status}: {count}")

    lines.append("")
    lines.append("final_status counts:")
    for status, count in case_df["final_status"].value_counts().sort_index().items():
        lines.append(f"  {status}: {count}")

    lines.append("")
    lines.append("plot_metric_policy counts:")
    for policy, count in case_df["plot_metric_policy"].value_counts().sort_index().items():
        lines.append(f"  {policy}: {count}")

    lines.append("")
    lines.append("For cases computed (metric_status=computed):")
    for col in METRIC_COLS:
        if col == "f_gray":
            lines.append(f"  {col}:{stats_block_with_median(computed[col])}")
        else:
            lines.append(f"  {col}:{stats_block(computed[col])}")

    lines.append("")
    lines.append("For all cases include_in_plots=True:")
    for col in METRIC_COLS:
        if col == "f_gray":
            lines.append(f"  {col}:{stats_block_with_median(in_plots[col])}")
        else:
            lines.append(f"  {col}:{stats_block(in_plots[col])}")

    lines.append("")
    lines.append("Strip counts (computed cases):")
    for col in ["n_strips_total", "n_white_strips", "n_black_strips", "n_gray_strips"]:
        vals = computed[col].dropna()
        if vals.empty:
            lines.append(f"  {col}: (no data)")
        else:
            lines.append(
                f"  {col}: sum={int(vals.sum())}  mean={vals.mean():.6g}  max={int(vals.max())}"
            )

    lines.append("")
    lines.append("Summary by scan_type (penumbra_metrics_by_plot_cell.csv):")
    scan_summary = (
        plot_df.groupby("scan_type", dropna=False)
        .agg(
            count=("plot_cell_id", "size"),
            mean_dR_total=("dR_total", "mean"),
            mean_dR_gray=("dR_gray", "mean"),
            mean_f_gray=("f_gray", "mean"),
        )
        .reset_index()
        .sort_values("scan_type")
    )
    for _, r in scan_summary.iterrows():
        lines.append(
            f"  {r['scan_type']}: count={int(r['count'])}  "
            f"mean_dR_total={r['mean_dR_total']:.6g}  "
            f"mean_dR_gray={r['mean_dR_gray']:.6g}  "
            f"mean_f_gray={r['mean_f_gray']:.6g}"
        )

    lines.append("")
    lines.append("Summary by final_status (penumbra_metrics_by_plot_cell.csv):")
    status_summary = (
        plot_df.groupby("final_status", dropna=False)
        .agg(
            count=("plot_cell_id", "size"),
            mean_dR_total=("dR_total", "mean"),
            mean_dR_gray=("dR_gray", "mean"),
            mean_f_gray=("f_gray", "mean"),
        )
        .reset_index()
        .sort_values("final_status")
    )
    for _, r in status_summary.iterrows():
        lines.append(
            f"  {r['final_status']}: count={int(r['count'])}  "
            f"mean_dR_total={r['mean_dR_total']:.6g}  "
            f"mean_dR_gray={r['mean_dR_gray']:.6g}  "
            f"mean_f_gray={r['mean_f_gray']:.6g}"
        )

    (out / OUTPUT_SUMMARY).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_consistency_checks(case_df: pd.DataFrame, plot_df: pd.DataFrame) -> None:
    interpolated_run = case_df["membership_scans"].astype(str).str.contains(
        "matlab_interpolated|interpolated_grid", na=False, regex=True
    ).any() or case_df["final_status"].astype(str).eq(
        "discarded_after_tolerance"
    ).any()
    if not interpolated_run and len(case_df) != EXPECTED_CASES:
        print(f"WARNING: expected {EXPECTED_CASES} cases, got {len(case_df)}")
    if not interpolated_run and len(plot_df) != EXPECTED_PLOT_CELLS:
        print(f"WARNING: expected {EXPECTED_PLOT_CELLS} plot cells, got {len(plot_df)}")

    fill_zero = case_df[case_df["plot_metric_policy"] == "fill_zero"]
    for col in METRIC_COLS:
        bad = fill_zero[~fill_zero[col].fillna(-1).eq(0)]
        if not bad.empty:
            print(f"WARNING: fill_zero cases with non-zero {col}: {len(bad)}")

    in_plots = case_df[case_df["include_in_plots"] == True]  # noqa: E712
    missing = in_plots[in_plots[METRIC_COLS].isna().any(axis=1)]
    if not missing.empty:
        print(f"WARNING: {len(missing)} include_in_plots=True rows with NaN metrics")

    closed_not_computed = case_df[
        (case_df["final_status"] == "closed") & (case_df["metric_status"] != "computed")
    ]
    if not closed_not_computed.empty:
        print(
            f"WARNING: {len(closed_not_computed)} closed case(s) without metric_status=computed"
        )
        for _, r in closed_not_computed.iterrows():
            print(f"  {r['case_id']}: {r['metric_status']}")

    computed = case_df[case_df["metric_status"] == "computed"]
    if not computed.empty:
        nan_f_gray = computed[computed["f_gray"].isna()]
        if not nan_f_gray.empty:
            print(f"WARNING: {len(nan_f_gray)} computed case(s) with NaN f_gray")

        bad_f_gray = computed[
            computed["f_gray"].notna()
            & ((computed["f_gray"] < 0) | (computed["f_gray"] > 1))
        ]
        if not bad_f_gray.empty:
            print(
                f"\n*** STRONG WARNING: {len(bad_f_gray)} computed case(s) "
                f"with f_gray outside [0,1] or NaN ***"
            )
            for _, r in bad_f_gray.iterrows():
                print(f"  {r['case_id']}: f_gray={r['f_gray']}")

    all_with_f_gray = case_df[case_df["f_gray"].notna()]
    if not all_with_f_gray.empty:
        fmin = all_with_f_gray["f_gray"].min()
        fmax = all_with_f_gray["f_gray"].max()
        if fmin < 0 or fmax > 1:
            print(
                f"\n*** STRONG WARNING: f_gray range [{fmin}, {fmax}] exceeds [0, 1] ***"
            )

    invalid_boundary = case_df[
        case_df["metric_status"].isin(
            ["invalid_closed_no_boundary", "invalid_closed_bad_bounds"]
        )
    ]
    if not invalid_boundary.empty:
        print(
            f"\n*** STRONG WARNING: {len(invalid_boundary)} case(s) with invalid closed bounds ***"
        )
        for _, r in invalid_boundary.iterrows():
            print(f"  {r['case_id']}: {r['metric_status']} — {r.get('notes', '')}")

    failed = case_df[case_df["metric_status"] == "failed"]
    if not failed.empty:
        print(f"\n*** STRONG WARNING: {len(failed)} case(s) with metric_status=failed ***")
        for _, r in failed.iterrows():
            print(f"  {r['case_id']}")


def main() -> None:
    out = run_dir()
    raw_final_path = resolve_raw_final(out)
    print(f"Run directory: {out}")
    print(f"GRAY_LOOKAHEAD_K: {GRAY_LOOKAHEAD_K}")
    print(f"DR_STEP_EXPECTED: {DR_STEP_EXPECTED}")
    print()

    required = [
        out / INPUT_CASE_CLASSIFICATION,
        out / INPUT_PLOT_CELL_CLASSIFICATION,
        out / INPUT_CASE_PLAN,
        out / INPUT_PLOT_CELL_PLAN,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    case_class = pd.read_csv(out / INPUT_CASE_CLASSIFICATION, dtype={"case_id": str})
    plot_class = pd.read_csv(
        out / INPUT_PLOT_CELL_CLASSIFICATION,
        dtype={"case_id": str, "plot_cell_id": str},
    )
    pd.read_csv(out / INPUT_CASE_PLAN, dtype={"case_id": str})
    pd.read_csv(out / INPUT_PLOT_CELL_PLAN, dtype={"case_id": str, "plot_cell_id": str})

    raw_grouped = load_raw_grouped(raw_final_path)

    case_rows: list[dict] = []
    interval_rows: list[dict] = []

    for _, cls_row in case_class.iterrows():
        case_id = cls_row["case_id"]
        final_status = cls_row["final_status"]
        policy = cls_row["plot_metric_policy"]
        include_in_plots = bool(cls_row["include_in_plots"])

        base = {
            "case_id": case_id,
            "membership_scans": cls_row["membership_scans"],
            "lat": cls_row["lat"],
            "lon": cls_row["lon"],
            "h0": cls_row["h0"],
            "zenith": cls_row["zenith"],
            "azimuth": cls_row["azimuth"],
            "final_status": final_status,
            "plot_metric_policy": policy,
        }

        if policy == "exclude":
            metrics = empty_metrics_row()
            metrics["metric_status"] = "excluded"
            metrics["notes"] = "discarded_after_tolerance"
            include_in_plots = False
        elif final_status == "failed":
            metrics = failed_metrics_row()
            include_in_plots = False
        elif policy == "fill_zero":
            raw_df = raw_grouped.get(case_id)
            n_raw = np.nan
            if raw_df is not None:
                n_raw = len(preprocess_case_raw(raw_df))
            metrics = fill_zero_metrics_row(n_raw_points=n_raw)
        elif policy == "compute_metrics":
            raw_df = raw_grouped.get(case_id)
            if raw_df is None or raw_df.empty:
                metrics = empty_metrics_row()
                metrics["metric_status"] = "missing_raw"
                metrics["notes"] = "case_id_missing_in_raw"
            else:
                metrics, strips = compute_closed_metrics(raw_df)
                if metrics["metric_status"] == "computed":
                    for strip in strips:
                        interval_rows.append(
                            {
                                "case_id": case_id,
                                **strip,
                                "lat": cls_row["lat"],
                                "lon": cls_row["lon"],
                                "h0": cls_row["h0"],
                                "zenith": cls_row["zenith"],
                                "azimuth": cls_row["azimuth"],
                                "final_status": final_status,
                            }
                        )
        else:
            metrics = empty_metrics_row()
            metrics["metric_status"] = "missing_raw"
            metrics["notes"] = f"unknown_plot_metric_policy={policy}"

        case_rows.append({**base, "include_in_plots": include_in_plots, **metrics})

    case_df = pd.DataFrame(case_rows)
    case_cols = [
        "case_id",
        "membership_scans",
        "lat",
        "lon",
        "h0",
        "zenith",
        "azimuth",
        "final_status",
        "plot_metric_policy",
        "include_in_plots",
        "metric_status",
        "R_L",
        "R_U",
        *METRIC_COLS,
        *DIAGNOSTIC_COLS,
        *COUNT_COLS,
        "notes",
    ]
    case_df = case_df[case_cols]
    case_df.to_csv(out / OUTPUT_METRICS_BY_CASE, index=False)

    interval_df = pd.DataFrame(interval_rows)
    if not interval_df.empty:
        interval_df = interval_df[
            [
                "case_id",
                "strip_index",
                "strip_label",
                "strip_start_R",
                "strip_end_R",
                "strip_width_GV",
                "is_ge_100MV",
                "is_ge_400MV",
                "lat",
                "lon",
                "h0",
                "zenith",
                "azimuth",
                "final_status",
            ]
        ]
    interval_df.to_csv(out / OUTPUT_INTERVAL_DETAILS, index=False)

    metric_merge_cols = ["case_id", "include_in_plots", "metric_status", *METRIC_COLS]
    case_metrics = case_df[metric_merge_cols].rename(
        columns={"include_in_plots": "include_in_plots_from_case"}
    )
    plot_df = plot_class.merge(case_metrics, on="case_id", how="left")
    plot_df["include_in_plots"] = plot_df["include_in_plots_from_case"]
    plot_df = plot_df.drop(columns=["include_in_plots_from_case"])

    plot_out_cols = [
        "plot_cell_id",
        "scan_type",
        "figure_group",
        "case_id",
        "plot_lat",
        "plot_lon",
        "plot_h0",
        "plot_zenith",
        "plot_azimuth",
        "lat",
        "lon",
        "h0",
        "zenith",
        "azimuth",
        "final_status",
        "plot_metric_policy",
        "include_in_plots",
        "metric_status",
        *METRIC_COLS,
    ]
    plot_df = plot_df[plot_out_cols]
    plot_df.to_csv(out / OUTPUT_METRICS_BY_PLOT_CELL, index=False)

    write_summary(out, case_df, plot_df)
    run_consistency_checks(case_df, plot_df)

    print()
    print("Output paths:")
    print(f"  {out / OUTPUT_METRICS_BY_CASE}")
    print(f"  {out / OUTPUT_METRICS_BY_PLOT_CELL}")
    print(f"  {out / OUTPUT_SUMMARY}")
    print(f"  {out / OUTPUT_INTERVAL_DETAILS}")
    print()
    print("metric_status counts:")
    for status, count in case_df["metric_status"].value_counts().sort_index().items():
        print(f"  {status}: {count}")
    print()
    print("final_status counts:")
    for status, count in case_df["final_status"].value_counts().sort_index().items():
        print(f"  {status}: {count}")
    print()
    computed = case_df[case_df["metric_status"] == "computed"]
    in_plots = case_df[case_df["include_in_plots"] == True]  # noqa: E712

    def print_f_gray_summary(label: str, subset: pd.DataFrame) -> None:
        vals = subset["f_gray"].dropna()
        print(f"{label}:")
        if vals.empty:
            print("  (no data)")
        else:
            print(
                f"  f_gray: min={vals.min():.6g}  mean={vals.mean():.6g}  "
                f"median={vals.median():.6g}  max={vals.max():.6g}"
            )

    print_f_gray_summary("f_gray summary (computed)", computed)
    print()
    print_f_gray_summary("f_gray summary (include_in_plots=True)", in_plots)
    print()

    all_f_gray = case_df["f_gray"].dropna()
    if not all_f_gray.empty and all_f_gray.min() >= 0 and all_f_gray.max() <= 1:
        print(
            f"f_gray in [0, 1]: confirmed "
            f"(min={all_f_gray.min():.6g}, max={all_f_gray.max():.6g})"
        )
    elif all_f_gray.empty:
        print("f_gray in [0, 1]: no non-NaN values to check")
    else:
        print(
            f"f_gray in [0, 1]: FAILED "
            f"(min={all_f_gray.min():.6g}, max={all_f_gray.max():.6g})"
        )


if __name__ == "__main__":
    main()
