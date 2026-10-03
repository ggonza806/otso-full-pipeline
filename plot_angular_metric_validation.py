#!/usr/bin/env python3
"""
plot_angular_metric_validation.py — Adaptable angular metric heatmap.

Set FIELD_NAME to any plottable penumbra metric:
  dR_total, dR_white, dR_black, dR_gray, f_gray, N_100, N_400

Shows one figure (no automatic save).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pipeline_paths import resolve_run_dir

from plot_metric_validation_common import (
    PLOTTABLE_FIELDS,
    apply_fill_zero_and_numeric,
    build_angular_mesh,
    color_limits,
    render_angular_figure,
    resolve_field_name,
)

RUN_DIR = resolve_run_dir()
INPUT_CSV = "penumbra_metrics_by_plot_cell.csv"

FIELD_NAME = "dR_total"

INVERT_ZENITH_AXIS = True

SCAN_TYPE = "angular"
PLOT_H0 = 400.0
PLOT_LAT = -25.2637
PLOT_LON = 302.4241


def load_and_filter(csv_path: Path, field_name: str) -> pd.DataFrame:
    field_name = resolve_field_name(field_name)
    df = pd.read_csv(csv_path, dtype={"plot_cell_id": str, "case_id": str})

    if field_name not in df.columns:
        raise ValueError(f"Column '{field_name}' not found in {csv_path}")

    interpolated = df["scan_type"].isin(["interpolated_grid", "angular_reference"])
    if interpolated.any():
        selection = (
            interpolated
            & df["plot_h0"].eq(PLOT_H0)
            & df["plot_lat"].eq(PLOT_LAT)
            & df["plot_lon"].eq(PLOT_LON)
        )
    else:
        selection = df["scan_type"].eq(SCAN_TYPE) & df["plot_h0"].eq(PLOT_H0)
    filtered = df[selection].copy()

    if filtered.empty:
        raise ValueError("No rows matched angular selection.")

    filtered = apply_fill_zero_and_numeric(filtered, field_name)

    dup = filtered.duplicated(subset=["plot_zenith", "plot_azimuth"], keep=False)
    if dup.any():
        raise ValueError(
            f"Duplicate (plot_zenith, plot_azimuth) pairs after filtering: {dup.sum()}"
        )

    return filtered


def plot_heatmap(df: pd.DataFrame, field_name: str) -> None:
    x_values, y_values, x_plot, y_plot, z_display = build_angular_mesh(
        df, field_name
    )
    vmin, vmax = color_limits([z_display], field_name)

    fig = render_angular_figure(
        x_plot,
        y_plot,
        z_display,
        field_name,
        x_values,
        y_values,
        vmin=vmin,
        vmax=vmax,
        invert_zenith_axis=INVERT_ZENITH_AXIS,
    )
    print(f"Color limits: {vmin:.6g} / {vmax:.6g}")
    print("Showing figure — save manually if needed.")
    plt.show()


def main() -> None:
    csv_path = RUN_DIR / INPUT_CSV
    field_name = resolve_field_name(FIELD_NAME)
    print("Reading plot-cell metrics:")
    print(f"  {csv_path.resolve()}")
    print(f"Plottable metrics: {', '.join(PLOTTABLE_FIELDS)}")
    print(f"Plotting: {field_name}")
    print()

    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = load_and_filter(csv_path, field_name)
    plot_heatmap(df, field_name)


if __name__ == "__main__":
    main()
