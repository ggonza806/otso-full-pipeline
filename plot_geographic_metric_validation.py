#!/usr/bin/env python3
"""
plot_geographic_metric_validation.py — Adaptable geographic metric heatmap.

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
    build_latlon_mesh,
    color_limits,
    render_latlon_figure,
    resolve_field_name,
)

RUN_DIR = resolve_run_dir()
INPUT_CSV = "penumbra_metrics_by_plot_cell.csv"

FIELD_NAME = "dR_total"

SCAN_TYPE = "geographic"
PLOT_H0 = 400.0
PLOT_ZENITH = 0.0
PLOT_AZIMUTH = 0.0

# Share color scale across all geographic directions for this metric.
SHARE_COLOR_SCALE_ACROSS_GEO_DIRECTIONS = True
GEO_DIRECTIONS = (
    (0.0, 0.0),
)


def load_and_filter(
    csv_path: Path,
    field_name: str,
    *,
    zenith: float,
    azimuth: float,
) -> pd.DataFrame:
    field_name = resolve_field_name(field_name)
    df = pd.read_csv(csv_path, dtype={"plot_cell_id": str, "case_id": str})

    if field_name not in df.columns:
        raise ValueError(f"Column '{field_name}' not found in {csv_path}")

    dense_grid = df["scan_type"].eq("interpolated_grid")
    source_mask = dense_grid if dense_grid.any() else df["scan_type"].eq(SCAN_TYPE)
    filtered = df[
        source_mask
        & (df["plot_h0"] == PLOT_H0)
        & (df["plot_zenith"] == zenith)
        & (df["plot_azimuth"] == azimuth)
    ].copy()

    if filtered.empty:
        raise ValueError("No rows matched geographic selection.")

    filtered = apply_fill_zero_and_numeric(filtered, field_name)

    dup = filtered.duplicated(subset=["plot_lat", "plot_lon"], keep=False)
    if dup.any():
        raise ValueError(
            f"Duplicate (plot_lat, plot_lon) pairs after filtering: {dup.sum()}"
        )

    return filtered


def shared_limits(
    csv_path: Path,
    field_name: str,
) -> tuple[float, float] | None:
    if not SHARE_COLOR_SCALE_ACROSS_GEO_DIRECTIONS:
        return None

    z_arrays = []
    for zenith, azimuth in GEO_DIRECTIONS:
        part = load_and_filter(
            csv_path, field_name, zenith=zenith, azimuth=azimuth
        )
        _, _, _, _, z_display = build_latlon_mesh(part, field_name)
        z_arrays.append(z_display)
    return color_limits(z_arrays, field_name)


def plot_heatmap(
    df: pd.DataFrame,
    field_name: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    x_values, y_values, x_plot, y_plot, z_display = build_latlon_mesh(df, field_name)
    if vmin is None or vmax is None:
        vmin, vmax = color_limits([z_display], field_name)

    fig = render_latlon_figure(
        x_plot,
        y_plot,
        z_display,
        field_name,
        x_values,
        y_values,
        vmin=vmin,
        vmax=vmax,
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

    limits = shared_limits(csv_path, field_name)
    df = load_and_filter(
        csv_path,
        field_name,
        zenith=PLOT_ZENITH,
        azimuth=PLOT_AZIMUTH,
    )
    if limits is None:
        plot_heatmap(df, field_name)
    else:
        vmin, vmax = limits
        plot_heatmap(df, field_name, vmin=vmin, vmax=vmax)


if __name__ == "__main__":
    main()
