#!/usr/bin/env python3
"""Export penumbra heatmaps (geographic, altitude, angular) to graficos_vkr/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from pipeline_paths import resolve_plot_output_dir, resolve_run_dir

from plot_metric_validation_common import (
    FIG_DPI,
    PLOTTABLE_FIELDS,
    apply_fill_zero_and_numeric,
    build_angular_mesh,
    build_latlon_mesh,
    color_limits,
    render_angular_figure,
    render_latlon_figure,
    smooth_factor_for_field,
    use_smooth_for_field,
)

RUN_DIR = resolve_run_dir()
INPUT_CSV = "penumbra_metrics_by_plot_cell.csv"
OUTPUT_DIR = resolve_plot_output_dir(RUN_DIR)

GEO_DIRECTIONS = (
    (0.0, 0.0),
)
ALT_H0_VALUES = (400,)
ANG_H0 = 400
ANG_LAT = -25.2637
ANG_LON = 302.4241


@dataclass
class MapSpec:
    kind: str
    field_name: str
    filename: str
    df: pd.DataFrame


def filter_rows(
    df: pd.DataFrame,
    field_name: str,
    *,
    scan_type: str,
    plot_h0: float,
    plot_zenith: float,
    plot_azimuth: float,
) -> pd.DataFrame:
    dense_grid = df["scan_type"].eq("interpolated_grid")
    source_mask = dense_grid if dense_grid.any() else df["scan_type"].eq(scan_type)
    filtered = df[
        source_mask
        & (df["plot_h0"] == plot_h0)
        & (df["plot_zenith"] == plot_zenith)
        & (df["plot_azimuth"] == plot_azimuth)
    ].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows for {scan_type} h0={plot_h0}, "
            f"zenith={plot_zenith}, azimuth={plot_azimuth}"
        )
    filtered = apply_fill_zero_and_numeric(filtered, field_name)
    if scan_type in {"geographic", "altitude"}:
        dup = filtered.duplicated(subset=["plot_lat", "plot_lon"], keep=False)
        key = "lat/lon"
    else:
        dup = filtered.duplicated(subset=["plot_zenith", "plot_azimuth"], keep=False)
        key = "zenith/azimuth"
    if dup.any():
        raise ValueError(f"Duplicate {key} pairs: {dup.sum()}")
    return filtered


def geo_filename(field_name: str, h0: int, zenith: float, azimuth: float) -> str:
    return f"{field_name}_geo_h{h0}_z{int(zenith)}_az{int(azimuth)}.png"


def alt_filename(field_name: str, h0: int) -> str:
    return f"{field_name}_alt_h{h0}_z0_az0.png"


def ang_filename(field_name: str, h0: int) -> str:
    return f"{field_name}_ang_h{h0}.png"


def collect_specs(df: pd.DataFrame, field_name: str) -> list[MapSpec]:
    specs: list[MapSpec] = []

    for zenith, azimuth in GEO_DIRECTIONS:
        part = filter_rows(
            df,
            field_name,
            scan_type="geographic",
            plot_h0=400.0,
            plot_zenith=zenith,
            plot_azimuth=azimuth,
        )
        specs.append(
            MapSpec(
                kind="latlon",
                field_name=field_name,
                filename=geo_filename(field_name, 400, zenith, azimuth),
                df=part,
            )
        )

    for h0 in ALT_H0_VALUES:
        part = filter_rows(
            df,
            field_name,
            scan_type="altitude",
            plot_h0=float(h0),
            plot_zenith=0.0,
            plot_azimuth=0.0,
        )
        specs.append(
            MapSpec(
                kind="latlon",
                field_name=field_name,
                filename=alt_filename(field_name, h0),
                df=part,
            )
        )

    interpolated = df["scan_type"].isin(["interpolated_grid", "angular_reference"])
    if interpolated.any():
        angular_selection = (
            interpolated
            & df["plot_h0"].eq(float(ANG_H0))
            & df["plot_lat"].eq(float(ANG_LAT))
            & df["plot_lon"].eq(float(ANG_LON))
        )
    else:
        angular_selection = (
            df["scan_type"].eq("angular")
            & df["plot_h0"].eq(float(ANG_H0))
        )
    part = df[angular_selection].copy()
    if part.empty:
        raise ValueError(
            "No angular rows for the configured fixed coordinates: "
            f"lat={ANG_LAT}, lon={ANG_LON}, h0={ANG_H0}. "
            "Choose coordinates present in penumbra_metrics_by_plot_cell.csv."
        )
    part = apply_fill_zero_and_numeric(part, field_name)
    dup = part.duplicated(subset=["plot_zenith", "plot_azimuth"], keep=False)
    if dup.any():
        raise ValueError(f"Duplicate zenith/azimuth pairs: {dup.sum()}")
    specs.append(
        MapSpec(
            kind="angular",
            field_name=field_name,
            filename=ang_filename(field_name, ANG_H0),
            df=part,
        )
    )
    return specs


def export_latlon_group(
    specs: list[MapSpec],
    field_name: str,
    out_dir: Path,
) -> int:
    meshes = []
    for spec in specs:
        x_values, y_values, x_plot, y_plot, z_display = build_latlon_mesh(
            spec.df, field_name
        )
        meshes.append((spec, x_values, y_values, x_plot, y_plot, z_display))

    vmin, vmax = color_limits([m[5] for m in meshes], field_name)
    saved = 0
    for spec, x_values, y_values, x_plot, y_plot, z_display in meshes:
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
        out_path = out_dir / spec.filename
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        import matplotlib.pyplot as plt

        plt.close(fig)
        print(
            f"Saved {spec.filename}  smooth={use_smooth_for_field(field_name)}  "
            f"limits=({vmin:.4g}, {vmax:.4g})"
        )
        saved += 1
    return saved


def export_angular_group(
    spec: MapSpec,
    field_name: str,
    out_dir: Path,
) -> int:
    x_values, y_values, x_plot, y_plot, z_display = build_angular_mesh(
        spec.df, field_name
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
        invert_zenith_axis=True,
    )
    out_path = out_dir / spec.filename
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(
        f"Saved {spec.filename}  smooth={use_smooth_for_field(field_name)}  "
        f"limits=({vmin:.4g}, {vmax:.4g})"
    )
    return 1


def main() -> None:
    csv_path = RUN_DIR / INPUT_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, dtype={"plot_cell_id": str, "case_id": str})

    print(f"Exporting to {OUTPUT_DIR.resolve()}")
    print(f"Metrics: {', '.join(PLOTTABLE_FIELDS)}")
    print(
        "Smooth rule: N_100/N_400 off; other metrics factor "
        f"{smooth_factor_for_field('dR_total')}"
    )
    print()

    total = 0
    for field_name in PLOTTABLE_FIELDS:
        specs = collect_specs(df, field_name)
        geo_specs = [s for s in specs if s.filename.startswith(f"{field_name}_geo_")]
        alt_specs = [s for s in specs if s.filename.startswith(f"{field_name}_alt_")]
        ang_specs = [s for s in specs if s.filename.startswith(f"{field_name}_ang_")]

        print(f"=== {field_name} ===")
        total += export_latlon_group(geo_specs, field_name, OUTPUT_DIR)
        total += export_latlon_group(alt_specs, field_name, OUTPUT_DIR)
        for spec in ang_specs:
            total += export_angular_group(spec, field_name, OUTPUT_DIR)
        print()

    print(f"Done. {total} files saved.")


if __name__ == "__main__":
    main()
