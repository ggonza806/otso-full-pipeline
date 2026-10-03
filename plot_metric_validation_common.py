"""Shared helpers for penumbra metric validation heatmaps."""

from __future__ import annotations

from typing import Literal

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    griddata = None
    SCIPY_AVAILABLE = False

DR_GV_FIELDS = {"dR_total", "dR_white", "dR_black", "dR_gray"}
COUNT_FIELDS = {"N_100", "N_400"}
FRACTION_FIELDS = {"f_gray"}
NONNEGATIVE_FIELDS = DR_GV_FIELDS | COUNT_FIELDS | FRACTION_FIELDS
SMOOTH_EXEMPT_FIELDS = COUNT_FIELDS  # N_100, N_400 — no interpolation smooth

PLOTTABLE_FIELDS = (
    "dR_total",
    "N_100",
    "N_400",
    "dR_gray",
    "f_gray",
    "dR_black",
    "dR_white",
)

SMOOTH_FACTOR = 4
SMOOTH_METHOD = "linear"

GEO_X_LABEL = "Долгота, град"
GEO_Y_LABEL = "Широта, град"
ANG_X_LABEL = "Азимутальный угол, град"
ANG_Y_LABEL = "Зенитный угол, град"

FIGSIZE = (12, 5)
FIG_DPI = 150


def resolve_field_name(field_name: str) -> str:
    name = field_name.strip()
    if name not in PLOTTABLE_FIELDS:
        raise ValueError(
            f"Unknown FIELD_NAME '{field_name}'. "
            f"Allowed: {', '.join(PLOTTABLE_FIELDS)}."
        )
    return name


def use_smooth_for_field(field_name: str) -> bool:
    return field_name not in SMOOTH_EXEMPT_FIELDS


def smooth_factor_for_field(field_name: str) -> int:
    return 0 if field_name in SMOOTH_EXEMPT_FIELDS else SMOOTH_FACTOR


def configure_matplotlib_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "DejaVu Serif",
                "Times New Roman",
                "Times",
                "STIXGeneral",
                "STIX",
            ],
            "font.size": 10,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def make_validation_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "low_violet_high_red",
        ["violet", "blue", "cyan", "green", "yellow", "orange", "red"],
        N=256,
    )


def colorbar_label(field_name: str) -> str:
    """Russian colorbar label with units where applicable."""
    labels = {
        "dR_total": r"$dR_{\mathrm{total}}$, ГВ",
        "dR_white": r"$dR_{\mathrm{white}}$, ГВ",
        "dR_black": r"$dR_{\mathrm{black}}$, ГВ",
        "dR_gray": r"$dR_{\mathrm{gray}}$, ГВ",
        "f_gray": r"$\eta_{\mathrm{gray}}$",
        "N_100": r"$N_{100}$",
        "N_400": r"$N_{400}$",
    }
    return labels.get(field_name, field_name)


def color_limits(
    z_arrays: list[np.ndarray],
    field_name: str,
) -> tuple[float, float]:
    """Shared vmin/vmax across comparable maps of the same metric."""
    flat_max = max(float(np.nanmax(z)) for z in z_arrays)
    if field_name in NONNEGATIVE_FIELDS:
        return 0.0, flat_max
    flat_min = min(float(np.nanmin(z)) for z in z_arrays)
    return flat_min, flat_max


def apply_fill_zero_and_numeric(df: pd.DataFrame, field_name: str) -> pd.DataFrame:
    out = df.copy()
    out[field_name] = pd.to_numeric(out[field_name], errors="coerce")
    out.loc[out["plot_metric_policy"] == "fill_zero", field_name] = (
        out.loc[out["plot_metric_policy"] == "fill_zero", field_name].fillna(0.0)
    )
    return out


def close_periodic_column(
    grid_df: pd.DataFrame,
    *,
    close_value: float = 0.0,
    close_label: float = 360.0,
) -> pd.DataFrame:
    if close_value in grid_df.columns:
        grid_df = grid_df.copy()
        grid_df[close_label] = grid_df[close_value]
    closed_cols = np.sort(grid_df.columns.to_numpy(dtype=float))
    return grid_df.reindex(columns=closed_cols)


def prepare_grid(
    df: pd.DataFrame,
    field_name: str,
    *,
    y_col: str,
    x_col: str,
    close_x_at: float | None = 360.0,
    close_x_from: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_values = np.sort(df[y_col].dropna().unique())
    x_values = np.sort(df[x_col].dropna().unique())

    grid_df = df.pivot(index=y_col, columns=x_col, values=field_name)
    grid_df = grid_df.reindex(index=y_values, columns=x_values)

    if close_x_at is not None:
        grid_df = close_periodic_column(
            grid_df,
            close_value=close_x_from,
            close_label=close_x_at,
        )
        x_values = np.sort(grid_df.columns.to_numpy(dtype=float))

    z = grid_df.to_numpy(dtype=float)
    if field_name in NONNEGATIVE_FIELDS:
        z = np.maximum(z, 0.0)

    return x_values, y_values, z


def build_interp_df_geographic(df: pd.DataFrame) -> pd.DataFrame:
    df_interp = df.copy()
    at_zero = df_interp[df_interp["plot_lon"] == 0.0].copy()
    if not at_zero.empty:
        wrap = at_zero.copy()
        wrap["plot_lon"] = 360.0
        df_interp = pd.concat([df_interp, wrap], ignore_index=True)
    return df_interp


def build_interp_df_angular(df: pd.DataFrame) -> pd.DataFrame:
    parts = [df.copy()]

    at_zero = df[df["plot_azimuth"] == 0.0].copy()
    if not at_zero.empty:
        wrap_az = at_zero.copy()
        wrap_az["plot_azimuth"] = 360.0
        parts.append(wrap_az)

    z_min = float(df["plot_zenith"].min())
    z_max = float(df["plot_zenith"].max())
    z_step = float(np.min(np.diff(np.sort(df["plot_zenith"].unique()))))

    at_min_z = df[df["plot_zenith"] == z_min].copy()
    if not at_min_z.empty:
        pad_low = at_min_z.copy()
        pad_low["plot_zenith"] = z_min - z_step
        parts.append(pad_low)

    at_max_z = df[df["plot_zenith"] == z_max].copy()
    if not at_max_z.empty:
        pad_high = at_max_z.copy()
        pad_high["plot_zenith"] = z_max + z_step
        parts.append(pad_high)

    df_interp = pd.concat(parts, ignore_index=True)
    extra_zero = df_interp[df_interp["plot_azimuth"] == 0.0].copy()
    if not extra_zero.empty:
        wrap_az2 = extra_zero.copy()
        wrap_az2["plot_azimuth"] = 360.0
        df_interp = pd.concat([df_interp, wrap_az2], ignore_index=True)

    return df_interp


def make_display_mesh(
    df: pd.DataFrame,
    field_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_raw: np.ndarray,
    *,
    x_col: str,
    y_col: str,
    use_smooth: bool | None = None,
    smooth: int | None = None,
    smooth_method: str = SMOOTH_METHOD,
    interp_mode: Literal["geographic", "angular"] = "geographic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if use_smooth is None:
        use_smooth = use_smooth_for_field(field_name)
    if smooth is None:
        smooth = smooth_factor_for_field(field_name)

    if use_smooth:
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for smoothed plots. Install scipy in your environment."
            )

        x_fine = np.linspace(
            x_values.min(),
            x_values.max(),
            len(x_values) * smooth,
        )
        y_fine = np.linspace(
            y_values.min(),
            y_values.max(),
            len(y_values) * smooth,
        )
        x_mesh, y_mesh = np.meshgrid(x_fine, y_fine)

        if interp_mode == "geographic":
            df_interp = build_interp_df_geographic(df)
        else:
            df_interp = build_interp_df_angular(df)

        points = df_interp[[x_col, y_col]].to_numpy(dtype=float)
        values = pd.to_numeric(df_interp[field_name], errors="coerce").to_numpy()
        finite = np.isfinite(points).all(axis=1) & np.isfinite(values)
        points = points[finite]
        values = values[finite]
        if len(values) < 3:
            raise ValueError(
                f"Only {len(values)} finite point(s) are available for interpolation"
            )

        z_plot = griddata(
            points,
            values,
            (x_mesh, y_mesh),
            method=smooth_method,
        )
        if field_name in NONNEGATIVE_FIELDS:
            z_plot = np.maximum(z_plot, 0.0)
        z_plot = np.ma.masked_invalid(z_plot)
        return x_fine, y_fine, z_plot

    return x_values, y_values, z_raw


def build_latlon_mesh(
    df: pd.DataFrame,
    field_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values, y_values, z_raw = prepare_grid(
        df,
        field_name,
        y_col="plot_lat",
        x_col="plot_lon",
        close_x_at=360.0,
        close_x_from=0.0,
    )
    x_plot, y_plot, z_display = make_display_mesh(
        df,
        field_name,
        x_values,
        y_values,
        z_raw,
        x_col="plot_lon",
        y_col="plot_lat",
        interp_mode="geographic",
    )
    return x_values, y_values, x_plot, y_plot, z_display


def build_angular_mesh(
    df: pd.DataFrame,
    field_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values, y_values, z_raw = prepare_grid(
        df,
        field_name,
        y_col="plot_zenith",
        x_col="plot_azimuth",
        close_x_at=360.0,
        close_x_from=0.0,
    )
    x_plot, y_plot, z_display = make_display_mesh(
        df,
        field_name,
        x_values,
        y_values,
        z_raw,
        x_col="plot_azimuth",
        y_col="plot_zenith",
        interp_mode="angular",
    )
    return x_values, y_values, x_plot, y_plot, z_display


def style_colorbar(cbar, field_name: str) -> None:
    cbar.ax.tick_params(labelsize=9.5)
    cbar.set_label(colorbar_label(field_name), fontsize=10)


def render_latlon_figure(
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    z_display: np.ndarray,
    field_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    vmin: float,
    vmax: float,
) -> Figure:
    import matplotlib.pyplot as plt

    configure_matplotlib_style()
    cmap = make_validation_colormap()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    mesh = ax.pcolormesh(
        x_plot,
        y_plot,
        z_display,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label(field_name), fontsize=10)
    cbar.ax.tick_params(labelsize=9.5)

    ax.set_xlabel(GEO_X_LABEL, fontsize=10.5)
    ax.set_ylabel(GEO_Y_LABEL, fontsize=10.5)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.set_xlim(float(x_values.min()), float(x_values.max()))
    ax.set_ylim(float(y_values.min()), float(y_values.max()))
    ax.grid(False)

    fig.tight_layout()
    return fig


def render_angular_figure(
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    z_display: np.ndarray,
    field_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    invert_zenith_axis: bool = True,
) -> Figure:
    import matplotlib.pyplot as plt

    configure_matplotlib_style()
    cmap = make_validation_colormap()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    mesh = ax.pcolormesh(
        x_plot,
        y_plot,
        z_display,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label(field_name), fontsize=10)
    cbar.ax.tick_params(labelsize=9.5)

    ax.set_xlabel(ANG_X_LABEL, fontsize=10.5)
    ax.set_ylabel(ANG_Y_LABEL, fontsize=10.5)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.set_xlim(float(x_values.min()), float(x_values.max()))
    ax.set_ylim(float(y_values.min()), float(y_values.max()))
    if invert_zenith_axis:
        ax.invert_yaxis()
    ax.grid(False)

    fig.tight_layout()
    return fig


def print_grid_diagnostics(
    *,
    field_name: str,
    n_filtered: int,
    x_values: np.ndarray,
    x_before_close: list[float],
    y_values: np.ndarray,
    z_raw: np.ndarray,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    z_display: np.ndarray,
    use_smooth: bool | None = None,
    smooth: int | None = None,
    smooth_method: str = SMOOTH_METHOD,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if use_smooth is None:
        use_smooth = use_smooth_for_field(field_name)
    if smooth is None:
        smooth = smooth_factor_for_field(field_name)

    print(f"FIELD_NAME: {field_name}")
    print(f"Filtered rows: {n_filtered}")
    print(f"Unique x: {x_values.tolist()}")
    print(f"Unique x (before closure): {x_before_close}")
    print(f"Unique y: {y_values.tolist()}")
    print(f"Final matrix shape (n_y, n_x): {z_raw.shape}")
    print(
        f"{field_name} min/max raw: "
        f"{np.nanmin(z_raw):.6g} / {np.nanmax(z_raw):.6g}"
    )
    print(
        f"{field_name} min/max displayed: "
        f"{float(np.nanmin(z_display)):.6g} / {float(np.nanmax(z_display)):.6g}"
    )
    if vmin is not None and vmax is not None:
        print(f"Shared color limits: {vmin:.6g} / {vmax:.6g}")
    print(f"Zero-value cells raw: {int(np.sum(z_raw == 0))}")
    print(f"USE_SMOOTH: {use_smooth}")
    print(f"SMOOTH: {smooth}")
    print(f"SMOOTH_METHOD: {smooth_method}")
    if use_smooth:
        print(f"Fine grid shape: ({len(y_plot)}, {len(x_plot)})")
