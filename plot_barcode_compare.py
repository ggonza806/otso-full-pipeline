#!/usr/bin/env python3
"""Comparative raw vs classified penumbra barcode for a single case."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from pipeline_paths import resolve_plot_output_dir, resolve_run_dir
from plot_metric_validation_common import FIG_DPI, configure_matplotlib_style

RUN_DIR = resolve_run_dir()
OUTPUT_DIR = resolve_plot_output_dir(RUN_DIR)

CASE_ID = "case_0581"
INPUT_RAW = "raw_barcodes_merged_final.csv"
INPUT_CASE_METRICS = "penumbra_metrics_by_case.csv"

GRAY_LOOKAHEAD_K = 10
WHITE_VALUE = 1
PENUMBRA_MARGIN_GV = 1.0

X_LABEL = "Жёсткость, ГВ"


def resolve_raw_path() -> Path:
    preferred = RUN_DIR / INPUT_RAW
    fallback = RUN_DIR / "raw_barcodes_merged.csv"
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"No merged raw file found in {RUN_DIR}: "
        f"{preferred.name} or {fallback.name}"
    )


def base_label(raw_filter: float) -> str:
    return "white" if raw_filter == WHITE_VALUE else "black"


def apply_gray_lookahead(labels: list[str], k: int) -> list[str]:
    n = len(labels)
    final = labels.copy()
    for i in range(n):
        lookahead = labels[i + 1 : i + k + 1]
        if any(lbl != labels[i] for lbl in lookahead):
            final[i] = "gray"
    return final


def label_to_code(label: str) -> int:
    return {"white": 0, "black": 1, "gray": 2}[label]


def load_case_data(case_id: str) -> tuple[pd.DataFrame, float, float, pd.Series]:
    raw_path = resolve_raw_path()
    metrics_path = RUN_DIR / INPUT_CASE_METRICS

    raw = pd.read_csv(raw_path, usecols=["case_id", "R_GV", "raw_filter"])
    case_raw = raw[raw["case_id"] == case_id].copy()
    if case_raw.empty:
        raise ValueError(f"No raw barcode rows for {case_id}")

    case_raw["R_GV"] = pd.to_numeric(case_raw["R_GV"], errors="coerce")
    case_raw["raw_filter"] = pd.to_numeric(case_raw["raw_filter"], errors="coerce")
    case_raw = case_raw[case_raw["R_GV"].notna() & case_raw["raw_filter"].notna()]
    case_raw["R_GV_round"] = case_raw["R_GV"].round(6)
    case_raw = case_raw.drop_duplicates(subset=["R_GV_round"], keep="first")
    case_raw = case_raw.sort_values("R_GV").reset_index(drop=True)

    metrics = pd.read_csv(metrics_path)
    row = metrics[metrics["case_id"] == case_id]
    if row.empty:
        raise ValueError(f"No metrics row for {case_id}")
    row = row.iloc[0]
    r_l = float(row["R_L"])
    r_u = float(row["R_U"])
    if not np.isfinite(r_l) or not np.isfinite(r_u):
        raise ValueError(f"Invalid R_L/R_U for {case_id}")

    return case_raw, r_l, r_u, row


def build_arrays(
    case_raw: pd.DataFrame,
    r_l: float,
    r_u: float,
    *,
    r_min: float,
    r_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = case_raw[
        (case_raw["R_GV"] >= r_min) & (case_raw["R_GV"] <= r_max)
    ].copy()
    if window.empty:
        raise ValueError("Empty barcode window.")

    r = window["R_GV"].to_numpy(dtype=float)
    raw_binary = np.array(
        [0 if v == WHITE_VALUE else 1 for v in window["raw_filter"].to_numpy()],
        dtype=int,
    )

    in_pen = (r >= r_l) & (r <= r_u)
    pen_idx = np.where(in_pen)[0]
    if len(pen_idx) == 0:
        raise ValueError("No penumbra points in display window.")

    pen_labels = [base_label(v) for v in window.loc[in_pen, "raw_filter"].tolist()]
    pen_class = apply_gray_lookahead(pen_labels, GRAY_LOOKAHEAD_K)

    classified = raw_binary.copy()
    for i, lbl in zip(pen_idx, pen_class, strict=True):
        classified[i] = label_to_code(lbl)

    return r, raw_binary, classified


def add_r_bounds(ax, r_l: float, r_u: float) -> None:
    for r_val, tag in ((r_l, r"$R_L$"), (r_u, r"$R_U$")):
        ax.axvline(r_val, color="0.25", linestyle="--", linewidth=0.9, zorder=3)
        ax.text(
            r_val,
            1.08,
            tag,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color="0.15",
        )


def plot_compare(case_id: str, out_path: Path) -> None:
    configure_matplotlib_style()
    case_raw, r_l, r_u, meta = load_case_data(case_id)

    r_min = r_l - PENUMBRA_MARGIN_GV
    r_max = r_u + PENUMBRA_MARGIN_GV
    r, raw_binary, classified = build_arrays(
        case_raw, r_l, r_u, r_min=r_min, r_max=r_max
    )

    raw_cmap = ListedColormap(["white", "black"])
    raw_norm = BoundaryNorm([-0.5, 0.5, 1.5], raw_cmap.N)
    class_cmap = ListedColormap(["white", "black", "#b8b8b8"])
    class_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], class_cmap.N)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 2.8),
        sharex=True,
        gridspec_kw={"hspace": 0.55, "height_ratios": [1, 1]},
    )
    ax_raw, ax_cls = axes

    extent = [float(r.min()), float(r.max()), 0, 1]

    ax_raw.imshow(
        raw_binary.reshape(1, -1),
        cmap=raw_cmap,
        norm=raw_norm,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )
    ax_cls.imshow(
        classified.reshape(1, -1),
        cmap=class_cmap,
        norm=class_norm,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )

    for ax in axes:
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_xlim(r_min, r_max)
        ax.tick_params(axis="x", labelsize=9.5)
        add_r_bounds(ax, r_l, r_u)

    ax_cls.set_xlabel(X_LABEL, fontsize=10.5)
    plt.setp(ax_raw.get_xticklabels(), visible=False)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {out_path}")
    print(f"Window: [{r_min:.2f}, {r_max:.2f}] GV")
    print(f"Penumbra: R_L={r_l:.2f}, R_U={r_u:.2f} GV")
    print(
        f"Metrics: dR_white={meta['dR_white']:.3g}, "
        f"dR_black={meta['dR_black']:.3g}, dR_gray={meta['dR_gray']:.3g} GV"
    )


def find_geographic_case_id(
    lat: float,
    lon: float,
    *,
    h0: float = 400.0,
    zenith: float = 60.0,
    azimuth: float = 0.0,
) -> tuple[str, float, float]:
    metrics_path = RUN_DIR / INPUT_CASE_METRICS
    metrics = pd.read_csv(metrics_path)
    geo = metrics[
        (metrics["h0"] == h0)
        & (metrics["zenith"] == zenith)
        & (metrics["azimuth"] == azimuth)
    ].copy()
    if geo.empty:
        raise ValueError("No geographic cases found for the requested angles.")

    lats = np.sort(geo["lat"].unique())
    lons = np.sort(geo["lon"].unique())
    lat_used = float(lats[np.argmin(np.abs(lats - lat))])
    lon_used = float(lons[np.argmin(np.abs(lons - lon))])

    row = geo[(geo["lat"] == lat_used) & (geo["lon"] == lon_used)]
    if row.empty:
        raise ValueError(f"No case for lat={lat_used}, lon={lon_used}")
    case_id = str(row.iloc[0]["case_id"])
    return case_id, lat_used, lon_used


def plot_compare_for_coords(
    lat: float,
    lon: float,
    out_path: Path,
    *,
    h0: float = 400.0,
    zenith: float = 60.0,
    azimuth: float = 0.0,
) -> tuple[str, float, float]:
    case_id, lat_used, lon_used = find_geographic_case_id(
        lat, lon, h0=h0, zenith=zenith, azimuth=azimuth
    )
    plot_compare(case_id, out_path)
    if lat_used != lat or lon_used != lon:
        print(
            f"Note: requested ({lat}, {lon}) -> nearest grid point "
            f"({lat_used}, {lon_used}), case_id={case_id}"
        )
    else:
        print(f"Matched grid point ({lat_used}, {lon_used}), case_id={case_id}")
    return case_id, lat_used, lon_used


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        batch_coords = [
            (40.0, 120.0),
            (40.0, 250.0),
            (60.0, 120.0),
        ]
        for lat, lon in batch_coords:
            out_path = OUTPUT_DIR / f"barcode_lat{int(lat)}_lon{int(lon)}.png"
            plot_compare_for_coords(lat, lon, out_path)
        return

    out_path = OUTPUT_DIR / f"barcode_compare_{CASE_ID}.png"
    plot_compare(CASE_ID, out_path)


if __name__ == "__main__":
    main()
