#!/usr/bin/env python3
"""
Load penumbra_boundaries.mat, clean rigidity windows, and densify by
linear interpolation. Does not run OTSO.

Rules:
  - If Rl or Ru is NaN → no window.
  - If Ru is at the source scan cap → open-high, not a physical window.
  - If 0 < Rl - Ru <= DR_CROSS_MAX → one-bin swap: R_low=min, R_high=max.
  - Larger inversions → no window.
  - Interpolate only inside the original grid; no extrapolation.
  - A densified point is valid only if every corner of its 4D cell is valid.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

PACKAGE_DIR = Path(__file__).resolve().parent
MAT_PATH = PACKAGE_DIR / "data" / "penumbra_boundaries.mat"

DR_CROSS_MAX = 0.02  # GV; original scan step is ~0.01
UPPER_SCAN_CAP_GV = 55.0
UPPER_SCAN_CAP_ATOL_GV = 1e-6
VERTICAL_WINDOW_TOLERANCE_GV = 0.005

# Denser query grid (still inside original bounds; no extrapolation).
LAT_OUT = np.arange(-60.0, 61.0, 10.0)
LON_OUT = np.arange(0.0, 360.0, 30.0)  # unique; 360 dropped as periodic
ZEN_OUT = np.arange(0.0, 181.0, 15.0)
AZ_OUT = np.arange(0.0, 360.0, 15.0)


def load_boundary_file(mat_path: Path = MAT_PATH) -> dict[str, np.ndarray]:
    raw = loadmat(mat_path, squeeze_me=True)
    lat = np.asarray(raw["latGrid"], dtype=float).ravel()
    lon = np.asarray(raw["lonGrid"], dtype=float).ravel()
    zen = np.asarray(raw["zenGrid"], dtype=float).ravel()
    az = np.asarray(raw["azGrid"], dtype=float).ravel()
    rl = np.asarray(raw["RlNet"], dtype=float)
    ru = np.asarray(raw["RuNet"], dtype=float)
    expected = (lat.size, lon.size, zen.size, az.size)
    if rl.shape != expected or ru.shape != expected:
        raise ValueError(
            f"Array shape mismatch: Rl{rl.shape} Ru{ru.shape} vs {expected}"
        )
    return {"lat": lat, "lon": lon, "zen": zen, "az": az, "Rl": rl, "Ru": ru}


def clean_windows(
    rl: np.ndarray,
    ru: np.ndarray,
    dr_cross_max: float = DR_CROSS_MAX,
    upper_scan_cap_gv: float | None = UPPER_SCAN_CAP_GV,
    upper_scan_cap_atol_gv: float = UPPER_SCAN_CAP_ATOL_GV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Return R_low, R_high (NaN where no window) and a boolean valid mask."""
    both_finite = np.isfinite(rl) & np.isfinite(ru)
    if upper_scan_cap_gv is None:
        at_upper_cap = np.zeros(rl.shape, dtype=bool)
    else:
        at_upper_cap = np.isfinite(ru) & np.isclose(
            ru,
            float(upper_scan_cap_gv),
            rtol=0.0,
            atol=float(upper_scan_cap_atol_gv),
        )
    usable = both_finite & ~at_upper_cap
    ordered = usable & (rl <= ru)
    crossed = usable & (rl > ru) & ((rl - ru) <= dr_cross_max)
    large_inv = usable & (rl > ru) & ((rl - ru) > dr_cross_max)

    r_low = np.full(rl.shape, np.nan, dtype=float)
    r_high = np.full(ru.shape, np.nan, dtype=float)

    r_low[ordered] = rl[ordered]
    r_high[ordered] = ru[ordered]
    r_low[crossed] = ru[crossed]
    r_high[crossed] = rl[crossed]

    valid = ordered | crossed
    stats = {
        "n_nodes": int(rl.size),
        "n_nan_either": int((~both_finite).sum()),
        "n_open_high_cap": int(at_upper_cap.sum()),
        "n_ordered": int(ordered.sum()),
        "n_crossed_swapped": int(crossed.sum()),
        "n_large_inversion_dropped": int(large_inv.sum()),
        "n_valid_windows": int(valid.sum()),
    }
    return r_low, r_high, valid, stats


def add_scan_bounds(
    windows: pd.DataFrame,
    margin_low_gv: float,
    margin_high_gv: float,
    rigidity_floor_gv: float,
    rigidity_step_gv: float,
) -> pd.DataFrame:
    """Convert interpolated physical bounds into Stage-1 scan bounds."""
    if min(margin_low_gv, margin_high_gv, rigidity_floor_gv, rigidity_step_gv) < 0:
        raise ValueError("Rigidity margins, floor, and step must be non-negative")
    if rigidity_step_gv == 0:
        raise ValueError("rigidity_step_gv must be greater than zero")

    out = windows.copy()
    step = float(rigidity_step_gv)
    raw_start = out["R_high"] + float(margin_high_gv)
    raw_end = np.maximum(
        out["R_low"] - float(margin_low_gv),
        float(rigidity_floor_gv),
    )
    # Snap outward to the OTSO rigidity lattice. Besides making the requested
    # margins conservative, this avoids passing long interpolation decimals to
    # the native OTSO/Fortran layer.
    out["R_start"] = np.ceil(raw_start / step - 1e-12) * step
    out["R_end"] = np.floor(raw_end / step + 1e-12) * step
    out["R_end"] = np.maximum(out["R_end"], float(rigidity_floor_gv))
    out["R_start"] = out["R_start"].round(12)
    out["R_end"] = out["R_end"].round(12)
    out["dR"] = step
    out["window_source"] = "matlab_linear_4d"
    out["window_status"] = "valid_interpolated"
    if (out["R_start"] <= out["R_end"]).any():
        raise ValueError("At least one interpolated scan window is empty or reversed")
    return out


def collapse_vertical_azimuths(
    windows: pd.DataFrame,
    tolerance_gv: float = VERTICAL_WINDOW_TOLERANCE_GV,
) -> pd.DataFrame:
    """Return one physical window per position at zenith 0/180 degrees.

    Azimuth is undefined for a vertical direction. The returned table uses
    azimuth=0 and a conservative envelope of the azimuthal estimates. The
    original rows remain suitable for plot cells and can all reference this
    canonical physical case.
    """
    required = {"lat", "lon", "zenith", "azimuth", "R_low", "R_high"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"Missing window columns: {sorted(missing)}")

    vertical = np.isclose(windows["zenith"], 0.0) | np.isclose(
        windows["zenith"], 180.0
    )
    nonvertical = windows.loc[~vertical].copy()
    canonical_rows = []

    for (lat, lon, zenith), group in windows.loc[vertical].groupby(
        ["lat", "lon", "zenith"], sort=False
    ):
        low_span = float(group["R_low"].max() - group["R_low"].min())
        high_span = float(group["R_high"].max() - group["R_high"].min())
        if max(low_span, high_span) > float(tolerance_gv):
            raise ValueError(
                "Inconsistent vertical rigidity estimates at "
                f"lat={lat}, lon={lon}, zenith={zenith}: "
                f"R_low span={low_span:.6g} GV, "
                f"R_high span={high_span:.6g} GV"
            )

        row = group.iloc[0].copy()
        row["azimuth"] = 0.0
        row["R_low"] = float(group["R_low"].min())
        row["R_high"] = float(group["R_high"].max())
        row["dR_window"] = row["R_high"] - row["R_low"]
        canonical_rows.append(row)

    canonical = pd.DataFrame(canonical_rows, columns=windows.columns)
    out = pd.concat([nonvertical, canonical], ignore_index=True)
    out = out.sort_values(
        ["lat", "lon", "zenith", "azimuth"], kind="stable"
    ).reset_index(drop=True)
    out.attrs["n_input_windows"] = int(len(windows))
    out.attrs["n_physical_windows"] = int(len(out))
    out.attrs["n_vertical_plot_windows"] = int(vertical.sum())
    out.attrs["n_vertical_physical_windows"] = int(len(canonical))
    return out


def _interval_indices(grid: np.ndarray, value: float) -> tuple[int, ...]:
    """Grid indices that bound `value`. Empty if outside [min, max]."""
    if value < grid[0] or value > grid[-1]:
        return ()
    if np.isclose(value, grid[0]):
        return (0,)
    if np.isclose(value, grid[-1]):
        return (grid.size - 1,)
    j = int(np.searchsorted(grid, value, side="right") - 1)
    if j < 0 or j >= grid.size - 1:
        return ()
    if np.isclose(value, grid[j]):
        return (j,)
    if np.isclose(value, grid[j + 1]):
        return (j + 1,)
    return (j, j + 1)


def hypercube_all_valid(
    valid: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    point: tuple[float, float, float, float],
) -> bool:
    index_sets = [_interval_indices(axis, coord) for axis, coord in zip(axes, point)]
    if any(len(s) == 0 for s in index_sets):
        return False
    for idx in product(*index_sets):
        if not valid[idx]:
            return False
    return True


def make_interpolators(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    r_low: np.ndarray,
    r_high: np.ndarray,
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    # Invalid vertices are gated by hypercube_all_valid before evaluation.
    # Replacing their NaNs here prevents scipy's 0*NaN propagation from
    # corrupting an otherwise valid query located exactly on an original node.
    safe_low = np.where(np.isfinite(r_low), r_low, 0.0)
    safe_high = np.where(np.isfinite(r_high), r_high, 0.0)
    kwargs = dict(
        bounds_error=False,
        fill_value=np.nan,
        method="linear",
    )
    interp_low = RegularGridInterpolator(axes, safe_low, **kwargs)
    interp_high = RegularGridInterpolator(axes, safe_high, **kwargs)
    return interp_low, interp_high


def densify_windows(
    grid: dict[str, np.ndarray],
    r_low: np.ndarray,
    r_high: np.ndarray,
    valid: np.ndarray,
    lat_out: np.ndarray = LAT_OUT,
    lon_out: np.ndarray = LON_OUT,
    zen_out: np.ndarray = ZEN_OUT,
    az_out: np.ndarray = AZ_OUT,
) -> pd.DataFrame:
    axes = (grid["lat"], grid["lon"], grid["zen"], grid["az"])
    for name, src, dst in (
        ("lat", axes[0], lat_out),
        ("lon", axes[1], lon_out),
        ("zen", axes[2], zen_out),
        ("az", axes[3], az_out),
    ):
        if dst.min() < src.min() - 1e-12 or dst.max() > src.max() + 1e-12:
            raise ValueError(
                f"{name} query [{dst.min()}, {dst.max()}] "
                f"is outside source [{src.min()}, {src.max()}]"
            )

    interp_low, interp_high = make_interpolators(axes, r_low, r_high)
    records = []
    n_query = 0
    n_outside_or_missing = 0

    for lat, lon, zen, az in product(lat_out, lon_out, zen_out, az_out):
        n_query += 1
        point = (float(lat), float(lon), float(zen), float(az))
        if not hypercube_all_valid(valid, axes, point):
            n_outside_or_missing += 1
            continue
        low = float(interp_low(point))
        high = float(interp_high(point))
        if not (np.isfinite(low) and np.isfinite(high) and high >= low):
            n_outside_or_missing += 1
            continue
        records.append(
            {
                "lat": point[0],
                "lon": point[1],
                "zenith": point[2],
                "azimuth": point[3],
                "R_low": low,
                "R_high": high,
                "dR_window": high - low,
            }
        )

    out = pd.DataFrame.from_records(records)
    out.attrs["n_query"] = n_query
    out.attrs["n_no_window"] = n_outside_or_missing
    return out


def original_window_table(
    grid: dict[str, np.ndarray],
    r_low: np.ndarray,
    r_high: np.ndarray,
    valid: np.ndarray,
) -> pd.DataFrame:
    lat, lon, zen, az = grid["lat"], grid["lon"], grid["zen"], grid["az"]
    rows = []
    for i, j, k, m in np.ndindex(valid.shape):
        if lon[j] == 360.0 or az[m] == 360.0:
            continue
        rows.append(
            {
                "lat": float(lat[i]),
                "lon": float(lon[j]),
                "zenith": float(zen[k]),
                "azimuth": float(az[m]),
                "has_window": bool(valid[i, j, k, m]),
                "R_low": float(r_low[i, j, k, m]) if valid[i, j, k, m] else np.nan,
                "R_high": float(r_high[i, j, k, m]) if valid[i, j, k, m] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _recover_original_nodes(grid, r_low, r_high, valid, interp_low, interp_high) -> dict:
    axes = (grid["lat"], grid["lon"], grid["zen"], grid["az"])
    max_err = 0.0
    n_ok = 0
    n_fail = 0
    for i, j, k, m in np.ndindex(valid.shape):
        point = (
            float(grid["lat"][i]),
            float(grid["lon"][j]),
            float(grid["zen"][k]),
            float(grid["az"][m]),
        )
        cube_ok = hypercube_all_valid(valid, axes, point)
        if valid[i, j, k, m]:
            if not cube_ok:
                n_fail += 1
                continue
            el = float(interp_low(point))
            eh = float(interp_high(point))
            err = max(abs(el - r_low[i, j, k, m]), abs(eh - r_high[i, j, k, m]))
            max_err = max(max_err, err)
            n_ok += 1
        else:
            if cube_ok:
                n_fail += 1
    return {"n_valid_recovered": n_ok, "n_fail": n_fail, "max_abs_err_gv": max_err}


def run_self_test() -> None:
    grid = load_boundary_file()
    r_low, r_high, valid, stats = clean_windows(grid["Rl"], grid["Ru"])

    print("=== original grid ===")
    print(f"file: {MAT_PATH}")
    print(
        "axes: lat",
        grid["lat"].tolist(),
        "| lon",
        grid["lon"].tolist(),
        "| zen",
        grid["zen"].tolist(),
        "| az",
        grid["az"].tolist(),
    )
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print(f"  unique nodes dropping lon=360 and az=360: {7 * 6 * 7 * 8}")

    orig = original_window_table(grid, r_low, r_high, valid)
    print(
        "  unique-physical has_window:",
        int(orig["has_window"].sum()),
        "/",
        len(orig),
    )

    axes = (grid["lat"], grid["lon"], grid["zen"], grid["az"])
    interp_low, interp_high = make_interpolators(axes, r_low, r_high)
    rec = _recover_original_nodes(grid, r_low, r_high, valid, interp_low, interp_high)
    print("\n=== recover original valid nodes by interpolation ===")
    print(f"  recovered: {rec['n_valid_recovered']}")
    print(f"  fail: {rec['n_fail']}")
    print(f"  max abs error GV: {rec['max_abs_err_gv']:.6e}")

    # No extrapolation: query outside lat range.
    outside = float(interp_low((-70.0, 0.0, 0.0, 0.0)))
    print("\n=== no extrapolation ===")
    print(f"  interp at lat=-70: {outside} (expect nan)")

    # Periodic copies.
    dlon = abs(float(interp_low((0.0, 0.0, 0.0, 0.0))) - float(interp_low((0.0, 360.0, 0.0, 0.0))))
    daz = abs(float(interp_low((-60.0, 0.0, 0.0, 0.0))) - float(interp_low((-60.0, 0.0, 0.0, 360.0))))
    print(f"  |R_low(lon=0)-R_low(lon=360)| at lat=0 zen=0 az=0: {dlon:.6e}")
    print(f"  |R_low(az=0)-R_low(az=360)| at lat=-60 zen=0 lon=0: {daz:.6e}")

    densified = densify_windows(grid, r_low, r_high, valid)
    print("\n=== densified query ===")
    print(f"  lat_out n={len(LAT_OUT)} {LAT_OUT.tolist()}")
    print(f"  lon_out n={len(LON_OUT)} {LON_OUT.tolist()}")
    print(f"  zen_out n={len(ZEN_OUT)} {ZEN_OUT.tolist()}")
    print(f"  az_out  n={len(AZ_OUT)}")
    print(f"  query points: {densified.attrs['n_query']}")
    print(f"  no window (NaN cell / invalid cube): {densified.attrs['n_no_window']}")
    print(f"  valid interpolated windows: {len(densified)}")
    if not densified.empty:
        print(
            "  dR_window min/median/max:",
            float(densified["dR_window"].min()),
            float(densified["dR_window"].median()),
            float(densified["dR_window"].max()),
        )
        print("  by zenith (valid counts):")
        print(densified.groupby("zenith").size().to_string())

    out_dir = PACKAGE_DIR / "data"
    orig_path = out_dir / "original_rigidity_windows.csv"
    dens_path = out_dir / "interpolated_rigidity_windows.csv"
    orig.to_csv(orig_path, index=False)
    densified.to_csv(dens_path, index=False)
    print(f"\nWrote {orig_path}")
    print(f"Wrote {dens_path}")

    assert stats["n_large_inversion_dropped"] == 0, "unexpected large inversions"
    assert rec["n_fail"] == 0
    assert rec["max_abs_err_gv"] < 1e-9
    assert np.isnan(outside)
    assert dlon < 1e-12 and daz < 1e-12
    assert len(densified) > 0
    assert densified.attrs["n_query"] == len(LAT_OUT) * len(LON_OUT) * len(ZEN_OUT) * len(AZ_OUT)
    print("\nSELF-TEST OK")


if __name__ == "__main__":
    run_self_test()
