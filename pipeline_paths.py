"""Portable path resolution shared by the pipeline and plotting scripts."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_BASE = PROJECT_DIR / "runs" / "5d_raw_default_requested_scans"
ACTIVE_RUN_FILE = PROJECT_DIR / "active_run.txt"


def runs_base() -> Path:
    """Return the run parent directory, optionally overridden by the user."""
    override = os.environ.get("PENUMBRA_RUNS_DIR")
    return Path(override).expanduser().resolve() if override else DEFAULT_RUNS_BASE


def activate_run(run_directory: Path) -> Path:
    """Record the current run for subsequent stages on this checkout."""
    resolved = Path(run_directory).expanduser().resolve()
    ACTIVE_RUN_FILE.write_text(f"{resolved}\n", encoding="utf-8")
    return resolved


def resolve_run_dir() -> Path:
    """Resolve a run from PENUMBRA_RUN_DIR or the local active-run pointer."""
    override = os.environ.get("PENUMBRA_RUN_DIR")
    if override:
        return Path(override).expanduser().resolve()

    if ACTIVE_RUN_FILE.is_file():
        value = ACTIVE_RUN_FILE.read_text(encoding="utf-8").strip()
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = PROJECT_DIR / path
            return path.resolve()

    raise RuntimeError(
        "No active run is configured. Run 5d_raw_scan.py first or set "
        "PENUMBRA_RUN_DIR to an existing run directory."
    )


def resolve_plot_output_dir(run_directory: Path | None = None) -> Path:
    """Return the plot directory, defaulting to <run>/plots."""
    override = os.environ.get("PENUMBRA_PLOT_OUTPUT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    run_path = Path(run_directory) if run_directory else resolve_run_dir()
    return run_path / "plots"
