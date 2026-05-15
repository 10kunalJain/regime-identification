"""Reliability-diagram figure for the calibrated crisis head, real data.

Reads the cross-validated OOF predictions written by `scripts/fit_crisis_head.py`
(`build/benchmarks/crisis_head.parquet`) and renders the empirical
calibration curve against the observable forward-drawdown indicator.
Bin counts are drawn on a secondary axis so over- and under-confident bins
are visible at a glance.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl
from _common import (
    FIGURE_HEIGHT_INCHES,
    FIGURE_WIDTH_INCHES,
    configure_style,
    parse_io_paths,
    save_and_close,
)

from regime.ensemble.calibration import reliability_curve

DEFAULT_INPUT = Path("build/benchmarks/crisis_head.parquet")


def main() -> int:
    inp, output = parse_io_paths(DEFAULT_INPUT)
    configure_style()

    df = pl.read_parquet(inp).drop_nulls(subset=["oof_calibrated", "label"])
    # `oof_calibrated` can contain NaN for the trailing-horizon rows where the
    # forward-drawdown label is unobservable; drop them before calibrating.
    proba = df["oof_calibrated"].to_numpy()
    realized = df["label"].to_numpy().astype(np.int64)
    finite = np.isfinite(proba)
    proba = proba[finite]
    realized = realized[finite]
    rc = reliability_curve(proba, realized, n_bins=10)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    ax.plot([0, 1], [0, 1], color="0.5", linestyle=":", linewidth=0.8, label="Perfect calibration")
    valid = ~np.isnan(rc.mean_predicted)
    ax.plot(
        rc.mean_predicted[valid],
        rc.mean_observed[valid],
        marker="o",
        color="black",
        linewidth=1.0,
        markersize=4,
        label="Empirical (OOF calibrated)",
    )
    counts_axis = ax.twinx()
    counts_axis.bar(
        (rc.bin_lower + rc.bin_upper) / 2.0,
        rc.bin_count,
        width=(rc.bin_upper - rc.bin_lower) * 0.9,
        color="0.85",
        zorder=0,
    )
    counts_axis.set_ylabel("Bin count")
    counts_axis.spines["top"].set_visible(False)

    ax.set_xlabel("Predicted P(crisis within 21d)")
    ax.set_ylabel("Empirical frequency")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Reliability diagram of the calibrated crisis head (OOF, walk-forward)")

    save_and_close(fig, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
