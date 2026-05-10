"""Reliability-diagram figure for the calibrated crisis head.

Synthetic data with a controllable miscalibration so the figure illustrates
the methodology before real-data results land. Real-data analogue: replace
the synthetic `(predicted, observed)` pair with the crisis-head's calibrated
output and the observable forward-drawdown indicator over the held-out folds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _common import (
    FIGURE_HEIGHT_INCHES,
    FIGURE_WIDTH_INCHES,
    configure_style,
    parse_output_path,
    save_and_close,
)

from regime.ensemble.calibration import reliability_curve


def main() -> int:
    output = parse_output_path()
    configure_style()

    rng = np.random.default_rng(0)
    n = 5000
    # Mildly miscalibrated: the model tends to over-confidence at the extremes.
    proba = rng.beta(0.7, 0.7, size=n)
    realized = (rng.uniform(0, 1, size=n) < proba**1.2).astype(np.int64)
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
        label="Empirical",
    )
    # Bin counts as a small bar chart on a secondary axis.
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
    ax.set_title("Reliability diagram of the calibrated crisis head (synthetic)")

    save_and_close(fig, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
