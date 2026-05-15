"""Headline figure: filtered vs smoothed crisis posterior over COVID, real data.

The central wedge of the paper: smoothed posteriors use information from the
entire fold and assign high crisis probability several days before any
real-time observer could. We show the joint-HMM-fit filtered and smoothed
crisis-state posteriors over the Feb-Jun 2020 window from the cached fit
in `build/joint_hmm_real.json`; the parquet that backs this figure is
materialized by `scripts/build_paper_inputs.py` so that this script is a
thin reader and runs in CI against the committed fixture.
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

DEFAULT_INPUT = Path("build/paper/posterior_covid.parquet")


def main() -> int:
    inp, output = parse_io_paths(DEFAULT_INPUT)
    configure_style()

    df = pl.read_parquet(inp).sort("data_time")
    dates = np.array(df["data_time"].to_list(), dtype="O")
    filtered = df["filtered_crisis"].to_numpy()
    smoothed = df["smoothed_crisis"].to_numpy()

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    ax.plot(
        dates,
        smoothed,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"Smoothed $P(s_t = \text{crisis} \mid y_{1:T})$",
    )
    ax.plot(
        dates,
        filtered,
        color="black",
        linestyle="-",
        linewidth=1.0,
        label=r"Filtered $P(s_t = \text{crisis} \mid y_{1:t})$",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Posterior probability")
    ax.set_ylim(-0.02, 1.02)
    if len(dates) > 0:
        ax.set_xlim(dates[0], dates[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Smoothed vs filtered crisis posterior (joint HMM, COVID window)")

    save_and_close(fig, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
