"""Full-history regime path: SPY price (log) + filtered crisis probability.

Reads `build/paper/regime_path.parquet` (data_time, state, crisis_prob,
spy_close), materialized by `scripts/build_paper_inputs.py`. The top panel
shades SPY's log-price by the most-likely regime per day; the bottom panel
overlays the filtered crisis-state probability with a 0.5 reference line.
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

DEFAULT_INPUT = Path("build/paper/regime_path.parquet")

REGIME_COLORS = {
    0: (0.737, 0.831, 0.902, 0.55),  # normal — pale blue
    1: (0.796, 0.878, 0.784, 0.55),  # calm bull — pale green
    2: (0.957, 0.722, 0.694, 0.85),  # crisis — pale red
}
REGIME_LABELS = {
    0: "State 0 — normal expansion",
    1: "State 1 — calm bull / low-vol",
    2: "State 2 — crisis",
}


def _contiguous_spans(seq: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (start, end_exclusive, value) runs for a 1-D integer sequence."""
    if seq.size == 0:
        return []
    spans: list[tuple[int, int, int]] = []
    cur = int(seq[0])
    start = 0
    for i in range(1, len(seq)):
        if int(seq[i]) != cur:
            spans.append((start, i, cur))
            start = i
            cur = int(seq[i])
    spans.append((start, len(seq), cur))
    return spans


def main() -> int:
    inp, output = parse_io_paths(DEFAULT_INPUT)
    configure_style()

    df = pl.read_parquet(inp).sort("data_time")
    dates = np.array(df["data_time"].to_list(), dtype="O")
    state_seq = df["state"].to_numpy().astype(np.int64)
    crisis_prob = df["crisis_prob"].to_numpy()
    spy_close = df["spy_close"].to_numpy()

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (ax_price, ax_crisis) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(FIGURE_WIDTH_INCHES * 1.6, FIGURE_HEIGHT_INCHES * 1.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    for s, e, k in _contiguous_spans(state_seq):
        ax_price.axvspan(dates[s], dates[e - 1], color=REGIME_COLORS[k], linewidth=0)
    ax_price.semilogy(dates, spy_close, color="black", linewidth=0.7)
    ax_price.set_ylabel("SPY (USD, log)")
    ax_price.set_title("Joint HMM regime path on US equities")
    legend_handles = [Patch(facecolor=REGIME_COLORS[k], label=REGIME_LABELS[k]) for k in (0, 1, 2)]
    ax_price.legend(handles=legend_handles, loc="upper left", framealpha=0.9, fontsize=7)

    ax_crisis.plot(dates, crisis_prob, color="#b22222", linewidth=0.7)
    ax_crisis.axhline(0.5, color="0.6", linestyle="--", linewidth=0.6)
    ax_crisis.set_ylabel("P(crisis)")
    ax_crisis.set_ylim(-0.02, 1.02)
    ax_crisis.set_xlabel("Date")
    ax_crisis.xaxis.set_major_locator(mdates.YearLocator(3))
    ax_crisis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    save_and_close(fig, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
