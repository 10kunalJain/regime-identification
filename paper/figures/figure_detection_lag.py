"""Cross-method × cross-crisis detection-lag heatmap.

Reads `build/benchmarks/methods_crisis_lag.parquet` (per-method × per-crisis
sustained-fire lag relative to the -5% drawdown anchor; see
`regime/eval/crises.py` for the canonical event registry) and renders a
methods × crises matrix with lag in trading days as the cell value. Cells
where the crisis fell outside the method's walk-forward eval window are
greyed out. Negative lags (lead) and positive lags (lag) share a divergent
colormap centered on zero.
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

DEFAULT_INPUT = Path("build/benchmarks/methods_crisis_lag.parquet")


def main() -> int:
    inp, output = parse_io_paths(DEFAULT_INPUT)
    configure_style()

    df = pl.read_parquet(inp)
    methods = df["method"].unique().sort().to_list()
    crises = df["crisis_name"].unique(maintain_order=True).to_list()
    method_idx = {m: i for i, m in enumerate(methods)}
    crisis_idx = {c: j for j, c in enumerate(crises)}

    lag_matrix = np.full((len(methods), len(crises)), np.nan, dtype=np.float64)
    in_window = np.zeros((len(methods), len(crises)), dtype=bool)
    for row in df.iter_rows(named=True):
        i = method_idx[row["method"]]
        j = crisis_idx[row["crisis_name"]]
        in_window[i, j] = bool(row["in_eval_window"])
        if row["in_eval_window"] and row["lag_m5"] is not None:
            lag_matrix[i, j] = float(row["lag_m5"])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES * 1.1, FIGURE_HEIGHT_INCHES * 1.1))
    # Divergent colormap centered at zero; clipped to the empirical extremes.
    finite = lag_matrix[np.isfinite(lag_matrix)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    im = ax.imshow(
        lag_matrix,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    # Annotate each cell — value if in-window-and-fired, "—" otherwise.
    for i in range(len(methods)):
        for j in range(len(crises)):
            if np.isfinite(lag_matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{int(lag_matrix[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if abs(lag_matrix[i, j]) < 0.5 * vmax else "white",
                )
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="0.5")

    ax.set_xticks(range(len(crises)))
    ax.set_xticklabels(crises, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_title("Detection lag (trading days vs −5% anchor) by method × crisis")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Lag (days)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    save_and_close(fig, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
