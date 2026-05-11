"""Visualize the regime path: SPY price + inferred regime + crisis probability.

Two-panel figure. Top panel: SPY price on a log scale with the background shaded
by the most-likely regime per day. Bottom panel: filtered crisis probability,
with the 8 historical-crisis m5 anchor dates marked as vertical lines for
visual sanity-checking against the registry.

Caches the fitted joint-HMM parameters in `build/joint_hmm_real.json` so the
plot can be re-run quickly without re-fitting (delete the cache to refit).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from regime.data.joint_dataset import (  # noqa: E402
    FF_COLUMNS,
    OBSERVATION_TICKERS,
    RENAMED_FF,
    build_wide_dataframe,
)
from regime.data.query import as_of  # noqa: E402
from regime.eval.crises import CRISES  # noqa: E402
from regime.models.joint_hmm import JointHmm  # noqa: E402

CACHE_PATH = Path("build/joint_hmm_real.json")
OUTPUT_PATH = Path("build/regime_path.png")


def _fit_or_load(
    df: pl.DataFrame, obs_cols: tuple[str, ...], fact_cols: tuple[str, ...]
) -> JointHmm:
    model = JointHmm(
        K=3,
        observation_columns=obs_cols,
        factor_columns=fact_cols,
        latent_factor_rank=3,
        n_restarts=3,
        max_iter=50,
        random_state=42,
    )
    if CACHE_PATH.exists():
        print(f"loading cached params from {CACHE_PATH}")
        with CACHE_PATH.open() as f:
            model.load_state_dict(json.load(f))
        return model

    print("fitting joint HMM on real data ...")
    model.fit(df, np.arange(df.height, dtype=np.int64))
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w") as f:
        json.dump(model.state_dict(), f)
    print(f"  cached params → {CACHE_PATH}")
    return model


def _spy_price_series(t: date) -> tuple[np.ndarray, np.ndarray]:
    """Return (dates, close) for SPY up to t."""
    spy = as_of("SPY", t)
    return (
        np.array(spy["data_time"].to_list(), dtype="O"),
        spy["close"].to_numpy(),
    )


def main() -> int:
    today = date.today()
    df = build_wide_dataframe(today)
    obs_cols = tuple(f"ret_{x}" for x in OBSERVATION_TICKERS)
    fact_cols = tuple(RENAMED_FF[f] for f in FF_COLUMNS)
    model = _fit_or_load(df, obs_cols, fact_cols)

    print("computing filtered posterior over full history ...")
    posterior = model.filter(df, np.arange(df.height, dtype=np.int64))
    state_seq = posterior.argmax(axis=1)
    crisis_prob = posterior[:, -1]  # state 2 = crisis (SPY-mean-descending sort)

    # SPY price for the same date range — align by date.
    df_dates = np.array(df["data_time"].to_list(), dtype="O")
    spy_dates, spy_close = _spy_price_series(today)
    # Restrict SPY to the joint-HMM date range.
    keep = np.isin(spy_dates, df_dates)
    spy_dates = spy_dates[keep]
    spy_close = spy_close[keep]

    print("rendering figure ...")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, (ax_price, ax_crisis) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top panel: SPY price (log scale) with regime-band shading.
    regime_colors = {
        0: to_rgba("#bcd4e6", 0.55),  # state 0: normal — pale blue
        1: to_rgba("#cbe0c8", 0.55),  # state 1: calm bull — pale green
        2: to_rgba("#f4b8b1", 0.85),  # state 2: crisis — pale red
    }
    # Plot regime bands as a series of vertical spans for contiguous runs.
    spans: list[tuple[int, int, int]] = []  # (start_idx, end_idx_exclusive, regime)
    cur = int(state_seq[0])
    start = 0
    for i in range(1, len(state_seq)):
        if int(state_seq[i]) != cur:
            spans.append((start, i, cur))
            start = i
            cur = int(state_seq[i])
    spans.append((start, len(state_seq), cur))

    for s, e, k in spans:
        ax_price.axvspan(df_dates[s], df_dates[e - 1], color=regime_colors[k], linewidth=0)

    ax_price.semilogy(spy_dates, spy_close, color="black", linewidth=0.8)
    ax_price.set_ylabel("SPY (USD, log scale)")
    ax_price.set_title(
        "Joint HMM regime path on US equities, 2003–2026 "
        "(SPY price; background colored by inferred regime)"
    )
    ax_price.spines["top"].set_visible(False)
    ax_price.spines["right"].set_visible(False)

    # Legend for regime colors.
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=regime_colors[0], label="State 0 — normal expansion"),
        Patch(facecolor=regime_colors[1], label="State 1 — calm bull / low-vol"),
        Patch(facecolor=regime_colors[2], label="State 2 — crisis"),
    ]
    ax_price.legend(handles=legend_handles, loc="upper left", framealpha=0.9, fontsize=8)

    # Bottom panel: crisis probability + crisis-registry m5 dates.
    ax_crisis.plot(df_dates, crisis_prob, color="#b22222", linewidth=0.7)
    ax_crisis.axhline(0.5, color="0.6", linestyle="--", linewidth=0.6)
    ax_crisis.set_ylabel("P(crisis)")
    ax_crisis.set_ylim(-0.02, 1.02)

    for crisis in CRISES:
        m5 = crisis.m5_date
        if df_dates[0] <= m5 <= df_dates[-1]:
            ax_crisis.axvline(m5, color="0.3", linewidth=0.5, linestyle=":")
            ax_crisis.text(
                m5,
                1.04,
                crisis.name.split()[0:2],  # short label
                ha="center",
                va="bottom",
                fontsize=6,
                color="0.3",
                rotation=0,
            )
    ax_crisis.spines["top"].set_visible(False)
    ax_crisis.spines["right"].set_visible(False)
    ax_crisis.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_crisis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_crisis.set_xlabel("Date")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUTPUT_PATH}")
    print(f"  size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
