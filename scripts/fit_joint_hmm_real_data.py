"""Fit the joint cross-sectional HMM on real data.

The headline visual: 23 years of daily US equity returns + Fama-French 5+Mom
factors → fit a 3-state joint HMM with FF-factor regime-switching means and
rank-3 latent factor covariance → inspect the discovered regimes.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from dotenv import load_dotenv

load_dotenv()

from regime.data.query import as_of, as_of_fama_french  # noqa: E402
from regime.features.registry import log_total_return  # noqa: E402
from regime.models.joint_hmm import JointHmm  # noqa: E402

OBSERVATION_TICKERS = ("SPY", "XLK", "XLF", "XLE", "XLV", "TLT")
FF_COLUMNS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")
RENAMED_FF = {f: f"ff_{f.lower().replace('-', '_')}" for f in FF_COLUMNS}


def build_wide_dataframe(t: date) -> pl.DataFrame:
    """Return DataFrame with columns: data_time, ret_<ticker>..., ff_<factor>..."""
    ticker_frames: list[pl.DataFrame] = []
    for ticker in OBSERVATION_TICKERS:
        ohlcv = as_of(ticker, t)
        rets = log_total_return(ohlcv).alias(f"ret_{ticker}")
        ticker_frames.append(ohlcv.select("data_time").with_columns(rets))

    combined = ticker_frames[0]
    for df in ticker_frames[1:]:
        combined = combined.join(df, on="data_time", how="inner")

    ff = as_of_fama_french(t).select("data_time", *FF_COLUMNS)
    combined = combined.join(ff, on="data_time", how="inner")
    combined = combined.rename(RENAMED_FF)
    return combined.drop_nulls().sort("data_time")


def main() -> int:
    print("=" * 70)
    print("Joint cross-sectional HMM on real data")
    print("=" * 70)

    today = date.today()
    df = build_wide_dataframe(today)
    print(f"Wide DataFrame: {df.height} rows × {df.width} columns")
    print(f"  date range: {df['data_time'].min()} → {df['data_time'].max()}")
    obs_cols = tuple(f"ret_{t}" for t in OBSERVATION_TICKERS)
    fact_cols = tuple(RENAMED_FF[f] for f in FF_COLUMNS)
    print(f"  observation columns: {obs_cols}")
    print(f"  factor columns:      {fact_cols}")

    print("\nFitting joint HMM (K=3, rank=3, n_restarts=3) ...")
    model = JointHmm(
        K=3,
        observation_columns=obs_cols,
        factor_columns=fact_cols,
        latent_factor_rank=3,
        n_restarts=3,
        max_iter=50,
        random_state=42,
    )
    train_idx = np.arange(df.height, dtype=np.int64)
    model.fit(df, train_idx)
    state = model.state_dict()
    print("done.\n")

    K = state["K"]
    alpha = np.array(state["alpha"])
    A = np.array(state["A"])
    pi = np.array(state["pi"])

    print("Stationary state distribution (initial fold):")
    print(f"  π = {pi.round(3)}\n")

    print("Transition matrix:")
    for i in range(K):
        row = "  ".join(f"{a:.3f}" for a in A[i])
        print(f"  state {i}: [{row}]")
    print()

    annualizer = 252.0
    print("Per-regime annualized intercept (α) by asset:")
    print(f"  {'asset':>6}  " + "  ".join(f"state{i:>2}" for i in range(K)))
    for j, ticker in enumerate(OBSERVATION_TICKERS):
        row = "  ".join(f"{alpha[k, j] * annualizer:>6.1%}" for k in range(K))
        print(f"  {ticker:>6}  {row}")
    print()

    # Filter the entire history; report the latest-state posterior.
    posterior = model.filter(df, train_idx)
    print("Most recent filtered posterior (last 5 trading days):")
    last_dates = df["data_time"].to_list()[-5:]
    for i, d in enumerate(last_dates):
        pp = posterior[-5 + i]
        line = " ".join(f"P(s={k})={pp[k]:.3f}" for k in range(K))
        print(f"  {d}  {line}")
    print()

    state_seq = posterior.argmax(axis=1)
    counts = np.bincount(state_seq, minlength=K)
    print("Empirical regime occupancy over full history:")
    for k in range(K):
        pct = counts[k] / len(state_seq)
        print(f"  state {k}: {counts[k]:5d} days ({pct:.1%})")
    print()

    # SPY annualized return per regime — the "what does each regime look like"
    # snapshot.
    spy_returns = df["ret_SPY"].to_numpy()
    print("Realized SPY behaviour by inferred regime:")
    print(f"  {'state':>6}  {'ann.ret':>10}  {'ann.vol':>10}  {'days':>6}")
    for k in range(K):
        mask = state_seq == k
        if mask.sum() > 1:
            ar = spy_returns[mask].mean() * annualizer
            av = spy_returns[mask].std(ddof=1) * np.sqrt(annualizer)
            n = int(mask.sum())
            print(f"  {k:>6}  {ar:>9.1%}  {av:>9.1%}  {n:>6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
