"""Regime-conditional Black-Litterman backtest driver.

Runs `RegimeBLStrategy` end-to-end on real-data posteriors and produces a
tearsheet under the central and stress cost models. Pre-registered
hyperparameters from `STRATEGY_HYPERPARAMETERS.md` §7–8 only; no tuning.

Inputs:
  - `build/benchmarks/methods.parquet` (Day 2) — joint_hmm walk-forward
    filtered posteriors at every test-day. Provides the 3-state regime
    posterior + argmax history the strategy consumes.
  - `regime.data.portfolio_returns.build_portfolio_returns(t, tickers)` —
    18-ETF (locked subset; see UNIVERSE below) arithmetic returns + notional
    ADV inputs for the engine's cost model.

Universe note: the canonical 19-ticker set (`SPY` + 11 SECTOR_SPDRS + 6
STYLE_FACTORS + `TLT`) is reduced to 17 here. XLC (started 2018-06) and
XLRE (started 2015-10) are dropped to keep a common-history window long
enough that the 5-year strategy lookback can engage by ~2018. Universe
composition is not pre-registered in `STRATEGY_HYPERPARAMETERS.md` §11
(it points to `PLAN.md` §2); this trim is a deliberate documented choice.

Equal-weight market-cap prior (1/17 per asset) per the user's Day-4 choice:
no yfinance API calls for `marketCap`, exact reproducibility from data on
disk.

Outputs:
  - `build/backtests/regime_bl_central.parquet` (NAV path + weights)
  - `build/backtests/regime_bl_stress.parquet`
  - `build/backtests/regime_bl_equity_curves.png`

Usage:
    uv run python scripts/backtest_regime_bl.py
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dotenv import load_dotenv

load_dotenv()

from regime.backtest.costs import central_cost_model, stress_cost_model  # noqa: E402
from regime.backtest.engine import BacktestResult, run_backtest  # noqa: E402
from regime.backtest.metrics import (  # noqa: E402
    PerformanceMetrics,
    performance_metrics,
)
from regime.data.portfolio_returns import build_portfolio_returns  # noqa: E402
from regime.strategy.regime_bl import RegimeBLConfig, RegimeBLStrategy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

METHODS_PARQUET = Path("build/benchmarks/methods.parquet")
OUT_DIR = Path("build/backtests")

# 17-ETF universe (canonical 19 minus XLC, XLRE — see module docstring).
UNIVERSE: tuple[str, ...] = (
    "SPY",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "MTUM",
    "QUAL",
    "USMV",
    "VLUE",
    "SIZE",
    "VTV",
    "TLT",
)
N_ASSETS = len(UNIVERSE)
LOOKBACK_DAYS = 252 * 5
REBALANCE_BAND_BP = 30.0


def main() -> int:
    if not METHODS_PARQUET.exists():
        print(f"ERROR: {METHODS_PARQUET} missing — run scripts/run_benchmark.py first.")
        return 1
    today = date.today()

    posteriors = _load_joint_hmm_posteriors(METHODS_PARQUET)
    print(
        f"joint_hmm posteriors: {posteriors.height} rows "
        f"({posteriors['data_time'].min()} → {posteriors['data_time'].max()})"
    )

    returns_df = build_portfolio_returns(today, tickers=UNIVERSE)
    print(
        f"portfolio returns: {returns_df.height} rows × {len(UNIVERSE)} assets "
        f"({returns_df['data_time'].min()} → {returns_df['data_time'].max()})"
    )

    df = posteriors.join(returns_df, on="data_time", how="inner").sort("data_time")
    print(f"joined: {df.height} rows ({df['data_time'].min()} → {df['data_time'].max()})")

    dates = df["data_time"].to_list()
    posterior_mat = _stack_columns(df, ("filtered_0", "filtered_1", "filtered_2"))
    regime_argmax = posterior_mat.argmax(axis=1)
    returns_mat = _stack_columns(df, tuple(f"ret_{t}" for t in UNIVERSE))
    notional_mat = _stack_columns(df, tuple(f"notional_{t}" for t in UNIVERSE))

    target_weights, weights_start = _compute_target_weights(
        posterior_mat=posterior_mat,
        regime_argmax=regime_argmax,
        returns=returns_mat,
    )
    print(
        f"strategy: lookback={LOOKBACK_DAYS}d, first rebalance day index "
        f"{weights_start} ({dates[weights_start]})"
    )

    backtest_dates = dates[weights_start:]
    backtest_returns = returns_mat[weights_start:]
    backtest_targets = target_weights[weights_start:]

    daily_vol = _per_asset_daily_vol(returns_mat, window=21, eval_slice=slice(weights_start, None))
    adv21 = _per_asset_adv21(notional_mat, window=21, eval_slice=slice(weights_start, None))
    print(
        f"per-asset daily_vol mean={daily_vol.mean():.4f}, "
        f"adv21 mean=${adv21.mean():.2e}, backtest rows={len(backtest_dates)}"
    )

    central = run_backtest(
        returns=backtest_returns,
        target_weights=backtest_targets,
        tickers=list(UNIVERSE),
        cost_model=central_cost_model(),
        daily_vol=daily_vol,
        adv21_notional=adv21,
        rebalance_band_bp=REBALANCE_BAND_BP,
    )
    stress = run_backtest(
        returns=backtest_returns,
        target_weights=backtest_targets,
        tickers=list(UNIVERSE),
        cost_model=stress_cost_model(),
        daily_vol=daily_vol,
        adv21_notional=adv21,
        rebalance_band_bp=REBALANCE_BAND_BP,
    )

    central_m = performance_metrics(central.nav)
    stress_m = performance_metrics(stress.nav)
    central_turnover = _mean_daily_turnover(central.weights, backtest_targets)
    stress_turnover = _mean_daily_turnover(stress.weights, backtest_targets)

    _print_summary(central, central_m, central_turnover, stress, stress_m, stress_turnover)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_parquet(backtest_dates, central, OUT_DIR / "regime_bl_central.parquet")
    _write_parquet(backtest_dates, stress, OUT_DIR / "regime_bl_stress.parquet")
    _plot_equity_curves(
        backtest_dates,
        central.nav,
        stress.nav,
        central_m,
        stress_m,
        OUT_DIR / "regime_bl_equity_curves.png",
    )
    print(f"\nwrote {OUT_DIR / 'regime_bl_central.parquet'}")
    print(f"wrote {OUT_DIR / 'regime_bl_stress.parquet'}")
    print(f"wrote {OUT_DIR / 'regime_bl_equity_curves.png'}")
    return 0


def _load_joint_hmm_posteriors(path: Path) -> pl.DataFrame:
    posterior = pl.read_parquet(path).filter(pl.col("method") == "joint_hmm").sort("data_time")
    if posterior.is_empty():
        raise RuntimeError(f"{path}: no rows for method=joint_hmm")
    cols = posterior["raw_features"].list.len().unique().to_list()
    if cols != [3]:
        raise RuntimeError(f"expected joint_hmm raw_features width = 3 (K=3), got {cols}")
    raw = np.array(posterior["raw_features"].to_list(), dtype=np.float64)
    return posterior.select("data_time").with_columns(
        pl.Series("filtered_0", raw[:, 0]),
        pl.Series("filtered_1", raw[:, 1]),
        pl.Series("filtered_2", raw[:, 2]),
    )


def _stack_columns(df: pl.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([df[c].to_numpy().astype(np.float64) for c in cols])


def _compute_target_weights(
    posterior_mat: np.ndarray,
    regime_argmax: np.ndarray,
    returns: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Strategy walk: for each t >= LOOKBACK_DAYS, compute the strategy's
    target weights using `posterior_mat[t]` and the prior `LOOKBACK_DAYS` of
    `returns` and `regime_argmax`. Earlier rows get equal-weight (placeholder;
    these rows are sliced away before the backtest engine runs).
    """
    n_t = returns.shape[0]
    market_cap = np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float64)
    config = RegimeBLConfig()
    strategy = RegimeBLStrategy(config=config, market_cap_weights=market_cap)

    weights = np.full((n_t, N_ASSETS), 1.0 / N_ASSETS, dtype=np.float64)
    first_rebalance = LOOKBACK_DAYS
    if n_t <= first_rebalance:
        raise RuntimeError(f"not enough joined data: n_t={n_t} <= LOOKBACK_DAYS={first_rebalance}")
    for t in range(first_rebalance, n_t):
        hist_returns = returns[t - LOOKBACK_DAYS : t]
        regime_hist = regime_argmax[t - LOOKBACK_DAYS : t]
        weights[t] = strategy.target_weights_at_t(
            regime_posterior=posterior_mat[t],
            historical_returns=hist_returns,
            regime_history=regime_hist,
        )
    return weights, first_rebalance


def _per_asset_daily_vol(returns: np.ndarray, window: int, eval_slice: slice) -> np.ndarray:
    """Per-asset daily vol = median of rolling `window`-day std over the
    evaluation slice. Constant across the backtest (engine signature requires
    a `(N,)` array, not `(T, N)`)."""
    T = returns.shape[0]
    if T < window:
        return np.std(returns, axis=0, ddof=1)
    rolled = np.array([returns[t - window : t].std(axis=0, ddof=1) for t in range(window, T)])
    start = max(eval_slice.start - window, 0) if eval_slice.start else 0
    eval_rolled = rolled[start:]
    if eval_rolled.shape[0] == 0:
        return rolled.mean(axis=0)
    return np.median(eval_rolled, axis=0)


def _per_asset_adv21(notional: np.ndarray, window: int, eval_slice: slice) -> np.ndarray:
    """Per-asset 21-day rolling mean notional, then median over the evaluation
    slice. Same shape-constant rationale as `_per_asset_daily_vol`."""
    T = notional.shape[0]
    if T < window:
        return notional.mean(axis=0)
    rolled = np.array([notional[t - window : t].mean(axis=0) for t in range(window, T)])
    start = max(eval_slice.start - window, 0) if eval_slice.start else 0
    eval_rolled = rolled[start:]
    if eval_rolled.shape[0] == 0:
        return rolled.mean(axis=0)
    return np.median(eval_rolled, axis=0)


def _mean_daily_turnover(actual_weights: np.ndarray, target_weights: np.ndarray) -> float:
    """Mean L1 turnover across rebalance attempts (post-band)."""
    if actual_weights.shape[0] < 2:
        return 0.0
    diffs = np.abs(np.diff(actual_weights, axis=0)).sum(axis=1)
    return float(diffs.mean())


def _print_summary(
    central: BacktestResult,
    central_m: PerformanceMetrics,
    central_turnover: float,
    stress: BacktestResult,
    stress_m: PerformanceMetrics,
    stress_turnover: float,
) -> None:
    def _fmt(m: PerformanceMetrics, turnover: float, rebalances: int) -> str:
        return (
            f"  ann_ret={m.annualized_return:>7.2%}   "
            f"ann_vol={m.annualized_vol:>6.2%}   "
            f"sharpe={m.sharpe:>5.2f}   "
            f"mdd={m.max_drawdown:>6.2%}   "
            f"turnover/day={turnover:>6.4f}   rebalances={rebalances}"
        )

    print("\nBacktest results:")
    print("  central cost:")
    print(_fmt(central_m, central_turnover, central.rebalance_count))
    print("  stress cost:")
    print(_fmt(stress_m, stress_turnover, stress.rebalance_count))


def _write_parquet(dates: list[date], result: BacktestResult, out_path: Path) -> None:
    pl.DataFrame(
        {
            "data_time": dates,
            "nav": result.nav,
            "cumulative_cost": result.cumulative_cost,
            **{f"w_{t}": result.weights[:, i] for i, t in enumerate(UNIVERSE)},
        },
        schema={
            "data_time": pl.Date,
            "nav": pl.Float64,
            "cumulative_cost": pl.Float64,
            **{f"w_{t}": pl.Float64 for t in UNIVERSE},
        },
    ).write_parquet(out_path)


def _plot_equity_curves(
    dates: list[date],
    nav_central: np.ndarray,
    nav_stress: np.ndarray,
    central_m: PerformanceMetrics,
    stress_m: PerformanceMetrics,
    out_path: Path,
) -> None:
    fig, (ax_nav, ax_dd) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_nav.plot(
        dates,
        nav_central,
        linewidth=1.6,
        color="#1f77b4",
        label=f"central (Sharpe={central_m.sharpe:.2f}, MDD={central_m.max_drawdown:.1%})",
    )
    ax_nav.plot(
        dates,
        nav_stress,
        linewidth=1.6,
        color="#d62728",
        label=f"stress (Sharpe={stress_m.sharpe:.2f}, MDD={stress_m.max_drawdown:.1%})",
    )
    ax_nav.set_ylabel("NAV (starts at 1.0)")
    ax_nav.set_title("Regime-conditional Black-Litterman — equity curves")
    ax_nav.set_yscale("log")
    ax_nav.legend(loc="upper left")
    ax_nav.grid(True, alpha=0.3)

    for nav, color, label in (
        (nav_central, "#1f77b4", "central"),
        (nav_stress, "#d62728", "stress"),
    ):
        peaks = np.maximum.accumulate(nav)
        dd = (peaks - nav) / np.maximum(peaks, 1e-12)
        ax_dd.fill_between(dates, 0.0, -dd, color=color, alpha=0.35, label=label)
    ax_dd.set_ylabel("drawdown")
    ax_dd.set_xlabel("date")
    ax_dd.legend(loc="lower left")
    ax_dd.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
