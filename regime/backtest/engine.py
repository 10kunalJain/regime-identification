"""Vectorized backtest engine.

Inputs:
  - returns:        shape (T, N) per-period asset returns (decimal)
  - target_weights: shape (T, N) target portfolio weights from the strategy
  - tickers:        list of N ticker strings (drives cost-tier lookup)
  - cost_model:     `CostModel` (central or stress)
  - daily_vol:      shape (N,) per-asset daily volatility (decimal) for impact
  - adv21_notional: shape (N,) per-asset 21-day average daily notional volume

The engine applies costs from turnover, with a turnover-band rebalance rule
(skip rebalance if total weight change < `rebalance_band_bp` bps). Initial
allocation at t=0 is applied without cost (assumes cash deployment, not a
rebalance from existing positions).

Output: `BacktestResult` with NAV path, post-trade weights, and cumulative cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regime.backtest.costs import CostModel


@dataclass(frozen=True)
class BacktestResult:
    nav: np.ndarray  # shape (T,)
    weights: np.ndarray  # shape (T, N) — actual weights after rebalance + drift
    cumulative_cost: np.ndarray  # shape (T,) cumulative cost in NAV units
    rebalance_count: int


def run_backtest(
    returns: np.ndarray,
    target_weights: np.ndarray,
    tickers: list[str],
    cost_model: CostModel,
    daily_vol: np.ndarray,
    adv21_notional: np.ndarray,
    initial_capital: float = 1.0,
    rebalance_band_bp: float = 30.0,
) -> BacktestResult:
    returns = np.asarray(returns, dtype=np.float64)
    target_weights = np.asarray(target_weights, dtype=np.float64)
    if returns.shape != target_weights.shape:
        raise ValueError(
            f"shape mismatch: returns {returns.shape} vs target_weights {target_weights.shape}"
        )
    T, N = returns.shape
    if len(tickers) != N:
        raise ValueError(f"tickers length {len(tickers)} != N={N}")

    daily_vol = np.asarray(daily_vol, dtype=np.float64).reshape(-1)
    adv21_notional = np.asarray(adv21_notional, dtype=np.float64).reshape(-1)

    nav = np.zeros(T, dtype=np.float64)
    weights = np.zeros((T, N), dtype=np.float64)
    cum_cost = np.zeros(T, dtype=np.float64)

    nav[0] = initial_capital
    weights[0] = target_weights[0]
    rebalance_count = 0
    band = rebalance_band_bp / 1e4

    for t in range(1, T):
        ret_t = returns[t]
        # Drift: held position grows by 1 + r_t
        gross = weights[t - 1] * (1.0 + ret_t)
        portfolio_return = float(weights[t - 1] @ ret_t)
        nav_after_return = nav[t - 1] * (1.0 + portfolio_return)
        denom = float(gross.sum()) if gross.sum() > 0 else 1.0
        drifted = gross / denom

        target = target_weights[t]
        turnover = float(np.abs(target - drifted).sum())
        if turnover > band:
            trade_notional_per_asset = nav_after_return * np.abs(target - drifted)
            cost_total = 0.0
            for i in range(N):
                if trade_notional_per_asset[i] <= 0.0:
                    continue
                cost_bps_i = cost_model.trade_cost_bps(
                    tickers[i],
                    float(trade_notional_per_asset[i]),
                    float(daily_vol[i]),
                    float(adv21_notional[i]),
                )
                cost_total += float(trade_notional_per_asset[i]) * cost_bps_i / 1e4
            nav[t] = nav_after_return - cost_total
            cum_cost[t] = cum_cost[t - 1] + cost_total
            weights[t] = target
            rebalance_count += 1
        else:
            nav[t] = nav_after_return
            cum_cost[t] = cum_cost[t - 1]
            weights[t] = drifted

    return BacktestResult(
        nav=nav,
        weights=weights,
        cumulative_cost=cum_cost,
        rebalance_count=rebalance_count,
    )
