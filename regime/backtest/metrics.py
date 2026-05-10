"""Performance metrics for backtest results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PerformanceMetrics:
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    final_nav: float
    n_periods: int


def equity_curve_to_returns(nav: np.ndarray) -> np.ndarray:
    nav = np.asarray(nav, dtype=np.float64)
    if len(nav) < 2:
        return np.array([], dtype=np.float64)
    return nav[1:] / nav[:-1] - 1.0


def annualized_return(nav: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    nav = np.asarray(nav, dtype=np.float64)
    if len(nav) < 2 or nav[0] <= 0.0:
        return 0.0
    total_ret = nav[-1] / nav[0]
    years = (len(nav) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    if total_ret <= 0.0:
        return -1.0
    return float(total_ret ** (1.0 / years) - 1.0)


def annualized_vol(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio."""
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0
    # If the input is (numerically) constant, Sharpe is undefined; return 0.
    if float(np.ptp(returns)) < 1e-15:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    sd = float(np.std(excess, ddof=1))
    if sd <= 1e-15:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def max_drawdown(nav: np.ndarray) -> float:
    """Worst peak-to-trough drawdown along the equity curve. Returns a non-negative fraction."""
    nav = np.asarray(nav, dtype=np.float64)
    if len(nav) == 0:
        return 0.0
    running_max = np.maximum.accumulate(nav)
    drawdown = (nav - running_max) / running_max
    return float(-drawdown.min()) if drawdown.size else 0.0


def performance_metrics(
    nav: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> PerformanceMetrics:
    nav = np.asarray(nav, dtype=np.float64)
    rets = equity_curve_to_returns(nav)
    return PerformanceMetrics(
        annualized_return=annualized_return(nav, periods_per_year),
        annualized_vol=annualized_vol(rets, periods_per_year),
        sharpe=sharpe_ratio(rets, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(nav),
        final_nav=float(nav[-1]) if len(nav) else 0.0,
        n_periods=len(nav),
    )
