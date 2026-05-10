"""Performance-metric tests."""

from __future__ import annotations

import numpy as np
import pytest

from regime.backtest.metrics import (
    annualized_return,
    annualized_vol,
    equity_curve_to_returns,
    max_drawdown,
    performance_metrics,
    sharpe_ratio,
)


def test_annualized_return_zero_drift_is_zero():
    nav = np.ones(252)
    assert annualized_return(nav) == pytest.approx(0.0)


def test_annualized_return_doubles_in_one_year():
    nav = np.linspace(1.0, 2.0, 253)
    ann = annualized_return(nav, periods_per_year=252)
    assert ann == pytest.approx(1.0, abs=0.01)


def test_max_drawdown_simple():
    nav = np.array([1.0, 1.1, 1.2, 0.9, 1.0])
    # Peak at 1.2 → trough at 0.9 → drawdown = (1.2 - 0.9) / 1.2 = 0.25.
    assert max_drawdown(nav) == pytest.approx(0.25)


def test_max_drawdown_monotonic_uptrend_is_zero():
    nav = np.linspace(1.0, 2.0, 100)
    assert max_drawdown(nav) == pytest.approx(0.0)


def test_sharpe_constant_returns_zero():
    rets = np.full(100, 0.001)
    # Zero std → Sharpe defined as 0 (per our convention).
    assert sharpe_ratio(rets) == pytest.approx(0.0)


def test_sharpe_positive_when_excess_positive():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.005, size=2000)
    assert sharpe_ratio(rets) > 0.0


def test_annualized_vol_scales_with_sqrt_periods():
    rets = np.full(100, 0.0)
    rets[0] = 0.01
    rets[1] = -0.01
    daily_std = float(np.std(rets, ddof=1))
    expected = daily_std * np.sqrt(252)
    assert annualized_vol(rets) == pytest.approx(expected)


def test_performance_metrics_smoke():
    rng = np.random.default_rng(1)
    daily = rng.normal(0.0005, 0.01, size=500)
    nav = np.concatenate(([1.0], np.cumprod(1.0 + daily)))
    pm = performance_metrics(nav)
    assert pm.n_periods == len(nav)
    assert pm.final_nav == pytest.approx(nav[-1])
    assert pm.max_drawdown >= 0.0


def test_equity_curve_to_returns_short_input():
    assert len(equity_curve_to_returns(np.array([1.0]))) == 0
