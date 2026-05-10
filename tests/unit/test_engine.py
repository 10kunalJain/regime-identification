"""Backtest-engine tests on synthetic returns + benchmark strategies."""

from __future__ import annotations

import numpy as np
import pytest

from regime.backtest.costs import central_cost_model, stress_cost_model
from regime.backtest.engine import run_backtest
from regime.backtest.strategies import (
    buy_and_hold,
    equal_weight,
    fixed_weights,
    naive_regime,
    risk_parity,
)


def _toy_returns(T: int = 500, N: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, size=(T, N))


def _zero_costs():
    """A cost model with zero commission, zero spread, zero impact."""
    from regime.backtest.costs import CostModel

    return CostModel(
        name="zero",
        commission_bps=0.0,
        half_spread_bps_by_tier={"spy": 0.0, "sector": 0.0, "factor": 0.0, "tlt": 0.0},
        impact_eta=0.0,
        volume_floor_multiplier=1.0,
    )


def test_buy_and_hold_matches_geometric_product_under_zero_costs():
    """Engine convention: nav[0] = initial; nav[t] = nav[t-1] * (1 + ret[t, asset]) for t >= 1.

    This convention treats `returns[0]` as undefined / unused (initial allocation
    happens at t=0 with no return earned). Returns at t >= 1 are realized.
    """
    rets = _toy_returns(T=300, N=4, seed=3)
    target = buy_and_hold(rets.shape[0], rets.shape[1], asset_idx=0)
    res = run_backtest(
        rets,
        target,
        tickers=["SPY", "XLF", "MTUM", "TLT"],
        cost_model=_zero_costs(),
        daily_vol=np.array([0.01, 0.012, 0.013, 0.005]),
        adv21_notional=np.array([1e9, 1e8, 5e7, 5e8]),
    )
    expected = np.concatenate(([1.0], np.cumprod(1.0 + rets[1:, 0])))
    np.testing.assert_allclose(res.nav, expected, rtol=1e-10)


def test_zero_cost_does_not_reduce_nav():
    rets = _toy_returns(T=200, N=3, seed=2)
    target = equal_weight(rets.shape[0], rets.shape[1])
    res_zero = run_backtest(
        rets,
        target,
        tickers=["SPY", "XLF", "TLT"],
        cost_model=_zero_costs(),
        daily_vol=np.full(3, 0.01),
        adv21_notional=np.full(3, 1e9),
    )
    assert res_zero.cumulative_cost[-1] == pytest.approx(0.0)


def test_central_costs_reduce_nav_relative_to_zero_costs():
    rets = _toy_returns(T=200, N=3, seed=4)
    # Forced rebalance every step by alternating between two target vectors.
    targets = np.zeros((rets.shape[0], 3))
    for t in range(rets.shape[0]):
        targets[t] = np.array([1.0, 0.0, 0.0]) if t % 2 == 0 else np.array([0.0, 1.0, 0.0])
    res_zero = run_backtest(
        rets,
        targets,
        tickers=["SPY", "XLF", "TLT"],
        cost_model=_zero_costs(),
        daily_vol=np.full(3, 0.01),
        adv21_notional=np.full(3, 1e9),
        rebalance_band_bp=0.0,  # rebalance every step
    )
    res_central = run_backtest(
        rets,
        targets,
        tickers=["SPY", "XLF", "TLT"],
        cost_model=central_cost_model(),
        daily_vol=np.full(3, 0.01),
        adv21_notional=np.full(3, 1e9),
        rebalance_band_bp=0.0,
    )
    assert res_central.nav[-1] < res_zero.nav[-1]
    assert res_central.cumulative_cost[-1] > 0.0


def test_stress_costs_more_than_central():
    rets = _toy_returns(T=200, N=3, seed=5)
    targets = np.zeros((rets.shape[0], 3))
    for t in range(rets.shape[0]):
        targets[t] = np.array([1.0, 0.0, 0.0]) if t % 2 == 0 else np.array([0.0, 1.0, 0.0])
    res_central = run_backtest(
        rets,
        targets,
        tickers=["SPY", "XLF", "TLT"],
        cost_model=central_cost_model(),
        daily_vol=np.full(3, 0.01),
        adv21_notional=np.full(3, 1e9),
        rebalance_band_bp=0.0,
    )
    res_stress = run_backtest(
        rets,
        targets,
        tickers=["SPY", "XLF", "TLT"],
        cost_model=stress_cost_model(),
        daily_vol=np.full(3, 0.01),
        adv21_notional=np.full(3, 1e9),
        rebalance_band_bp=0.0,
    )
    assert res_stress.cumulative_cost[-1] > res_central.cumulative_cost[-1]
    assert res_stress.nav[-1] < res_central.nav[-1]


def test_turnover_band_skips_rebalance():
    """A wide turnover band should produce zero rebalances if drift stays small."""
    rets = np.full((100, 3), 0.0)  # zero returns → no drift
    targets = np.tile(np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]), (100, 1))  # constant target
    res = run_backtest(
        rets,
        targets,
        tickers=["SPY", "XLF", "TLT"],
        cost_model=central_cost_model(),
        daily_vol=np.full(3, 0.01),
        adv21_notional=np.full(3, 1e9),
        rebalance_band_bp=10000.0,  # huge band, never rebalance
    )
    # Initial allocation only; no further rebalances.
    assert res.rebalance_count == 0


def test_strategy_shape_validation():
    rets = _toy_returns(T=100, N=3, seed=0)
    bad_target = np.zeros((50, 3))  # wrong T
    with pytest.raises(ValueError, match="shape mismatch"):
        run_backtest(
            rets,
            bad_target,
            tickers=["SPY", "XLF", "TLT"],
            cost_model=_zero_costs(),
            daily_vol=np.full(3, 0.01),
            adv21_notional=np.full(3, 1e9),
        )


def test_naive_regime_strategy():
    cp = np.array([0.1, 0.2, 0.6, 0.4, 0.7])
    weights = naive_regime(cp, n_assets=3, risk_on_idx=0, defensive_idx=2, threshold=0.5)
    # cp > 0.5 → defensive; else risk_on.
    np.testing.assert_array_equal(weights[:, 0], np.array([1.0, 1.0, 0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(weights[:, 2], np.array([0.0, 0.0, 1.0, 0.0, 1.0]))
    # All rows sum to 1.
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_fixed_weights_validates_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        fixed_weights(10, np.array([0.6, 0.3]))


def test_risk_parity_inverse_vol():
    """High-vol asset should get lower weight than low-vol asset."""
    rng = np.random.default_rng(7)
    n_lookback = 80
    n_total = 200
    # Asset 0 is more volatile than asset 1.
    rets = np.column_stack(
        [
            rng.normal(0.0, 0.02, size=n_total),
            rng.normal(0.0, 0.005, size=n_total),
        ]
    )
    weights = risk_parity(rets, lookback=n_lookback)
    # After lookback, weight on the lower-vol asset (1) > weight on higher-vol asset (0).
    assert (weights[n_lookback:, 1] > weights[n_lookback:, 0]).all()
    # Weights sum to 1.
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_60_40_constant_weights_under_zero_costs():
    """60% SPY / 40% TLT held constant under zero costs should yield NAV close to
    the geometric product of the weighted period returns. Drift between
    rebalance bands is the only source of small deviation."""
    rets = _toy_returns(T=300, N=2, seed=10)  # SPY, TLT
    target = fixed_weights(rets.shape[0], np.array([0.6, 0.4]))
    res = run_backtest(
        rets,
        target,
        tickers=["SPY", "TLT"],
        cost_model=_zero_costs(),
        daily_vol=np.array([0.01, 0.005]),
        adv21_notional=np.array([1e9, 5e8]),
        rebalance_band_bp=0.0,  # rebalance every step
    )
    # With band=0 and zero costs, every period is fully rebalanced — nav follows
    # geometric product of the weighted period returns from t=1 forward.
    portfolio_returns = (rets * np.array([0.6, 0.4])).sum(axis=1)
    expected = np.concatenate(([1.0], np.cumprod(1.0 + portfolio_returns[1:])))
    np.testing.assert_allclose(res.nav, expected, rtol=1e-10)
