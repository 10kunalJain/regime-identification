"""End-to-end tests for the regime-conditional BL strategy."""

from __future__ import annotations

import numpy as np
import pytest

from regime.strategy.regime_bl import (
    RegimeBLConfig,
    RegimeBLStrategy,
    regime_conditional_means,
)


def _synthetic_returns(T: int = 1260, N: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, size=(T, N))


def _synthetic_regimes(T: int = 1260, K: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, K, size=T)


def test_regime_conditional_means_per_regime():
    rets = np.array(
        [
            [0.01, 0.02],
            [0.01, 0.02],
            [0.01, 0.02],
            [-0.01, -0.02],
            [-0.01, -0.02],
            [-0.01, -0.02],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    regimes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    means = regime_conditional_means(rets, regimes, K=3, annualization=252, min_samples=2)
    np.testing.assert_allclose(means[0], np.array([0.01 * 252, 0.02 * 252]))
    np.testing.assert_allclose(means[1], np.array([-0.01 * 252, -0.02 * 252]))
    np.testing.assert_allclose(means[2], np.array([0.0, 0.0]))


def test_regime_conditional_means_falls_back_for_thin_regimes():
    """A regime with too few samples uses the overall mean."""
    rets = np.array([[0.01, 0.02], [0.01, 0.02], [0.01, 0.02], [-0.01, -0.02]])
    regimes = np.array([0, 0, 0, 1])
    overall = rets.mean(axis=0) * 252
    means = regime_conditional_means(rets, regimes, K=3, annualization=252, min_samples=2)
    # Regime 0 has 3 samples → its own mean.
    np.testing.assert_allclose(means[0], np.array([0.01 * 252, 0.02 * 252]))
    # Regime 1 has 1 sample (< min_samples=2) → overall mean.
    np.testing.assert_allclose(means[1], overall)
    # Regime 2 has 0 samples → overall mean.
    np.testing.assert_allclose(means[2], overall)


def test_regime_bl_strategy_smoke():
    rets = _synthetic_returns(T=1260, N=4, seed=1)
    regimes = _synthetic_regimes(T=1260, K=3, seed=2)
    market_caps = np.array([0.5, 0.2, 0.2, 0.1])
    cfg = RegimeBLConfig(lookback_days=252 * 5)
    strat = RegimeBLStrategy(cfg, market_caps)

    posterior = np.array([0.5, 0.3, 0.2])
    weights = strat.target_weights_at_t(posterior, rets, regimes)

    assert weights.shape == (4,)
    assert (weights >= -1e-9).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    # Each weight is bounded by asset_cap.
    assert (weights <= cfg.asset_cap + 1e-6).all()
    # Realized vol respects vol target.
    sigma = np.cov(rets, rowvar=False, ddof=1) * cfg.annualization
    realized_var = float(weights @ sigma @ weights)
    assert realized_var <= cfg.vol_target_annual**2 + 1e-6


def test_regime_bl_validates_market_cap_weights():
    cfg = RegimeBLConfig()
    with pytest.raises(ValueError, match="must sum to 1"):
        RegimeBLStrategy(cfg, np.array([0.5, 0.4]))  # sums to 0.9
    with pytest.raises(ValueError, match="non-negative"):
        RegimeBLStrategy(cfg, np.array([1.0, -0.0001, 0.0001]))


def test_regime_bl_dimension_mismatch_raises():
    rets = _synthetic_returns(T=600, N=4, seed=0)
    regimes = _synthetic_regimes(T=600, K=3, seed=0)
    market_caps = np.array([0.5, 0.5])  # wrong length
    cfg = RegimeBLConfig(lookback_days=600)
    strat = RegimeBLStrategy(cfg, market_caps)
    with pytest.raises(ValueError, match="market_cap_weights"):
        strat.target_weights_at_t(np.array([0.5, 0.3, 0.2]), rets, regimes)


def test_regime_bl_pre_registered_defaults_match_spec():
    """Pin the pre-registered Q8 / Q11 / Q9 defaults so silent edits trip CI."""
    cfg = RegimeBLConfig()
    assert cfg.tau == 0.05
    assert cfg.risk_aversion == 2.5
    assert cfg.vol_target_annual == 0.10
    assert cfg.asset_cap == 0.40
    assert cfg.lookback_days == 252 * 5
    assert cfg.annualization == 252
