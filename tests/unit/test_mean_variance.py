"""Tests for the constrained mean-variance optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from regime.strategy.mean_variance import constrained_mean_variance


def test_optimizer_outputs_long_only_simplex():
    mu = np.array([0.10, 0.05, 0.07])
    sigma = np.diag([0.04, 0.09, 0.16])
    w = constrained_mean_variance(mu, sigma, risk_aversion=2.5, asset_cap=1.0)
    assert (w >= -1e-9).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_optimizer_respects_asset_cap():
    mu = np.array([0.20, 0.05, 0.05])  # asset 0 dominates
    sigma = np.diag([0.04, 0.04, 0.04])
    w = constrained_mean_variance(mu, sigma, risk_aversion=1.0, asset_cap=0.40)
    assert w[0] <= 0.40 + 1e-6


def test_optimizer_lower_risk_aversion_means_higher_risk():
    """A lower risk aversion should produce a higher portfolio variance."""
    mu = np.array([0.15, 0.05])
    sigma = np.diag([0.04, 0.01])
    w_high = constrained_mean_variance(mu, sigma, risk_aversion=10.0, asset_cap=1.0)
    w_low = constrained_mean_variance(mu, sigma, risk_aversion=0.5, asset_cap=1.0)
    var_high = float(w_high @ sigma @ w_high)
    var_low = float(w_low @ sigma @ w_low)
    assert var_low > var_high


def test_optimizer_vol_target_active():
    """Vol-target ceiling should keep portfolio variance below target²."""
    # Asset 0: σ=20% (var=0.04). Asset 1: σ=5% (var=0.0025). Min-vol portfolio
    # is 100% asset 1 (5% vol), so the 10% vol target is achievable.
    mu = np.array([0.20, 0.05])
    sigma = np.diag([0.04, 0.0025])
    target = 0.10  # 10% annual vol → var cap = 0.01
    w = constrained_mean_variance(
        mu,
        sigma,
        risk_aversion=0.5,  # low → would otherwise prefer high-return asset 0
        asset_cap=1.0,
        vol_target_annual=target,
    )
    var = float(w @ sigma @ w)
    assert var <= target**2 + 1e-6


def test_optimizer_rejects_too_tight_cap():
    """With cap × N < 1, sum-to-one is infeasible."""
    mu = np.zeros(3)
    sigma = np.eye(3)
    with pytest.raises(ValueError, match="too tight"):
        constrained_mean_variance(mu, sigma, asset_cap=0.10)


def test_optimizer_returns_fallback_on_infeasible():
    """If the optimizer fails, it should return the fallback unchanged."""
    fallback = np.array([0.5, 0.5])
    mu = np.array([0.0, 0.0])
    sigma = np.eye(2) * np.nan  # certain to break the optimizer
    w = constrained_mean_variance(
        mu, sigma, risk_aversion=2.5, asset_cap=1.0, fallback_weights=fallback
    )
    np.testing.assert_array_equal(w, fallback)
