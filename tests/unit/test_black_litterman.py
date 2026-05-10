"""Tests for Black-Litterman primitives."""

from __future__ import annotations

import numpy as np
import pytest

from regime.strategy.black_litterman import (
    idzorek_omega,
    implied_returns,
    posterior_returns,
)


def test_implied_returns_shape_and_sign():
    sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    w_mkt = np.array([0.6, 0.4])
    pi = implied_returns(w_mkt, sigma, risk_aversion=2.5)
    assert pi.shape == (2,)
    # Positive market-cap weights and PSD sigma → positive implied returns.
    assert (pi > 0).all()


def test_implied_returns_dimension_mismatch_raises():
    sigma = np.eye(2)
    w_mkt = np.array([0.6, 0.4, 0.0])  # wrong size
    with pytest.raises(ValueError, match="matching dimension"):
        implied_returns(w_mkt, sigma)


def test_idzorek_omega_diagonal():
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    omega = idzorek_omega(P, sigma, tau=0.05)
    assert omega.shape == (2, 2)
    # Diagonal: τ × diag(PΣP^T) = 0.05 × [0.04, 0.09]
    np.testing.assert_allclose(np.diag(omega), [0.05 * 0.04, 0.05 * 0.09])
    # Off-diagonals are zero.
    np.testing.assert_allclose(omega - np.diag(np.diag(omega)), 0.0)


def test_posterior_equals_prior_when_view_matches_prior():
    """If Q = P Π, the posterior should equal the prior."""
    sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    w_mkt = np.array([0.6, 0.4])
    Pi = implied_returns(w_mkt, sigma)
    P = np.eye(2)
    Q = P @ Pi  # exactly the prior
    Omega = idzorek_omega(P, sigma, tau=0.05)
    mu_star = posterior_returns(Pi, P, Q, Omega, tau=0.05, sigma=sigma)
    np.testing.assert_allclose(mu_star, Pi, rtol=1e-10)


def test_posterior_shifts_toward_view_with_low_uncertainty():
    """A low-Ω view pulls the posterior toward Q; large-Ω leaves it near Π."""
    sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    w_mkt = np.array([0.6, 0.4])
    Pi = implied_returns(w_mkt, sigma)
    P = np.eye(2)
    Q = Pi + np.array([0.10, 0.0])  # strong positive view on asset 0

    omega_tight = np.diag([1e-6, 1e-6])
    mu_tight = posterior_returns(Pi, P, Q, omega_tight, tau=0.05, sigma=sigma)
    omega_loose = np.diag([1.0, 1.0])
    mu_loose = posterior_returns(Pi, P, Q, omega_loose, tau=0.05, sigma=sigma)

    assert mu_tight[0] > Pi[0]
    # Tight-Ω posterior is closer to Q than loose-Ω posterior.
    assert abs(mu_tight[0] - Q[0]) < abs(mu_loose[0] - Q[0])


def test_posterior_shape_validation():
    sigma = np.eye(3)
    Pi = np.zeros(3)
    P = np.eye(2)  # wrong column count
    Q = np.zeros(2)
    Omega = np.eye(2)
    with pytest.raises(ValueError, match="P columns"):
        posterior_returns(Pi, P, Q, Omega, tau=0.05, sigma=sigma)
