"""Black-Litterman primitives.

Standard BL setup:
  - Π: equilibrium expected returns implied by market-cap weights via reverse
    mean-variance optimization: Π = λ Σ w_mkt.
  - τ: scalar prior uncertainty.
  - P: K_views × N "pick" matrix (which assets each view talks about).
  - Q: K_views vector of view expected returns.
  - Ω: K_views × K_views diagonal view-uncertainty matrix.

Posterior expected returns (numerically-stable form using a single solve):
  μ* = Π + τΣ Pᵀ (P τΣ Pᵀ + Ω)⁻¹ (Q - P Π)

`idzorek_omega` builds Ω = diag(τ × diag(P Σ Pᵀ)) per Idzorek (2007),
matching the choice locked in STRATEGY_HYPERPARAMETERS.md §7.
"""

from __future__ import annotations

import numpy as np


def implied_returns(
    market_cap_weights: np.ndarray, sigma: np.ndarray, risk_aversion: float = 2.5
) -> np.ndarray:
    """Reverse-optimization implied returns Π = λ Σ w_mkt."""
    market_cap_weights = np.asarray(market_cap_weights, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != market_cap_weights.size:
        raise ValueError("sigma and market_cap_weights must have matching dimension")
    return float(risk_aversion) * sigma @ market_cap_weights


def idzorek_omega(P: np.ndarray, sigma: np.ndarray, tau: float) -> np.ndarray:
    """Idzorek 2007 view uncertainty: Ω = diag(τ × diag(P Σ Pᵀ))."""
    P = np.asarray(P, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    raw = np.diag(P @ sigma @ P.T)
    return np.diag(float(tau) * raw)


def posterior_returns(
    Pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Closed-form Black-Litterman posterior expected returns.

    μ* = Π + τΣ Pᵀ (P τΣ Pᵀ + Ω)⁻¹ (Q - P Π)
    """
    Pi = np.asarray(Pi, dtype=np.float64).reshape(-1)
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64).reshape(-1)
    Omega = np.asarray(Omega, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if P.shape[1] != sigma.shape[0]:
        raise ValueError("P columns must equal sigma dimension")
    if P.shape[0] != Q.size:
        raise ValueError("P rows must equal Q length")

    tau_sigma = float(tau) * sigma
    middle = P @ tau_sigma @ P.T + Omega
    rhs = Q - P @ Pi
    update = tau_sigma @ P.T @ np.linalg.solve(middle, rhs)
    return Pi + update
