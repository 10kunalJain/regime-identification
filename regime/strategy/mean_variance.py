"""Mean-variance optimization with constraints.

Solves:

    max_w  w^T μ - (λ/2) w^T Σ w
    s.t.   sum_i w_i = 1
           0 ≤ w_i ≤ asset_cap
           w^T Σ w ≤ vol_target_annual²  (annualized variance cap)

The vol-target inequality is enforced by an SLSQP constraint. If infeasible
(e.g., the risk-aversion / cap combination cannot reach the vol target with
positive weights), the optimizer falls back to the equilibrium prior.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def constrained_mean_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = 2.5,
    asset_cap: float = 0.40,
    vol_target_annual: float | None = None,
    fallback_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Long-only sum-to-one mean-variance with optional vol-target ceiling."""
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    if sigma.shape != (n, n):
        raise ValueError(f"sigma shape {sigma.shape} != ({n}, {n})")
    if asset_cap * n < 1.0:
        raise ValueError(f"asset_cap={asset_cap} too tight to satisfy sum-to-one with N={n} assets")

    def objective(w: np.ndarray) -> float:
        return float(-mu @ w + 0.5 * risk_aversion * w @ sigma @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return np.asarray(-mu + risk_aversion * sigma @ w, dtype=np.float64)

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
    ]
    if vol_target_annual is not None:
        var_cap = float(vol_target_annual) ** 2
        constraints.append({"type": "ineq", "fun": lambda w: float(var_cap - w @ sigma @ w)})

    bounds = [(0.0, float(asset_cap))] * n

    if fallback_weights is None:
        x0 = np.full(n, 1.0 / n)
    else:
        x0 = np.asarray(fallback_weights, dtype=np.float64).reshape(-1).copy()

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if not result.success:
        if fallback_weights is not None:
            return np.asarray(fallback_weights, dtype=np.float64).reshape(-1)
        return np.full(n, 1.0 / n)

    weights = np.asarray(result.x, dtype=np.float64)
    # Clip tiny negatives from optimizer numerical noise.
    weights = np.maximum(weights, 0.0)
    s = float(weights.sum())
    if s > 0.0:
        weights = weights / s
    return weights
