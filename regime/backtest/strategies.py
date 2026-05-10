"""Benchmark strategy weight generators.

Each function takes `returns` and metadata and returns a (T, N) array of target
weights. Strategies are deliberately stateless and pure — the regime-conditional
strategy in Week 8 will plug into the same shape.
"""

from __future__ import annotations

import numpy as np


def buy_and_hold(t_count: int, n_assets: int, asset_idx: int) -> np.ndarray:
    """100% in `asset_idx` from t=0 forward."""
    weights = np.zeros((t_count, n_assets), dtype=np.float64)
    weights[:, asset_idx] = 1.0
    return weights


def fixed_weights(t_count: int, weight_vec: np.ndarray) -> np.ndarray:
    """Constant-target weights at every t."""
    weight_vec = np.asarray(weight_vec, dtype=np.float64).reshape(-1)
    if not np.isclose(weight_vec.sum(), 1.0, atol=1e-9):
        raise ValueError(f"weight_vec must sum to 1; got {weight_vec.sum():.6f}")
    return np.tile(weight_vec, (t_count, 1))


def equal_weight(t_count: int, n_assets: int) -> np.ndarray:
    """1/N in every asset."""
    return np.full((t_count, n_assets), 1.0 / n_assets, dtype=np.float64)


def risk_parity(returns: np.ndarray, lookback: int = 63) -> np.ndarray:
    """Inverse-vol weighting from rolling realized vol.

    Weights start at equal-weight for the first `lookback` periods (insufficient
    history) and switch to inverse-vol thereafter, normalized to sum to 1.
    """
    returns = np.asarray(returns, dtype=np.float64)
    T, N = returns.shape
    weights = np.full((T, N), 1.0 / N, dtype=np.float64)
    for t in range(lookback, T):
        window = returns[t - lookback : t]
        vol = np.std(window, axis=0, ddof=1)
        inv_vol = 1.0 / np.maximum(vol, 1e-9)
        w = inv_vol / float(inv_vol.sum())
        weights[t] = w
    return weights


def naive_regime(
    crisis_prob: np.ndarray,
    n_assets: int,
    risk_on_idx: int,
    defensive_idx: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Risk-on (100% `risk_on_idx`) when crisis_prob ≤ threshold; else 100% `defensive_idx`."""
    crisis_prob = np.asarray(crisis_prob, dtype=np.float64).reshape(-1)
    T = len(crisis_prob)
    weights = np.zeros((T, n_assets), dtype=np.float64)
    risk_off = crisis_prob > threshold
    weights[~risk_off, risk_on_idx] = 1.0
    weights[risk_off, defensive_idx] = 1.0
    return weights
