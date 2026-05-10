"""Regime-conditional Black-Litterman strategy.

Wires the Week 6 ensemble crisis posterior to a Black-Litterman view at each
rebalance. Hyperparameters are pre-registered in STRATEGY_HYPERPARAMETERS.md
§7 (BL τ=0.05, Idzorek Ω, 10% vol target, 40% per-asset cap, 5y lookback).

For each rebalance date t, given:
  - regime posterior P(s_t = k) from the ensemble
  - rolling-window historical returns over the past `lookback_days`
  - rolling-window regime sequence over the past `lookback_days`
  - per-asset market-cap weights

The strategy:
  1. Computes Σ from the rolling window (annualized).
  2. Computes Π = λ Σ w_mkt (reverse optimization).
  3. Computes per-regime historical mean returns μ̂_k (annualized).
  4. Builds the view: P = I_N, Q = Σ_k posterior[k] · μ̂_k.
  5. Builds Idzorek Ω = diag(τ × diag(P Σ Pᵀ)).
  6. Computes posterior μ* via Black-Litterman.
  7. Solves long-only sum-to-one mean-variance with vol-target ceiling and
     per-asset cap; returns the target weights.

The strategy class is stateless w.r.t. previous holdings (the backtest engine
tracks NAV / actual weights / turnover); each `target_weights_at_t` call is a
pure function of inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regime.strategy.black_litterman import (
    idzorek_omega,
    implied_returns,
    posterior_returns,
)
from regime.strategy.mean_variance import constrained_mean_variance


@dataclass(frozen=True)
class RegimeBLConfig:
    """Pre-registered hyperparameters per STRATEGY_HYPERPARAMETERS.md §7."""

    tau: float = 0.05
    risk_aversion: float = 2.5
    vol_target_annual: float = 0.10
    asset_cap: float = 0.40
    lookback_days: int = 252 * 5  # 5 years rolling
    annualization: int = 252
    min_regime_samples: int = 20  # if fewer than this, fall back to overall mean


def regime_conditional_means(
    historical_returns: np.ndarray,
    regime_history: np.ndarray,
    K: int,
    annualization: int,
    min_samples: int,
) -> np.ndarray:
    """Per-regime annualized mean returns over the lookback window.

    Returns shape (K, N). If a regime has fewer than `min_samples` observations
    in the window, fall back to the overall mean for that regime.
    """
    historical_returns = np.asarray(historical_returns, dtype=np.float64)
    regime_history = np.asarray(regime_history, dtype=np.int64).reshape(-1)
    if len(regime_history) != len(historical_returns):
        raise ValueError("regime_history must align with historical_returns by row")
    overall = historical_returns.mean(axis=0) * annualization
    means = np.tile(overall, (K, 1))
    for k in range(K):
        mask = regime_history == k
        if int(mask.sum()) >= min_samples:
            means[k] = historical_returns[mask].mean(axis=0) * annualization
    return means


class RegimeBLStrategy:
    """Regime-conditional Black-Litterman target-weight generator."""

    def __init__(self, config: RegimeBLConfig, market_cap_weights: np.ndarray) -> None:
        self.config = config
        mcw = np.asarray(market_cap_weights, dtype=np.float64).reshape(-1)
        if not np.isclose(mcw.sum(), 1.0, atol=1e-9):
            raise ValueError(f"market_cap_weights must sum to 1; got {mcw.sum():.6f}")
        if (mcw < 0).any():
            raise ValueError("market_cap_weights must be non-negative")
        self.market_cap_weights = mcw

    def target_weights_at_t(
        self,
        regime_posterior: np.ndarray,
        historical_returns: np.ndarray,
        regime_history: np.ndarray,
    ) -> np.ndarray:
        """Compute target weights at a single rebalance time.

        Args:
            regime_posterior: shape (K,) — P(s_t = k) for the current period.
            historical_returns: shape (lookback_days, N) — daily returns over window.
            regime_history: shape (lookback_days,) — argmax regime per timestep.

        Returns:
            shape (N,) target weights summing to 1.
        """
        regime_posterior = np.asarray(regime_posterior, dtype=np.float64).reshape(-1)
        historical_returns = np.asarray(historical_returns, dtype=np.float64)
        n_assets = historical_returns.shape[1]
        K = len(regime_posterior)
        cfg = self.config

        if n_assets != self.market_cap_weights.size:
            raise ValueError(
                f"historical_returns has {n_assets} assets;"
                f" market_cap_weights has {self.market_cap_weights.size}"
            )

        sigma = np.cov(historical_returns, rowvar=False, ddof=1) * cfg.annualization
        sigma = 0.5 * (sigma + sigma.T)  # symmetrize for numerical safety

        Pi = implied_returns(self.market_cap_weights, sigma, cfg.risk_aversion)

        mu_per_regime = regime_conditional_means(
            historical_returns,
            regime_history,
            K=K,
            annualization=cfg.annualization,
            min_samples=cfg.min_regime_samples,
        )
        Q = (regime_posterior[:, None] * mu_per_regime).sum(axis=0)

        P = np.eye(n_assets)
        Omega = idzorek_omega(P, sigma, cfg.tau)

        try:
            mu_star = posterior_returns(Pi, P, Q, Omega, cfg.tau, sigma)
        except np.linalg.LinAlgError:
            return self.market_cap_weights.copy()

        return constrained_mean_variance(
            mu=mu_star,
            sigma=sigma,
            risk_aversion=cfg.risk_aversion,
            asset_cap=cfg.asset_cap,
            vol_target_annual=cfg.vol_target_annual,
            fallback_weights=self.market_cap_weights,
        )
