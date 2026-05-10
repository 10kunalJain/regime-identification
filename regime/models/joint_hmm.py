"""Joint cross-sectional HMM — the headline benchmarked method (Q4 lock).

Multivariate observation r_t in R^d (d ETF returns). Latent state s_t in {1..K}
shared across assets. Per-regime emission:

    r_t | s_t = k, f_t^FF ~ N(α_k + B_k^obs f_t^FF,  L_k L_k^T + diag(D_k))

where:
  - f_t^FF in R^m are the daily Fama-French 5+Mom factor returns (observed).
  - α_k in R^d, B_k^obs in R^{d×m} are regime-conditional intercepts and FF betas.
  - L_k in R^{d×r}, D_k in R^d are the rank-r latent-factor structure of the
    regime-conditional residual covariance, with rank r locked at 3 per
    STRATEGY_HYPERPARAMETERS.md.

Custom EM:
  - E-step: forward-backward over states with our pure-numpy primitives.
  - M-step:
      - π from γ_0; A from ξ.
      - α_k, B_k^obs from weighted multivariate regression with weights γ_t,k.
      - L_k, D_k from weighted factor-analyzer EM (inner loop, 5-10 iters)
        on the regime-conditional residuals.

Multi-restart with best-log-likelihood selection. Regime-collapse detection
(`RegimeCollapseError`) raised if any regime's posterior mass falls below 5%
during the run. State labels are sorted by SPY-component mean (first
observation column by convention) at the end of fit for stability across folds;
the W₂ + Hungarian alignment to a canonical fold is wired through the existing
`regime/models/alignment.py` from Week 3.

Implements `StateRegimeModel`. The factor vector `f_t^FF` has to be present in
the input DataFrame as a configurable group of columns (default = the six
`ff_*` columns produced by `regime/features/definitions/factors.py`).
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from regime.models._hmm_core import forward_backward, forward_backward_xi, forward_filter

_LOG = logging.getLogger(__name__)
_LOG_2_PI = float(np.log(2.0 * np.pi))
_TINY = 1e-12


class RegimeCollapseError(RuntimeError):
    """Raised when a regime's posterior mass falls below the collapse threshold."""


class JointHmm:
    """Joint cross-sectional HMM with FF-factor mean + factor-cov residual structure."""

    K: int

    def __init__(
        self,
        K: int = 3,
        observation_columns: tuple[str, ...] = ("SPY",),
        factor_columns: tuple[str, ...] = (
            "ff_mkt_rf",
            "ff_smb",
            "ff_hml",
            "ff_rmw",
            "ff_cma",
            "ff_mom",
        ),
        latent_factor_rank: int = 3,
        n_restarts: int = 10,
        max_iter: int = 100,
        tol: float = 1e-4,
        inner_fa_iter: int = 8,
        regime_collapse_threshold: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.K = K
        self.observation_columns = observation_columns
        self.factor_columns = factor_columns
        self.latent_factor_rank = int(latent_factor_rank)
        self.n_restarts = int(n_restarts)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.inner_fa_iter = int(inner_fa_iter)
        self.regime_collapse_threshold = float(regime_collapse_threshold)
        self.random_state = random_state
        self._params: dict | None = None

    # ------------------------------------------------------------------
    # StateRegimeModel API

    def fit(self, features: pl.DataFrame, train_idx: np.ndarray) -> None:
        Y, F = self._extract(features, train_idx)
        T = Y.shape[0]
        if T < 5 * self.K:
            raise ValueError(f"need at least {5 * self.K} obs to fit K={self.K} JointHmm, got {T}")

        best_log_lik = -np.inf
        best_params: dict | None = None
        for offset in range(self.n_restarts):
            seed = self.random_state + offset
            try:
                params, log_lik = self._em_one_restart(Y, F, seed)
            except RegimeCollapseError as exc:
                _LOG.debug("JointHmm restart %d collapsed: %s", offset, exc)
                continue
            except Exception:
                _LOG.debug("JointHmm restart %d failed", offset, exc_info=True)
                continue
            if np.isfinite(log_lik) and log_lik > best_log_lik:
                best_log_lik = log_lik
                best_params = params

        if best_params is None:
            raise RuntimeError("all JointHmm restarts failed to converge")

        # Stable label ordering: sort regimes by mean of the first observation
        # column (SPY by convention) under the historical FF-factor mean.
        f_mean = F.mean(axis=0)
        spy_means = np.array(
            [best_params["alpha"][k, 0] + best_params["B"][k, 0] @ f_mean for k in range(self.K)]
        )
        order = np.argsort(-spy_means)
        self._params = _permute_params(best_params, order)

    def filter(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        Y, F = self._extract(features, idx)
        log_pi, log_A, log_emissions = self._log_pieces(Y, F)
        return forward_filter(log_emissions, log_pi, log_A)

    def smooth(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        Y, F = self._extract(features, idx)
        log_pi, log_A, log_emissions = self._log_pieces(Y, F)
        return forward_backward(log_emissions, log_pi, log_A)

    def state_dict(self) -> dict:
        if self._params is None:
            return {"fitted": False}
        p = self._params
        return {
            "fitted": True,
            "K": self.K,
            "observation_columns": list(self.observation_columns),
            "factor_columns": list(self.factor_columns),
            "latent_factor_rank": self.latent_factor_rank,
            "pi": p["pi"].tolist(),
            "A": p["A"].tolist(),
            "alpha": p["alpha"].tolist(),
            "B": p["B"].tolist(),
            "L": p["L"].tolist(),
            "D": p["D"].tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        if not state.get("fitted"):
            self._params = None
            return
        self.K = int(state["K"])
        self.observation_columns = tuple(state["observation_columns"])
        self.factor_columns = tuple(state["factor_columns"])
        self.latent_factor_rank = int(state["latent_factor_rank"])
        self._params = {
            "pi": np.asarray(state["pi"], dtype=np.float64),
            "A": np.asarray(state["A"], dtype=np.float64),
            "alpha": np.asarray(state["alpha"], dtype=np.float64),
            "B": np.asarray(state["B"], dtype=np.float64),
            "L": np.asarray(state["L"], dtype=np.float64),
            "D": np.asarray(state["D"], dtype=np.float64),
        }

    # ------------------------------------------------------------------
    # Internals

    def _extract(self, features: pl.DataFrame, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols = list(self.observation_columns) + list(self.factor_columns)
        mask = np.zeros(features.height, dtype=bool)
        mask[idx] = True
        sub = features.select(*cols).filter(pl.Series(mask)).drop_nulls()
        if sub.height == 0:
            raise ValueError("no rows after filtering for non-null observation/factor columns")
        arr = sub.to_numpy().astype(np.float64)
        d = len(self.observation_columns)
        Y = arr[:, :d]
        F = arr[:, d:]
        return Y, F

    def _log_pieces(
        self, Y: np.ndarray, F: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._params is None:
            raise RuntimeError("model not fit")
        p = self._params
        log_pi = np.log(np.maximum(p["pi"], _TINY))
        log_A = np.log(np.maximum(p["A"], _TINY))
        log_emissions = _log_emissions(Y, F, p["alpha"], p["B"], p["L"], p["D"])
        return log_pi, log_A, log_emissions

    def _em_one_restart(self, Y: np.ndarray, F: np.ndarray, seed: int) -> tuple[dict, float]:
        T = Y.shape[0]
        K = self.K
        r = self.latent_factor_rank
        rng = np.random.default_rng(seed)

        # Initialize: random soft-assignment, then M-step from those weights.
        gamma_init = rng.dirichlet(np.ones(K), size=T)  # (T, K)
        pi = gamma_init.mean(axis=0)
        A = np.full((K, K), 1.0 / K)
        alpha, B, L, D = _m_step_emissions(Y, F, gamma_init, r)

        log_lik_prev = -np.inf
        log_lik = -np.inf
        for _ in range(self.max_iter):
            log_pi = np.log(np.maximum(pi, _TINY))
            log_A = np.log(np.maximum(A, _TINY))
            log_emissions = _log_emissions(Y, F, alpha, B, L, D)
            gamma, xi, log_lik = forward_backward_xi(log_emissions, log_pi, log_A)

            mass = gamma.mean(axis=0)
            if np.any(mass < self.regime_collapse_threshold):
                raise RegimeCollapseError(
                    f"regime mass below threshold: min={mass.min():.4f},"
                    f" threshold={self.regime_collapse_threshold:.4f}"
                )

            pi = gamma[0] / max(float(gamma[0].sum()), _TINY)
            xi_sum = xi.sum(axis=0)
            A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), _TINY)
            alpha, B, L, D = _m_step_emissions(Y, F, gamma, r, prev_L=L, prev_D=D)

            if np.isfinite(log_lik) and abs(log_lik - log_lik_prev) < self.tol:
                break
            log_lik_prev = log_lik

        params = {
            "pi": pi,
            "A": A,
            "alpha": alpha,
            "B": B,
            "L": L,
            "D": D,
        }
        return params, log_lik


# ----------------------------------------------------------------------
# Module-level helpers (testable in isolation)


def _log_emissions(
    Y: np.ndarray,
    F: np.ndarray,
    alpha: np.ndarray,
    B: np.ndarray,
    L: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Per-timestep, per-regime log emission density.

    Args:
        Y: shape (T, d) observations.
        F: shape (T, m) factor regressors.
        alpha: shape (K, d) regime intercepts.
        B: shape (K, d, m) regime FF betas.
        L: shape (K, d, r) regime latent factor loadings.
        D: shape (K, d) regime diagonal residual variances.

    Returns:
        shape (T, K) log p(y_t | s_t = k, f_t).
    """
    T, d = Y.shape
    K = alpha.shape[0]
    log_emit = np.full((T, K), -np.inf, dtype=np.float64)
    for k in range(K):
        cov = L[k] @ L[k].T + np.diag(D[k])
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0 or not np.isfinite(logdet):
            continue
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        mean = alpha[k][None, :] + F @ B[k].T  # (T, d)
        diff = Y - mean
        quad = np.einsum("td,de,te->t", diff, cov_inv, diff)
        log_emit[:, k] = -0.5 * (d * _LOG_2_PI + logdet + quad)
    return log_emit


def _m_step_emissions(
    Y: np.ndarray,
    F: np.ndarray,
    gamma: np.ndarray,
    rank: int,
    prev_L: np.ndarray | None = None,
    prev_D: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-regime weighted regression for (α, B) and FA-EM for (L, D)."""
    T, d = Y.shape
    K = gamma.shape[1]
    m = F.shape[1]
    alpha = np.zeros((K, d), dtype=np.float64)
    B = np.zeros((K, d, m), dtype=np.float64)
    L = np.zeros((K, d, rank), dtype=np.float64)
    D = np.ones((K, d), dtype=np.float64)
    X = np.column_stack([np.ones(T), F])  # (T, m+1)

    for k in range(K):
        w = gamma[:, k]
        w_total = float(w.sum())
        if w_total < _TINY:
            continue
        # Weighted linear regression: solve (X^T W X) θ = X^T W Y, θ shape (m+1, d).
        XtWX = (X * w[:, None]).T @ X  # (m+1, m+1)
        XtWX += 1e-9 * np.eye(m + 1)
        XtWY = (X * w[:, None]).T @ Y  # (m+1, d)
        try:
            theta = np.linalg.solve(XtWX, XtWY)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(XtWX, XtWY, rcond=None)[0]
        alpha[k] = theta[0]
        B[k] = theta[1:].T

        # Residuals and weighted FA-EM.
        residuals = Y - (alpha[k][None, :] + F @ B[k].T)
        L0 = prev_L[k] if prev_L is not None else None
        D0 = prev_D[k] if prev_D is not None else None
        L[k], D[k] = _weighted_factor_analyzer_em(
            residuals, w, rank, max_iter=8, init_L=L0, init_D=D0
        )

    return alpha, B, L, D


def _weighted_factor_analyzer_em(
    residuals: np.ndarray,
    weights: np.ndarray,
    rank: int,
    max_iter: int = 8,
    init_L: np.ndarray | None = None,
    init_D: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form weighted FA-EM (Ghahramani-Hinton 1997)."""
    d = residuals.shape[1]
    w = weights
    w_total = float(w.sum())
    if w_total < _TINY:
        return np.zeros((d, rank)), np.ones(d)

    # Weighted second-moment matrix C_w (residuals are already centered by the
    # M-step regression; mean ≈ 0).
    weighted = residuals * w[:, None]
    C_w = (residuals.T @ weighted) / w_total
    C_w = 0.5 * (C_w + C_w.T)  # symmetrize for numerical safety

    if init_L is not None and init_D is not None:
        L = init_L.copy()
        D = np.maximum(init_D.copy(), _TINY)
    else:
        eigvals, eigvecs = np.linalg.eigh(C_w)
        eigvals = np.maximum(eigvals, _TINY)
        order = np.argsort(eigvals)[-rank:]
        L = eigvecs[:, order] * np.sqrt(eigvals[order])[None, :]
        residual_diag = np.maximum(np.diag(C_w) - np.diag(L @ L.T), _TINY)
        D = residual_diag

    for _ in range(max_iter):
        D_inv = 1.0 / np.maximum(D, _TINY)
        # G = I_r + L^T D^{-1} L
        G = np.eye(rank) + L.T @ (D_inv[:, None] * L)
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            break
        beta = G_inv @ (L.T * D_inv[None, :])  # (r, d)
        Ezz = np.eye(rank) - beta @ L + beta @ C_w @ beta.T  # (r, r)
        Ezz = 0.5 * (Ezz + Ezz.T)
        try:
            Ezz_inv = np.linalg.inv(Ezz + 1e-9 * np.eye(rank))
        except np.linalg.LinAlgError:
            break
        L_new = C_w @ beta.T @ Ezz_inv  # (d, r)
        D_new = np.maximum(np.diag(C_w) - np.diag(L_new @ beta @ C_w), _TINY)
        if np.allclose(L_new, L, atol=1e-9) and np.allclose(D_new, D, atol=1e-9):
            L, D = L_new, D_new
            break
        L, D = L_new, D_new
    return L, D


def _permute_params(params: dict, perm: np.ndarray) -> dict:
    """Reorder regimes in a parameters dict according to permutation `perm`."""
    return {
        "pi": params["pi"][perm],
        "A": params["A"][np.ix_(perm, perm)],
        "alpha": params["alpha"][perm],
        "B": params["B"][perm],
        "L": params["L"][perm],
        "D": params["D"][perm],
    }
