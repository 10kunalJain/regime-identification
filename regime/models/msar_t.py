"""Markov-switching model with Student-t emissions (univariate, AR(0)).

The "MS-AR" baseline in the fair-evaluation benchmark. Univariate observation
(typically `ret_1d`); per-state mean μ_k and scale σ_k; shared, fixed degrees-
of-freedom ν (default 5, the typical value for daily equity returns).

The point of t-emissions vs Gaussian is heavy-tailed regime-conditional return
distributions — fits financial data better than Gaussian and is the standard
robustification of MS-AR in the literature (Fonseca-Loretan 2002, Lopes-Salazar
2006). For v1 we use AR(0) (no autoregressive term); the design grill (Q4)
established that AR dynamics belong in the joint-HMM benchmarked method, not
the univariate baseline.

Custom EM with data-augmented Student-t M-step:
    y_t | s_t=k, u_t ~ Normal(μ_k, σ_k² / u_t)
    u_t | s_t=k    ~ Gamma(ν/2, ν/2)
giving Student-t marginals and closed-form weighted M-step updates.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM

from regime.models._hmm_core import (
    forward_backward,
    forward_backward_xi,
    forward_filter,
    student_t_log_emissions_univariate,
)

_LOG = logging.getLogger(__name__)
_LOG_EPS = 1e-12


class MsarT:
    """Markov-switching mean / scale with Student-t emissions. Implements StateRegimeModel."""

    K: int

    def __init__(
        self,
        K: int = 3,
        feature_columns: tuple[str, ...] = ("ret_1d",),
        nu: float = 5.0,
        n_restarts: int = 5,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int = 42,
    ) -> None:
        self.K = K
        self.feature_columns = feature_columns
        self.nu = nu
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._params: dict | None = None  # {"pi", "A", "mu", "sigma", "nu"}

    def fit(self, features: pl.DataFrame, train_idx: np.ndarray) -> None:
        if len(self.feature_columns) != 1:
            raise ValueError(f"MsarT is univariate; got {len(self.feature_columns)} cols")
        y = self._extract(features, train_idx)
        if len(y) < 2 * self.K:
            raise ValueError(
                f"need at least {2 * self.K} obs to fit K={self.K} MsarT, got {len(y)}"
            )

        best_log_lik = -np.inf
        best_params: dict | None = None
        for offset in range(self.n_restarts):
            seed = self.random_state + offset
            try:
                pi, A, mu, sigma = self._init_for_seed(y, seed)
                pi, A, mu, sigma, log_lik = self._em_student_t(y, pi, A, mu, sigma)
            except Exception:
                _LOG.debug("MsarT restart %d failed", offset, exc_info=True)
                continue
            if np.isfinite(log_lik) and log_lik > best_log_lik:
                best_log_lik = log_lik
                best_params = {
                    "pi": pi,
                    "A": A,
                    "mu": mu,
                    "sigma": sigma,
                    "nu": float(self.nu),
                }
        if best_params is None:
            raise RuntimeError("all MsarT restarts failed to converge")
        self._params = best_params

    def filter(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        y = self._extract(features, idx)
        log_pi, log_A, log_emissions = self._log_pieces(y)
        return forward_filter(log_emissions, log_pi, log_A)

    def smooth(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        y = self._extract(features, idx)
        log_pi, log_A, log_emissions = self._log_pieces(y)
        return forward_backward(log_emissions, log_pi, log_A)

    def state_dict(self) -> dict:
        if self._params is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "K": self.K,
            "feature_columns": list(self.feature_columns),
            "nu": float(self._params["nu"]),
            "pi": self._params["pi"].tolist(),
            "A": self._params["A"].tolist(),
            "mu": self._params["mu"].tolist(),
            "sigma": self._params["sigma"].tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        if not state.get("fitted"):
            self._params = None
            return
        self.K = int(state["K"])
        self.feature_columns = tuple(state["feature_columns"])
        self.nu = float(state["nu"])
        self._params = {
            "pi": np.asarray(state["pi"], dtype=np.float64),
            "A": np.asarray(state["A"], dtype=np.float64),
            "mu": np.asarray(state["mu"], dtype=np.float64),
            "sigma": np.asarray(state["sigma"], dtype=np.float64),
            "nu": float(state["nu"]),
        }

    # ------------------------------------------------------------------

    def _extract(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        col = self.feature_columns[0]
        mask = np.zeros(features.height, dtype=bool)
        mask[idx] = True
        sub = features.select(col).filter(pl.Series(mask)).drop_nulls()
        return sub.to_numpy().reshape(-1)

    def _init_for_seed(
        self, y: np.ndarray, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize EM. The first restart uses GaussianHMM; later restarts
        use random initialization seeded by `seed` to break symmetry."""
        if seed == self.random_state:
            try:
                return self._init_from_gaussian_hmm(y, seed)
            except Exception:
                _LOG.exception("Gaussian-HMM init failed; using random fallback")
        return self._init_random(y, seed)

    def _init_from_gaussian_hmm(
        self, y: np.ndarray, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = GaussianHMM(
            n_components=self.K,
            covariance_type="diag",
            n_iter=50,
            random_state=seed,
            init_params="stmc",
        )
        m.fit(y.reshape(-1, 1))
        pi = np.asarray(m.startprob_, dtype=np.float64).copy()
        A = np.asarray(m.transmat_, dtype=np.float64).copy()
        mu = np.asarray(m.means_, dtype=np.float64).reshape(-1).copy()
        sigma = np.sqrt(np.asarray(m.covars_, dtype=np.float64).reshape(-1)).copy()
        sigma = np.maximum(sigma, _LOG_EPS)
        return pi, A, mu, sigma

    def _init_random(
        self, y: np.ndarray, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        pi = np.full(self.K, 1.0 / self.K)
        A = np.full((self.K, self.K), 1.0 / self.K)
        # Spread initial means across observed quantiles for coverage.
        quantiles = np.linspace(0.1, 0.9, self.K)
        mu = np.quantile(y, quantiles).astype(np.float64)
        mu += rng.normal(0.0, 0.01 * float(np.std(y)) + _LOG_EPS, size=self.K)
        sigma = np.full(self.K, float(np.std(y)) + _LOG_EPS)
        return pi, A, mu, sigma

    def _em_student_t(
        self,
        y: np.ndarray,
        pi: np.ndarray,
        A: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        log_lik_prev = -np.inf
        log_lik = -np.inf
        for _ in range(self.max_iter):
            log_pi = np.log(np.maximum(pi, _LOG_EPS))
            log_A = np.log(np.maximum(A, _LOG_EPS))
            log_emissions = student_t_log_emissions_univariate(y, mu, sigma, self.nu)

            gamma, xi, log_lik = forward_backward_xi(log_emissions, log_pi, log_A)

            scaled_resid_sq = ((y[:, None] - mu[None, :]) / sigma[None, :]) ** 2
            u = (self.nu + 1.0) / (self.nu + scaled_resid_sq)

            pi = gamma[0] / max(float(gamma[0].sum()), _LOG_EPS)
            xi_sum = xi.sum(axis=0)
            A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), _LOG_EPS)

            w = gamma * u
            w_sum = w.sum(axis=0)
            mu_new = (w * y[:, None]).sum(axis=0) / np.maximum(w_sum, _LOG_EPS)

            resid_sq = (y[:, None] - mu_new[None, :]) ** 2
            num = (gamma * u * resid_sq).sum(axis=0)
            den = np.maximum(gamma.sum(axis=0), _LOG_EPS)
            sigma_new = np.sqrt(np.maximum(num / den, _LOG_EPS))

            mu = mu_new
            sigma = sigma_new

            if np.isfinite(log_lik) and abs(log_lik - log_lik_prev) < self.tol:
                break
            log_lik_prev = log_lik

        return pi, A, mu, sigma, log_lik

    def _log_pieces(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._params is None:
            raise RuntimeError("model not fit")
        p = self._params
        log_pi = np.log(np.maximum(p["pi"], _LOG_EPS))
        log_A = np.log(np.maximum(p["A"], _LOG_EPS))
        log_emissions = student_t_log_emissions_univariate(y, p["mu"], p["sigma"], float(p["nu"]))
        return log_pi, log_A, log_emissions
