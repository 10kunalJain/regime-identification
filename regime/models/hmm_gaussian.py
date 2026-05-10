"""Multivariate Gaussian HMM (K=3 locked) implementing StateRegimeModel.

Fitting via hmmlearn's Baum-Welch with multiple random restarts; filter and
smooth use our own pure-numpy forward / forward-backward (see
`regime/models/_hmm_core.py`). hmmlearn is the most robust EM available; we
don't call its `predict_proba` because that's smoothed-only.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM

from regime.models._hmm_core import (
    forward_backward,
    forward_filter,
    gaussian_log_emissions,
)

_LOG = logging.getLogger(__name__)


class HmmGaussian:
    """Gaussian HMM regime model. Implements `StateRegimeModel`.

    Attributes:
        K: number of regimes (3).
        feature_columns: which feature columns to use as the observation vector.
        n_restarts: random-restart count for fitting.
    """

    K: int

    def __init__(
        self,
        K: int = 3,
        feature_columns: tuple[str, ...] = ("ret_1d", "rv_21d"),
        n_restarts: int = 10,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.K = K
        self.feature_columns = feature_columns
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._params: dict | None = None  # {"pi", "A", "means", "covs"}

    def fit(self, features: pl.DataFrame, train_idx: np.ndarray) -> None:
        X = self._extract(features, train_idx)
        if len(X) < 2 * self.K:
            raise ValueError(f"need at least {2 * self.K} obs to fit K={self.K} HMM, got {len(X)}")

        best_score = -np.inf
        best_model: GaussianHMM | None = None
        for offset in range(self.n_restarts):
            seed = self.random_state + offset
            m = GaussianHMM(
                n_components=self.K,
                covariance_type="full",
                n_iter=self.max_iter,
                tol=self.tol,
                random_state=seed,
                init_params="stmc",
            )
            try:
                m.fit(X)
                score = float(m.score(X))
            except Exception:
                _LOG.debug("restart %d failed", offset, exc_info=True)
                continue
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_model = m

        if best_model is None:
            raise RuntimeError("all HMM restarts failed to converge")

        self._params = {
            "pi": np.asarray(best_model.startprob_, dtype=np.float64),
            "A": np.asarray(best_model.transmat_, dtype=np.float64),
            "means": np.asarray(best_model.means_, dtype=np.float64),
            "covs": np.asarray(best_model.covars_, dtype=np.float64),
        }

    def filter(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        X = self._extract(features, idx)
        log_pi, log_A, log_emissions = self._log_pieces(X)
        return forward_filter(log_emissions, log_pi, log_A)

    def smooth(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        X = self._extract(features, idx)
        log_pi, log_A, log_emissions = self._log_pieces(X)
        return forward_backward(log_emissions, log_pi, log_A)

    def state_dict(self) -> dict:
        if self._params is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "K": self.K,
            "feature_columns": list(self.feature_columns),
            "pi": self._params["pi"].tolist(),
            "A": self._params["A"].tolist(),
            "means": self._params["means"].tolist(),
            "covs": self._params["covs"].tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        if not state.get("fitted"):
            self._params = None
            return
        self.K = int(state["K"])
        self.feature_columns = tuple(state["feature_columns"])
        self._params = {
            "pi": np.asarray(state["pi"], dtype=np.float64),
            "A": np.asarray(state["A"], dtype=np.float64),
            "means": np.asarray(state["means"], dtype=np.float64),
            "covs": np.asarray(state["covs"], dtype=np.float64),
        }

    def _extract(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        cols = list(self.feature_columns)
        mask = np.zeros(features.height, dtype=bool)
        mask[idx] = True
        sub = features.select(*cols).filter(pl.Series(mask)).drop_nulls()
        return sub.to_numpy()

    def _log_pieces(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._params is None:
            raise RuntimeError("model not fit")
        p = self._params
        with np.errstate(divide="ignore"):
            log_pi = np.log(p["pi"])
            log_A = np.log(p["A"])
        log_emissions = gaussian_log_emissions(X, p["means"], p["covs"])
        return log_pi, log_A, log_emissions
