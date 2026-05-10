"""Sparse Jump Model (Shu 2024; Bemporad-style mode partitioning).

A regime-clustering method that augments k-means with a jump penalty for
state transitions. Loss:

    sum_t ||x_t - mu_{s_t}||² + jump_penalty * #{t : s_t != s_{t-1}}

Implements `StateRegimeModel`. Output posteriors are one-hot (deterministic
clustering): `filter(t)` is online greedy state assignment using only data
up to t; `smooth(t)` is the optimal Viterbi-like DP over the full block.

After fitting, cluster centers are sorted by the first feature column's mean
so that state labels are stable across folds without external alignment.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

_LOG = logging.getLogger(__name__)


class SparseJumpModel:
    """Sparse jump model with deterministic state assignment. Implements StateRegimeModel."""

    K: int

    def __init__(
        self,
        K: int = 3,
        feature_columns: tuple[str, ...] = ("ret_1d", "rv_21d"),
        jump_penalty: float = 0.5,
        n_iter: int = 50,
        n_restarts: int = 5,
        tol: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        self.K = K
        self.feature_columns = feature_columns
        self.jump_penalty = float(jump_penalty)
        self.n_iter = int(n_iter)
        self.n_restarts = int(n_restarts)
        self.tol = float(tol)
        self.random_state = random_state
        self._params: dict | None = None

    def fit(self, features: pl.DataFrame, train_idx: np.ndarray) -> None:
        X = self._extract(features, train_idx)
        if len(X) < 2 * self.K:
            raise ValueError(f"need at least {2 * self.K} obs to fit K={self.K} SJM, got {len(X)}")

        best_loss = np.inf
        best_centers: np.ndarray | None = None
        for offset in range(self.n_restarts):
            seed = self.random_state + offset
            try:
                centers, loss = self._fit_one(X, seed)
            except Exception:
                _LOG.debug("SJM restart %d failed", offset, exc_info=True)
                continue
            if loss < best_loss:
                best_loss = loss
                best_centers = centers

        if best_centers is None:
            raise RuntimeError("all SJM restarts failed")

        # Sort centers by first-feature mean for stable state labelling.
        order = np.argsort(-best_centers[:, 0])
        self._params = {"centers": best_centers[order]}

    def filter(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        """Online greedy assignment using only data up to t (one-hot output)."""
        X = self._extract(features, idx)
        if self._params is None:
            raise RuntimeError("model not fit")
        states = _online_assign(X, self._params["centers"], self.jump_penalty)
        return _to_one_hot(states, self.K)

    def smooth(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        """DP-optimal assignment over the full block (one-hot output)."""
        X = self._extract(features, idx)
        if self._params is None:
            raise RuntimeError("model not fit")
        states = _dp_assign(X, self._params["centers"], self.jump_penalty)
        return _to_one_hot(states, self.K)

    def state_dict(self) -> dict:
        return {
            "fitted": self._params is not None,
            "K": self.K,
            "feature_columns": list(self.feature_columns),
            "jump_penalty": self.jump_penalty,
            "centers": (self._params["centers"].tolist() if self._params is not None else None),
        }

    def load_state_dict(self, state: dict) -> None:
        if not state.get("fitted"):
            self._params = None
            return
        self.K = int(state["K"])
        self.feature_columns = tuple(state["feature_columns"])
        self.jump_penalty = float(state["jump_penalty"])
        self._params = {"centers": np.asarray(state["centers"], dtype=np.float64)}

    # ------------------------------------------------------------------

    def _extract(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        cols = list(self.feature_columns)
        mask = np.zeros(features.height, dtype=bool)
        mask[idx] = True
        sub = features.select(*cols).filter(pl.Series(mask)).drop_nulls()
        return sub.to_numpy().astype(np.float64)

    def _fit_one(self, X: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(seed)
        # Initialize centers: random samples from the data (k-means++ light).
        idx0 = rng.integers(0, len(X), size=1)[0]
        centers = [X[idx0]]
        for _ in range(self.K - 1):
            dists = np.min(
                np.linalg.norm(X[:, None, :] - np.array(centers)[None, :, :], axis=2) ** 2,
                axis=1,
            )
            probs = dists / max(float(dists.sum()), 1e-12)
            idx_next = int(rng.choice(len(X), p=probs))
            centers.append(X[idx_next])
        centers = np.array(centers, dtype=np.float64)

        prev_loss = np.inf
        for _ in range(self.n_iter):
            states = _dp_assign(X, centers, self.jump_penalty)
            new_centers = _update_centers(X, states, self.K, fallback=centers)
            loss = _compute_loss(X, new_centers, states, self.jump_penalty)
            centers = new_centers
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return centers, float(prev_loss)


def _dp_assign(X: np.ndarray, centers: np.ndarray, jump_penalty: float) -> np.ndarray:
    """Viterbi-like DP for optimal SJM state assignment."""
    T = len(X)
    K = centers.shape[0]
    local = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2) ** 2  # (T, K)

    cost = np.empty((T, K), dtype=np.float64)
    backptr = np.zeros((T, K), dtype=np.int64)
    cost[0] = local[0]
    for t in range(1, T):
        # candidate[j, k] = cost[t-1, j] + jump_penalty * (j != k)
        prev = cost[t - 1]
        candidate = prev[:, None] + jump_penalty * (1.0 - np.eye(K))
        best_j = np.argmin(candidate, axis=0)
        best_val = candidate[best_j, np.arange(K)]
        cost[t] = best_val + local[t]
        backptr[t] = best_j

    states = np.empty(T, dtype=np.int64)
    states[-1] = int(np.argmin(cost[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]
    return states


def _online_assign(X: np.ndarray, centers: np.ndarray, jump_penalty: float) -> np.ndarray:
    """Greedy online state assignment using only data up to t."""
    T = len(X)
    K = centers.shape[0]
    states = np.empty(T, dtype=np.int64)
    local = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2) ** 2  # (T, K)
    states[0] = int(np.argmin(local[0]))
    for t in range(1, T):
        cost = local[t].copy()
        switch_mask = np.arange(K) != states[t - 1]
        cost = cost + jump_penalty * switch_mask.astype(np.float64)
        states[t] = int(np.argmin(cost))
    return states


def _update_centers(X: np.ndarray, states: np.ndarray, K: int, fallback: np.ndarray) -> np.ndarray:
    """Recompute cluster centers as means within each assigned state."""
    centers = np.empty_like(fallback)
    for k in range(K):
        members = X[states == k]
        if len(members) == 0:
            centers[k] = fallback[k]
        else:
            centers[k] = members.mean(axis=0)
    return centers


def _compute_loss(
    X: np.ndarray, centers: np.ndarray, states: np.ndarray, jump_penalty: float
) -> float:
    sq = np.linalg.norm(X - centers[states], axis=1) ** 2
    jumps = int(np.sum(states[1:] != states[:-1]))
    return float(sq.sum() + jump_penalty * jumps)


def _to_one_hot(states: np.ndarray, K: int) -> np.ndarray:
    out = np.zeros((len(states), K), dtype=np.float64)
    out[np.arange(len(states)), states] = 1.0
    return out
