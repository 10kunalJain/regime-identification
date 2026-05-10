"""Wasserstein k-means clustering on rolling distributional windows.

Each timestep t is represented by a window of W observations ending at t,
treated as an empirical distribution. Cluster the windows by Wasserstein-2
distance between empirical distributions; native features per timestep are the
distances from that timestep's window to each cluster medoid.

Implements `ChangePointModel`. The output is a (T, K) distance matrix, *not*
P(state) — the ensemble crisis-head consumes these distances as auxiliary
features per Q7's two-protocol design.

Univariate fast path: sort each window and use the L2 distance between order
statistics (closed form for W_2 between empirical 1D distributions).
Multivariate: sliced Wasserstein with random unit projections.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

_LOG = logging.getLogger(__name__)


class WassersteinKmeans:
    """Wasserstein k-means medoid clustering. Implements ChangePointModel."""

    def __init__(
        self,
        K: int = 3,
        feature_columns: tuple[str, ...] = ("ret_1d",),
        window: int = 21,
        n_iter: int = 20,
        n_restarts: int = 3,
        n_projections: int = 50,
        random_state: int = 42,
    ) -> None:
        self.K = int(K)
        self.feature_columns = feature_columns
        self.window = int(window)
        self.n_iter = int(n_iter)
        self.n_restarts = int(n_restarts)
        self.n_projections = int(n_projections)
        self.random_state = random_state
        self._params: dict | None = None  # {"medoids": ndarray of shape (K, W, D)}

    def fit(self, features: pl.DataFrame, train_idx: np.ndarray) -> None:
        X = self._extract(features, train_idx)
        if len(X) < self.window + self.K:
            raise ValueError(f"need at least {self.window + self.K} obs; got {len(X)}")
        windows = _build_windows(X, self.window)
        n = len(windows)
        if n < self.K:
            raise ValueError(f"need ≥ K={self.K} windows; got {n}")

        best_loss = np.inf
        best_medoids: np.ndarray | None = None
        for offset in range(self.n_restarts):
            seed = self.random_state + offset
            try:
                medoids, loss = self._fit_one(windows, seed)
            except Exception:
                _LOG.debug("WKmeans restart %d failed", offset, exc_info=True)
                continue
            if loss < best_loss:
                best_loss = loss
                best_medoids = medoids
        if best_medoids is None:
            raise RuntimeError("all WassersteinKmeans restarts failed")
        self._params = {"medoids": best_medoids}

    def native_features(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("model not fit")
        X = self._extract(features, idx)
        T = len(X)
        out = np.full((T, self.K), np.nan, dtype=np.float64)
        rng = np.random.default_rng(self.random_state + 7919)
        for t in range(self.window - 1, T):
            w_t = X[t - self.window + 1 : t + 1]
            for k in range(self.K):
                out[t, k] = _wasserstein_distance(
                    w_t, self._params["medoids"][k], self.n_projections, rng
                )
        return out

    def state_dict(self) -> dict:
        return {
            "fitted": self._params is not None,
            "K": self.K,
            "feature_columns": list(self.feature_columns),
            "window": self.window,
            "n_projections": self.n_projections,
            "medoids": (self._params["medoids"].tolist() if self._params is not None else None),
        }

    def load_state_dict(self, state: dict) -> None:
        if not state.get("fitted"):
            self._params = None
            return
        self.K = int(state["K"])
        self.feature_columns = tuple(state["feature_columns"])
        self.window = int(state["window"])
        self.n_projections = int(state["n_projections"])
        self._params = {"medoids": np.asarray(state["medoids"], dtype=np.float64)}

    # ------------------------------------------------------------------

    def _extract(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        cols = list(self.feature_columns)
        mask = np.zeros(features.height, dtype=bool)
        mask[idx] = True
        sub = features.select(*cols).filter(pl.Series(mask)).drop_nulls()
        return sub.to_numpy().astype(np.float64)

    def _fit_one(self, windows: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(seed)
        n = len(windows)
        # Initialize: kmeans++-flavoured medoid selection
        first = int(rng.integers(0, n))
        medoid_indices = [first]
        for _ in range(self.K - 1):
            dists_to_existing = np.array(
                [
                    min(
                        _wasserstein_distance(windows[i], windows[m], self.n_projections, rng)
                        for m in medoid_indices
                    )
                    for i in range(n)
                ]
            )
            sq = dists_to_existing**2
            probs = sq / max(float(sq.sum()), 1e-12)
            medoid_indices.append(int(rng.choice(n, p=probs)))
        medoids = windows[medoid_indices].copy()

        prev_assignments: np.ndarray | None = None
        for _ in range(self.n_iter):
            dists = np.empty((n, self.K), dtype=np.float64)
            for k in range(self.K):
                for i in range(n):
                    dists[i, k] = _wasserstein_distance(
                        windows[i], medoids[k], self.n_projections, rng
                    )
            assignments = dists.argmin(axis=1)
            if prev_assignments is not None and np.array_equal(prev_assignments, assignments):
                break
            prev_assignments = assignments

            for k in range(self.K):
                members_idx = np.where(assignments == k)[0]
                if len(members_idx) == 0:
                    continue
                # Medoid = within-cluster minimum total distance.
                m = len(members_idx)
                intra = np.zeros(m, dtype=np.float64)
                for i in range(m):
                    for j in range(m):
                        if i == j:
                            continue
                        intra[i] += _wasserstein_distance(
                            windows[members_idx[i]],
                            windows[members_idx[j]],
                            self.n_projections,
                            rng,
                        )
                medoids[k] = windows[members_idx[int(np.argmin(intra))]]

        loss = float(
            sum(
                _wasserstein_distance(windows[i], medoids[assignments[i]], self.n_projections, rng)
                for i in range(n)
            )
        )
        return medoids, loss


def _build_windows(X: np.ndarray, window: int) -> np.ndarray:
    """Sliding windows of length `window`. Returns shape (T - window + 1, window, D)."""
    T = len(X)
    if T < window:
        return np.empty((0, window, X.shape[1]), dtype=np.float64)
    return np.stack([X[i : i + window] for i in range(T - window + 1)], axis=0)


def _wasserstein_distance(
    a: np.ndarray, b: np.ndarray, n_projections: int, rng: np.random.Generator
) -> float:
    """W₂ between two equal-size empirical distributions.

    Univariate fast path: sort and L2-difference order stats.
    Multivariate: sliced Wasserstein with `n_projections` random unit directions.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if a.shape[1] == 1:
        sa = np.sort(a[:, 0])
        sb = np.sort(b[:, 0])
        return float(np.sqrt(np.mean((sa - sb) ** 2)))
    d = a.shape[1]
    total = 0.0
    for _ in range(n_projections):
        direction = rng.normal(size=d)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        direction = direction / norm
        pa = np.sort(a @ direction)
        pb = np.sort(b @ direction)
        total += float(np.mean((pa - pb) ** 2))
    return float(np.sqrt(total / max(n_projections, 1)))
