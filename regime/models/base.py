"""Model interface protocols.

Two distinct protocols, per the design grill (Q7):

  - State-based methods (HMM, MS-AR, SJM, joint HMM) implement
    `StateRegimeModel` and output P(state).
  - Change-point methods (BOCPD, Wasserstein k-means) implement
    `ChangePointModel` and output native features (run-length distribution
    summaries, cluster distances). They do NOT pretend to output P(state).

The ensemble stacker consumes both schemas. See ARCHITECTURE.md §4.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl


@runtime_checkable
class StateRegimeModel(Protocol):
    """For HMM-family methods that output P(state) per timestep."""

    K: int

    def fit(self, features: pl.DataFrame, train_idx: np.ndarray) -> None:
        """Fit on the rows indexed by train_idx within `features`."""
        ...

    def filter(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        """Return P(s_t | y_{1:t}) for each t in idx. Shape (len(idx), K).

        Filtered (online, no peeking at future). Use this for walk-forward eval.
        """
        ...

    def smooth(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        """Return P(s_t | y_{1:T}). Shape (len(idx), K).

        Smoothed (uses the entire fold). Use only for descriptive plots and
        the historical-explorer panel; never in walk-forward evaluation.
        """
        ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


@runtime_checkable
class ChangePointModel(Protocol):
    """For change-point methods (BOCPD, Wasserstein k-means)."""

    def fit(self, features: pl.DataFrame, train_idx: np.ndarray) -> None: ...

    def native_features(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        """Return native change-point features. Shape (len(idx), F).

        For BOCPD: run-length distribution summary statistics + change probability.
        For Wasserstein k-means: distances to cluster centroids + soft assignment.
        Not a P(state). The ensemble crisis-head consumes these as auxiliary
        features.
        """
        ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...
