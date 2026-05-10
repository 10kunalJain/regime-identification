"""Meta-ensemble stacker: combines the four state-based methods.

Three-state posterior is the (un-calibrated in v1) average of Hungarian-aligned
filtered outputs. Each underlying state-based model already enforces a stable
state-label ordering (by SPY-mean for HmmGaussian / MsarT / SparseJumpModel /
JointHmm), so the average is meaningful without an explicit per-fold alignment
step. The crisis probability is read off as the last column (lowest-mean
regime) by convention.

Per Q7 of the design grill: this stacker only handles state-based methods.
Change-point methods feed their native features directly into the crisis head
(see `regime/ensemble/crisis_head.py`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from regime.models.base import StateRegimeModel


class EnsembleStacker:
    """Average filtered posteriors from a list of `StateRegimeModel`s."""

    def __init__(self, models: Sequence[StateRegimeModel]) -> None:
        if not models:
            raise ValueError("EnsembleStacker requires at least one model")
        self.models = list(models)
        K = self.models[0].K
        for m in self.models[1:]:
            if m.K != K:
                raise ValueError(
                    f"all stacked models must share K; got {[m.K for m in self.models]}"
                )
        self.K = K

    def fit(self, features: pl.DataFrame, train_idx: np.ndarray) -> None:
        for m in self.models:
            m.fit(features, train_idx)

    def filter(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        """Average the filtered (T, K) outputs across all models."""
        outs = [m.filter(features, idx) for m in self.models]
        # All models must produce the same number of rows; if a model dropped
        # null rows, take the trailing min length.
        min_len = min(o.shape[0] for o in outs)
        outs = [o[-min_len:] for o in outs]
        stacked = np.stack(outs, axis=0)
        return stacked.mean(axis=0)

    def smooth(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        outs = [m.smooth(features, idx) for m in self.models]
        min_len = min(o.shape[0] for o in outs)
        outs = [o[-min_len:] for o in outs]
        return np.stack(outs, axis=0).mean(axis=0)

    def crisis_prob(self, features: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
        """Marginal crisis-state probability = last column of the averaged posterior."""
        return self.filter(features, idx)[:, -1]
