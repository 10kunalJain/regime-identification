"""Tests for the crisis-onset early-warning head."""

from __future__ import annotations

import numpy as np
import pytest

from regime.ensemble.crisis_head import CrisisHead
from regime.eval.labels import UNOBSERVABLE


def _separable_data(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Linearly separable 5-feature classification problem with ~10% positive rate."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 5))
    # True logit: positive when first two features sum positive
    score = X[:, 0] + 0.5 * X[:, 1] - 1.5
    proba_true = 1.0 / (1.0 + np.exp(-score))
    y = (rng.uniform(0, 1, size=n) < proba_true).astype(np.int64)
    return X, y


def test_crisis_head_predicts_in_zero_one():
    X, y = _separable_data(n=600, seed=1)
    head = CrisisHead()
    head.fit(X, y)
    p = head.predict_proba(X)
    assert (p >= 0.0).all() and (p <= 1.0).all()


def test_crisis_head_better_than_baserate_on_separable():
    """On linearly-separable data the head should clearly beat a constant baserate predictor."""
    X, y = _separable_data(n=2000, seed=2)
    head = CrisisHead()
    head.fit(X, y)
    p = head.predict_proba(X)
    base = float(y.mean())
    base_brier = float(np.mean((np.full_like(p, base) - y) ** 2))
    head_brier = float(np.mean((p - y) ** 2))
    assert head_brier < base_brier


def test_crisis_head_drops_unobservable_rows():
    """UNOBSERVABLE-labelled rows must be excluded from training."""
    X, y_obs = _separable_data(n=400, seed=3)
    # Mark the trailing 50 rows as unobservable.
    y = y_obs.copy()
    y[-50:] = UNOBSERVABLE
    head = CrisisHead()
    head.fit(X, y)
    # Predicting on the unobservable rows should still work — the head only
    # rejected them at training time.
    p = head.predict_proba(X[-50:])
    assert (p >= 0.0).all() and (p <= 1.0).all()


def test_crisis_head_state_dict_roundtrip():
    X, y = _separable_data(n=400, seed=4)
    head = CrisisHead()
    head.fit(X, y)
    p_before = head.predict_proba(X)

    state = head.state_dict()
    restored = CrisisHead()
    restored.load_state_dict(state)
    p_after = restored.predict_proba(X)

    np.testing.assert_allclose(p_before, p_after, atol=1e-9)


def test_crisis_head_requires_both_classes():
    """All-zero labels should raise."""
    X = np.random.RandomState(0).normal(size=(20, 3))
    y = np.zeros(20, dtype=np.int64)
    head = CrisisHead()
    with pytest.raises(ValueError, match="positive and negative"):
        head.fit(X, y)


def test_crisis_head_unfitted_predict_raises():
    head = CrisisHead()
    with pytest.raises(RuntimeError, match="not fit"):
        head.predict_proba(np.zeros((1, 3)))
