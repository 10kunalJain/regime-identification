"""Tests for calibration metrics: reliability curve, ECE, Brier, PR-AUC."""

from __future__ import annotations

import numpy as np
import pytest

from regime.ensemble.calibration import (
    brier_score,
    expected_calibration_error,
    positive_base_rate,
    pr_auc,
    reliability_curve,
)


def test_reliability_curve_perfectly_calibrated():
    """If predictions equal observed rates per bin, ECE ≈ 0."""
    rng = np.random.default_rng(0)
    proba = rng.uniform(0.0, 1.0, size=10000)
    y = (rng.uniform(0.0, 1.0, size=10000) < proba).astype(np.int64)
    ece = expected_calibration_error(proba, y, n_bins=10)
    assert ece < 0.05


def test_reliability_curve_miscalibrated():
    """A constant 0.5 prediction on a 90%-positive sample → mean_observed ≫ predicted."""
    proba = np.full(1000, 0.5)
    y = np.ones(1000, dtype=np.int64)
    rc = reliability_curve(proba, y, n_bins=10)
    # All predictions in the [0.4, 0.5) or [0.5, 0.6) bin
    nonempty = rc.bin_count > 0
    assert (rc.mean_observed[nonempty] - rc.mean_predicted[nonempty] > 0.4).any()


def test_brier_perfect_zero():
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1, 0, 1, 0])
    assert brier_score(p, y) == pytest.approx(0.0)


def test_brier_constant_half_on_balanced():
    p = np.full(100, 0.5)
    y = np.tile([0, 1], 50)
    assert brier_score(p, y) == pytest.approx(0.25)


def test_brier_shape_mismatch_raises():
    with pytest.raises(ValueError):
        brier_score(np.array([0.5]), np.array([0, 1]))


def test_pr_auc_high_for_good_classifier():
    """A classifier whose probability is exactly the true label has PR-AUC = 1."""
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    proba = y.astype(np.float64)
    assert pr_auc(proba, y) == pytest.approx(1.0)


def test_pr_auc_random_near_baserate():
    """A random classifier on imbalanced data should have PR-AUC near base rate."""
    rng = np.random.default_rng(0)
    n = 10000
    y = (rng.uniform(0, 1, size=n) < 0.1).astype(np.int64)
    proba = rng.uniform(0, 1, size=n)
    auc = pr_auc(proba, y)
    base = positive_base_rate(y)
    assert auc > 0.05
    assert auc < base + 0.05  # close to baseline


def test_pr_auc_handles_all_zero_or_all_one():
    y_zero = np.zeros(10, dtype=np.int64)
    proba = np.array([0.5] * 10)
    assert pr_auc(proba, y_zero) == 0.0


def test_positive_base_rate():
    y = np.array([0, 0, 1, 1, 0])
    assert positive_base_rate(y) == pytest.approx(0.4)
