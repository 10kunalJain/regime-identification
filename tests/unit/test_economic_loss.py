"""Tests for the economic-loss threshold curve."""

from __future__ import annotations

import numpy as np
import pytest

from regime.ensemble.economic_loss import (
    DEFAULT_FALSE_ALARM_COST_BP,
    DEFAULT_MISSED_CRISIS_COST_BP,
    economic_loss_curve,
    optimal_threshold,
)


def test_loss_curve_zero_threshold_all_alarms():
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0, 0, 1, 0, 0])
    curve = economic_loss_curve(proba, y, thresholds=np.array([-0.01]))
    # threshold below all predictions → predict 1 everywhere → false alarms = #negatives.
    assert curve.false_alarms[0] == 4
    assert curve.missed_crises[0] == 0


def test_loss_curve_high_threshold_all_misses():
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0, 0, 1, 0, 0])
    curve = economic_loss_curve(proba, y, thresholds=np.array([1.01]))
    # threshold above all predictions → predict 0 everywhere → missed = #positives.
    assert curve.false_alarms[0] == 0
    assert curve.missed_crises[0] == 1


def test_optimal_threshold_under_asymmetric_loss():
    """With our default 10:120 cost ratio, the optimal threshold should fire on
    a feature even when its empirical positive rate is low."""
    rng = np.random.default_rng(0)
    n = 5000
    base_rate = 0.05
    y = (rng.uniform(0, 1, size=n) < base_rate).astype(np.int64)
    # Predictor that's slightly correlated with y.
    proba = 0.3 * y + rng.uniform(0, 0.5, size=n)
    threshold, curve = optimal_threshold(proba, y)
    assert 0.0 <= threshold <= 1.0
    assert curve.losses.min() == curve.losses[np.argmin(np.abs(curve.thresholds - threshold))]


def test_loss_proportional_to_cost_ratios():
    proba = np.array([0.1, 0.6])
    y = np.array([0, 1])
    # threshold = 0.5: predict [0, 1] → 0 false alarms, 0 missed.
    curve = economic_loss_curve(
        proba,
        y,
        false_alarm_cost_bp=DEFAULT_FALSE_ALARM_COST_BP,
        missed_crisis_cost_bp=DEFAULT_MISSED_CRISIS_COST_BP,
        thresholds=np.array([0.5]),
    )
    assert curve.losses[0] == pytest.approx(0.0)
