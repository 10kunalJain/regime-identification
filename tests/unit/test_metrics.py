"""Tests for evaluation metrics."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from regime.eval.crises import CrisisEvent
from regime.eval.metrics import (
    brier_score,
    detection_lag,
    first_sustained_fire,
    regime_dwell_times,
    transition_matrix_frobenius,
)


def _seq(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


# ---------- first_sustained_fire ----------


def test_sustained_fire_returns_first_day_of_streak():
    dates = _seq(date(2020, 1, 1), 20)
    probs = np.zeros(20)
    probs[5:10] = 0.7  # 5-day streak above 0.5 starting at index 5
    fire = first_sustained_fire(dates, probs, after=date(2020, 1, 1))
    assert fire == dates[5]


def test_sustained_fire_requires_three_consecutive():
    dates = _seq(date(2020, 1, 1), 10)
    probs = np.array([0.0, 0.7, 0.7, 0.0, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0])
    fire = first_sustained_fire(dates, probs, after=date(2020, 1, 1))
    # First 2-day streak at indices 1-2 doesn't qualify; 3-day streak at 4-6 does.
    assert fire == dates[4]


def test_sustained_fire_returns_none_when_no_streak():
    dates = _seq(date(2020, 1, 1), 10)
    probs = np.array([0.7, 0.7, 0.0, 0.7, 0.7, 0.0, 0.7, 0.7, 0.0, 0.0])
    fire = first_sustained_fire(dates, probs, after=date(2020, 1, 1))
    assert fire is None


def test_sustained_fire_respects_after_cutoff():
    dates = _seq(date(2020, 1, 1), 20)
    probs = np.zeros(20)
    probs[2:8] = 0.7  # streak starting at index 2
    # After cutoff in the middle of the streak — the streak before the cutoff doesn't count.
    fire = first_sustained_fire(dates, probs, after=dates[5])
    # The streak from index 2 doesn't continue past index 5; probs[5:8] = 0.7 (3 days).
    # The streak starting at 2 is broken by the after-cutoff restart.
    assert fire == dates[5]


def test_sustained_fire_threshold_is_strict():
    """Threshold > is strict: a value exactly equal to threshold does NOT fire."""
    dates = _seq(date(2020, 1, 1), 5)
    probs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    fire = first_sustained_fire(dates, probs, after=date(2020, 1, 1), threshold=0.5)
    assert fire is None


# ---------- detection_lag ----------


def test_detection_lag_basic():
    crisis = CrisisEvent(
        name="test",
        peak_date=date(2020, 1, 5),
        m5_date=date(2020, 1, 10),
        m10_date=date(2020, 1, 15),
        bottom_date=date(2020, 1, 20),
        note="x" * 30,
    )
    dates = _seq(date(2020, 1, 1), 30)
    probs = np.zeros(30)
    # Fire 5 trading days after m5_date (which is dates[9]); streak starts at dates[14]
    probs[14:18] = 0.7
    res = detection_lag(dates, probs, crisis)
    assert res.first_fire_date == dates[14]
    assert res.lag_trading_days == 5  # dates[14] - dates[9] = 5 days


def test_detection_lag_no_fire():
    crisis = CrisisEvent(
        name="test",
        peak_date=date(2020, 1, 5),
        m5_date=date(2020, 1, 10),
        m10_date=date(2020, 1, 15),
        bottom_date=date(2020, 1, 20),
        note="x" * 30,
    )
    dates = _seq(date(2020, 1, 1), 30)
    probs = np.zeros(30)
    res = detection_lag(dates, probs, crisis)
    assert res.first_fire_date is None
    assert res.lag_trading_days is None


# ---------- brier_score ----------


def test_brier_perfect_is_zero():
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(p, y) == pytest.approx(0.0)


def test_brier_random_is_quarter():
    """Brier score of always-0.5 forecast on balanced labels = 0.25."""
    p = np.full(100, 0.5)
    y = np.tile([0.0, 1.0], 50)
    assert brier_score(p, y) == pytest.approx(0.25)


def test_brier_mismatched_shapes_raises():
    with pytest.raises(ValueError):
        brier_score(np.array([0.5]), np.array([0.0, 1.0]))


# ---------- transition_matrix_frobenius ----------


def test_frobenius_zero_when_identical():
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert transition_matrix_frobenius(a, a) == pytest.approx(0.0)


def test_frobenius_positive_when_different():
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    b = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert transition_matrix_frobenius(a, b) > 0


# ---------- regime_dwell_times ----------


def test_dwell_times_simple():
    seq = np.array([0, 0, 0, 1, 1, 0, 2, 2, 2, 2])
    dwells = regime_dwell_times(seq)
    assert dwells[0] == [3, 1]
    assert dwells[1] == [2]
    assert dwells[2] == [4]


def test_dwell_times_empty():
    assert regime_dwell_times(np.array([], dtype=np.int64)) == {}
