"""Tests for the forward-drawdown crisis-label generator."""

from __future__ import annotations

import numpy as np

from regime.eval.labels import (
    UNOBSERVABLE,
    forward_drawdown_indicator,
    observable_mask,
)


def test_label_shapes_and_unobservable_tail():
    close = np.linspace(100.0, 90.0, 100)
    y = forward_drawdown_indicator(close, horizon=21, threshold=0.10)
    assert y.shape == (100,)
    # Last 21 rows are unobservable.
    assert (y[-21:] == UNOBSERVABLE).all()


def test_label_no_drawdown_returns_zero():
    close = np.linspace(100.0, 200.0, 100)  # monotonic uptrend
    y = forward_drawdown_indicator(close, horizon=21, threshold=0.10)
    observable = y[y != UNOBSERVABLE]
    assert (observable == 0).all()


def test_label_strong_drawdown_triggers_one():
    close = np.full(100, 100.0)
    # Plant a 20% drop at t=50 → labels for t=29..49 should be 1
    close[50:71] = 80.0
    y = forward_drawdown_indicator(close, horizon=21, threshold=0.10)
    # At t=49, the next 21 days include t=50 (close=80), so drawdown = 20% > 10%.
    assert y[49] == 1
    # At t=20, close from 21..41 is still 100, no drawdown.
    assert y[20] == 0


def test_observable_mask_masks_unobservable():
    y = np.array([0, 1, 0, UNOBSERVABLE, UNOBSERVABLE])
    mask = observable_mask(y)
    np.testing.assert_array_equal(mask, np.array([True, True, True, False, False]))


def test_short_series_returns_all_unobservable():
    close = np.array([100.0, 99.0, 98.0])
    y = forward_drawdown_indicator(close, horizon=21, threshold=0.10)
    assert (y == UNOBSERVABLE).all()


def test_threshold_sensitivity():
    close = np.full(50, 100.0)
    close[25:46] = 95.0  # 5% drop
    y_strict = forward_drawdown_indicator(close, horizon=21, threshold=0.10)
    y_loose = forward_drawdown_indicator(close, horizon=21, threshold=0.04)
    # At t=24 with a 5% drop ahead, strict threshold (10%) → 0, loose (4%) → 1.
    assert y_strict[24] == 0
    assert y_loose[24] == 1
