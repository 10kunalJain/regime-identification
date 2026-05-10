"""In-memory state store tests."""

from __future__ import annotations

from datetime import date

import pytest

from regime.api.schemas import RegimePosterior
from regime.runtime.state import InMemoryStateStore


def _post(d: date, crisis: float = 0.05) -> RegimePosterior:
    return RegimePosterior(
        as_of=d,
        regime_probs_uncal={"calm_bull": 0.7, "neutral": 0.25, "crisis": 0.05},
        crisis_prob_21d_cal=crisis,
        confidence=0.8,
        method="ensemble",
        version="abc123",
    )


def test_empty_store_returns_none():
    store = InMemoryStateStore()
    assert store.get_latest_posterior() is None
    assert store.get_filter_state() is None
    assert store.get_posterior_path(None, None) == []


def test_put_and_get_latest():
    store = InMemoryStateStore()
    store.put_posterior(_post(date(2020, 1, 1)))
    store.put_posterior(_post(date(2020, 1, 3)))
    store.put_posterior(_post(date(2020, 1, 2)))
    latest = store.get_latest_posterior()
    assert latest is not None
    assert latest.as_of == date(2020, 1, 3)


def test_get_posterior_path_filters():
    store = InMemoryStateStore()
    for d in (date(2020, 1, 1), date(2020, 1, 5), date(2020, 1, 10)):
        store.put_posterior(_post(d))
    path = store.get_posterior_path(date(2020, 1, 3), date(2020, 1, 8))
    assert [p.as_of for p in path] == [date(2020, 1, 5)]


def test_get_posterior_path_unbounded():
    store = InMemoryStateStore()
    for d in (date(2020, 1, 1), date(2020, 1, 5)):
        store.put_posterior(_post(d))
    assert len(store.get_posterior_path(None, None)) == 2


def test_filter_state_roundtrip():
    store = InMemoryStateStore()
    state = {"log_pi": [0.1, 0.5], "log_alpha": [0.0, 0.0], "t": 7}
    store.put_filter_state(state)
    loaded = store.get_filter_state()
    assert loaded == state
    # Returned dict is a copy — mutating it should not affect the store.
    assert loaded is not None
    loaded["t"] = 999
    refetched = store.get_filter_state()
    assert refetched is not None
    assert refetched["t"] == 7


def test_posterior_schema_validates_crisis_prob_range():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RegimePosterior(
            as_of=date(2020, 1, 1),
            regime_probs_uncal={"calm_bull": 1.0},
            crisis_prob_21d_cal=1.5,  # > 1.0 → validation error
            confidence=0.5,
            method="x",
            version="x",
        )
