"""FastAPI endpoint tests using the in-memory state store."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from regime.api.app import create_app
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


@pytest.fixture
def store_with_seed_data() -> InMemoryStateStore:
    store = InMemoryStateStore()
    store.put_posterior(_post(date(2020, 1, 1), crisis=0.05))
    store.put_posterior(_post(date(2020, 1, 5), crisis=0.10))
    store.put_posterior(_post(date(2020, 3, 9), crisis=0.45))
    return store


def test_healthz():
    app = create_app(InMemoryStateStore())
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_regime_now_returns_404_when_empty():
    app = create_app(InMemoryStateStore())
    client = TestClient(app)
    r = client.get("/regime/now")
    assert r.status_code == 404
    assert "no posterior" in r.json()["detail"]


def test_regime_now_returns_latest(store_with_seed_data):
    app = create_app(store_with_seed_data)
    client = TestClient(app)
    r = client.get("/regime/now")
    assert r.status_code == 200
    body = r.json()
    assert body["as_of"] == "2020-03-09"
    assert body["crisis_prob_21d_cal"] == 0.45
    assert body["method"] == "ensemble"


def test_regime_path_returns_all_when_unbounded(store_with_seed_data):
    app = create_app(store_with_seed_data)
    client = TestClient(app)
    r = client.get("/regime/path")
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_regime_path_filters_by_from_to(store_with_seed_data):
    app = create_app(store_with_seed_data)
    client = TestClient(app)
    r = client.get("/regime/path", params={"from": "2020-01-04", "to": "2020-02-01"})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["as_of"] == "2020-01-05"


def test_regime_path_rejects_inverted_range(store_with_seed_data):
    app = create_app(store_with_seed_data)
    client = TestClient(app)
    r = client.get("/regime/path", params={"from": "2020-03-01", "to": "2020-01-01"})
    assert r.status_code == 400
    assert "cannot be after" in r.json()["detail"]


def test_forecast_returns_501():
    app = create_app(InMemoryStateStore())
    client = TestClient(app)
    r = client.get("/forecast", params={"horizon": 21})
    assert r.status_code == 501


def test_forecast_validates_horizon():
    app = create_app(InMemoryStateStore())
    client = TestClient(app)
    r = client.get("/forecast", params={"horizon": 100})
    # Pydantic Query validation rejects horizon > 63.
    assert r.status_code == 422
