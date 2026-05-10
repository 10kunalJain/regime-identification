"""Tests for the FastAPI /metrics endpoint integration."""

from __future__ import annotations

from fastapi.testclient import TestClient

from regime.api.app import create_app
from regime.monitoring.metrics import MetricsRegistry
from regime.runtime.state import InMemoryStateStore


def test_metrics_endpoint_serves_text():
    metrics = MetricsRegistry()
    metrics.set_gauge("regime_crisis_prob_21d_calibrated", 0.42)
    metrics.set_gauge("regime_posterior_prob", 0.7, labels={"regime": "calm_bull"})
    app = create_app(InMemoryStateStore(), metrics=metrics)
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert "regime_crisis_prob_21d_calibrated 0.42" in body
    assert 'regime_posterior_prob{regime="calm_bull"} 0.7' in body


def test_metrics_endpoint_default_registry_empty():
    app = create_app(InMemoryStateStore())
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    # Empty registry renders to just a trailing newline.
    assert r.text == "\n"
