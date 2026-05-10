"""Alert-rule + WebhookAlerter tests."""

from __future__ import annotations

import httpx

from regime.monitoring.alerts import (
    AlertPayload,
    WebhookAlerter,
    crisis_alert,
    drift_alert,
    latency_alert,
    oracle_idle_alert,
    refresh_failure_alert,
)


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"ok": True})


def _failing_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "boom"})


def _network_failing_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("offline")


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


# ---------- alert-rule pure functions ----------


def test_crisis_alert_fires_above_threshold():
    p = crisis_alert(crisis_prob=0.55, threshold=0.5)
    assert p is not None
    assert p.key == "crisis"
    assert p.severity == "critical"


def test_crisis_alert_silent_below_threshold():
    assert crisis_alert(crisis_prob=0.3, threshold=0.5) is None


def test_drift_alert_picks_worst_feature():
    panel = {"ret_1d": 0.05, "rv_21d": 0.40, "vix_term_structure": 0.15}
    p = drift_alert(panel, threshold=0.25)
    assert p is not None
    assert "rv_21d" in p.message
    assert p.severity == "warning"


def test_drift_alert_silent_when_all_below():
    panel = {"a": 0.01, "b": 0.05}
    assert drift_alert(panel, threshold=0.25) is None


def test_drift_alert_silent_on_empty_panel():
    assert drift_alert({}, threshold=0.25) is None


def test_latency_alert():
    assert latency_alert(0.05) is None
    p = latency_alert(0.150)
    assert p is not None
    assert "150" in p.message


def test_refresh_failure_alert():
    assert refresh_failure_alert(3600.0) is None
    p = refresh_failure_alert(2 * 86400.0)
    assert p is not None and p.severity == "critical"


def test_oracle_idle_alert():
    assert oracle_idle_alert(0.30) is None
    p = oracle_idle_alert(0.10, threshold=0.18)
    assert p is not None
    assert "10" in p.message


# ---------- WebhookAlerter ----------


def test_alerter_fires_payload_to_webhook():
    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(200)

    alerter = WebhookAlerter(
        webhook_url="http://test/hook",
        client=_client(handler),
        clock=lambda: 1000.0,
    )
    payload = AlertPayload(key="crisis", title="x", message="y", severity="critical")
    sent = alerter.fire(payload)
    assert sent is True
    assert len(requests) == 1
    body = requests[0].read().decode()
    assert '"crisis"' in body
    assert '"critical"' in body


def test_alerter_dedups_within_window():
    counter = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(200)

    times = iter([1000.0, 1100.0, 2900.0])  # 100s, then 1900s later

    alerter = WebhookAlerter(
        webhook_url="http://test/hook",
        client=_client(handler),
        dedup_window_seconds=1800.0,  # 30 min
        clock=lambda: next(times),
    )
    p = AlertPayload(key="crisis", title="x", message="y")
    assert alerter.fire(p) is True  # t=1000 → fires
    assert alerter.fire(p) is False  # t=1100, within window → dedup'd
    assert alerter.fire(p) is True  # t=2900, outside window → fires
    assert counter["n"] == 2


def test_alerter_returns_false_on_5xx():
    alerter = WebhookAlerter(
        webhook_url="http://test/hook",
        client=_client(_failing_handler),
        clock=lambda: 1000.0,
    )
    sent = alerter.fire(AlertPayload(key="x", title="t", message="m"))
    assert sent is False


def test_alerter_returns_false_on_network_error():
    alerter = WebhookAlerter(
        webhook_url="http://test/hook",
        client=_client(_network_failing_handler),
        clock=lambda: 1000.0,
    )
    sent = alerter.fire(AlertPayload(key="x", title="t", message="m"))
    assert sent is False


def test_alerter_failed_send_does_not_record_dedup():
    """If a fire fails, the next attempt should not be dedup'd."""
    attempts = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        # First fails, second succeeds.
        return httpx.Response(500 if attempts["n"] == 1 else 200)

    alerter = WebhookAlerter(
        webhook_url="http://test/hook",
        client=_client(handler),
        dedup_window_seconds=1800.0,
        clock=lambda: 1000.0,
    )
    p = AlertPayload(key="crisis", title="x", message="y")
    assert alerter.fire(p) is False  # first attempt fails
    assert alerter.fire(p) is True  # second attempt should not be dedup'd
    assert attempts["n"] == 2
