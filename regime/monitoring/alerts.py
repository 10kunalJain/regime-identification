"""Webhook alert dispatcher with deduplication.

Per ARCHITECTURE.md §9, five alert rules drive operational reliability:

  - Crisis alert (P(crisis) > 0.5 AND no fire in 30d)
  - Drift alert (max PSI > 0.25 for 1h)
  - Latency degradation (p99 filter latency > 0.1s)
  - Refresh failure (data_refresh_last_success_timestamp > 24h ago)
  - Oracle idle warning (avg CPU over 24h < 0.18) — fires ~6 days before
    Oracle Cloud Always Free reclamation kicks in.

The dispatcher:
  - Posts a JSON body to a webhook URL (Slack-incoming-webhook compatible).
  - Deduplicates: an alert with the same `key` is not re-fired within
    `dedup_window_seconds` (default 30 min, matching ARCHITECTURE.md §9
    "no fire in 30d" for crisis but configurable per call site).
  - Records the last-fire time per key in memory; for production, swap in a
    persisted `AlertHistory` (the deploy-phase Postgres adapter).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

_LOG = logging.getLogger(__name__)
DEFAULT_DEDUP_WINDOW_SECONDS = 30 * 60  # 30 minutes


@dataclass(frozen=True)
class AlertPayload:
    """A single webhook payload."""

    key: str  # dedup key (e.g., "crisis", "drift")
    title: str
    message: str
    severity: str = "info"  # "info" | "warning" | "critical"


class WebhookAlerter:
    """Thread-unsafe by design — one instance per scheduler / API process."""

    def __init__(
        self,
        webhook_url: str,
        client: httpx.Client | None = None,
        dedup_window_seconds: float = DEFAULT_DEDUP_WINDOW_SECONDS,
        clock: callable = time.time,  # type: ignore[type-arg]
    ) -> None:
        self.webhook_url = webhook_url
        self._client = client or httpx.Client(timeout=5.0)
        self.dedup_window_seconds = float(dedup_window_seconds)
        self._last_fire: dict[str, float] = {}
        self._clock = clock

    def fire(self, payload: AlertPayload) -> bool:
        """Return True if the webhook was sent (and got a 2xx); False if
        dedup'd or HTTP failed."""
        now = float(self._clock())
        last = self._last_fire.get(payload.key)
        if last is not None and (now - last) < self.dedup_window_seconds:
            _LOG.debug("alert %r dedup'd (last fire %.1fs ago)", payload.key, now - last)
            return False
        body = {
            "key": payload.key,
            "title": payload.title,
            "text": payload.message,
            "severity": payload.severity,
        }
        try:
            response = self._client.post(self.webhook_url, json=body)
        except httpx.HTTPError as exc:
            _LOG.warning("webhook POST failed for alert %r: %s", payload.key, exc)
            return False
        if response.status_code >= 400:
            _LOG.warning(
                "webhook returned %d for alert %r: %s",
                response.status_code,
                payload.key,
                response.text[:200],
            )
            return False
        self._last_fire[payload.key] = now
        return True


def crisis_alert(crisis_prob: float, threshold: float = 0.5) -> AlertPayload | None:
    """Build a crisis-alert payload if `crisis_prob` exceeds `threshold`."""
    if crisis_prob <= threshold:
        return None
    return AlertPayload(
        key="crisis",
        title="Crisis-onset probability above threshold",
        message=f"P(crisis within 21d) = {crisis_prob:.2%} ≥ {threshold:.0%}",
        severity="critical",
    )


def drift_alert(drift_panel: dict[str, float], threshold: float = 0.25) -> AlertPayload | None:
    """Build a drift-alert payload if any feature's PSI exceeds `threshold`."""
    if not drift_panel:
        return None
    worst_feature, worst_value = max(drift_panel.items(), key=lambda kv: kv[1])
    if worst_value <= threshold:
        return None
    return AlertPayload(
        key="drift",
        title="Feature drift above threshold",
        message=f"{worst_feature}: PSI = {worst_value:.3f} ≥ {threshold:.2f}",
        severity="warning",
    )


def latency_alert(p99_seconds: float, threshold_seconds: float = 0.1) -> AlertPayload | None:
    """Build a latency-degradation payload if p99 exceeds `threshold_seconds`."""
    if p99_seconds <= threshold_seconds:
        return None
    return AlertPayload(
        key="latency",
        title="Filter p99 latency degraded",
        message=f"p99 = {p99_seconds * 1000:.1f} ms ≥ {threshold_seconds * 1000:.0f} ms",
        severity="warning",
    )


def refresh_failure_alert(
    seconds_since_last_refresh: float, threshold_seconds: float = 86400.0
) -> AlertPayload | None:
    """Fire if the daily refresh hasn't succeeded within `threshold_seconds`."""
    if seconds_since_last_refresh <= threshold_seconds:
        return None
    hours = seconds_since_last_refresh / 3600.0
    return AlertPayload(
        key="refresh_failure",
        title="Data refresh failed",
        message=f"Last successful refresh was {hours:.1f}h ago",
        severity="critical",
    )


def oracle_idle_alert(
    cpu_utilization_avg_24h: float, threshold: float = 0.18
) -> AlertPayload | None:
    """Fire if Oracle CPU utilization 24h average < threshold (early-warning
    for Oracle Cloud Always Free reclamation; threshold matches the official
    20% idle floor with a 2-percentage-point margin)."""
    if cpu_utilization_avg_24h >= threshold:
        return None
    return AlertPayload(
        key="oracle_idle",
        title="Oracle CPU utilization below idle threshold",
        message=(
            f"Avg 24h CPU = {cpu_utilization_avg_24h:.1%} < {threshold:.0%};"
            " Oracle reclamation possible in ~6 days if this persists."
        ),
        severity="warning",
    )
