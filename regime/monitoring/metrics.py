"""Lightweight Prometheus exposition.

We don't pull in `prometheus_client` to keep the dep tree small — gauges and
counters are plain attributes on a registry, and the registry renders to
Prometheus text format on demand. This is sufficient for our scrape model
(Grafana Cloud Agent pulls `/metrics` once per minute).

Format reference: https://github.com/prometheus/docs/blob/main/content/docs/instrumenting/exposition_formats.md
"""

from __future__ import annotations

from collections.abc import Mapping
from threading import Lock


def _format_labels(labels: Mapping[str, str] | None) -> str:
    if not labels:
        return ""
    pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + pairs + "}"


class MetricsRegistry:
    """Thread-safe registry of gauges, counters, and labelled samples."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._gauges: dict[str, float] = {}
        self._counters: dict[str, float] = {}
        self._labelled: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._helps: dict[str, str] = {}
        self._types: dict[str, str] = {}

    def set_gauge(
        self, name: str, value: float, *, help: str = "", labels: Mapping[str, str] | None = None
    ) -> None:
        """Set a gauge (last value wins). Optionally with labels."""
        with self._lock:
            self._helps.setdefault(name, help)
            self._types[name] = "gauge"
            if labels:
                key = (name, tuple(sorted(labels.items())))
                self._labelled[key] = float(value)
            else:
                self._gauges[name] = float(value)

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        *,
        help: str = "",
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Increment a counter (monotone-only). Optionally with labels."""
        with self._lock:
            self._helps.setdefault(name, help)
            self._types[name] = "counter"
            if labels:
                key = (name, tuple(sorted(labels.items())))
                self._labelled[key] = self._labelled.get(key, 0.0) + float(value)
            else:
                self._counters[name] = self._counters.get(name, 0.0) + float(value)

    def render(self) -> str:
        """Return the Prometheus text-format exposition."""
        with self._lock:
            lines: list[str] = []
            seen: set[str] = set()

            for name, value in sorted({**self._gauges, **self._counters}.items()):
                if name not in seen:
                    seen.add(name)
                    if self._helps.get(name):
                        lines.append(f"# HELP {name} {self._helps[name]}")
                    lines.append(f"# TYPE {name} {self._types.get(name, 'gauge')}")
                lines.append(f"{name} {value}")

            for (name, lbls), value in sorted(self._labelled.items()):
                if name not in seen:
                    seen.add(name)
                    if self._helps.get(name):
                        lines.append(f"# HELP {name} {self._helps[name]}")
                    lines.append(f"# TYPE {name} {self._types.get(name, 'gauge')}")
                lines.append(f"{name}{_format_labels(dict(lbls))} {value}")
            return "\n".join(lines) + "\n"


# Project-level metric names — single source of truth used by the rule
# definitions in `regime/monitoring/alerts.py` and the Grafana dashboard
# config (Week 11).
METRIC_REGIME_PROB = "regime_posterior_prob"
METRIC_CRISIS_PROB = "regime_crisis_prob_21d_calibrated"
METRIC_FILTER_LATENCY = "regime_filter_latency_seconds"
METRIC_FEATURE_DRIFT_PSI = "feature_drift_psi"
METRIC_DATA_REFRESH_TS = "data_refresh_last_success_timestamp"
METRIC_ORACLE_CPU = "oracle_cpu_utilization_avg_24h"
METRIC_TRANSITION_COUNT = "regime_transitions_total"
