"""FastAPI app factory.

Per ARCHITECTURE.md §5: public read-only API. The state store is injected
so production can use Postgres while tests use the in-memory implementation.

Endpoints:
  - `GET /healthz`           — liveness probe.
  - `GET /regime/now`        — latest calibrated posterior.
  - `GET /regime/path`       — historical posterior path; optional from/to.
  - `GET /forecast`          — placeholder; returns NotImplemented for v1.
  - `GET /metrics`           — Prometheus exposition (added in Week 10).
"""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse

from regime.api.schemas import RegimePosterior
from regime.monitoring.metrics import MetricsRegistry
from regime.runtime.state import StateStore


def create_app(
    state_store: StateStore,
    metrics: MetricsRegistry | None = None,
) -> FastAPI:
    """Build a FastAPI app bound to the given `state_store` and metrics registry."""
    app = FastAPI(
        title="Regime Identification Service",
        description="Real-time market regime identification for US equities.",
        version="0.1.0",
    )
    metrics = metrics or MetricsRegistry()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics() -> str:
        return metrics.render()

    @app.get("/regime/now", response_model=RegimePosterior)
    def regime_now() -> RegimePosterior:
        latest = state_store.get_latest_posterior()
        if latest is None:
            raise HTTPException(
                status_code=404,
                detail="no posterior available yet — run `regime data refresh` to seed state",
            )
        return latest

    @app.get("/regime/path", response_model=list[RegimePosterior])
    def regime_path(
        from_date: date | None = Query(default=None, alias="from"),
        to_date: date | None = Query(default=None, alias="to"),
    ) -> list[RegimePosterior]:
        if from_date is not None and to_date is not None and from_date > to_date:
            raise HTTPException(
                status_code=400, detail=f"from={from_date} cannot be after to={to_date}"
            )
        return state_store.get_posterior_path(from_date, to_date)

    @app.get("/forecast")
    def forecast(horizon: int = Query(default=21, ge=1, le=63)) -> dict[str, str]:
        # Forecast endpoint is a placeholder for v2 — Week 9 ships posterior endpoints
        # only. The Week-11 dashboard does not depend on this endpoint.
        raise HTTPException(
            status_code=501,
            detail=f"forecast(horizon={horizon}) not yet implemented",
        )

    return app
