"""Runtime state store interface + in-memory implementation.

The interface is intentionally minimal so a Postgres adapter (the production
deployment per ARCHITECTURE.md §10) can drop in without touching the API or
runtime modules.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol

from regime.api.schemas import RegimePosterior


class StateStore(Protocol):
    """Persistence layer for streaming inference state."""

    def put_posterior(self, posterior: RegimePosterior) -> None: ...

    def get_latest_posterior(self) -> RegimePosterior | None: ...

    def get_posterior_path(
        self, from_date: date | None, to_date: date | None
    ) -> list[RegimePosterior]: ...

    def put_filter_state(self, state: dict) -> None: ...

    def get_filter_state(self) -> dict | None: ...


class InMemoryStateStore:
    """In-memory implementation. Used in tests and for local development."""

    def __init__(self) -> None:
        self._posteriors: list[RegimePosterior] = []
        self._filter_state: dict | None = None

    def put_posterior(self, posterior: RegimePosterior) -> None:
        self._posteriors.append(posterior)
        self._posteriors.sort(key=lambda p: p.as_of)

    def get_latest_posterior(self) -> RegimePosterior | None:
        return self._posteriors[-1] if self._posteriors else None

    def get_posterior_path(
        self, from_date: date | None, to_date: date | None
    ) -> list[RegimePosterior]:
        out: list[RegimePosterior] = []
        for p in self._posteriors:
            d = p.as_of if isinstance(p.as_of, date) else _parse_iso_date(str(p.as_of))
            if from_date is not None and d < from_date:
                continue
            if to_date is not None and d > to_date:
                continue
            out.append(p)
        return out

    def put_filter_state(self, state: dict) -> None:
        self._filter_state = dict(state)

    def get_filter_state(self) -> dict | None:
        return None if self._filter_state is None else dict(self._filter_state)


def _parse_iso_date(s: str) -> date:
    if "T" in s:
        return datetime.fromisoformat(s).date()
    return date.fromisoformat(s)
