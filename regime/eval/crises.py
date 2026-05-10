"""Canonical crisis-event registry.

Eight historical events used in the fair-evaluation benchmark. Each entry
commits the (peak, m5, m10, bottom) anchor dates with a one-line citation.
The registry is *static data* — never derived from the underlying price data —
so detection-lag snapshot tests stay stable across data revisions.

The headline detection-lag policy uses `m5_date` as the anchor and a
3-consecutive-day sustained-fire rule at threshold 0.5 (per Q6 of the design
grill). Multi-anchor sensitivity (peak / m5 / m10) appears in the writeup
appendix.

Anchor dates are best-effort estimates pinned from public news sources; they
will be cross-checked against actual SPY data during Week 5/6 evaluation and
revised in a single, documented commit if the price-data check disagrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CrisisEvent:
    name: str
    peak_date: date
    m5_date: date
    m10_date: date
    bottom_date: date
    note: str


CRISES: tuple[CrisisEvent, ...] = (
    CrisisEvent(
        name="Aug 2007 Quant Quake",
        peak_date=date(2007, 7, 19),
        m5_date=date(2007, 8, 9),
        m10_date=date(2007, 8, 16),
        bottom_date=date(2007, 8, 16),
        note="Aug 6-9 2007 quant-fund liquidation cascade; SPY tested -10% briefly.",
    ),
    CrisisEvent(
        name="Sep 2008 Lehman / GFC",
        peak_date=date(2007, 10, 9),
        m5_date=date(2007, 11, 7),
        m10_date=date(2008, 1, 15),
        bottom_date=date(2009, 3, 9),
        note="Lehman bankruptcy filed Sep 15 2008; bear market started Oct 2007.",
    ),
    CrisisEvent(
        name="Aug 2011 US Debt Ceiling",
        peak_date=date(2011, 4, 29),
        m5_date=date(2011, 8, 1),
        m10_date=date(2011, 8, 4),
        bottom_date=date(2011, 10, 3),
        note="S&P US sovereign downgrade Aug 5 2011; -19% SPY drawdown.",
    ),
    CrisisEvent(
        name="Aug 2015 China Devaluation",
        peak_date=date(2015, 5, 21),
        m5_date=date(2015, 8, 21),
        m10_date=date(2015, 8, 24),
        bottom_date=date(2016, 2, 11),
        note="Yuan devaluation Aug 11 2015; Black Monday Aug 24.",
    ),
    CrisisEvent(
        name="Feb 2018 Volmageddon",
        peak_date=date(2018, 1, 26),
        m5_date=date(2018, 2, 2),
        m10_date=date(2018, 2, 8),
        bottom_date=date(2018, 2, 8),
        note="XIV blow-up Feb 5 2018; short-vol unwind cascade.",
    ),
    CrisisEvent(
        name="Mar 2020 COVID",
        peak_date=date(2020, 2, 19),
        m5_date=date(2020, 2, 25),
        m10_date=date(2020, 3, 9),
        bottom_date=date(2020, 3, 23),
        note="WHO pandemic declaration Mar 11 2020; circuit breakers Mar 9/12/16.",
    ),
    CrisisEvent(
        name="Sep 2022 LDI / UK Gilt",
        peak_date=date(2022, 8, 16),
        m5_date=date(2022, 9, 1),
        m10_date=date(2022, 9, 23),
        bottom_date=date(2022, 10, 12),
        note="UK mini-budget Sep 23 2022; LDI margin spiral; BoE intervention.",
    ),
    CrisisEvent(
        name="Aug 2024 Yen Carry Unwind",
        peak_date=date(2024, 7, 16),
        m5_date=date(2024, 8, 2),
        m10_date=date(2024, 8, 5),
        bottom_date=date(2024, 8, 5),
        note="BoJ rate hike Jul 31 2024; JPY-funded carry unwind; Aug 5 risk-off.",
    ),
)


def by_name(name: str) -> CrisisEvent:
    for c in CRISES:
        if c.name == name:
            return c
    raise KeyError(f"crisis {name!r} not in registry")


def all_names() -> tuple[str, ...]:
    return tuple(c.name for c in CRISES)
