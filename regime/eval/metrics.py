"""Evaluation metrics for regime models.

Headline metric: detection lag (trading days from `m5_date` to first
3-consecutive-days where P(crisis) > threshold). Other metrics: Brier score
for one-step-ahead probabilistic forecasts; transition-matrix Frobenius
distance for cross-fold stability; per-regime dwell-time distributions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from regime.eval.crises import CrisisEvent

SUSTAINED_FIRE_DAYS = 3
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class DetectionLagResult:
    crisis_name: str
    anchor_date: date
    first_fire_date: date | None
    lag_trading_days: int | None


def first_sustained_fire(
    dates: Sequence[date],
    crisis_prob: np.ndarray,
    after: date,
    threshold: float = DEFAULT_THRESHOLD,
    sustained_days: int = SUSTAINED_FIRE_DAYS,
) -> date | None:
    """First date >= `after` where crisis_prob > threshold for `sustained_days` consecutive days.

    Returns the date of the first day in the streak (the "fire" date), not the
    last. If no streak of length `sustained_days` exists, return None.
    """
    n = len(dates)
    if n != len(crisis_prob):
        raise ValueError(f"len(dates)={n} != len(crisis_prob)={len(crisis_prob)}")

    consecutive = 0
    streak_start: int | None = None
    for i in range(n):
        if dates[i] < after:
            consecutive = 0
            streak_start = None
            continue
        if crisis_prob[i] > threshold:
            if consecutive == 0:
                streak_start = i
            consecutive += 1
            if consecutive >= sustained_days and streak_start is not None:
                return dates[streak_start]
        else:
            consecutive = 0
            streak_start = None
    return None


def detection_lag(
    dates: Sequence[date],
    crisis_prob: np.ndarray,
    crisis_event: CrisisEvent,
    threshold: float = DEFAULT_THRESHOLD,
    sustained_days: int = SUSTAINED_FIRE_DAYS,
) -> DetectionLagResult:
    """Detection lag from a crisis's m5_date anchor to the first sustained fire."""
    anchor = crisis_event.m5_date
    fire = first_sustained_fire(
        dates, crisis_prob, after=anchor, threshold=threshold, sustained_days=sustained_days
    )
    if fire is None:
        return DetectionLagResult(
            crisis_name=crisis_event.name,
            anchor_date=anchor,
            first_fire_date=None,
            lag_trading_days=None,
        )

    anchor_idx = next((i for i, d in enumerate(dates) if d >= anchor), None)
    if anchor_idx is None:
        return DetectionLagResult(
            crisis_name=crisis_event.name,
            anchor_date=anchor,
            first_fire_date=fire,
            lag_trading_days=None,
        )
    fire_idx = list(dates).index(fire)
    return DetectionLagResult(
        crisis_name=crisis_event.name,
        anchor_date=anchor,
        first_fire_date=fire,
        lag_trading_days=fire_idx - anchor_idx,
    )


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Brier score = mean((p - y)²). Lower is better; 0 is perfect."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"p shape {p.shape} != y shape {y.shape}")
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def transition_matrix_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius distance between two transition matrices."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b, ord="fro"))


def regime_dwell_times(state_seq: np.ndarray) -> dict[int, list[int]]:
    """Per-regime list of dwell-time durations from a state sequence."""
    state_seq = np.asarray(state_seq)
    if state_seq.size == 0:
        return {}
    out: dict[int, list[int]] = {}
    current = int(state_seq[0])
    length = 1
    for s in state_seq[1:]:
        if int(s) == current:
            length += 1
        else:
            out.setdefault(current, []).append(length)
            current = int(s)
            length = 1
    out.setdefault(current, []).append(length)
    return out
