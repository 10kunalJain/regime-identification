"""Economic-loss threshold selection for the crisis head.

Per Q10 of the design grill: the headline detection-lag metric uses the
un-cherrypicked threshold τ = 0.5, but the *strategy* uses the
economic-loss-optimal threshold under the asymmetric cost:

    loss(τ) = c_FA * #FalseAlarms(τ) + c_MC * #MissedCrises(τ)

with c_FA = 10 bp (round-trip to defensive allocation) and c_MC = 120 bp
(empirical excess drawdown beyond initial 10% × ~10 unprotected days). The
optimal threshold under symmetric Bayes-decision is c_FA / (c_FA + c_MC) =
10 / 130 ≈ 0.077; we report the empirical optimum on the actual loss curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_FALSE_ALARM_COST_BP = 10.0
DEFAULT_MISSED_CRISIS_COST_BP = 120.0


@dataclass(frozen=True)
class LossCurve:
    thresholds: np.ndarray
    losses: np.ndarray
    false_alarms: np.ndarray
    missed_crises: np.ndarray


def economic_loss_curve(
    proba: np.ndarray,
    y: np.ndarray,
    false_alarm_cost_bp: float = DEFAULT_FALSE_ALARM_COST_BP,
    missed_crisis_cost_bp: float = DEFAULT_MISSED_CRISIS_COST_BP,
    thresholds: np.ndarray | None = None,
) -> LossCurve:
    """Total loss across thresholds for a binary classifier."""
    proba = np.asarray(proba, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    losses = np.zeros_like(thresholds)
    fa = np.zeros_like(thresholds, dtype=np.int64)
    mc = np.zeros_like(thresholds, dtype=np.int64)
    for i, t in enumerate(thresholds):
        pred = (proba > t).astype(np.int64)
        fa[i] = int(((pred == 1) & (y == 0)).sum())
        mc[i] = int(((pred == 0) & (y == 1)).sum())
        losses[i] = false_alarm_cost_bp * fa[i] + missed_crisis_cost_bp * mc[i]
    return LossCurve(thresholds=thresholds, losses=losses, false_alarms=fa, missed_crises=mc)


def optimal_threshold(
    proba: np.ndarray,
    y: np.ndarray,
    false_alarm_cost_bp: float = DEFAULT_FALSE_ALARM_COST_BP,
    missed_crisis_cost_bp: float = DEFAULT_MISSED_CRISIS_COST_BP,
    thresholds: np.ndarray | None = None,
) -> tuple[float, LossCurve]:
    """Threshold minimizing the asymmetric economic loss; returns (τ*, full curve)."""
    curve = economic_loss_curve(proba, y, false_alarm_cost_bp, missed_crisis_cost_bp, thresholds)
    best_idx = int(np.argmin(curve.losses))
    return float(curve.thresholds[best_idx]), curve
