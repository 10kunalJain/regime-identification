"""Calibration evaluation: reliability diagram, Brier, ECE.

Per Q10 of the design grill: PR-AUC is the headline classification metric (AUC
is misleading at 5-10% positive prevalence); reliability diagram + Brier are
the headline calibration evidence. Implementations stay numpy-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityCurve:
    bin_lower: np.ndarray
    bin_upper: np.ndarray
    mean_predicted: np.ndarray
    mean_observed: np.ndarray
    bin_count: np.ndarray


def reliability_curve(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> ReliabilityCurve:
    """Bin predictions into `n_bins` equal-width bins; report calibration in each."""
    proba = np.asarray(proba, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if proba.shape != y.shape:
        raise ValueError(f"shape mismatch {proba.shape} vs {y.shape}")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(proba, bins[1:-1], right=False), 0, n_bins - 1)
    mean_predicted = np.full(n_bins, np.nan, dtype=np.float64)
    mean_observed = np.full(n_bins, np.nan, dtype=np.float64)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = bin_ids == b
        if m.any():
            mean_predicted[b] = float(proba[m].mean())
            mean_observed[b] = float(y[m].mean())
            bin_count[b] = int(m.sum())
    return ReliabilityCurve(
        bin_lower=bins[:-1],
        bin_upper=bins[1:],
        mean_predicted=mean_predicted,
        mean_observed=mean_observed,
        bin_count=bin_count,
    )


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """ECE: weighted average |mean_predicted - mean_observed| across bins."""
    rc = reliability_curve(proba, y, n_bins)
    total = int(rc.bin_count.sum())
    if total == 0:
        return 0.0
    weights = rc.bin_count.astype(np.float64) / total
    diffs = np.abs(rc.mean_predicted - rc.mean_observed)
    diffs[np.isnan(diffs)] = 0.0
    return float(np.sum(weights * diffs))


def brier_score(proba: np.ndarray, y: np.ndarray) -> float:
    """Brier score = mean((proba - y)^2). Lower is better; 0 is perfect."""
    proba = np.asarray(proba, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if proba.shape != y.shape:
        raise ValueError(f"shape mismatch {proba.shape} vs {y.shape}")
    if proba.size == 0:
        return 0.0
    return float(np.mean((proba - y) ** 2))


def pr_auc(proba: np.ndarray, y: np.ndarray) -> float:
    """Area under the precision-recall curve. Headline metric for crisis-head
    evaluation under class imbalance (Q10 lock).

    The random baseline equals the positive base rate, so a meaningful result
    is always presented as `(pr_auc, base_rate)`.
    """
    proba = np.asarray(proba, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if proba.shape != y.shape:
        raise ValueError(f"shape mismatch {proba.shape} vs {y.shape}")
    if proba.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return 0.0
    order = np.argsort(-proba, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    # Prepend (recall=0, precision=1) so trapezoid integration gives the
    # standard AP-style PR-AUC.
    rec = np.concatenate(([0.0], recall))
    prec = np.concatenate(([1.0], precision))
    return float(np.trapezoid(prec, rec))


def positive_base_rate(y: np.ndarray) -> float:
    y = np.asarray(y).reshape(-1)
    n = max(len(y), 1)
    return float(np.sum(y == 1) / n)
