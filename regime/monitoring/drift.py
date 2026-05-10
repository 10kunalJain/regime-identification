"""Population-Stability-Index drift detector.

Standard PSI formula:

    PSI = Σ_i (actual_i - expected_i) × ln(actual_i / expected_i)

where `expected_i` and `actual_i` are the per-bin proportions of the training
and current distributions. Bin edges are derived from quantiles of the
expected (training) distribution so the metric is scale-invariant and respects
heavy-tailed features (returns, vol).

Standard interpretation thresholds (Banking / credit-scoring convention):
  - PSI < 0.10 — no meaningful change.
  - 0.10 ≤ PSI < 0.25 — moderate drift; investigate.
  - PSI ≥ 0.25 — significant drift; alert (the threshold used in the project's
    drift-alert rule, ARCHITECTURE.md §9).

Per-feature drift is reported as a `dict[str, float]`; the alert rule fires
on `max(values)`.
"""

from __future__ import annotations

import numpy as np

DEFAULT_N_BINS = 10
_EPS = 1e-6


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """PSI between an `expected` (training) and an `actual` (current) sample.

    Returns 0.0 when either sample is empty or when the expected distribution
    is degenerate (e.g., constant value with no usable bin edges).
    """
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    edges = np.quantile(expected, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.maximum(expected_pct, _EPS)
    actual_pct = np.maximum(actual_pct, _EPS)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def feature_drift_panel(
    expected_features: dict[str, np.ndarray],
    actual_features: dict[str, np.ndarray],
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, float]:
    """PSI per feature — input is two same-keyed dicts of arrays."""
    out: dict[str, float] = {}
    for name in expected_features:
        if name not in actual_features:
            continue
        out[name] = population_stability_index(
            expected_features[name], actual_features[name], n_bins=n_bins
        )
    return out
