"""PSI drift-detector tests."""

from __future__ import annotations

import numpy as np
import pytest

from regime.monitoring.drift import (
    feature_drift_panel,
    population_stability_index,
)


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=10_000)
    psi = population_stability_index(sample, sample)
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_mean_shift():
    rng = np.random.default_rng(0)
    expected = rng.normal(0.0, 1.0, size=10_000)
    small_shift = rng.normal(0.2, 1.0, size=10_000)
    big_shift = rng.normal(2.0, 1.0, size=10_000)
    psi_small = population_stability_index(expected, small_shift)
    psi_big = population_stability_index(expected, big_shift)
    assert psi_big > psi_small > 0.0


def test_psi_above_alert_threshold_under_significant_drift():
    rng = np.random.default_rng(0)
    expected = rng.normal(0.0, 1.0, size=5_000)
    actual = rng.normal(2.0, 0.5, size=5_000)
    psi = population_stability_index(expected, actual)
    # Significant drift should easily exceed the 0.25 alert threshold.
    assert psi > 0.25


def test_psi_handles_empty_inputs():
    assert population_stability_index(np.array([]), np.array([1.0, 2.0])) == 0.0
    assert population_stability_index(np.array([1.0, 2.0]), np.array([])) == 0.0


def test_psi_handles_constant_expected():
    """Constant expected → no usable bin edges → returns 0."""
    expected = np.full(100, 5.0)
    actual = np.linspace(0.0, 10.0, 100)
    psi = population_stability_index(expected, actual)
    assert psi == 0.0


def test_psi_ignores_nan_and_inf():
    expected = np.array([1.0, 2.0, np.nan, np.inf, 3.0, 4.0, 5.0])
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Should compute a finite PSI on the finite portion.
    psi = population_stability_index(expected, actual)
    assert np.isfinite(psi)


def test_feature_drift_panel():
    rng = np.random.default_rng(0)
    expected = {
        "ret_1d": rng.normal(0.0, 1.0, size=2000),
        "rv_21d": rng.normal(0.0, 1.0, size=2000),
    }
    actual = {
        "ret_1d": rng.normal(0.0, 1.0, size=2000),
        "rv_21d": rng.normal(2.0, 1.0, size=2000),  # heavily drifted
    }
    panel = feature_drift_panel(expected, actual)
    assert "ret_1d" in panel and "rv_21d" in panel
    assert panel["rv_21d"] > panel["ret_1d"]


def test_feature_drift_panel_skips_missing_keys():
    expected = {"a": np.array([1.0, 2.0, 3.0])}
    actual = {"b": np.array([1.0, 2.0, 3.0])}
    panel = feature_drift_panel(expected, actual)
    assert panel == {}
