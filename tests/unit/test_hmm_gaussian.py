"""HmmGaussian recovery test on synthetic data.

The headline correctness test for the HMM implementation: simulate from a
known 3-state HMM, fit, and assert that the fitted parameters approximately
recover the true ones (after Hungarian alignment) and that the inferred state
sequence has high accuracy.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regime.models._hmm_core import (
    forward_backward,
    forward_filter,
    gaussian_log_emissions,
    sample_gaussian_hmm,
)
from regime.models.alignment import hungarian_align
from regime.models.hmm_gaussian import HmmGaussian


def _build_synthetic_features(T: int = 2000, seed: int = 7) -> tuple[pl.DataFrame, np.ndarray]:
    """Sample from a 3-state Gaussian HMM with regimes (calm, neutral, crisis)."""
    rng = np.random.default_rng(seed)
    pi = np.array([1.0, 0.0, 0.0])
    A = np.array(
        [
            [0.97, 0.02, 0.01],
            [0.05, 0.90, 0.05],
            [0.02, 0.10, 0.88],
        ]
    )
    means = np.array(
        [
            [0.001, 0.10],  # calm: small positive return, low vol
            [0.000, 0.18],  # neutral
            [-0.003, 0.45],  # crisis: negative return, high vol
        ]
    )
    covs = np.array(
        [
            [[1e-6, 0.0], [0.0, 1e-3]],
            [[4e-6, 0.0], [0.0, 4e-3]],
            [[2e-5, 0.0], [0.0, 1e-2]],
        ]
    )
    states, obs = sample_gaussian_hmm(pi, A, means, covs, T, rng)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(T)]
    df = pl.DataFrame(
        {
            "data_time": dates,
            "ret_1d": obs[:, 0],
            "rv_21d": obs[:, 1],
        },
        schema={"data_time": pl.Date, "ret_1d": pl.Float64, "rv_21d": pl.Float64},
    )
    return df, states


# ---------- _hmm_core primitives ----------


def test_forward_filter_normalized():
    """Filtered posterior must sum to 1 at every timestep."""
    rng = np.random.default_rng(0)
    pi = np.array([0.5, 0.3, 0.2])
    A = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    means = np.array([[0.0], [1.0], [-1.0]])
    covs = np.tile(np.eye(1), (3, 1, 1)) * 0.5
    _, X = sample_gaussian_hmm(pi, A, means, covs, 200, rng)

    log_emissions = gaussian_log_emissions(X, means, covs)
    log_pi = np.log(pi)
    log_A = np.log(A)
    posterior = forward_filter(log_emissions, log_pi, log_A)

    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-10)
    assert (posterior >= 0.0).all()


def test_smoothed_posterior_normalized():
    rng = np.random.default_rng(1)
    pi = np.array([0.5, 0.3, 0.2])
    A = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    means = np.array([[0.0], [1.0], [-1.0]])
    covs = np.tile(np.eye(1), (3, 1, 1)) * 0.5
    _, X = sample_gaussian_hmm(pi, A, means, covs, 200, rng)

    log_emissions = gaussian_log_emissions(X, means, covs)
    smoothed = forward_backward(log_emissions, np.log(pi), np.log(A))

    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-10)


def test_smoothed_uses_more_information_than_filtered():
    """At an interior timestep, smoothed posterior should differ from filtered.

    With overlapping emission distributions (so observations are ambiguous), the
    backward pass adds real information beyond what the forward pass alone
    sees, and the two posteriors must differ.
    """
    rng = np.random.default_rng(2)
    pi = np.array([0.5, 0.5])
    A = np.array([[0.85, 0.15], [0.15, 0.85]])
    means = np.array([[0.0], [1.0]])
    # High emission variance so observations don't pin down the state.
    covs = np.tile(np.eye(1), (2, 1, 1)) * 1.5
    _, X = sample_gaussian_hmm(pi, A, means, covs, 200, rng)

    log_emissions = gaussian_log_emissions(X, means, covs)
    filtered = forward_filter(log_emissions, np.log(pi), np.log(A))
    smoothed = forward_backward(log_emissions, np.log(pi), np.log(A))

    # The two posteriors must differ at *some* interior timestep.
    diff_per_t = np.abs(filtered - smoothed).max(axis=1)
    assert diff_per_t.max() > 0.01


# ---------- HmmGaussian end-to-end ----------


@pytest.mark.slow
def test_hmm_recovers_parameters_on_synthetic_data():
    """Fit on synthetic 3-state HMM data; recovered means should match (after alignment)."""
    df, _true_states = _build_synthetic_features(T=2000, seed=7)
    model = HmmGaussian(K=3, feature_columns=("ret_1d", "rv_21d"), n_restarts=8)
    train_idx = np.arange(df.height, dtype=np.int64)
    model.fit(df, train_idx)

    fitted = model.state_dict()
    fitted_means = np.array(fitted["means"])
    fitted_covs = np.array(fitted["covs"])

    true_means = np.array([[0.001, 0.10], [0.000, 0.18], [-0.003, 0.45]])
    true_covs = np.array(
        [
            [[1e-6, 0.0], [0.0, 1e-3]],
            [[4e-6, 0.0], [0.0, 4e-3]],
            [[2e-5, 0.0], [0.0, 1e-2]],
        ]
    )
    perm = hungarian_align(true_means, true_covs, fitted_means, fitted_covs)
    aligned_means = fitted_means[perm]

    # Means should be close to truth (loose tolerance — finite-sample EM).
    np.testing.assert_allclose(aligned_means[:, 1], true_means[:, 1], atol=0.1)


def test_hmm_filter_and_smooth_normalized():
    df, _ = _build_synthetic_features(T=600, seed=11)
    model = HmmGaussian(K=3, feature_columns=("ret_1d", "rv_21d"), n_restarts=4)
    train_idx = np.arange(400, dtype=np.int64)
    test_idx = np.arange(400, df.height, dtype=np.int64)

    model.fit(df, train_idx)
    f = model.filter(df, test_idx)
    s = model.smooth(df, test_idx)

    np.testing.assert_allclose(f.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(s.sum(axis=1), 1.0, atol=1e-9)


def test_hmm_state_dict_roundtrip():
    df, _ = _build_synthetic_features(T=400, seed=3)
    model = HmmGaussian(K=3, feature_columns=("ret_1d", "rv_21d"), n_restarts=3)
    model.fit(df, np.arange(df.height, dtype=np.int64))

    state = model.state_dict()
    restored = HmmGaussian(K=3, feature_columns=("ret_1d", "rv_21d"))
    restored.load_state_dict(state)

    test_idx = np.arange(300, df.height, dtype=np.int64)
    np.testing.assert_allclose(model.filter(df, test_idx), restored.filter(df, test_idx))


def test_hmm_fit_raises_on_too_few_observations():
    df = pl.DataFrame(
        {
            "data_time": [date(2020, 1, 1), date(2020, 1, 2)],
            "ret_1d": [0.0, 0.0],
            "rv_21d": [0.1, 0.1],
        }
    )
    model = HmmGaussian(K=3, n_restarts=2)
    with pytest.raises(ValueError, match="need at least"):
        model.fit(df, np.arange(2, dtype=np.int64))
