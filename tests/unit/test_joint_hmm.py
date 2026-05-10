"""JointHmm tests on synthetic FF-factor regime-switching data.

The headline correctness check: simulate from a known 3-state joint HMM with
FF-factor regime-switching means and rank-3 factor covariance, fit, and assert
that the recovered means and structure are recognizable. Plus filter / smooth
normalization, state-dict roundtrip, and the regime-collapse safety net.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regime.models.joint_hmm import (
    JointHmm,
    RegimeCollapseError,
    _log_emissions,
    _weighted_factor_analyzer_em,
)


def _build_synthetic_df(
    T: int = 1500, K: int = 3, d: int = 4, m: int = 6, seed: int = 17
) -> tuple[pl.DataFrame, dict]:
    """Sample from a 3-state joint HMM with FF-factor mean + factor cov.

    Returns (DataFrame with observation + factor columns, ground-truth dict).
    """
    rng = np.random.default_rng(seed)
    pi = np.array([0.5, 0.3, 0.2])
    A = np.array(
        [
            [0.95, 0.04, 0.01],
            [0.05, 0.92, 0.03],
            [0.03, 0.07, 0.90],
        ]
    )
    # Regime 0 is "bull" with positive intercepts; regime 2 "crisis" with
    # negative; regime 1 neutral. Decreases in magnitude across observation
    # columns to reflect the cross-section having a dominant first asset (SPY).
    alpha_template = np.array(
        [
            [1.0, 0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05][:d],
            [0.0] * d,
            [-1.0, -0.6, -0.3, -0.2, -0.05, -0.05, -0.05, -0.05][:d],
        ]
    )
    alpha = alpha_template
    B = rng.normal(0.0, 0.3, size=(K, d, m))
    rank = 2
    L = rng.normal(0.0, 0.2, size=(K, d, rank))
    D = np.full((K, d), 0.05)

    F = rng.normal(0.0, 0.5, size=(T, m))
    states = np.empty(T, dtype=np.int64)
    obs = np.empty((T, d), dtype=np.float64)
    states[0] = rng.choice(K, p=pi)
    for t in range(T):
        if t > 0:
            states[t] = rng.choice(K, p=A[states[t - 1]])
        s = states[t]
        cov = L[s] @ L[s].T + np.diag(D[s])
        mean = alpha[s] + B[s] @ F[t]
        obs[t] = rng.multivariate_normal(mean, cov)

    dates = [date(2010, 1, 1) + timedelta(days=i) for i in range(T)]
    obs_cols = {f"y{i}": obs[:, i] for i in range(d)}
    factor_cols = {f"f{j}": F[:, j] for j in range(m)}
    df = pl.DataFrame(
        {"data_time": dates, **obs_cols, **factor_cols},
        schema={
            "data_time": pl.Date,
            **{f"y{i}": pl.Float64 for i in range(d)},
            **{f"f{j}": pl.Float64 for j in range(m)},
        },
    )
    truth = {
        "alpha": alpha,
        "B": B,
        "L": L,
        "D": D,
        "states": states,
        "K": K,
        "d": d,
        "m": m,
    }
    return df, truth


# ---------- Helpers ----------


def test_log_emissions_normalized_via_finite_difference():
    """log_emissions of N(0, I) at a grid integrates to ~1."""
    grid = np.linspace(-10.0, 10.0, 10001).reshape(-1, 1)  # (T, d=1)
    F = np.zeros((grid.shape[0], 1))
    alpha = np.zeros((1, 1))
    B = np.zeros((1, 1, 1))
    L = np.zeros((1, 1, 0))  # rank=0 → cov = D
    D = np.ones((1, 1))
    log_emit = _log_emissions(grid, F, alpha, B, L, D)
    integral = float(np.trapezoid(np.exp(log_emit[:, 0]), grid[:, 0]))
    assert integral == pytest.approx(1.0, rel=1e-3)


def test_weighted_factor_analyzer_em_recovers_low_rank_cov():
    """Recover a known low-rank-plus-diagonal covariance from sampled residuals."""
    rng = np.random.default_rng(0)
    d = 5
    rank = 2
    L_true = rng.normal(0.0, 0.5, size=(d, rank))
    D_true = np.full(d, 0.1)
    Sigma = L_true @ L_true.T + np.diag(D_true)
    n = 5000
    samples = rng.multivariate_normal(np.zeros(d), Sigma, size=n)
    weights = np.ones(n)
    L_fit, D_fit = _weighted_factor_analyzer_em(samples, weights, rank, max_iter=30)
    Sigma_fit = L_fit @ L_fit.T + np.diag(D_fit)
    np.testing.assert_allclose(Sigma_fit, Sigma, atol=0.1, rtol=0.1)


# ---------- JointHmm end-to-end ----------


@pytest.mark.slow
def test_joint_hmm_finds_three_regimes():
    df, truth = _build_synthetic_df(T=1500, K=3, d=4, m=6, seed=17)
    obs_cols = tuple(f"y{i}" for i in range(truth["d"]))
    factor_cols = tuple(f"f{j}" for j in range(truth["m"]))
    model = JointHmm(
        K=3,
        observation_columns=obs_cols,
        factor_columns=factor_cols,
        latent_factor_rank=2,
        n_restarts=4,
        max_iter=80,
    )
    model.fit(df, np.arange(df.height, dtype=np.int64))
    state = model.state_dict()
    assert state["fitted"]
    fitted_alpha = np.array(state["alpha"])
    # After SPY-mean ordering (descending on first observation column), the
    # first regime should have the highest alpha[0] and the last the lowest.
    assert fitted_alpha[0, 0] > fitted_alpha[1, 0]
    assert fitted_alpha[1, 0] > fitted_alpha[2, 0]
    # Top regime alpha[0] should be on the positive side, bottom on negative.
    assert fitted_alpha[0, 0] > 0.2
    assert fitted_alpha[2, 0] < -0.2


def test_joint_hmm_filter_smooth_normalized():
    df, truth = _build_synthetic_df(T=600, K=3, d=3, m=4, seed=4)
    obs_cols = tuple(f"y{i}" for i in range(truth["d"]))
    factor_cols = tuple(f"f{j}" for j in range(truth["m"]))
    model = JointHmm(
        K=3,
        observation_columns=obs_cols,
        factor_columns=factor_cols,
        latent_factor_rank=2,
        n_restarts=2,
        max_iter=30,
    )
    train_idx = np.arange(400, dtype=np.int64)
    test_idx = np.arange(400, df.height, dtype=np.int64)
    model.fit(df, train_idx)
    f = model.filter(df, test_idx)
    s = model.smooth(df, test_idx)
    np.testing.assert_allclose(f.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(s.sum(axis=1), 1.0, atol=1e-9)


def test_joint_hmm_state_dict_roundtrip():
    df, truth = _build_synthetic_df(T=400, K=3, d=3, m=4, seed=5)
    obs_cols = tuple(f"y{i}" for i in range(truth["d"]))
    factor_cols = tuple(f"f{j}" for j in range(truth["m"]))
    model = JointHmm(
        K=3,
        observation_columns=obs_cols,
        factor_columns=factor_cols,
        latent_factor_rank=2,
        n_restarts=2,
        max_iter=20,
    )
    model.fit(df, np.arange(df.height, dtype=np.int64))
    state = model.state_dict()

    restored = JointHmm(
        K=3,
        observation_columns=obs_cols,
        factor_columns=factor_cols,
    )
    restored.load_state_dict(state)
    test_idx = np.arange(300, df.height, dtype=np.int64)
    np.testing.assert_allclose(model.filter(df, test_idx), restored.filter(df, test_idx))


def test_joint_hmm_regime_collapse_path_works():
    """An impossibly-high collapse threshold must trigger RegimeCollapseError
    on every restart and propagate up as a RuntimeError ('all restarts failed')."""
    df, truth = _build_synthetic_df(T=400, K=3, d=3, m=4, seed=8)
    obs_cols = tuple(f"y{i}" for i in range(truth["d"]))
    factor_cols = tuple(f"f{j}" for j in range(truth["m"]))
    # Threshold = 0.99 means every regime would need ≥99% mass simultaneously,
    # which is impossible — so every restart collapses and the outer fit raises.
    model = JointHmm(
        K=3,
        observation_columns=obs_cols,
        factor_columns=factor_cols,
        latent_factor_rank=2,
        n_restarts=2,
        max_iter=10,
        regime_collapse_threshold=0.99,
    )
    with pytest.raises(RuntimeError, match="all JointHmm restarts failed"):
        model.fit(df, np.arange(df.height, dtype=np.int64))


def test_joint_hmm_rejects_too_few_observations():
    df = pl.DataFrame(
        {
            "data_time": [date(2020, 1, 1)] * 5,
            "y0": [0.0] * 5,
            "f0": [0.0] * 5,
        }
    )
    model = JointHmm(K=3, observation_columns=("y0",), factor_columns=("f0",))
    with pytest.raises(ValueError, match="need at least"):
        model.fit(df, np.arange(5, dtype=np.int64))


def test_regime_collapse_error_is_runtime_error_subclass():
    assert issubclass(RegimeCollapseError, RuntimeError)
