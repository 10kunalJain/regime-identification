"""Sparse Jump Model tests on synthetic 3-state data."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regime.models.sparse_jump import SparseJumpModel


def _build_three_state_df(T: int = 600, seed: int = 7) -> pl.DataFrame:
    """Generate clearly-separated 3-state data."""
    rng = np.random.default_rng(seed)
    pi = np.array([0.5, 0.3, 0.2])
    A = np.array(
        [
            [0.96, 0.03, 0.01],
            [0.05, 0.92, 0.03],
            [0.03, 0.07, 0.90],
        ]
    )
    means = np.array([[2.0, 0.5], [0.0, 0.3], [-2.0, 0.8]])
    states = np.empty(T, dtype=np.int64)
    obs = np.empty((T, 2), dtype=np.float64)
    states[0] = rng.choice(3, p=pi)
    obs[0] = means[states[0]] + rng.normal(0.0, 0.2, size=2)
    for t in range(1, T):
        states[t] = rng.choice(3, p=A[states[t - 1]])
        obs[t] = means[states[t]] + rng.normal(0.0, 0.2, size=2)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(T)]
    return pl.DataFrame(
        {"data_time": dates, "ret_1d": obs[:, 0], "rv_21d": obs[:, 1]},
        schema={"data_time": pl.Date, "ret_1d": pl.Float64, "rv_21d": pl.Float64},
    )


def test_sjm_fit_and_assign():
    df = _build_three_state_df(T=600, seed=1)
    model = SparseJumpModel(
        K=3, feature_columns=("ret_1d", "rv_21d"), jump_penalty=0.5, n_restarts=3
    )
    model.fit(df, np.arange(df.height, dtype=np.int64))
    state = model.state_dict()
    assert state["fitted"]
    centers = np.array(state["centers"])
    # Centers sorted by first-feature mean (descending). State 0 should have the
    # highest mean, state 2 the lowest.
    assert centers[0, 0] > centers[1, 0] > centers[2, 0]


def test_sjm_filter_returns_one_hot():
    df = _build_three_state_df(T=400, seed=2)
    model = SparseJumpModel(K=3, jump_penalty=0.5, n_restarts=2)
    model.fit(df, np.arange(df.height, dtype=np.int64))
    posterior = model.filter(df, np.arange(df.height, dtype=np.int64))
    # Each row has exactly one 1.0
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0)
    assert ((posterior == 0.0) | (posterior == 1.0)).all()


def test_sjm_smooth_loss_at_most_filter_loss():
    """DP smoothing must achieve at most the loss of online greedy on the same data."""
    df = _build_three_state_df(T=400, seed=3)
    model = SparseJumpModel(K=3, jump_penalty=0.5, n_restarts=2)
    idx = np.arange(df.height, dtype=np.int64)
    model.fit(df, idx)

    # Compute loss for each path under the SJM objective.
    centers = np.array(model.state_dict()["centers"])
    feature_arr = (
        df.select("ret_1d", "rv_21d")
        .filter(pl.Series(np.ones(df.height, dtype=bool)))
        .drop_nulls()
        .to_numpy()
        .astype(np.float64)
    )
    states_f = np.argmax(model.filter(df, idx), axis=1)
    states_s = np.argmax(model.smooth(df, idx), axis=1)

    def _loss(states: np.ndarray) -> float:
        sq = float((np.linalg.norm(feature_arr - centers[states], axis=1) ** 2).sum())
        jumps = int((states[1:] != states[:-1]).sum())
        return sq + 0.5 * jumps

    # DP-optimal path must be at least as good as the greedy path.
    assert _loss(states_s) <= _loss(states_f) + 1e-9


def test_sjm_jump_penalty_reduces_switches():
    """Higher jump penalty should produce fewer state switches."""
    df = _build_three_state_df(T=400, seed=4)
    idx = np.arange(df.height, dtype=np.int64)

    model_low = SparseJumpModel(K=3, jump_penalty=0.0, n_restarts=2)
    model_low.fit(df, idx)
    s_low = np.argmax(model_low.smooth(df, idx), axis=1)

    model_high = SparseJumpModel(K=3, jump_penalty=10.0, n_restarts=2)
    model_high.fit(df, idx)
    s_high = np.argmax(model_high.smooth(df, idx), axis=1)

    switches_low = int((s_low[1:] != s_low[:-1]).sum())
    switches_high = int((s_high[1:] != s_high[:-1]).sum())
    assert switches_high <= switches_low


def test_sjm_state_dict_roundtrip():
    df = _build_three_state_df(T=300, seed=5)
    model = SparseJumpModel(K=3, jump_penalty=0.5, n_restarts=2)
    idx = np.arange(df.height, dtype=np.int64)
    model.fit(df, idx)
    state = model.state_dict()

    restored = SparseJumpModel(K=3)
    restored.load_state_dict(state)
    f_a = model.filter(df, idx)
    f_b = restored.filter(df, idx)
    np.testing.assert_array_equal(f_a, f_b)


def test_sjm_rejects_too_few_observations():
    df = pl.DataFrame(
        {
            "data_time": [date(2020, 1, 1), date(2020, 1, 2)],
            "ret_1d": [0.0, 0.1],
            "rv_21d": [0.1, 0.1],
        }
    )
    model = SparseJumpModel(K=3)
    with pytest.raises(ValueError, match="need at least"):
        model.fit(df, np.arange(2, dtype=np.int64))
