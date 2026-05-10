"""Bocpd correctness on a synthetic step-change scenario.

Generate observations from N(0, 1) for the first half, then N(2, 0.5) for the
second half. Bocpd's `change_prob` should spike around the step boundary.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regime.models.bocpd import Bocpd


def _step_change_df(T: int = 400, change_at: int = 200, seed: int = 13) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    pre = rng.normal(0.0, 1.0, size=change_at)
    post = rng.normal(2.0, 0.5, size=T - change_at)
    y = np.concatenate([pre, post])
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(T)]
    return pl.DataFrame(
        {"data_time": dates, "ret_1d": y},
        schema={"data_time": pl.Date, "ret_1d": pl.Float64},
    )


def test_bocpd_detects_step_change():
    """The headline BOCPD signal is the *expected run length* collapsing after
    a change, not change_prob spiking. Under Adams-MacKay's formulation,
    change_prob is bounded by the hazard rate (~1/lambda); the model conveys
    "change detected" by the run-length distribution shifting toward zero.
    """
    df = _step_change_df(T=400, change_at=200, seed=13)
    model = Bocpd(feature_columns=("ret_1d",), hazard_lambda=100.0)
    model.fit(df, np.arange(50, dtype=np.int64))
    feats = model.native_features(df, np.arange(df.height, dtype=np.int64))
    expected_rl = feats[:, 1]

    # Before the change, the expected run length grows roughly linearly with t
    # (no change → run keeps extending). After the change, it should collapse.
    pre_change_max = float(expected_rl[150:200].max())
    post_change_min = float(expected_rl[205:240].min())
    assert post_change_min < pre_change_max / 5, (
        f"expected_rl did not collapse after change: pre={pre_change_max:.1f},"
        f" post_min={post_change_min:.1f}"
    )


def test_bocpd_outputs_three_features():
    df = _step_change_df(T=200, seed=1)
    model = Bocpd(feature_columns=("ret_1d",), hazard_lambda=100.0)
    model.fit(df, np.arange(50, dtype=np.int64))
    feats = model.native_features(df, np.arange(df.height, dtype=np.int64))
    assert feats.shape == (200, 3)
    # change_prob in [0, 1]
    assert (feats[:, 0] >= 0.0).all()
    assert (feats[:, 0] <= 1.0 + 1e-9).all()
    # Expected run length is non-negative
    assert (feats[:, 1] >= 0.0).all()
    # Entropy is non-negative
    assert (feats[:, 2] >= -1e-9).all()


def test_bocpd_change_prob_bounded_by_hazard():
    """Under steady-state, change_prob ≈ hazard rate (= 1/lambda)."""
    df = _step_change_df(T=300, seed=7)
    model = Bocpd(feature_columns=("ret_1d",), hazard_lambda=100.0)
    model.fit(df, np.arange(50, dtype=np.int64))
    feats = model.native_features(df, np.arange(df.height, dtype=np.int64))
    change_prob = feats[:, 0]
    # First step has change_prob equal to the hazard rate by construction.
    assert change_prob[0] == pytest.approx(1.0 / 100.0, abs=1e-9)


def test_bocpd_state_dict_roundtrip():
    df = _step_change_df(T=300, seed=2)
    model = Bocpd(feature_columns=("ret_1d",), hazard_lambda=200.0)
    model.fit(df, np.arange(100, dtype=np.int64))
    feats_a = model.native_features(df, np.arange(df.height, dtype=np.int64))

    state = model.state_dict()
    restored = Bocpd(feature_columns=("ret_1d",))
    restored.load_state_dict(state)
    feats_b = restored.native_features(df, np.arange(df.height, dtype=np.int64))

    np.testing.assert_allclose(feats_a, feats_b)


def test_bocpd_rejects_multivariate():
    with pytest.raises(ValueError, match="univariate"):
        Bocpd(feature_columns=("ret_1d", "rv_21d"))


def test_bocpd_max_run_length_bounded():
    """Truncating to a small max_run_length must not break recursion or the
    qualitative change-detection signal (expected-run-length collapse)."""
    df = _step_change_df(T=400, change_at=200, seed=13)
    model = Bocpd(feature_columns=("ret_1d",), hazard_lambda=100.0, max_run_length=50)
    model.fit(df, np.arange(50, dtype=np.int64))
    feats = model.native_features(df, np.arange(df.height, dtype=np.int64))
    expected_rl = feats[:, 1]
    pre_change_max = float(expected_rl[150:200].max())
    post_change_min = float(expected_rl[205:240].min())
    assert post_change_min < pre_change_max / 5
