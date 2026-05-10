"""Tests for EnsembleStacker."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regime.ensemble.stacker import EnsembleStacker
from regime.models._hmm_core import sample_gaussian_hmm
from regime.models.hmm_gaussian import HmmGaussian


def _build_df(T: int = 400, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    pi = np.array([0.6, 0.3, 0.1])
    A = np.array(
        [
            [0.95, 0.04, 0.01],
            [0.10, 0.85, 0.05],
            [0.05, 0.20, 0.75],
        ]
    )
    means = np.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 1.0]])
    covs = np.tile(np.eye(2) * 0.1, (3, 1, 1))
    _, obs = sample_gaussian_hmm(pi, A, means, covs, T, rng)
    dates = [date(2010, 1, 1) + timedelta(days=i) for i in range(T)]
    return pl.DataFrame(
        {"data_time": dates, "ret_1d": obs[:, 0], "rv_21d": obs[:, 1]},
        schema={"data_time": pl.Date, "ret_1d": pl.Float64, "rv_21d": pl.Float64},
    )


def test_stacker_filter_normalizes():
    df = _build_df(T=300, seed=7)
    m1 = HmmGaussian(K=3, feature_columns=("ret_1d", "rv_21d"), n_restarts=1, random_state=11)
    m2 = HmmGaussian(K=3, feature_columns=("ret_1d", "rv_21d"), n_restarts=1, random_state=23)
    stacker = EnsembleStacker([m1, m2])
    train_idx = np.arange(200, dtype=np.int64)
    test_idx = np.arange(200, df.height, dtype=np.int64)
    stacker.fit(df, train_idx)
    f = stacker.filter(df, test_idx)
    np.testing.assert_allclose(f.sum(axis=1), 1.0, atol=1e-9)


def test_stacker_crisis_prob_is_last_column():
    df = _build_df(T=300, seed=8)
    m1 = HmmGaussian(K=3, feature_columns=("ret_1d", "rv_21d"), n_restarts=1, random_state=5)
    stacker = EnsembleStacker([m1])
    train_idx = np.arange(200, dtype=np.int64)
    test_idx = np.arange(200, df.height, dtype=np.int64)
    stacker.fit(df, train_idx)
    full = stacker.filter(df, test_idx)
    crisis = stacker.crisis_prob(df, test_idx)
    np.testing.assert_allclose(crisis, full[:, -1])


def test_stacker_rejects_empty_models():
    with pytest.raises(ValueError, match="at least one model"):
        EnsembleStacker([])


def test_stacker_rejects_mismatched_k():
    m1 = HmmGaussian(K=3)
    m2 = HmmGaussian(K=2)
    with pytest.raises(ValueError, match="must share K"):
        EnsembleStacker([m1, m2])
