"""WassersteinKmeans tests on synthetic distributional regimes."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regime.models.wasserstein_kmeans import (
    WassersteinKmeans,
    _wasserstein_distance,
)


def _two_regime_df(T: int = 600, change_at: int = 300, seed: int = 5) -> pl.DataFrame:
    """First half: N(0, 1). Second half: N(0, 0.3) — same mean, different scale."""
    rng = np.random.default_rng(seed)
    pre = rng.normal(0.0, 1.0, size=change_at)
    post = rng.normal(0.0, 0.3, size=T - change_at)
    y = np.concatenate([pre, post])
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(T)]
    return pl.DataFrame(
        {"data_time": dates, "ret_1d": y},
        schema={"data_time": pl.Date, "ret_1d": pl.Float64},
    )


def _three_regime_df(T: int = 600, seed: int = 9) -> pl.DataFrame:
    """Three sequential regimes with different scales."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=T // 3)
    b = rng.normal(0.0, 0.3, size=T // 3)
    c = rng.normal(0.0, 2.0, size=T - 2 * (T // 3))
    y = np.concatenate([a, b, c])
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(T)]
    return pl.DataFrame(
        {"data_time": dates, "ret_1d": y},
        schema={"data_time": pl.Date, "ret_1d": pl.Float64},
    )


# ---------- distance primitive ----------


def test_wasserstein_zero_for_identical_samples():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 1))
    d = _wasserstein_distance(a, a, n_projections=1, rng=rng)
    assert d == pytest.approx(0.0, abs=1e-12)


def test_wasserstein_grows_with_mean_shift_univariate():
    rng = np.random.default_rng(0)
    a = np.sort(rng.normal(0.0, 1.0, size=200)).reshape(-1, 1)
    b = np.sort(rng.normal(2.0, 1.0, size=200)).reshape(-1, 1)
    c = np.sort(rng.normal(5.0, 1.0, size=200)).reshape(-1, 1)
    d_ab = _wasserstein_distance(a, b, n_projections=1, rng=rng)
    d_ac = _wasserstein_distance(a, c, n_projections=1, rng=rng)
    assert d_ac > d_ab


def test_wasserstein_multivariate_via_slicing():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(50, 3))
    b = rng.normal(size=(50, 3)) + 5.0
    d_ab = _wasserstein_distance(a, b, n_projections=20, rng=rng)
    d_aa = _wasserstein_distance(a, a, n_projections=20, rng=rng)
    assert d_ab > d_aa


def test_wasserstein_shape_mismatch_raises():
    rng = np.random.default_rng(2)
    a = np.zeros((10, 1))
    b = np.zeros((20, 1))
    with pytest.raises(ValueError, match="shape mismatch"):
        _wasserstein_distance(a, b, n_projections=1, rng=rng)


# ---------- WassersteinKmeans ----------


def test_wkmeans_fit_and_features():
    df = _three_regime_df(T=600, seed=9)
    model = WassersteinKmeans(K=3, feature_columns=("ret_1d",), window=20, n_restarts=2, n_iter=10)
    idx = np.arange(df.height, dtype=np.int64)
    model.fit(df, idx)
    feats = model.native_features(df, idx)
    assert feats.shape == (df.height, 3)
    # First (window - 1) rows are NaN (insufficient data).
    assert np.isnan(feats[:19]).all()
    # All other rows have finite distances.
    assert np.isfinite(feats[19:]).all()


def test_wkmeans_distances_smaller_within_regime():
    """A window inside a regime should be closer to the medoid that summarizes
    that regime than to other medoids."""
    df = _three_regime_df(T=600, seed=11)
    model = WassersteinKmeans(K=3, feature_columns=("ret_1d",), window=20, n_restarts=2, n_iter=10)
    idx = np.arange(df.height, dtype=np.int64)
    model.fit(df, idx)
    feats = model.native_features(df, idx)

    # Drop NaN early rows and check assignments by min-distance differ across regimes.
    finite = feats[19:]
    assignments = np.argmin(finite, axis=1)
    # Three regimes were generated sequentially of length ~T/3.
    seg = (df.height - 19) // 3
    early = assignments[:seg]
    late = assignments[2 * seg :]
    # The most common cluster in early-segment vs late-segment should differ.
    most_common_early = int(np.bincount(early).argmax())
    most_common_late = int(np.bincount(late).argmax())
    assert most_common_early != most_common_late


def test_wkmeans_state_dict_roundtrip():
    df = _two_regime_df(T=400, seed=3)
    model = WassersteinKmeans(K=2, window=20, n_restarts=2, n_iter=5)
    idx = np.arange(df.height, dtype=np.int64)
    model.fit(df, idx)
    feats_a = model.native_features(df, idx)
    state = model.state_dict()
    restored = WassersteinKmeans(K=2, window=20)
    restored.load_state_dict(state)
    feats_b = restored.native_features(df, idx)
    np.testing.assert_allclose(feats_a, feats_b)


def test_wkmeans_rejects_too_few_observations():
    df = pl.DataFrame(
        {
            "data_time": [date(2020, 1, d + 1) for d in range(10)],
            "ret_1d": [0.0] * 10,
        }
    )
    model = WassersteinKmeans(K=3, window=20)
    with pytest.raises(ValueError, match="need at least"):
        model.fit(df, np.arange(10, dtype=np.int64))
