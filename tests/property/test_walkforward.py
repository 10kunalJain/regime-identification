"""Walk-forward harness tests.

Two contracts to verify:
  - Determinism: running the same harness twice on the same input produces
    identical output (under fixed seeds + single-threaded BLAS).
  - No-leak: predictions in fold k only depend on rows < fold k's start; adding
    rows after a fold's predictions don't change those predictions.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regime.eval.walkforward import WalkForwardConfig, walk_forward
from regime.models._hmm_core import sample_gaussian_hmm
from regime.models.hmm_gaussian import HmmGaussian


def _make_synthetic_df(T: int, seed: int) -> pl.DataFrame:
    """Well-conditioned 3-state HMM data for the walk-forward harness tests.

    Uses well-separated means and equal modest variances so EM converges to the
    same optimum across runs (the harness test isn't trying to stress the
    fitter's edge cases — it's testing the harness's data-handling).
    """
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
        {
            "data_time": dates,
            "ret_1d": obs[:, 0],
            "rv_21d": obs[:, 1],
        },
        schema={"data_time": pl.Date, "ret_1d": pl.Float64, "rv_21d": pl.Float64},
    )


def _factory(seed: int = 42):
    """Single-restart factory: removes best-of-N tie-breaking sensitivity."""
    return lambda: HmmGaussian(
        K=3, feature_columns=("ret_1d", "rv_21d"), n_restarts=1, random_state=seed
    )


def _df_close(a: pl.DataFrame, b: pl.DataFrame, tol: float = 1e-2) -> bool:
    """Tolerance equality for DataFrames with float columns.

    Per ARCHITECTURE.md §8.1, fitted-parameter outputs go in the *tolerance*
    contract bucket, not bit-exact: hmmlearn's full-cov EM is numerically
    sensitive near degenerate states and produces sub-1% perturbations across
    nominally-identical runs. Date / integer columns are compared exactly.
    """
    if a.shape != b.shape or a.columns != b.columns:
        return False
    for col in a.columns:
        col_a = a.get_column(col)
        col_b = b.get_column(col)
        if col_a.dtype.is_float():
            diff_array = (col_a - col_b).to_numpy()
            max_abs = float(np.abs(diff_array).max())
            if max_abs > tol:
                return False
        else:
            if not (col_a == col_b).all():
                return False
    return True


@pytest.mark.slow
def test_walkforward_deterministic():
    df = _make_synthetic_df(T=400, seed=1)
    cfg = WalkForwardConfig(initial_train_rows=200, refit_every_rows=80)
    out1 = walk_forward(df, _factory(seed=42), cfg)
    out2 = walk_forward(df, _factory(seed=42), cfg)
    assert _df_close(out1, out2)


@pytest.mark.slow
def test_walkforward_extending_data_does_not_change_earlier_folds():
    """Adding rows after the harness's last fold doesn't change earlier folds' predictions."""
    df_short = _make_synthetic_df(T=400, seed=2)
    cfg = WalkForwardConfig(initial_train_rows=200, refit_every_rows=80)
    out_short = walk_forward(df_short, _factory(seed=11), cfg)

    # Build an extended DF with the same first 400 rows and 100 extra rows.
    df_long = _make_synthetic_df(T=500, seed=2)
    out_long = walk_forward(df_long, _factory(seed=11), cfg)

    short_dates = set(out_short["data_time"].to_list())
    out_long_restricted = out_long.filter(pl.col("data_time").is_in(list(short_dates))).sort(
        "data_time"
    )
    out_short_sorted = out_short.sort("data_time")

    # The restricted long output should match the short output to within tolerance.
    assert _df_close(out_long_restricted, out_short_sorted)


def test_walkforward_returns_empty_when_train_too_small():
    df = _make_synthetic_df(T=100, seed=3)
    cfg = WalkForwardConfig(initial_train_rows=200, refit_every_rows=63)
    out = walk_forward(df, _factory(), cfg)
    assert out.is_empty()


@pytest.mark.slow
def test_walkforward_filtered_columns_normalized():
    df = _make_synthetic_df(T=400, seed=5)
    cfg = WalkForwardConfig(initial_train_rows=200, refit_every_rows=100)
    out = walk_forward(df, _factory(seed=7), cfg)

    filtered_cols = [c for c in out.columns if c.startswith("filtered_")]
    assert len(filtered_cols) == 3
    arr = np.column_stack([out.get_column(c).to_numpy() for c in filtered_cols])
    row_sum = arr.sum(axis=1)
    np.testing.assert_allclose(row_sum, 1.0, atol=1e-9)
