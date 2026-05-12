"""Smoke test for the Day-2 cross-method benchmark.

Goal: verify the registry round-trips end-to-end. For each registered method
we exercise:
  - factory construction with the same (obs_cols, factor_cols) the real
    benchmark script passes;
  - fit on training rows + filter / native_features over the full window;
  - per-fold metric computation + posterior row emission;
  - parquet-ready schema of the long-format posterior frame.

Hermetic: no PIT data store; synthetic 450-day two-regime generator inline.
1 fold of ~150 test bars (≈ "1-year" in the smoke-test sense). Wall-clock
target ≤ 90 s on M1.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from regime.eval.registry import (
    REGISTRY,
    UNIVARIATE_RETURN_COLUMN,
    UNIVARIATE_RV_COLUMN,
)
from regime.eval.runner import (
    WalkForwardConfig,
    method_crisis_lag_to_dataframe,
    method_folds_to_dataframe,
    run_cross_method_walkforward,
)

N_DAYS = 450
SMOKE_TRAIN_ROWS = 300
SMOKE_TEST_ROWS = 150
OBS_TICKERS: tuple[str, ...] = ("SPY", "XLK", "XLF", "XLE", "XLV", "TLT")
FF_FACTORS: tuple[str, ...] = (
    "ff_mkt_rf",
    "ff_smb",
    "ff_hml",
    "ff_rmw",
    "ff_cma",
    "ff_mom",
)


def _synthetic_wide_frame(seed: int = 0) -> pl.DataFrame:
    """Generate a two-regime synthetic wide df with the same column schema as
    `regime.data.joint_dataset.build_wide_dataframe`, plus `rv_21d_SPY`."""
    rng = np.random.default_rng(seed)

    # Two-regime SPY return generator: calm (μ=0.05 bp/d, σ=50 bp) vs.
    # crisis (μ=-10 bp/d, σ=250 bp). Geometric dwell time, mean 40 days.
    spy = np.empty(N_DAYS, dtype=np.float64)
    state = 0
    dwell_remaining = int(rng.geometric(p=1.0 / 40.0))
    for t in range(N_DAYS):
        if dwell_remaining <= 0:
            state = 1 - state
            dwell_remaining = int(rng.geometric(p=1.0 / 40.0))
        mu, sigma = (0.0005, 0.005) if state == 0 else (-0.001, 0.025)
        spy[t] = float(rng.normal(mu, sigma))
        dwell_remaining -= 1

    # Other tickers: correlated-with-SPY common factor + idiosyncratic noise.
    rets: dict[str, np.ndarray] = {"ret_SPY": spy}
    for ticker in OBS_TICKERS[1:]:
        idio = rng.normal(0.0, 0.004, size=N_DAYS)
        rets[f"ret_{ticker}"] = 0.7 * spy + idio

    factors: dict[str, np.ndarray] = {f: rng.normal(0.0, 0.005, size=N_DAYS) for f in FF_FACTORS}

    base_date = date(2003, 1, 2)
    dates = [base_date + timedelta(days=i) for i in range(N_DAYS)]

    df = pl.DataFrame(
        {"data_time": dates, **rets, **factors},
        schema={
            "data_time": pl.Date,
            **{f"ret_{t}": pl.Float64 for t in OBS_TICKERS},
            **{f: pl.Float64 for f in FF_FACTORS},
        },
    )

    # rv_21d_SPY computed by the benchmark script in real life; mirror that here.
    df = df.with_columns(
        pl.col(UNIVARIATE_RETURN_COLUMN).rolling_std(window_size=21).alias(UNIVARIATE_RV_COLUMN)
    ).drop_nulls()
    return df


def test_registry_round_trips_on_synthetic_smoke() -> None:
    df = _synthetic_wide_frame(seed=0)
    assert df.height > SMOKE_TRAIN_ROWS + SMOKE_TEST_ROWS // 2
    assert UNIVARIATE_RETURN_COLUMN in df.columns
    assert UNIVARIATE_RV_COLUMN in df.columns

    obs_cols = tuple(f"ret_{t}" for t in OBS_TICKERS)
    cfg = WalkForwardConfig(
        initial_train_rows=SMOKE_TRAIN_ROWS,
        refit_every_rows=SMOKE_TEST_ROWS,
    )
    result = run_cross_method_walkforward(
        df,
        obs_cols=obs_cols,
        factor_cols=FF_FACTORS,
        config=cfg,
    )

    method_names = [m.name for m in REGISTRY]
    assert {f.method_name for f in result.folds} == set(method_names)

    # 1 fold per method on this short window
    fold_ids_per_method = {
        name: sorted(f.fold_id for f in result.folds if f.method_name == name)
        for name in method_names
    }
    assert all(ids == [0] for ids in fold_ids_per_method.values()), fold_ids_per_method

    # Posterior schema sanity
    assert result.posterior.columns == [
        "method",
        "kind",
        "data_time",
        "fold_id",
        "crisis_score",
        "label",
        "raw_features",
    ]
    assert result.posterior.schema["raw_features"] == pl.List(pl.Float64)

    # Per-method posterior shape + raw-feature width matches registry declaration.
    for method in REGISTRY:
        sub = result.posterior.filter(pl.col("method") == method.name)
        assert sub.height > 0, f"no rows for {method.name}"
        assert sub["kind"].unique().to_list() == [method.kind]

        raw_widths = {len(v) for v in sub["raw_features"].to_list()}
        assert raw_widths == {len(method.raw_feature_names)}, (
            f"{method.name}: raw_features widths {raw_widths} "
            f"!= declared {len(method.raw_feature_names)}"
        )

        # crisis_score finite for every test bar
        scores = sub["crisis_score"].to_numpy()
        assert np.all(np.isfinite(scores)), f"{method.name} produced non-finite crisis_score"

    # State methods must report a crisis_state index in [0, K-1] for K=3;
    # changepoint methods must report None.
    for f in result.folds:
        if f.method_kind == "state":
            assert f.crisis_state is not None
            assert 0 <= f.crisis_state <= 2
        else:
            assert f.crisis_state is None

    # Summary table has exactly one row per method with the expected columns.
    assert result.summary.height == len(REGISTRY)
    assert set(result.summary["method"].to_list()) == set(method_names)
    assert result.summary.columns == [
        "method",
        "mean_lag_m5",
        "mean_brier",
        "mean_pr_auc",
        "agreement",
    ]

    # Parquet adapters round-trip without raising.
    folds_df = method_folds_to_dataframe(result.folds)
    assert folds_df.height == len(result.folds)
    crisis_df = method_crisis_lag_to_dataframe(result.crisis_lag)
    # 6 methods × 8 crises = 48 crisis-lag rows (regardless of in-window status)
    assert crisis_df.height == len(REGISTRY) * 8
