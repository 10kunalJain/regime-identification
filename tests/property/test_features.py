"""Property tests for the feature pipeline.

The headline guarantee: extending the underlying data past `t` never changes
the feature values at any date `<= t`. This is the no-leak contract carried
forward from the PIT query layer.

Per-feature stationarity sanity checks live alongside.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from regime.features.pipeline import build_features_for_ticker
from regime.features.registry import all_features


def _ramp_prices(
    start: date, n: int, base: float = 100.0, step: float = 0.5
) -> list[tuple[date, float]]:
    """Linear ramp; produces non-degenerate but deterministic feature values."""
    return [(start + timedelta(days=i), base + step * i) for i in range(n)]


def _wiggly_prices(start: date, n: int) -> list[tuple[date, float]]:
    """Sinusoidal-ish ramp so RSI and MA distance aren't pinned at extremes."""
    return [
        (start + timedelta(days=i), 100.0 + 5.0 * math.sin(i / 7.0) + 0.1 * i) for i in range(n)
    ]


@given(extra_days=st.integers(min_value=1, max_value=60))
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_feature_unchanged_when_future_added(synthetic_ticker_factory, extra_days):
    """Adding data after `t` must not change any feature value at or before `t`."""
    base_prices = _wiggly_prices(date(2020, 1, 1), 250)
    pivot = base_prices[200][0]

    synthetic_ticker_factory("FEAT", base_prices)
    df_short = build_features_for_ticker("FEAT", pivot)

    extended = base_prices + [
        (base_prices[-1][0] + timedelta(days=i + 1), 999.0 + i) for i in range(extra_days)
    ]
    synthetic_ticker_factory("FEAT", extended)
    df_long = build_features_for_ticker("FEAT", pivot)

    # Same as_of cutoff → identical feature DataFrames row-for-row.
    assert df_short.equals(df_long)


def test_no_feature_value_after_t(synthetic_ticker_factory):
    """The pipeline returns no row with data_time > t."""
    prices = _wiggly_prices(date(2020, 1, 1), 100)
    synthetic_ticker_factory("FEAT", prices)
    pivot = prices[60][0]
    df = build_features_for_ticker("FEAT", pivot)
    max_dt = df["data_time"].max()
    assert isinstance(max_dt, date)
    assert max_dt <= pivot


def test_returns_window_lengths(synthetic_ticker_factory):
    """ret_Nd is null for the first N rows and non-null thereafter."""
    prices = _ramp_prices(date(2020, 1, 1), 100)
    synthetic_ticker_factory("FEAT", prices)
    df = build_features_for_ticker("FEAT", prices[-1][0])
    for n in (1, 5, 21, 63):
        col = f"ret_{n}d"
        # First n rows should have null (need n+1 prices → n returns → rolling sum of n).
        nulls = df[col].is_null().sum()
        assert nulls >= n, f"{col} has {nulls} nulls, expected ≥ {n}"


def test_realized_vol_is_non_negative(synthetic_ticker_factory):
    prices = _wiggly_prices(date(2020, 1, 1), 100)
    synthetic_ticker_factory("FEAT", prices)
    df = build_features_for_ticker("FEAT", prices[-1][0])
    for col in ("rv_5d", "rv_21d", "rv_63d", "vol_of_vol_21d"):
        non_null = df[col].drop_nulls()
        if non_null.len() == 0:
            continue
        assert (non_null >= 0.0).all(), f"{col} has negative values"


def test_rsi_in_zero_to_hundred(synthetic_ticker_factory):
    prices = _wiggly_prices(date(2020, 1, 1), 100)
    synthetic_ticker_factory("FEAT", prices)
    df = build_features_for_ticker("FEAT", prices[-1][0])
    rsi = df["rsi_14"].drop_nulls()
    if rsi.len() == 0:
        return
    assert (rsi >= 0.0).all()
    assert (rsi <= 100.0).all()


def test_rsi_at_100_on_monotonic_uptrend(synthetic_ticker_factory):
    """RSI should saturate near 100 when prices only go up."""
    prices = _ramp_prices(date(2020, 1, 1), 100, step=0.5)
    synthetic_ticker_factory("FEAT", prices)
    df = build_features_for_ticker("FEAT", prices[-1][0])
    rsi = df["rsi_14"].drop_nulls()
    last_rsi = rsi.tail(1).item()
    assert last_rsi == pytest.approx(100.0, abs=1e-6)


def test_dist_high_is_non_positive(synthetic_ticker_factory):
    prices = _wiggly_prices(date(2020, 1, 1), 100)
    synthetic_ticker_factory("FEAT", prices)
    df = build_features_for_ticker("FEAT", prices[-1][0])
    vals = df["dist_high_21d"].drop_nulls()
    if vals.len() == 0:
        return
    # Must be ≤ 0 by construction (close ≤ rolling max).
    assert (vals <= 1e-12).all()


def test_dist_low_is_non_negative(synthetic_ticker_factory):
    prices = _wiggly_prices(date(2020, 1, 1), 100)
    synthetic_ticker_factory("FEAT", prices)
    df = build_features_for_ticker("FEAT", prices[-1][0])
    vals = df["dist_low_21d"].drop_nulls()
    if vals.len() == 0:
        return
    assert (vals >= -1e-12).all()


def test_registry_is_populated():
    """Sanity: importing the package registers every feature module."""
    names = {f.name for f in all_features()}
    expected_subset = {
        "ret_1d",
        "ret_5d",
        "ret_21d",
        "ret_63d",
        "rv_5d",
        "rv_21d",
        "rv_63d",
        "vol_of_vol_21d",
        "rsi_14",
        "ma_dist_50",
        "ma_dist_200",
        "dist_high_21d",
        "dist_low_21d",
    }
    assert expected_subset.issubset(names), f"missing: {expected_subset - names}"


def test_feature_pipeline_is_deterministic(synthetic_ticker_factory, tmp_path):
    """Bit-exact contract: building twice produces byte-identical Parquet output."""
    prices = _wiggly_prices(date(2020, 1, 1), 200)
    synthetic_ticker_factory("FEAT", prices)
    df1 = build_features_for_ticker("FEAT", prices[-1][0])
    df2 = build_features_for_ticker("FEAT", prices[-1][0])

    # In-memory comparison (Polars equality)
    assert df1.equals(df2)

    # Round-trip Parquet bit-exactness under fixed compression / sort.
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    df1.sort("data_time").write_parquet(p1, compression="zstd", statistics=True)
    df2.sort("data_time").write_parquet(p2, compression="zstd", statistics=True)
    assert p1.read_bytes() == p2.read_bytes()


def test_empty_when_no_data(tmp_data_root):
    df = build_features_for_ticker("UNKNOWN", date(2020, 1, 1))
    assert df.is_empty()
    # Schema still has the registered feature columns.
    feature_names = {f.name for f in all_features()}
    assert feature_names.issubset(set(df.columns))


def test_feature_columns_are_float64(synthetic_ticker_factory):
    prices = _wiggly_prices(date(2020, 1, 1), 100)
    synthetic_ticker_factory("FEAT", prices)
    df = build_features_for_ticker("FEAT", prices[-1][0])
    for f in all_features():
        col = pl.col(f.name)
        assert df.select(col).dtypes[0] == pl.Float64, f"{f.name} not Float64"
