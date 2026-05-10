"""Unit tests for cross-sectional, VIX, macro, and Fama-French feature builders.

These features have dedicated builders (not registry-based) because they
operate on multi-ticker / multi-series inputs, so each gets its own dedicated
test file. The shared no-leak property test for the master pipeline lives in
tests/property/test_master_pipeline.py.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from regime.data.universe import SECTOR_SPDRS
from regime.features.definitions.dispersion import build_dispersion_features
from regime.features.definitions.factors import build_factor_features
from regime.features.definitions.macro import build_macro_features
from regime.features.definitions.vix import build_vix_features


def _flat_prices(start: date, n: int, p: float) -> list[tuple[date, float]]:
    return [(start + timedelta(days=i), p) for i in range(n)]


def _wiggly_prices(
    start: date, n: int, base: float, amp: float, phase: int
) -> list[tuple[date, float]]:
    import math

    return [(start + timedelta(days=i), base + amp * math.sin((i + phase) / 7.0)) for i in range(n)]


# ---------- Cross-sectional dispersion ----------


def test_dispersion_zero_when_all_sectors_identical(synthetic_ticker_factory):
    """If every sector has identical prices, daily returns are identical and dispersion is 0."""
    prices = _flat_prices(date(2020, 1, 1), 30, 100.0)
    for ticker in SECTOR_SPDRS:
        synthetic_ticker_factory(ticker, prices)
    df = build_dispersion_features(date(2020, 12, 31))
    disp = df["sector_dispersion"].drop_nulls()
    assert disp.len() > 0
    # Returns are all log(100/100) = 0 → std across sectors = 0.
    assert (disp.abs() < 1e-12).all()


def test_dispersion_non_negative(synthetic_ticker_factory):
    for i, ticker in enumerate(SECTOR_SPDRS):
        prices = _wiggly_prices(date(2020, 1, 1), 60, 100.0, 5.0, phase=i)
        synthetic_ticker_factory(ticker, prices)
    df = build_dispersion_features(date(2020, 12, 31))
    disp = df["sector_dispersion"].drop_nulls()
    assert (disp >= 0.0).all()


def test_dispersion_no_leak_on_extension(synthetic_ticker_factory):
    """Adding sector data after t must not change dispersion at or before t."""
    short_prices = [
        _wiggly_prices(date(2020, 1, 1), 60, 100.0, 5.0, phase=i) for i in range(len(SECTOR_SPDRS))
    ]
    pivot = short_prices[0][50][0]
    for ticker, prices in zip(SECTOR_SPDRS, short_prices, strict=True):
        synthetic_ticker_factory(ticker, prices)
    df_short = build_dispersion_features(pivot)

    extended = [
        prices + [(prices[-1][0] + timedelta(days=k + 1), 999.0 + k) for k in range(20)]
        for prices in short_prices
    ]
    for ticker, prices in zip(SECTOR_SPDRS, extended, strict=True):
        synthetic_ticker_factory(ticker, prices)
    df_long = build_dispersion_features(pivot)

    assert df_short.equals(df_long)


# ---------- VIX complex ----------


def test_vix_term_structure_sign(synthetic_ticker_factory):
    """vix_term_structure = vix3m_level - vix_level."""
    vix_prices = _flat_prices(date(2020, 1, 1), 30, 15.0)
    vix3m_prices = _flat_prices(date(2020, 1, 1), 30, 18.0)
    vvix_prices = _flat_prices(date(2020, 1, 1), 30, 90.0)
    synthetic_ticker_factory("^VIX", vix_prices)
    synthetic_ticker_factory("^VIX3M", vix3m_prices)
    synthetic_ticker_factory("^VVIX", vvix_prices)

    df = build_vix_features(date(2020, 12, 31))
    ts = df["vix_term_structure"].drop_nulls()
    assert ts.len() > 0
    assert ((ts - 3.0).abs() < 1e-12).all()


def test_vix_5d_change_correctness(synthetic_ticker_factory):
    """vix_5d_change[t] = vix_level[t] - vix_level[t-5]."""
    prices = [(date(2020, 1, 1) + timedelta(days=i), 10.0 + i) for i in range(30)]
    synthetic_ticker_factory("^VIX", prices)
    synthetic_ticker_factory("^VIX3M", prices)
    synthetic_ticker_factory("^VVIX", prices)
    df = build_vix_features(date(2020, 12, 31))
    chg = df["vix_5d_change"].drop_nulls()
    # Each 5d change on a unit-step ramp is exactly 5.
    assert ((chg - 5.0).abs() < 1e-12).all()


def test_vix_features_empty_without_vix_data(tmp_data_root):
    df = build_vix_features(date(2020, 1, 1))
    assert df.is_empty()
    assert {"vix_level", "vix_term_structure", "vvix_level", "vix_5d_change"}.issubset(
        set(df.columns)
    )


# ---------- Macro from FRED ----------


def test_macro_t10y2y_passthrough(synthetic_fred_factory):
    rows = [
        (date(2020, 1, 1) + timedelta(days=i), date(2020, 1, 2) + timedelta(days=i), 0.5)
        for i in range(30)
    ]
    synthetic_fred_factory("T10Y2Y", rows)
    synthetic_fred_factory("BAMLH0A0HYM2", rows)
    synthetic_fred_factory("DTWEXBGS", rows)
    df = build_macro_features(date(2020, 12, 31))
    assert ((df["t10y2y"].drop_nulls() - 0.5).abs() < 1e-12).all()


def test_macro_pit_respects_publication_lag(synthetic_fred_factory):
    """A FRED row with knowledge_time = t+1 must not appear when querying at t."""
    rows = [
        (date(2020, 1, 1), date(2020, 1, 2), 0.5),  # knowable from Jan 2
        (date(2020, 1, 5), date(2020, 1, 6), 0.6),  # knowable from Jan 6
    ]
    synthetic_fred_factory("T10Y2Y", rows)
    df_jan1 = build_macro_features(date(2020, 1, 1))
    assert df_jan1.is_empty() or df_jan1["t10y2y"].drop_nulls().len() == 0

    df_jan2 = build_macro_features(date(2020, 1, 2))
    vals = df_jan2["t10y2y"].drop_nulls()
    assert vals.len() == 1
    assert vals.item() == pytest.approx(0.5)


def test_macro_hy_oas_21d_change(synthetic_fred_factory):
    """hy_oas_21d_chg[t] = hy_oas[t] - hy_oas[t-21]."""
    rows_t10 = [
        (date(2020, 1, 1) + timedelta(days=i), date(2020, 1, 1) + timedelta(days=i), 1.0)
        for i in range(40)
    ]
    rows_oas = [
        (date(2020, 1, 1) + timedelta(days=i), date(2020, 1, 1) + timedelta(days=i), float(i))
        for i in range(40)
    ]
    rows_usd = [
        (date(2020, 1, 1) + timedelta(days=i), date(2020, 1, 1) + timedelta(days=i), 100.0)
        for i in range(40)
    ]
    synthetic_fred_factory("T10Y2Y", rows_t10)
    synthetic_fred_factory("BAMLH0A0HYM2", rows_oas)
    synthetic_fred_factory("DTWEXBGS", rows_usd)

    df = build_macro_features(date(2020, 12, 31))
    chg = df["hy_oas_21d_chg"].drop_nulls()
    # On a unit-step ramp the 21-day change is constant 21.
    assert ((chg - 21.0).abs() < 1e-12).all()


# ---------- Fama-French factors ----------


def test_factor_passthrough_and_rv(synthetic_fama_french_factory):
    rows = []
    for i in range(60):
        d = date(2020, 1, 1) + timedelta(days=i)
        rows.append(
            {
                "data_time": d,
                "knowledge_time": d,
                "Mkt-RF": 0.001,
                "SMB": 0.0,
                "HML": 0.0,
                "RMW": 0.0,
                "CMA": 0.0,
                "Mom": 0.0,
            }
        )
    synthetic_fama_french_factory(rows)
    df = build_factor_features(date(2020, 12, 31))
    # Constant Mkt-RF returns → 21d realized vol is 0.
    assert ((df["ff_mkt_rf"].drop_nulls() - 0.001).abs() < 1e-9).all()
    rv = df["ff_mkt_rf_rv_21d"].drop_nulls()
    assert (rv.abs() < 1e-12).all()


def test_factor_features_empty_without_data(tmp_data_root):
    df = build_factor_features(date(2020, 1, 1))
    assert df.is_empty()
    expected = {f"ff_{c}" for c in ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")}
    assert expected.issubset(set(df.columns))
