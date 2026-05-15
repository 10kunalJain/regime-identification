"""Tests for the multi-asset portfolio returns + ADV builder."""

from __future__ import annotations

from datetime import date

import pytest

from regime.data.portfolio_returns import PORTFOLIO_TICKERS, build_portfolio_returns


def _seed_two_tickers(factory, dates: list[date], close_a: float, close_b: float) -> None:
    factory("AAA", [(d, close_a) for d in dates])
    factory("BBB", [(d, close_b) for d in dates])


def test_portfolio_tickers_covers_universe():
    # The 18-ETF universe = SPY + 11 sector SPDRs + 6 style factor ETFs + TLT.
    assert len(PORTFOLIO_TICKERS) == 19
    assert "SPY" in PORTFOLIO_TICKERS
    assert "TLT" in PORTFOLIO_TICKERS


def test_build_portfolio_returns_columns_and_arithmetic(synthetic_ticker_factory):
    dates = [date(2020, 1, d) for d in range(1, 6)]
    _seed_two_tickers(synthetic_ticker_factory, dates, close_a=100.0, close_b=50.0)

    df = build_portfolio_returns(date(2020, 12, 31), tickers=("AAA", "BBB"))

    assert df.columns == ["data_time", "ret_AAA", "notional_AAA", "ret_BBB", "notional_BBB"]
    # Flat price series → arithmetic returns are zero; first row dropped by drop_nulls.
    assert df.height == len(dates) - 1
    assert (df["ret_AAA"] == 0.0).all()
    assert (df["ret_BBB"] == 0.0).all()
    # synthetic_ticker_factory sets volume=1000 for every row.
    assert df["notional_AAA"].item(0) == pytest.approx(100.0 * 1000)
    assert df["notional_BBB"].item(0) == pytest.approx(50.0 * 1000)


def test_build_portfolio_returns_uses_dividend_in_total_return(synthetic_ticker_factory):
    dates = [date(2020, 1, d) for d in range(1, 5)]
    synthetic_ticker_factory(
        "DIV",
        [(d, 100.0) for d in dates],
        actions=[{"ex_date": date(2020, 1, 3), "kind": "dividend", "ratio": 0.0, "amount": 2.0}],
    )

    df = build_portfolio_returns(date(2020, 12, 31), tickers=("DIV",))
    ret_on_div = df.filter(df["data_time"] == date(2020, 1, 3))["ret_DIV"].item()
    # (close + dividend) / prev_close - 1 = (100 + 2) / 100 - 1 = 0.02
    assert ret_on_div == pytest.approx(0.02)


def test_build_portfolio_returns_inner_join_aligns_to_common_dates(synthetic_ticker_factory):
    synthetic_ticker_factory("AAA", [(date(2020, 1, d), 100.0) for d in range(1, 6)])
    synthetic_ticker_factory("BBB", [(date(2020, 1, d), 50.0) for d in range(3, 8)])

    df = build_portfolio_returns(date(2020, 12, 31), tickers=("AAA", "BBB"))
    # Common range is Jan 3-5; first row of each side is dropped by the return
    # shift, leaving only the two dates where both tickers have a return.
    assert df["data_time"].to_list() == [date(2020, 1, 4), date(2020, 1, 5)]


def test_build_portfolio_returns_sorted_ascending(synthetic_ticker_factory):
    synthetic_ticker_factory("AAA", [(date(2020, 2, 1), 100.0), (date(2020, 1, 15), 90.0)])

    df = build_portfolio_returns(date(2020, 12, 31), tickers=("AAA",))
    times = df["data_time"].to_list()
    assert times == sorted(times)


def test_build_portfolio_returns_raises_on_missing_ticker(tmp_data_root):
    with pytest.raises(ValueError, match="no PIT data for 'AAA'"):
        build_portfolio_returns(date(2020, 1, 1), tickers=("AAA",))
