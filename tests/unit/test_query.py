"""Basic query-layer unit tests."""

from __future__ import annotations

from datetime import date

import pytest

from regime.data.query import as_of


def test_query_returns_expected_columns(synthetic_ticker_factory):
    prices = [(date(2020, 1, 1), 100.0), (date(2020, 1, 2), 101.0)]
    synthetic_ticker_factory("ABC", prices)
    df = as_of("ABC", date(2020, 1, 5))
    expected = {
        "data_time",
        "knowledge_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dividend_amount",
    }
    assert expected.issubset(set(df.columns))


def test_query_unknown_ticker_returns_empty(tmp_data_root):
    df = as_of("UNKNOWN", date(2020, 1, 1))
    assert df.is_empty()


def test_dividend_appears_only_on_ex_date(synthetic_ticker_factory):
    prices = [(date(2020, 1, d), 100.0) for d in range(1, 11)]
    actions = [{"ex_date": date(2020, 1, 5), "kind": "dividend", "ratio": 0.0, "amount": 1.5}]
    synthetic_ticker_factory("DIV", prices, actions=actions)

    df = as_of("DIV", date(2020, 12, 31))
    div_row = df.filter(df["data_time"] == date(2020, 1, 5))
    assert div_row["dividend_amount"].item() == pytest.approx(1.5)
    other = df.filter(df["data_time"] != date(2020, 1, 5))
    assert (other["dividend_amount"] == 0.0).all()


def test_dividend_not_visible_before_ex_date(synthetic_ticker_factory):
    prices = [(date(2020, 1, d), 100.0) for d in range(1, 11)]
    actions = [{"ex_date": date(2020, 1, 5), "kind": "dividend", "ratio": 0.0, "amount": 1.5}]
    synthetic_ticker_factory("DIV", prices, actions=actions)

    df = as_of("DIV", date(2020, 1, 4))
    assert (df["dividend_amount"] == 0.0).all()
