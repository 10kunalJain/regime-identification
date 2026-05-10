"""Synthetic-split scenario test (Q1 lock — the headline correctness test).

Given a ticker with a 2-for-1 split on a known ex_date:
  - Querying before ex_date returns ORIGINAL prices and ORIGINAL volume.
  - Querying on or after ex_date returns price/2 and volume*2 for all data_time
    strictly before ex_date; rows on/after ex_date are unchanged.
  - A pre-split dividend, observed after the split, is divided by the split ratio
    so it is on the same share basis as the adjusted prices.
"""

from __future__ import annotations

from datetime import date

import pytest

from regime.data.query import as_of

PRE_SPLIT_PRICE = 100.0
POST_SPLIT_PRICE = 50.0
SPLIT_DATE = date(2020, 6, 15)


@pytest.fixture
def split_ticker(synthetic_ticker_factory):
    pre = [(date(2020, 5, d), PRE_SPLIT_PRICE) for d in range(1, 11)]
    post = [(date(2020, 6, d), POST_SPLIT_PRICE) for d in range(16, 26)]
    actions = [
        {"ex_date": SPLIT_DATE, "kind": "split", "ratio": 2.0, "amount": 0.0},
    ]
    synthetic_ticker_factory("SPLIT", pre + post, actions=actions)


def test_before_ex_date_returns_original(split_ticker):
    df = as_of("SPLIT", date(2020, 6, 14))
    assert df.height == 10
    assert df["close"].max() == pytest.approx(PRE_SPLIT_PRICE)
    assert df["close"].min() == pytest.approx(PRE_SPLIT_PRICE)
    assert df["volume"].max() == 1000


def test_on_ex_date_applies_split(split_ticker):
    df = as_of("SPLIT", SPLIT_DATE)
    pre_rows = df.filter(df["data_time"] < SPLIT_DATE)
    assert pre_rows["close"].max() == pytest.approx(PRE_SPLIT_PRICE / 2.0)
    assert pre_rows["volume"].max() == 2000


def test_after_ex_date_split_persists(split_ticker):
    df = as_of("SPLIT", date(2020, 12, 31))
    pre_rows = df.filter(df["data_time"] < SPLIT_DATE)
    post_rows = df.filter(df["data_time"] >= SPLIT_DATE)
    assert pre_rows["close"].max() == pytest.approx(PRE_SPLIT_PRICE / 2.0)
    assert post_rows["close"].max() == pytest.approx(POST_SPLIT_PRICE)
    assert pre_rows["volume"].max() == 2000
    assert post_rows["volume"].max() == 1000


def test_no_action_with_ex_date_in_future_is_applied(split_ticker):
    """A query strictly before the split must not apply it anywhere."""
    df = as_of("SPLIT", date(2020, 5, 30))
    assert df["close"].min() == pytest.approx(PRE_SPLIT_PRICE)
    assert df["close"].max() == pytest.approx(PRE_SPLIT_PRICE)
    assert df["volume"].max() == 1000


def test_dividend_split_consistency(synthetic_ticker_factory):
    """A $2 pre-split dividend, observed after the split, becomes $1 in adjusted terms."""
    pre = [(date(2020, 5, d), 100.0) for d in range(1, 11)]
    post = [(date(2020, 6, d), 50.0) for d in range(16, 26)]
    actions = [
        {"ex_date": date(2020, 5, 5), "kind": "dividend", "ratio": 0.0, "amount": 2.0},
        {"ex_date": SPLIT_DATE, "kind": "split", "ratio": 2.0, "amount": 0.0},
    ]
    synthetic_ticker_factory("DIVSPL", pre + post, actions=actions)

    # Before the split, the dividend is $2 (original face value).
    df_before = as_of("DIVSPL", date(2020, 5, 5))
    div_row = df_before.filter(df_before["data_time"] == date(2020, 5, 5))
    assert div_row["dividend_amount"].item() == pytest.approx(2.0)

    # After the split, the dividend in adjusted-share terms is $1.
    df_after = as_of("DIVSPL", date(2020, 7, 1))
    div_row = df_after.filter(df_after["data_time"] == date(2020, 5, 5))
    assert div_row["dividend_amount"].item() == pytest.approx(1.0)
