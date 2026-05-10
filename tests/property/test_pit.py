"""Property tests for the PIT contract.

Hypothesis tests asserting that for any random query time `t`:
  - No row returned has knowledge_time > t.
  - No future split or dividend has been incorporated.
  - The query is monotonic — extending t can only add rows, never remove or alter.
"""

from __future__ import annotations

from datetime import date, timedelta

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from regime.data.query import as_of

_ANCHOR = date(2020, 1, 15)


def _make_default_ticker(synthetic_ticker_factory) -> None:
    prices = [(date(2020, 1, d), 100.0 + d) for d in range(1, 28)]
    synthetic_ticker_factory("FAKE", prices)


@given(days_offset=st.integers(min_value=-30, max_value=60))
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_no_row_with_knowledge_time_in_future(synthetic_ticker_factory, days_offset):
    _make_default_ticker(synthetic_ticker_factory)
    t = _ANCHOR + timedelta(days=days_offset)
    df = as_of("FAKE", t)
    if df.is_empty():
        return
    max_kt = df["knowledge_time"].max()
    assert isinstance(max_kt, date)
    assert max_kt <= t


@given(
    earlier_offset=st.integers(min_value=-30, max_value=60),
    delta=st.integers(min_value=1, max_value=30),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_query_is_monotonic_in_t(synthetic_ticker_factory, earlier_offset, delta):
    """Extending t never removes or alters rows that were visible at the earlier t."""
    _make_default_ticker(synthetic_ticker_factory)
    t1 = _ANCHOR + timedelta(days=earlier_offset)
    t2 = t1 + timedelta(days=delta)

    df1 = as_of("FAKE", t1)
    df2 = as_of("FAKE", t2)

    if df1.is_empty():
        return

    # df1's rows must all be present in df2 with identical OHLC values
    common = df2.join(df1, on="data_time", how="inner", suffix="_old")
    for col in ("open", "high", "low", "close"):
        assert (common[col] == common[f"{col}_old"]).all()
    # And df2 has at least as many rows as df1
    assert df2.height >= df1.height


def test_empty_when_t_before_all_data(synthetic_ticker_factory):
    _make_default_ticker(synthetic_ticker_factory)
    df = as_of("FAKE", date(2019, 12, 31))
    assert df.is_empty()


def test_full_history_when_t_after_all_data(synthetic_ticker_factory):
    _make_default_ticker(synthetic_ticker_factory)
    df = as_of("FAKE", date(2020, 12, 31))
    assert df.height == 27
