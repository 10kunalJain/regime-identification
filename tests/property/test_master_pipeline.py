"""Master-pipeline property test.

End-to-end no-leak: extending any underlying data source past `t` must not
change any feature value at or before `t` for any ticker.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from regime.data.universe import EQUITIES
from regime.features.pipeline import build_features


def _build_universe(synthetic_ticker_factory, n: int) -> date:
    """Wire a small synthetic universe: every equity ticker plus the VIX complex."""
    base = date(2020, 1, 1)
    for i, ticker in enumerate(EQUITIES):
        prices = [
            (base + timedelta(days=k), 100.0 + 5.0 * math.sin((k + i) / 7.0) + 0.05 * k)
            for k in range(n)
        ]
        synthetic_ticker_factory(ticker, prices)
    for j, ticker in enumerate(("^VIX", "^VIX3M", "^VVIX")):
        prices = [
            (base + timedelta(days=k), 15.0 + 0.5 * math.sin((k + j) / 5.0)) for k in range(n)
        ]
        synthetic_ticker_factory(ticker, prices)
    # TLT (defensive)
    synthetic_ticker_factory(
        "TLT", [(base + timedelta(days=k), 130.0 - 0.02 * k) for k in range(n)]
    )
    return base + timedelta(days=n - 1)


@given(extra_days=st.integers(min_value=1, max_value=30))
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_master_pipeline_no_leak(synthetic_ticker_factory, extra_days):
    """Extending any data source past `t` must not change features at or before `t`."""
    n_short = 100
    last_short = _build_universe(synthetic_ticker_factory, n_short)
    pivot = last_short - timedelta(days=20)
    df_short = build_features(pivot)

    # Extend every ticker past last_short with junk values.
    base = date(2020, 1, 1)
    for i, ticker in enumerate(EQUITIES):
        all_prices = [
            (base + timedelta(days=k), 100.0 + 5.0 * math.sin((k + i) / 7.0) + 0.05 * k)
            for k in range(n_short)
        ] + [(last_short + timedelta(days=k + 1), 9999.0 + k) for k in range(extra_days)]
        synthetic_ticker_factory(ticker, all_prices)
    df_long = build_features(pivot)

    assert df_short.equals(df_long)


def test_master_pipeline_smoke(synthetic_ticker_factory):
    last = _build_universe(synthetic_ticker_factory, 60)
    df = build_features(last)
    assert df.height > 0
    # Should have one row per (data_time, ticker) for tickers we set up.
    assert "ticker" in df.columns
    tickers_present = set(df["ticker"].unique().to_list())
    assert {"SPY", "TLT"}.issubset(tickers_present)
    # All registered per-ticker features are present.
    expected_cols = {
        "data_time",
        "ticker",
        "ret_1d",
        "rv_21d",
        "rsi_14",
        "ma_dist_50",
        "vix_level",
        "vix_term_structure",
    }
    assert expected_cols.issubset(set(df.columns))
