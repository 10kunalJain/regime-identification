"""Shared test fixtures.

`tmp_data_root` redirects the data store to a tmp directory.
`synthetic_ticker_factory` builds a hermetic ticker with controllable price + action
history so PIT and split tests don't hit yfinance.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def tmp_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REGIME_DATA_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def synthetic_fred_factory(tmp_data_root: Path) -> Callable[..., None]:
    """Build a hermetic FRED series.

    Args (to the returned callable):
      series_id: FRED series ID.
      rows: list of (data_time, knowledge_time, value).
    """

    def _make(series_id: str, rows: list[tuple[date, date, float]]) -> None:
        from regime.data import store

        df = pl.DataFrame(
            [{"data_time": d, "knowledge_time": k, "value": v} for d, k, v in rows],
            schema={
                "data_time": pl.Date,
                "knowledge_time": pl.Date,
                "value": pl.Float64,
            },
        )
        store.write_parquet(df, store.fred_path(series_id))

    return _make


@pytest.fixture
def synthetic_fama_french_factory(tmp_data_root: Path) -> Callable[..., None]:
    """Build a hermetic Fama-French factors file.

    Args (to the returned callable):
      rows: list of dicts with keys data_time, knowledge_time, Mkt-RF, SMB, HML,
            RMW, CMA, Mom (each a daily decimal return).
    """

    def _make(rows: list[dict[str, object]]) -> None:
        from regime.data import store

        schema = {
            "data_time": pl.Date,
            "knowledge_time": pl.Date,
            "Mkt-RF": pl.Float64,
            "SMB": pl.Float64,
            "HML": pl.Float64,
            "RMW": pl.Float64,
            "CMA": pl.Float64,
            "Mom": pl.Float64,
        }
        df = pl.DataFrame(rows, schema=schema)
        store.write_parquet(df, store.fama_french_path())

    return _make


@pytest.fixture
def synthetic_ticker_factory(
    tmp_data_root: Path,
) -> Callable[..., Path]:
    """Build a hermetic ticker.

    Args (to the returned callable):
      ticker: ticker name.
      prices: list of (data_time, close) — open=high=low=close, volume=1000.
      actions: optional list of dicts with keys ex_date, kind ('split' or 'dividend'),
               ratio (for splits), amount (for dividends).
    """

    def _make(
        ticker: str,
        prices: list[tuple[date, float]],
        actions: list[dict[str, object]] | None = None,
    ) -> Path:
        from regime.data import store

        rows = [
            {
                "data_time": d,
                "knowledge_time": d,
                "open": p,
                "high": p,
                "low": p,
                "close": p,
                "volume": 1000,
            }
            for d, p in prices
        ]
        raw_schema = {
            "data_time": pl.Date,
            "knowledge_time": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
        raw_df = pl.DataFrame(rows, schema=raw_schema)
        for year in sorted({d.year for d, _ in prices}):
            part = raw_df.filter(pl.col("data_time").dt.year() == year)
            store.write_parquet(part, store.yfinance_raw_path(ticker, year))

        action_schema = {
            "ex_date": pl.Date,
            "kind": pl.Utf8,
            "ratio": pl.Float64,
            "amount": pl.Float64,
        }
        actions_df = (
            pl.DataFrame(actions, schema=action_schema)
            if actions
            else pl.DataFrame(schema=action_schema)
        )
        store.write_parquet(actions_df, store.yfinance_actions_path(ticker))
        return tmp_data_root

    return _make
