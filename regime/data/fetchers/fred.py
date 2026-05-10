"""FRED fetcher.

Each row is tagged with knowledge_time = data_time + publication_lag (calendar days).
Refine to business days in v2 if needed.
"""

from __future__ import annotations

import logging
import os

import polars as pl
from fredapi import Fred

from regime.data import store
from regime.data.universe import FRED_PUB_LAG_DAYS, FRED_SERIES

_LOG = logging.getLogger(__name__)


def refresh_all() -> None:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set; copy .env.example → .env and add a free key")
    fred = Fred(api_key=api_key)
    for sid in FRED_SERIES:
        try:
            refresh_one(fred, sid)
        except Exception:
            _LOG.exception("failed to refresh %s", sid)


def refresh_one(fred: Fred, series_id: str) -> None:
    _LOG.info("refresh fred %s", series_id)
    s = fred.get_series(series_id).reset_index()
    s.columns = ["data_time", "value"]
    df = (
        pl.from_pandas(s)
        .with_columns(
            pl.col("data_time").cast(pl.Date),
            pl.col("value").cast(pl.Float64),
        )
        .drop_nulls("value")
    )
    lag = FRED_PUB_LAG_DAYS.get(series_id, 1)
    df = df.with_columns((pl.col("data_time") + pl.duration(days=lag)).alias("knowledge_time"))
    store.write_parquet(df, store.fred_path(series_id))
