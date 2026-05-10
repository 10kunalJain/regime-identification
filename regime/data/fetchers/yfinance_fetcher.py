"""yfinance fetcher.

Fetches UNADJUSTED OHLCV (auto_adjust=False) and the action log (splits + dividends)
into the Parquet store, partitioned by year. Raw OHLCV is append-only; the action log
is the source of truth for retrospective adjustments.
"""

from __future__ import annotations

import logging

import polars as pl
import yfinance as yf

from regime.data import store
from regime.data.universe import YF_TICKERS

_LOG = logging.getLogger(__name__)


def refresh_all(start: str = "2003-01-01") -> None:
    for t in YF_TICKERS:
        try:
            refresh_one(t, start)
        except Exception:
            _LOG.exception("failed to refresh %s", t)


def refresh_one(ticker: str, start: str = "2003-01-01") -> None:
    _LOG.info("refresh %s", ticker)
    tk = yf.Ticker(ticker)
    hist = tk.history(start=start, auto_adjust=False, actions=True)
    if hist.empty:
        _LOG.warning("no history for %s", ticker)
        return

    df = pl.from_pandas(hist.reset_index())
    rename_map = {
        "Date": "data_time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Dividends": "_div",
        "Stock Splits": "_split",
    }
    have = {c: rename_map[c] for c in df.columns if c in rename_map}
    df = df.rename(have)
    df = df.with_columns(pl.col("data_time").cast(pl.Date))

    raw = df.select(
        "data_time",
        pl.col("data_time").alias("knowledge_time"),
        "open",
        "high",
        "low",
        "close",
        pl.col("volume").cast(pl.Int64),
    )

    for year in sorted({d.year for d in raw["data_time"]}):
        part = raw.filter(pl.col("data_time").dt.year() == year)
        store.write_parquet(part, store.yfinance_raw_path(ticker, year))

    actions: list[dict[str, object]] = []
    if "_split" in df.columns:
        for row in df.filter(pl.col("_split") > 0).iter_rows(named=True):
            actions.append(
                {
                    "ex_date": row["data_time"],
                    "kind": "split",
                    "ratio": float(row["_split"]),
                    "amount": 0.0,
                }
            )
    if "_div" in df.columns:
        for row in df.filter(pl.col("_div") > 0).iter_rows(named=True):
            actions.append(
                {
                    "ex_date": row["data_time"],
                    "kind": "dividend",
                    "ratio": 0.0,
                    "amount": float(row["_div"]),
                }
            )
    schema = {
        "ex_date": pl.Date,
        "kind": pl.Utf8,
        "ratio": pl.Float64,
        "amount": pl.Float64,
    }
    actions_df = pl.DataFrame(actions, schema=schema) if actions else pl.DataFrame(schema=schema)
    store.write_parquet(actions_df, store.yfinance_actions_path(ticker))
