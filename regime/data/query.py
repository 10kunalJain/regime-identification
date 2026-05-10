"""Point-in-time query layer.

The single read API for every consumer of stored data — yfinance OHLCV, FRED
series, and Fama-French factors. All three apply the same PIT contract: only
rows with `knowledge_time <= t` are returned.

For yfinance OHLCV, splits and dividends are applied from the action log at
query time. Raw OHLCV is never mutated; the action log is append-only.

PIT contract enforced here:
  - Only rows with knowledge_time <= t are returned.
  - All splits with ex_date <= t are applied (cumulatively) to all returned prices.
  - dividend_amount[r] is the dividend with ex_date == r.data_time if such a dividend
    exists and ex_date <= t, else 0. Pre-split dividends are themselves divided by
    the cumulative post-dividend split ratio so price and dividend are on the same
    share basis (matters only if a ticker actually splits; ETFs in our universe
    rarely do, but the math is implemented correctly).
  - No split or dividend with ex_date > t affects the output.
  - FRED rows have knowledge_time = data_time + publication_lag.
  - Fama-French rows have knowledge_time = data_time (Ken French publishes daily
    factors at end-of-day; for v1 we treat them as same-day knowable).
"""

from __future__ import annotations

from datetime import date

import polars as pl

from regime.data import store

OHLC_COLS = ("open", "high", "low", "close")


def as_of(ticker: str, t: date) -> pl.DataFrame:
    """Return the PIT-correct OHLCV view for `ticker` as of `t`."""
    raw = _read_raw(ticker, t)
    actions = _read_actions(ticker)
    return _apply_actions(raw, actions, t)


def as_of_fred(series_id: str, t: date) -> pl.DataFrame:
    """Return PIT-correct FRED rows for `series_id` as of `t`.

    Columns: data_time, knowledge_time, value. Filtered by knowledge_time <= t.
    """
    p = store.fred_path(series_id)
    schema = {
        "data_time": pl.Date,
        "knowledge_time": pl.Date,
        "value": pl.Float64,
    }
    if not p.exists():
        return pl.DataFrame(schema=schema)
    return pl.read_parquet(p).filter(pl.col("knowledge_time") <= t).sort("data_time")


def as_of_fama_french(t: date) -> pl.DataFrame:
    """Return PIT-correct Fama-French daily factors as of `t`.

    Columns: data_time, knowledge_time, Mkt-RF, SMB, HML, RMW, CMA, Mom (each as
    daily decimal returns, i.e. divided by 100 from the published percent form).
    Filtered by knowledge_time <= t.
    """
    p = store.fama_french_path()
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
    if not p.exists():
        return pl.DataFrame(schema=schema)
    return pl.read_parquet(p).filter(pl.col("knowledge_time") <= t).sort("data_time")


def _read_raw(ticker: str, t: date) -> pl.DataFrame:
    base = store.yfinance_raw_dir(ticker)
    if not base.exists():
        return _empty_raw()
    parts = sorted(base.glob("*.parquet"))
    if not parts:
        return _empty_raw()
    frames = [pl.read_parquet(p) for p in parts]
    df = pl.concat(frames, how="vertical_relaxed")
    return df.filter(pl.col("knowledge_time") <= t).sort("data_time")


def _read_actions(ticker: str) -> pl.DataFrame:
    p = store.yfinance_actions_path(ticker)
    schema = {
        "ex_date": pl.Date,
        "kind": pl.Utf8,
        "ratio": pl.Float64,
        "amount": pl.Float64,
    }
    if not p.exists():
        return pl.DataFrame(schema=schema)
    return pl.read_parquet(p).sort("ex_date")


def _empty_raw() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "data_time": pl.Date,
            "knowledge_time": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
    )


def _apply_actions(raw: pl.DataFrame, actions: pl.DataFrame, t: date) -> pl.DataFrame:
    """Apply splits cumulatively and attach dividends, respecting the PIT cutoff t."""
    if raw.is_empty():
        return raw.with_columns(pl.lit(0.0).alias("dividend_amount"))

    relevant = actions.filter(pl.col("ex_date") <= t).sort("ex_date")
    splits = relevant.filter(pl.col("kind") == "split").select("ex_date", "ratio")
    divs = relevant.filter(pl.col("kind") == "dividend").select(
        pl.col("ex_date").alias("data_time"),
        pl.col("amount").alias("dividend_amount"),
    )

    out = raw

    for split_row in splits.iter_rows(named=True):
        ex_d = split_row["ex_date"]
        r = float(split_row["ratio"])
        if r <= 0.0:
            continue
        pre_mask = pl.col("data_time") < ex_d
        price_exprs = [
            pl.when(pre_mask).then(pl.col(c) / r).otherwise(pl.col(c)).alias(c) for c in OHLC_COLS
        ]
        volume_expr = (
            pl.when(pre_mask)
            .then((pl.col("volume").cast(pl.Float64) * r).cast(pl.Int64))
            .otherwise(pl.col("volume"))
            .alias("volume")
        )
        out = out.with_columns(*price_exprs, volume_expr)
        if not divs.is_empty():
            divs = divs.with_columns(
                pl.when(pl.col("data_time") < ex_d)
                .then(pl.col("dividend_amount") / r)
                .otherwise(pl.col("dividend_amount"))
                .alias("dividend_amount")
            )

    out = out.join(divs, on="data_time", how="left").with_columns(
        pl.col("dividend_amount").fill_null(0.0)
    )
    return out.sort("data_time")
