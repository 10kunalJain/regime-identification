"""Macro features from FRED: yield-curve slope, HY OAS, broad USD."""

from __future__ import annotations

from datetime import date

import polars as pl

from regime.data.query import as_of_fred

_EMPTY_SCHEMA = {
    "data_time": pl.Date,
    "t10y2y": pl.Float64,
    "hy_oas": pl.Float64,
    "hy_oas_21d_chg": pl.Float64,
    "usd_broad": pl.Float64,
    "usd_21d_logret": pl.Float64,
}


def build_macro_features(t: date) -> pl.DataFrame:
    """FRED macro features for every date <= t (respecting publication lag)."""
    t10y2y = as_of_fred("T10Y2Y", t).select("data_time", pl.col("value").alias("t10y2y"))
    if t10y2y.is_empty():
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    df = t10y2y

    hyoas = as_of_fred("BAMLH0A0HYM2", t).select("data_time", pl.col("value").alias("hy_oas"))
    if not hyoas.is_empty():
        df = df.join(hyoas, on="data_time", how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("hy_oas"))

    usd = as_of_fred("DTWEXBGS", t).select("data_time", pl.col("value").alias("usd_broad"))
    if not usd.is_empty():
        df = df.join(usd, on="data_time", how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("usd_broad"))

    df = df.with_columns(
        (pl.col("hy_oas") - pl.col("hy_oas").shift(21)).alias("hy_oas_21d_chg"),
        (pl.col("usd_broad").log() - pl.col("usd_broad").log().shift(21)).alias("usd_21d_logret"),
    )
    return df.select(
        "data_time",
        "t10y2y",
        "hy_oas",
        "hy_oas_21d_chg",
        "usd_broad",
        "usd_21d_logret",
    ).sort("data_time")
