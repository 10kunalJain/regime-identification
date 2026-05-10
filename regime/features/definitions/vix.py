"""VIX complex features: VIX level, term structure, VVIX, 5-day change."""

from __future__ import annotations

from datetime import date

import polars as pl

from regime.data.query import as_of

_EMPTY_SCHEMA = {
    "data_time": pl.Date,
    "vix_level": pl.Float64,
    "vix_term_structure": pl.Float64,
    "vvix_level": pl.Float64,
    "vix_5d_change": pl.Float64,
}


def build_vix_features(t: date) -> pl.DataFrame:
    """VIX complex features for every date <= t."""
    vix = as_of("^VIX", t)
    if vix.is_empty():
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    df = vix.select("data_time", pl.col("close").alias("vix_level"))

    vix3m = as_of("^VIX3M", t)
    if not vix3m.is_empty():
        df = df.join(
            vix3m.select("data_time", pl.col("close").alias("vix3m_level")),
            on="data_time",
            how="left",
        )
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("vix3m_level"))

    vvix = as_of("^VVIX", t)
    if not vvix.is_empty():
        df = df.join(
            vvix.select("data_time", pl.col("close").alias("vvix_level")),
            on="data_time",
            how="left",
        )
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("vvix_level"))

    df = df.with_columns(
        (pl.col("vix3m_level") - pl.col("vix_level")).alias("vix_term_structure"),
        (pl.col("vix_level") - pl.col("vix_level").shift(5)).alias("vix_5d_change"),
    )
    return df.select(
        "data_time", "vix_level", "vix_term_structure", "vvix_level", "vix_5d_change"
    ).sort("data_time")
