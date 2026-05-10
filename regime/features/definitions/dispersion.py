"""Cross-sectional dispersion features.

Computed across the 11 sector SPDR ETFs. Operates on multiple tickers, so it
has its own builder rather than registering against the per-ticker pipeline.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from regime.data.query import as_of
from regime.data.universe import SECTOR_SPDRS
from regime.features.registry import log_total_return

ROLLING_Z_WINDOW = 252


_EMPTY_SCHEMA = {
    "data_time": pl.Date,
    "sector_dispersion": pl.Float64,
    "sector_dispersion_z": pl.Float64,
}


def build_dispersion_features(t: date) -> pl.DataFrame:
    """Per-day cross-sectional dispersion across the sector SPDR universe.

    Columns:
      - data_time
      - sector_dispersion: std of single-day log total returns across sectors.
      - sector_dispersion_z: 252-day rolling z-score of sector_dispersion.
    """
    return_dfs: list[pl.DataFrame] = []
    for ticker in SECTOR_SPDRS:
        ohlcv = as_of(ticker, t)
        if ohlcv.is_empty():
            continue
        r = log_total_return(ohlcv).alias(f"r_{ticker}")
        return_dfs.append(ohlcv.select("data_time").with_columns(r))

    if not return_dfs:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    combined = return_dfs[0]
    for df in return_dfs[1:]:
        combined = combined.join(df, on="data_time", how="full", coalesce=True)

    return_cols = [c for c in combined.columns if c.startswith("r_")]

    long = combined.unpivot(
        index="data_time", on=return_cols, variable_name="ticker", value_name="r"
    )
    disp = (
        long.group_by("data_time")
        .agg(pl.col("r").std().alias("sector_dispersion"))
        .sort("data_time")
    )

    disp = disp.with_columns(
        pl.col("sector_dispersion").rolling_mean(window_size=ROLLING_Z_WINDOW).alias("_disp_mean"),
        pl.col("sector_dispersion").rolling_std(window_size=ROLLING_Z_WINDOW).alias("_disp_std"),
    )
    disp = disp.with_columns(
        ((pl.col("sector_dispersion") - pl.col("_disp_mean")) / pl.col("_disp_std")).alias(
            "sector_dispersion_z"
        )
    ).drop("_disp_mean", "_disp_std")

    return disp.select("data_time", "sector_dispersion", "sector_dispersion_z")
