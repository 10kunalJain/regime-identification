"""Joint cross-sectional dataset builder.

Builds the wide DataFrame used by the joint HMM and downstream walk-forward
evaluation: one row per trading day, log-total-returns for each ETF in
`OBSERVATION_TICKERS`, plus the Fama-French 5 + Momentum factor returns
under the `ff_<name>` namespace.

Lives in `regime/data/` (not `scripts/`) so it is importable by both the CLI
and the standalone scripts under `scripts/`. Pure PIT — every read goes
through `regime.data.query`.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from regime.data.query import as_of, as_of_fama_french
from regime.features.registry import log_total_return

OBSERVATION_TICKERS: tuple[str, ...] = ("SPY", "XLK", "XLF", "XLE", "XLV", "TLT")
FF_COLUMNS: tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")
RENAMED_FF: dict[str, str] = {f: f"ff_{f.lower().replace('-', '_')}" for f in FF_COLUMNS}


def build_wide_dataframe(t: date) -> pl.DataFrame:
    """Return DataFrame with columns: data_time, ret_<ticker>..., ff_<factor>...

    All reads go through the PIT layer (`as_of(...)`). Joins are inner so any
    day missing a single column is dropped — yfinance trading calendar +
    Fama-French calendar must agree, and the leading rows are clipped to the
    latest start across all sources.
    """
    ticker_frames: list[pl.DataFrame] = []
    for ticker in OBSERVATION_TICKERS:
        ohlcv = as_of(ticker, t)
        rets = log_total_return(ohlcv).alias(f"ret_{ticker}")
        ticker_frames.append(ohlcv.select("data_time").with_columns(rets))

    combined = ticker_frames[0]
    for df in ticker_frames[1:]:
        combined = combined.join(df, on="data_time", how="inner")

    ff = as_of_fama_french(t).select("data_time", *FF_COLUMNS)
    combined = combined.join(ff, on="data_time", how="inner")
    combined = combined.rename(RENAMED_FF)
    return combined.drop_nulls().sort("data_time")


__all__ = [
    "FF_COLUMNS",
    "OBSERVATION_TICKERS",
    "RENAMED_FF",
    "build_wide_dataframe",
]
