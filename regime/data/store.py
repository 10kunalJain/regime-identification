"""Parquet partition layout and write helpers.

Layout:
    <root>/yfinance/<ticker>/raw/<year>.parquet      # immutable raw OHLCV
    <root>/yfinance/<ticker>/actions.parquet         # split + dividend log
    <root>/fred/<series>.parquet                     # FRED series, append-only
    <root>/fama_french/factors.parquet               # FF 5+Mom daily

The store does not adjust prices. The query layer in regime.data.query reconstructs
adjusted prices from the action log at query time. See ARCHITECTURE.md §3.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

PARQUET_COMPRESSION = "zstd"


def data_root() -> Path:
    """Project data root; override via REGIME_DATA_ROOT env var."""
    return Path(os.environ.get("REGIME_DATA_ROOT", "data")).resolve()


def yfinance_raw_path(ticker: str, year: int) -> Path:
    return data_root() / "yfinance" / ticker / "raw" / f"{year}.parquet"


def yfinance_raw_dir(ticker: str) -> Path:
    return data_root() / "yfinance" / ticker / "raw"


def yfinance_actions_path(ticker: str) -> Path:
    return data_root() / "yfinance" / ticker / "actions.parquet"


def fred_path(series_id: str) -> Path:
    return data_root() / "fred" / f"{series_id}.parquet"


def fama_french_path() -> Path:
    return data_root() / "fama_french" / "factors.parquet"


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a Polars DataFrame to Parquet with deterministic settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sort_col = df.columns[0]
    df.sort(sort_col).write_parquet(
        path,
        compression=PARQUET_COMPRESSION,
        statistics=True,
    )


def read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)
