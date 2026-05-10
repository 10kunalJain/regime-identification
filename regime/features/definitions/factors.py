"""Fama-French daily factor features and their 21-day realized vols."""

from __future__ import annotations

import math
from datetime import date

import polars as pl

from regime.data.query import as_of_fama_french

ANNUALIZATION = 252
RV_WINDOW = 21
FF_FACTORS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")


def _normalize(name: str) -> str:
    return name.lower().replace("-", "_")


def _empty_schema() -> dict[str, type[pl.DataType]]:
    schema: dict[str, type[pl.DataType]] = {"data_time": pl.Date}
    for f in FF_FACTORS:
        col = _normalize(f)
        schema[f"ff_{col}"] = pl.Float64
        schema[f"ff_{col}_rv_21d"] = pl.Float64
    return schema


def build_factor_features(t: date) -> pl.DataFrame:
    """Fama-French daily factor returns and their 21-day realized vols."""
    df = as_of_fama_french(t)
    if df.is_empty():
        return pl.DataFrame(schema=_empty_schema())

    out = df.select("data_time")
    for f in FF_FACTORS:
        if f not in df.columns:
            continue
        col = _normalize(f)
        ret = df[f]
        rv = ret.rolling_std(window_size=RV_WINDOW) * math.sqrt(ANNUALIZATION)
        out = out.with_columns(
            ret.alias(f"ff_{col}"),
            rv.alias(f"ff_{col}_rv_21d"),
        )
    return out.sort("data_time")
