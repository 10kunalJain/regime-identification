"""Feature pipeline.

Two entrypoints:

  - `build_features_for_ticker(ticker, t)` — per-ticker registry features,
    operating on PIT-correct OHLCV from `regime.data.query.as_of`.
  - `build_features(t)` — master orchestrator that joins per-ticker features
    across the full equity universe with cross-sectional, VIX, macro, and
    Fama-French features (each from its own builder).

PIT contract is inherited from the underlying queries; this module never reads
raw partitions directly. Any leakage in a feature definition would show up as
the feature value at `t` changing when data after `t` is added — see
`tests/property/test_features.py` for the explicit checks.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from regime.data.query import as_of
from regime.data.universe import DEFENSIVE, EQUITIES
from regime.features.definitions.dispersion import build_dispersion_features
from regime.features.definitions.factors import build_factor_features
from regime.features.definitions.macro import build_macro_features
from regime.features.definitions.vix import build_vix_features
from regime.features.registry import all_features


def build_features_for_ticker(ticker: str, t: date) -> pl.DataFrame:
    """Return a wide DataFrame of all registered per-ticker features for `ticker` as of `t`.

    Columns: data_time, knowledge_time, plus one column per registered feature.
    Rows are sorted ascending by data_time. Early rows have null feature values
    where insufficient history is available for the rolling window.
    """
    raw = as_of(ticker, t)
    if raw.is_empty():
        return pl.DataFrame(
            schema={
                "data_time": pl.Date,
                "knowledge_time": pl.Date,
                **{f.name: pl.Float64 for f in all_features()},
            }
        )

    out = raw.select("data_time", "knowledge_time")
    for feature in all_features():
        try:
            values = feature.fn(raw)
        except Exception as exc:
            raise RuntimeError(f"feature {feature.name!r} failed at t={t}") from exc
        out = out.with_columns(values.alias(feature.name).cast(pl.Float64))

    return out.sort("data_time")


def build_features(t: date) -> pl.DataFrame:
    """Master pipeline: every feature, every ticker, as of `t`.

    Returns a wide DataFrame with rows = (data_time, ticker) and columns =
    per-ticker registered features + cross-sectional dispersion + VIX complex
    + macro (FRED) + Fama-French factors. Cross-sectional / global features are
    broadcast to every ticker via left join on `data_time`.
    """
    ticker_dfs: list[pl.DataFrame] = []
    for ticker in (*EQUITIES, *DEFENSIVE):
        df = build_features_for_ticker(ticker, t)
        if df.is_empty():
            continue
        df = df.with_columns(pl.lit(ticker).alias("ticker"))
        ticker_dfs.append(df)

    if not ticker_dfs:
        return pl.DataFrame()

    base = pl.concat(ticker_dfs, how="vertical_relaxed")

    cs = build_dispersion_features(t)
    if not cs.is_empty():
        base = base.join(cs, on="data_time", how="left")

    vix = build_vix_features(t)
    if not vix.is_empty():
        base = base.join(vix, on="data_time", how="left")

    macro = build_macro_features(t)
    if not macro.is_empty():
        base = base.join(macro, on="data_time", how="left")

    ff = build_factor_features(t)
    if not ff.is_empty():
        base = base.join(ff, on="data_time", how="left")

    return base.sort(["data_time", "ticker"])
