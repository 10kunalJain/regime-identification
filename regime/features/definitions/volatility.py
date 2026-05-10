"""Realized-volatility features."""

from __future__ import annotations

import math

import polars as pl

from regime.features.registry import ComputeFn, Feature, log_total_return, register

ANNUALIZATION = 252
_RV_WINDOWS = (5, 21, 63)


def _make_rv_fn(n: int) -> ComputeFn:
    def fn(df: pl.DataFrame) -> pl.Series:
        r = log_total_return(df)
        return r.rolling_std(window_size=n) * math.sqrt(ANNUALIZATION)

    return fn


def _vol_of_vol_21d(df: pl.DataFrame) -> pl.Series:
    r = log_total_return(df)
    rv5 = r.rolling_std(window_size=5) * math.sqrt(ANNUALIZATION)
    return rv5.rolling_std(window_size=21)


for _n in _RV_WINDOWS:
    register(
        Feature(
            name=f"rv_{_n}d",
            fn=_make_rv_fn(_n),
            inputs=("close", "dividend_amount"),
            window=_n + 1,
            description=f"Annualized realized volatility over {_n}d (using log total returns).",
        )
    )

register(
    Feature(
        name="vol_of_vol_21d",
        fn=_vol_of_vol_21d,
        inputs=("close", "dividend_amount"),
        window=5 + 21 + 1,
        description="21d std of rolling 5d realized volatility (vol of vol).",
    )
)
