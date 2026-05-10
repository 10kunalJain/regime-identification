"""Return features: log total returns at multiple horizons."""

from __future__ import annotations

import polars as pl

from regime.features.registry import ComputeFn, Feature, log_total_return, register

_RETURN_HORIZONS = (1, 5, 21, 63)


def _make_return_fn(n: int) -> ComputeFn:
    def fn(df: pl.DataFrame) -> pl.Series:
        # Multi-horizon log return = rolling sum of single-day log returns.
        return log_total_return(df).rolling_sum(window_size=n)

    return fn


for _n in _RETURN_HORIZONS:
    register(
        Feature(
            name=f"ret_{_n}d",
            fn=_make_return_fn(_n),
            inputs=("close", "dividend_amount"),
            window=_n + 1,  # +1 for the lag in log_total_return
            description=f"Log total return over {_n} trading day(s).",
        )
    )
