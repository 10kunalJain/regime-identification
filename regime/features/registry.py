"""Feature registry.

Every feature is a `Feature` instance registered in `REGISTRY`. The pipeline
iterates the registry and computes each feature against PIT-correct input data.
New features go through this registry; never compute features ad-hoc inside
model code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

ComputeFn = Callable[[pl.DataFrame], pl.Series]


@dataclass(frozen=True)
class Feature:
    """Definition of a single feature.

    Attributes:
        name: unique identifier, also used as the output column name.
        fn: callable taking a per-ticker OHLCV DataFrame (with `dividend_amount`
            column attached by the PIT query layer) and returning a Series
            aligned to the input rows.
        inputs: the columns of the input DataFrame this feature reads. Used in
            FEATURES.md to document dependencies; not enforced at runtime.
        window: rolling window size in trading days; 0 for non-rolling features.
            Used to assert that early rows are null and to gate stationarity
            tests until enough history accumulates.
        description: one-line human-readable description.
    """

    name: str
    fn: ComputeFn
    inputs: tuple[str, ...]
    window: int
    description: str


REGISTRY: dict[str, Feature] = {}


def register(feature: Feature) -> Feature:
    """Add a feature to the registry; raise if a name is already registered."""
    if feature.name in REGISTRY:
        raise ValueError(f"feature {feature.name!r} already registered")
    REGISTRY[feature.name] = feature
    return feature


def all_features() -> list[Feature]:
    return list(REGISTRY.values())


def log_total_return(df: pl.DataFrame) -> pl.Series:
    """Single-day log total return: log((P_t + D_t) / P_{t-1}).

    Treats dividends as part of total return — matches the PIT query layer's
    contract that `dividend_amount` is attached on ex-date and is on the same
    share basis as the (split-adjusted) prices.
    """
    p = df["close"]
    d = df["dividend_amount"]
    return ((p + d) / p.shift(1)).log()
