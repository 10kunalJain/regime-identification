"""Trend / mean-reversion features: RSI, distance from MAs, distance from rolling extrema."""

from __future__ import annotations

import polars as pl

from regime.features.registry import ComputeFn, Feature, register

RSI_PERIOD = 14
_MA_WINDOWS = (50, 200)
_HILO_WINDOW = 21


def _rsi_wilder(df: pl.DataFrame, period: int = RSI_PERIOD) -> pl.Series:
    """Wilder's RSI using exponential smoothing with alpha = 1/period.

    Returns values in [0, 100]; 100 on a strictly monotonic uptrend, 0 on a
    strictly monotonic downtrend, 50 when gains and losses balance.
    """
    delta = df["close"].diff()
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False)
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _make_ma_dist_fn(n: int) -> ComputeFn:
    def fn(df: pl.DataFrame) -> pl.Series:
        ma = df["close"].rolling_mean(window_size=n)
        sd = df["close"].rolling_std(window_size=n)
        return (df["close"] - ma) / sd

    return fn


def _dist_from_high(df: pl.DataFrame) -> pl.Series:
    """Fractional distance from rolling 21d high; <= 0."""
    rolling_max = df["close"].rolling_max(window_size=_HILO_WINDOW)
    return (df["close"] - rolling_max) / rolling_max


def _dist_from_low(df: pl.DataFrame) -> pl.Series:
    """Fractional distance from rolling 21d low; >= 0."""
    rolling_min = df["close"].rolling_min(window_size=_HILO_WINDOW)
    return (df["close"] - rolling_min) / rolling_min


register(
    Feature(
        name="rsi_14",
        fn=_rsi_wilder,
        inputs=("close",),
        window=RSI_PERIOD + 1,
        description="Wilder's RSI(14) on close-to-close changes.",
    )
)

for _n in _MA_WINDOWS:
    register(
        Feature(
            name=f"ma_dist_{_n}",
            fn=_make_ma_dist_fn(_n),
            inputs=("close",),
            window=_n,
            description=f"Z-score of close vs rolling {_n}-day SMA.",
        )
    )

register(
    Feature(
        name="dist_high_21d",
        fn=_dist_from_high,
        inputs=("close",),
        window=_HILO_WINDOW,
        description="Fractional distance from rolling 21-day high (≤ 0).",
    )
)

register(
    Feature(
        name="dist_low_21d",
        fn=_dist_from_low,
        inputs=("close",),
        window=_HILO_WINDOW,
        description="Fractional distance from rolling 21-day low (≥ 0).",
    )
)
