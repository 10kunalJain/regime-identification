"""Multi-asset portfolio returns + ADV for backtest consumers.

Builds the wide arithmetic-return dataframe over the 18-ETF universe locked
in `regime.data.universe` (SPY + 11 sector SPDRs + 6 style factor ETFs + TLT),
plus the raw 21-day notional ADV per asset needed by the cost model's impact
term. Pure PIT — every read goes through `regime.data.query.as_of`.

The Day-2 `joint_dataset.build_wide_dataframe` is the joint-HMM dataset and
covers a different (smaller, factor-flavoured) ETF subset; this builder is
the portfolio-side companion.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from regime.data.query import as_of
from regime.data.universe import DEFENSIVE, EQUITIES

PORTFOLIO_TICKERS: tuple[str, ...] = (*EQUITIES, *DEFENSIVE)


def build_portfolio_returns(
    t: date, tickers: tuple[str, ...] = PORTFOLIO_TICKERS
) -> pl.DataFrame:
    """Wide arithmetic-return + notional-volume dataframe for `tickers`.

    Columns: `data_time`, then for each ticker:
      - `ret_<ticker>`: arithmetic total return, i.e. (close + dividend) /
        close_{t-1} - 1.  The backtest engine consumes these directly.
      - `notional_<ticker>`: close × volume, in dollars (the cost model's
        ADV21 input — caller is expected to rolling-mean over 21 days).

    Joins are inner across all tickers + drop_nulls applied, so the output
    starts on the first trading day where every ETF has both a return and a
    notional. Rows are sorted by `data_time`. Default `tickers` is the full
    19-ETF universe; callers wanting a longer common history (e.g., the
    Day-4 backtest dropping XLC/XLRE which started 2018/2015) should pass a
    subset and document the choice at the call site.
    """
    frames: list[pl.DataFrame] = []
    for ticker in tickers:
        ohlcv = as_of(ticker, t)
        if ohlcv.is_empty():
            raise ValueError(f"no PIT data for {ticker!r} as of {t}")
        ret_expr = (
            (pl.col("close") + pl.col("dividend_amount")) / pl.col("close").shift(1) - 1.0
        ).alias(f"ret_{ticker}")
        notional_expr = (pl.col("close") * pl.col("volume").cast(pl.Float64)).alias(
            f"notional_{ticker}"
        )
        sub = ohlcv.with_columns(ret_expr, notional_expr).select(
            "data_time", f"ret_{ticker}", f"notional_{ticker}"
        )
        frames.append(sub)

    combined = frames[0]
    for df in frames[1:]:
        combined = combined.join(df, on="data_time", how="inner")
    return combined.drop_nulls().sort("data_time")


__all__ = ["PORTFOLIO_TICKERS", "build_portfolio_returns"]
