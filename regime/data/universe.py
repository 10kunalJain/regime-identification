"""Universe of tickers and series — locked per PLAN.md §2."""

from __future__ import annotations

from typing import Final

# Equity ETFs (18 tickers)
SPY: Final[str] = "SPY"
SECTOR_SPDRS: Final[tuple[str, ...]] = (
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
)
STYLE_FACTORS: Final[tuple[str, ...]] = (
    "MTUM",
    "QUAL",
    "USMV",
    "VLUE",
    "SIZE",
    "VTV",
)
DEFENSIVE: Final[tuple[str, ...]] = ("TLT",)

EQUITIES: Final[tuple[str, ...]] = (SPY, *SECTOR_SPDRS, *STYLE_FACTORS)

# Volatility complex (yfinance — note ^ prefix)
VOL_COMPLEX: Final[tuple[str, ...]] = ("^VIX", "^VIX3M", "^VVIX")

# All yfinance tickers
YF_TICKERS: Final[tuple[str, ...]] = (*EQUITIES, *DEFENSIVE, *VOL_COMPLEX)

# FRED series IDs
FRED_SERIES: Final[tuple[str, ...]] = (
    "DGS10",  # 10y Treasury yield
    "DGS2",  # 2y Treasury yield
    "T10Y2Y",  # 10y-2y slope
    "BAMLH0A0HYM2",  # HY OAS
    "DCOILWTICO",  # WTI crude
    "DEXUSEU",  # USD-EUR
    "DTWEXBGS",  # Broad USD index
)

# FRED publication lags (calendar days; conservative — refine to business days in v2)
FRED_PUB_LAG_DAYS: Final[dict[str, int]] = {
    "DGS10": 1,
    "DGS2": 1,
    "T10Y2Y": 1,
    "BAMLH0A0HYM2": 1,
    "DCOILWTICO": 1,
    "DEXUSEU": 1,
    "DTWEXBGS": 2,
}

# Fama-French factors (6: Mkt-RF, SMB, HML, RMW, CMA, Mom)
FAMA_FRENCH_FACTORS: Final[tuple[str, ...]] = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")
