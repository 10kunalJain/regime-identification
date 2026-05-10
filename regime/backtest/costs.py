"""Cost model — central + pessimistic-stress columns per Q9 lock.

Each parameter is tagged with its empirical source. Both columns are reported
in every backtest result (the design grill's Q9: "headline strategy must beat
benchmarks under *both* central and stress costs").

Cost components:
  - commission: per trade, flat bps
  - half_spread_bps: tier-dependent (SPY < sectors < factor ETFs)
  - impact: Frazzini-Israel-Moskowitz square-root law,
       Δp / p (bps) = η * σ_daily * sqrt(Q / V_daily) * 1e4

Tier assignment is per ticker via `tier_for_ticker`. Default tiers in this
project's universe:
  - "spy"   → SPY (high liquidity)
  - "sector"→ XLB/XLC/XLE/XLF/XLI/XLK/XLP/XLRE/XLU/XLV/XLY (sector SPDRs)
  - "factor"→ MTUM, QUAL, USMV, VLUE, SIZE, VTV
  - "tlt"   → TLT
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ----- tier mapping for the project universe -----

_DEFAULT_TIER_MAP: dict[str, str] = {
    "SPY": "spy",
    **{
        t: "sector"
        for t in ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")
    },
    **{t: "factor" for t in ("MTUM", "QUAL", "USMV", "VLUE", "SIZE", "VTV")},
    "TLT": "tlt",
}


def tier_for_ticker(ticker: str, tier_map: dict[str, str] | None = None) -> str:
    """Return the cost-tier label for a ticker. Unknowns default to 'factor' (highest spread)."""
    m = tier_map or _DEFAULT_TIER_MAP
    return m.get(ticker, "factor")


@dataclass(frozen=True)
class CostModel:
    """A complete cost-model parameterization.

    Both central and stress instances share this shape; differences are in
    `commission_bps`, `half_spread_bps_by_tier`, `impact_eta`, and
    `volume_floor_multiplier`.
    """

    name: str  # "central" or "stress"
    commission_bps: float  # source: see docstring
    half_spread_bps_by_tier: dict[str, float] = field(default_factory=dict)
    impact_eta: float = 0.15  # Frazzini-Israel-Moskowitz 2018, Table 4
    volume_floor_multiplier: float = 1.0  # max(ADV21, multiplier × creation_unit)
    tier_map: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_TIER_MAP))

    def trade_cost_bps(
        self,
        ticker: str,
        trade_notional: float,
        daily_vol: float,
        adv21_notional: float,
    ) -> float:
        """Total round-trip cost in bps for a single one-way trade."""
        tier = tier_for_ticker(ticker, self.tier_map)
        commission = self.commission_bps
        half_spread = self.half_spread_bps_by_tier.get(tier, 5.0)
        v_floor = max(adv21_notional, self.volume_floor_multiplier * 1.0)
        if v_floor <= 0.0 or trade_notional <= 0.0:
            impact_bps = 0.0
        else:
            impact_bps = self.impact_eta * daily_vol * np.sqrt(trade_notional / v_floor) * 1e4
        return float(commission + half_spread + impact_bps)


# ----- pre-registered parameter sets per STRATEGY_HYPERPARAMETERS.md §8 -----


def central_cost_model() -> CostModel:
    """Central case — best estimate from cited industry / academic sources.

    Sources (committed in STRATEGY_HYPERPARAMETERS.md §8):
      - Commission: Interactive Brokers tiered ETF pricing for >$1M ADV ETFs.
      - Half-spreads: NYSE TAQ-derived peer-reviewed median quoted half-spread.
      - Impact η: Frazzini, Israel & Moskowitz (2018) "Trading Costs," Table 4.
      - Volume floor: 1× creation-unit notional, accounts for AP-mediated liquidity.
    """
    return CostModel(
        name="central",
        commission_bps=0.5,
        half_spread_bps_by_tier={
            "spy": 0.5,
            "sector": 1.5,
            "factor": 4.0,
            "tlt": 1.0,
        },
        impact_eta=0.15,
        volume_floor_multiplier=1.0,
    )


def stress_cost_model() -> CostModel:
    """Pessimistic stress case — 3× spreads, η at literature worst end, no V floor."""
    return CostModel(
        name="stress",
        commission_bps=1.0,
        half_spread_bps_by_tier={
            "spy": 1.5,
            "sector": 4.5,
            "factor": 12.0,
            "tlt": 3.0,
        },
        impact_eta=1.0,
        volume_floor_multiplier=0.0,
    )
