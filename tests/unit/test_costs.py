"""Tests for the cost model (central + stress)."""

from __future__ import annotations

import pytest

from regime.backtest.costs import (
    CostModel,
    central_cost_model,
    stress_cost_model,
    tier_for_ticker,
)


def test_tier_assignment_for_universe():
    assert tier_for_ticker("SPY") == "spy"
    assert tier_for_ticker("XLK") == "sector"
    assert tier_for_ticker("MTUM") == "factor"
    assert tier_for_ticker("TLT") == "tlt"
    # Unknowns default to factor (most conservative).
    assert tier_for_ticker("ZZZZ") == "factor"


def test_central_cheaper_than_stress_for_same_trade():
    central = central_cost_model()
    stress = stress_cost_model()
    c_cost = central.trade_cost_bps("SPY", 1_000_000.0, 0.01, 1e9)
    s_cost = stress.trade_cost_bps("SPY", 1_000_000.0, 0.01, 1e9)
    assert s_cost > c_cost


def test_costs_increase_with_trade_size():
    """Square-root impact → larger trades cost more bps."""
    central = central_cost_model()
    small = central.trade_cost_bps("SPY", 10_000.0, 0.01, 1e9)
    big = central.trade_cost_bps("SPY", 1_000_000_000.0, 0.01, 1e9)
    assert big > small


def test_factor_etfs_cost_more_than_spy():
    central = central_cost_model()
    spy_cost = central.trade_cost_bps("SPY", 1e6, 0.01, 1e9)
    factor_cost = central.trade_cost_bps("MTUM", 1e6, 0.01, 1e8)
    # Factor ETF spread is 4 vs SPY 0.5 → factor cost > SPY cost.
    assert factor_cost > spy_cost


def test_zero_trade_notional_zero_impact():
    central = central_cost_model()
    cost = central.trade_cost_bps("SPY", 0.0, 0.01, 1e9)
    # Commission + half-spread but no impact.
    assert cost == pytest.approx(central.commission_bps + central.half_spread_bps_by_tier["spy"])


def test_cost_model_is_immutable():
    central = central_cost_model()
    # Frozen dataclass — assignment must raise FrozenInstanceError.
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        central.commission_bps = 99.0  # type: ignore[misc]


def test_custom_cost_model():
    custom = CostModel(
        name="custom",
        commission_bps=2.0,
        half_spread_bps_by_tier={"sector": 5.0, "spy": 2.0, "factor": 7.0, "tlt": 3.0},
        impact_eta=0.5,
    )
    cost = custom.trade_cost_bps("XLK", 1e5, 0.012, 1e8)
    assert cost > custom.commission_bps + custom.half_spread_bps_by_tier["sector"]


def test_central_and_stress_pre_registered_values_match_spec():
    """Pin the pre-registered Q9 values so silent edits trip CI."""
    central = central_cost_model()
    assert central.commission_bps == 0.5
    assert central.half_spread_bps_by_tier == {
        "spy": 0.5,
        "sector": 1.5,
        "factor": 4.0,
        "tlt": 1.0,
    }
    assert central.impact_eta == 0.15

    stress = stress_cost_model()
    assert stress.commission_bps == 1.0
    assert stress.half_spread_bps_by_tier == {
        "spy": 1.5,
        "sector": 4.5,
        "factor": 12.0,
        "tlt": 3.0,
    }
    assert stress.impact_eta == 1.0
