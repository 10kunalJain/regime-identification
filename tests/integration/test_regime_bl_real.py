"""Integration test: regime-BL Sharpe under stress costs is at least 60% of
Sharpe under central costs.

Sanity bar (not headline) — the strategy must remain meaningful when the
pessimistic-stress cost column is applied; a strategy that collapses entirely
under realistic-worst trading frictions isn't shippable, regardless of how
strong its central-case backtest looks.

Guard: if central Sharpe is below `MIN_CENTRAL_SHARPE` (i.e. small/zero/
negative), the 60% ratio rule degenerates ("ratio passes vacuously") so we
skip — central performance is the precondition to even ask the cost-stress
question. The skip is loud so it surfaces in CI logs.

Truncated 8-y window — uses the parquet inputs as-is from the latest
`scripts/backtest_regime_bl.py` run; no in-test recomputation of the
full backtest. Slow-marked and skipped when the parquets are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from regime.backtest.metrics import performance_metrics

CENTRAL_PARQUET = Path("build/backtests/regime_bl_central.parquet")
STRESS_PARQUET = Path("build/backtests/regime_bl_stress.parquet")

MIN_CENTRAL_SHARPE = 0.20
STRESS_SHARPE_FLOOR_FRACTION = 0.60


@pytest.mark.slow
def test_stress_sharpe_at_least_60pct_of_central_sharpe() -> None:
    if not CENTRAL_PARQUET.exists() or not STRESS_PARQUET.exists():
        pytest.skip(
            "regime-BL backtest parquets missing; run "
            "scripts/backtest_regime_bl.py first"
        )

    central_nav = pl.read_parquet(CENTRAL_PARQUET)["nav"].to_numpy().astype(np.float64)
    stress_nav = pl.read_parquet(STRESS_PARQUET)["nav"].to_numpy().astype(np.float64)
    assert central_nav.shape == stress_nav.shape, (
        f"central and stress NAV shapes differ: {central_nav.shape} vs {stress_nav.shape}"
    )

    central_metrics = performance_metrics(central_nav)
    stress_metrics = performance_metrics(stress_nav)

    if central_metrics.sharpe < MIN_CENTRAL_SHARPE:
        pytest.skip(
            f"central Sharpe {central_metrics.sharpe:.3f} below precondition "
            f"floor {MIN_CENTRAL_SHARPE} — the 60% rule is meaningless when "
            "central performance is itself unsatisfactory"
        )

    floor = STRESS_SHARPE_FLOOR_FRACTION * central_metrics.sharpe
    assert stress_metrics.sharpe >= floor, (
        f"stress Sharpe {stress_metrics.sharpe:.3f} below "
        f"{int(STRESS_SHARPE_FLOOR_FRACTION * 100)}% of central Sharpe "
        f"{central_metrics.sharpe:.3f} (floor = {floor:.3f}). "
        f"central MDD={central_metrics.max_drawdown:.3f}, "
        f"stress MDD={stress_metrics.max_drawdown:.3f}"
    )
