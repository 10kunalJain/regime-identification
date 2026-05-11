"""Integration test: joint-HMM walk-forward on real data, 2003-2010.

Window covers the GFC and the Aug 2007 quant quake — the two earliest events
in the canonical crisis registry. Three expanding folds (test years 2008,
2009, 2010). Integer per-crisis detection-lag values are committed as a JSON
snapshot under `tests/snapshots/walkforward_joint_hmm_lag.json` and asserted
bit-exact on every run.

When the numbers shift (intentionally or otherwise), CI fails loudly with a
diff. Regenerate explicitly with:

    uv run pytest tests/integration/test_walkforward_real.py --regenerate-snapshots

…and document the reason in the commit message.

This test is opt-out under `slow` and self-skips when the PIT data store is
empty (e.g. fresh CI checkout that hasn't run `regime data refresh`).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from regime.data import store
from regime.data.joint_dataset import build_wide_dataframe
from regime.eval.runner import (
    WalkForwardConfig,
    run_joint_hmm_walkforward,
)
from regime.models.joint_hmm import JointHmm

SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent / "snapshots" / "walkforward_joint_hmm_lag.json"
)
WINDOW_END = date(2010, 12, 31)
WINDOW_START = date(2003, 1, 1)


def _data_available() -> bool:
    """Skip the integration test if the PIT store isn't populated."""
    try:
        root = store.data_root()
    except Exception:
        return False
    return root.exists() and any(root.iterdir())


@pytest.mark.slow
def test_detection_lag_snapshot_real_data(regenerate_snapshots: bool) -> None:
    if not _data_available():
        pytest.skip("PIT data store unavailable; run `uv run regime data refresh` first")

    df = build_wide_dataframe(WINDOW_END)
    df = df.filter(
        (df["data_time"] >= WINDOW_START) & (df["data_time"] <= WINDOW_END)
    )
    assert df.height > 1500, f"expected >1500 trading days in 2003-2010, got {df.height}"

    obs_cols = tuple(c for c in df.columns if c.startswith("ret_"))
    fact_cols = tuple(c for c in df.columns if c.startswith("ff_"))

    def factory() -> JointHmm:
        return JointHmm(
            K=3,
            observation_columns=obs_cols,
            factor_columns=fact_cols,
            latent_factor_rank=3,
            n_restarts=3,
            max_iter=50,
            random_state=42,
        )

    cfg = WalkForwardConfig(initial_train_rows=1260, refit_every_rows=252)
    result = run_joint_hmm_walkforward(df, factory, cfg)

    observed = {
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "n_folds": len(result.folds),
        "n_test_rows": int(result.posterior.height),
        "crisis_lag": [
            {
                "crisis_name": c.crisis_name,
                "in_eval_window": c.in_eval_window,
                "first_fire_date": (
                    c.first_fire_date.isoformat() if c.first_fire_date is not None else None
                ),
                "lag_m5": c.lag_m5,
                "lag_m10": c.lag_m10,
            }
            for c in result.crisis_lag
        ],
    }

    if regenerate_snapshots:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(observed, indent=2) + "\n")
        pytest.skip(f"regenerated snapshot at {SNAPSHOT_PATH}")

    assert SNAPSHOT_PATH.exists(), (
        f"snapshot file missing at {SNAPSHOT_PATH}. "
        "Run with --regenerate-snapshots to create it."
    )
    expected = json.loads(SNAPSHOT_PATH.read_text())
    assert observed == expected, (
        "detection-lag snapshot mismatch.\n"
        f"  expected: {json.dumps(expected, indent=2)}\n"
        f"  observed: {json.dumps(observed, indent=2)}\n"
        "If the change is intentional, re-run with --regenerate-snapshots "
        "and explain why in the commit message."
    )
