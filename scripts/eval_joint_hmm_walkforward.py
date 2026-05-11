"""Single-shot driver for the joint-HMM walk-forward evaluation.

Runs the expanding-window walk-forward with pre-registered defaults over
the full real-data history, prints the fold + crisis tables, and writes the
long-format posterior + crisis-lag + per-fold parquets under `build/eval/`.

Thin wrapper around `regime eval walkforward` — provided so the script
remains discoverable in `scripts/` alongside the existing `fit_*` /
`plot_*` scripts, and as a single entry point the Day 2 benchmark + Day 5
figures can `python` directly.

Usage:
    uv run python scripts/eval_joint_hmm_walkforward.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from regime.data.joint_dataset import build_wide_dataframe  # noqa: E402
from regime.eval.runner import (  # noqa: E402
    INITIAL_TRAIN_ROWS,
    REFIT_EVERY_ROWS,
    WalkForwardConfig,
    crisis_lag_to_dataframe,
    folds_to_dataframe,
    format_crisis_lag_table,
    format_fold_summary,
    run_joint_hmm_walkforward,
)
from regime.models.joint_hmm import JointHmm  # noqa: E402

OUTPUT_DIR = Path("build/eval")


def main() -> int:
    today = date.today()
    df = build_wide_dataframe(today)
    print(
        f"loaded {df.height} rows × {df.width} cols  "
        f"({df['data_time'].min()} → {df['data_time'].max()})"
    )

    obs_cols = tuple(c for c in df.columns if c.startswith("ret_"))
    fact_cols = tuple(c for c in df.columns if c.startswith("ff_"))
    print(f"  observation columns: {obs_cols}")
    print(f"  factor columns:      {fact_cols}")

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

    cfg = WalkForwardConfig(
        initial_train_rows=INITIAL_TRAIN_ROWS,
        refit_every_rows=REFIT_EVERY_ROWS,
    )
    print(
        f"\nrunning walk-forward (initial_train={cfg.initial_train_rows} rows,"
        f" refit_every={cfg.refit_every_rows} rows) ..."
    )
    result = run_joint_hmm_walkforward(df, factory, cfg)

    print("\nPer-fold metrics:")
    print(format_fold_summary(result.folds))
    print("\nPer-crisis detection lag (m5 anchor primary, m10 sensitivity):")
    print(format_crisis_lag_table(result.crisis_lag))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    posterior_path = OUTPUT_DIR / "walkforward_joint_hmm.parquet"
    crisis_path = OUTPUT_DIR / "walkforward_joint_hmm_crisis_lag.parquet"
    folds_path = OUTPUT_DIR / "walkforward_joint_hmm_folds.parquet"
    result.posterior.write_parquet(posterior_path)
    crisis_lag_to_dataframe(result.crisis_lag).write_parquet(crisis_path)
    folds_to_dataframe(result.folds).write_parquet(folds_path)
    print(f"\nwrote {posterior_path}  ({posterior_path.stat().st_size / 1024:.1f} KB)")
    print(f"wrote {crisis_path}")
    print(f"wrote {folds_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
