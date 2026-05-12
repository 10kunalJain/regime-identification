"""Day-2 cross-method benchmark driver.

Runs the six registered methods (HmmGaussian, MsarT, SparseJumpModel,
JointHmm, Bocpd, WassersteinKmeans) through the walk-forward harness on the
full real-data 2003-2026 window, writes `build/benchmarks/methods.parquet`,
prints the 6×4 console summary, and dumps per-fold + per-crisis-lag
parquets for downstream consumption (Day 3 crisis head, Day 5 figures).

Usage:
    uv run python scripts/run_benchmark.py
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

load_dotenv()

# Per-method/per-fold progress logs from `regime.eval.runner`. Stream-flush each
# line so the in-flight wall-time table stays visible under tee / SIGKILL.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

from regime.data.joint_dataset import (  # noqa: E402
    FF_COLUMNS,
    OBSERVATION_TICKERS,
    RENAMED_FF,
    build_wide_dataframe,
)
from regime.eval.registry import (  # noqa: E402
    REGISTRY,
    UNIVARIATE_RETURN_COLUMN,
    UNIVARIATE_RV_COLUMN,
)
from regime.eval.runner import (  # noqa: E402
    INITIAL_TRAIN_ROWS,
    REFIT_EVERY_ROWS,
    WalkForwardConfig,
    format_cross_method_summary,
    method_crisis_lag_to_dataframe,
    method_folds_to_dataframe,
    run_cross_method_walkforward,
)

OUTPUT_DIR = Path("build/benchmarks")


def main() -> int:
    today = date.today()
    df_raw = build_wide_dataframe(today)
    df = df_raw.with_columns(
        pl.col(UNIVARIATE_RETURN_COLUMN).rolling_std(window_size=21).alias(UNIVARIATE_RV_COLUMN)
    ).drop_nulls()

    print(
        f"loaded {df.height} rows × {df.width} cols  "
        f"({df['data_time'].min()} → {df['data_time'].max()})"
    )

    obs_cols = tuple(f"ret_{t}" for t in OBSERVATION_TICKERS)
    factor_cols = tuple(RENAMED_FF[f] for f in FF_COLUMNS)
    print(f"  observation columns: {obs_cols}")
    print(f"  factor columns:      {factor_cols}")
    print(f"  methods:             {[m.name for m in REGISTRY]}")

    cfg = WalkForwardConfig(
        initial_train_rows=INITIAL_TRAIN_ROWS,
        refit_every_rows=REFIT_EVERY_ROWS,
    )
    print(
        f"\nrunning cross-method walk-forward (initial_train={cfg.initial_train_rows} rows, "
        f"refit_every={cfg.refit_every_rows} rows) ..."
    )
    result = run_cross_method_walkforward(
        df,
        obs_cols=obs_cols,
        factor_cols=factor_cols,
        config=cfg,
    )

    print("\nPer-method summary (6×4):")
    print(format_cross_method_summary(result.summary))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    posterior_path = OUTPUT_DIR / "methods.parquet"
    folds_path = OUTPUT_DIR / "methods_folds.parquet"
    crisis_path = OUTPUT_DIR / "methods_crisis_lag.parquet"
    summary_path = OUTPUT_DIR / "methods_summary.parquet"

    result.posterior.write_parquet(posterior_path)
    method_folds_to_dataframe(result.folds).write_parquet(folds_path)
    method_crisis_lag_to_dataframe(result.crisis_lag).write_parquet(crisis_path)
    result.summary.write_parquet(summary_path)

    print(f"\nwrote {posterior_path}  ({posterior_path.stat().st_size / 1024:.1f} KB)")
    print(f"wrote {folds_path}")
    print(f"wrote {crisis_path}")
    print(f"wrote {summary_path}")

    n_folds_per_method = (
        method_folds_to_dataframe(result.folds).group_by("method").agg(pl.len().alias("n_folds"))
    )
    print(
        "\nfolds-per-method sanity:\n  "
        + n_folds_per_method.to_pandas().to_string(index=False).replace("\n", "\n  ")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
