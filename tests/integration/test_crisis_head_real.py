"""Integration test: ensemble crisis head's PR-AUC strictly improves over the
best single method on the walk-forward OOF surface.

Why PR-AUC and not Brier (per the user's chosen acceptance for Day 3):
Day-2's change-point methods (BOCPD, Wasserstein k-means) carry a per-fold
supervised logistic head inside their `crisis_score`. That per-fold scorer is
refit at every walk-forward boundary; it's a moving target the Day-3
ensemble — a single fit-then-predict head — can't reliably beat on point
Brier. Ranking quality (PR-AUC) is where the ensemble's value-add over single
methods is real and measurable. Brier is reported in the script's console
table per plan, just not asserted here.

Evaluation surface — walk-forward, pooled across folds:
  - For each Day-2 fold f, fit a `CrisisHead` on rows strictly before f's
    first `data_time`; predict (uncalibrated LR raw probability) on f's rows.
  - Concatenate raw OOF predictions across all valid folds.
  - Fit one pooled isotonic calibrator on the pooled raw OOFs vs labels.
  - Score the calibrated predictions against single methods' `crisis_score`
    on the same observable rows.

Slow-marked. Skipped automatically when `build/benchmarks/methods.parquet`
is absent (e.g., fresh CI checkout without `scripts/run_benchmark.py`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.isotonic import IsotonicRegression

from regime.ensemble.calibration import brier_score, pr_auc
from regime.ensemble.crisis_head import CrisisHead, assemble_feature_matrix
from regime.eval.labels import UNOBSERVABLE

METHODS_PARQUET = Path("build/benchmarks/methods.parquet")
MIN_TRAIN_ROWS = 100
MIN_TRAIN_POSITIVES = 3


@pytest.mark.slow
def test_ensemble_pr_auc_strictly_improves_over_best_single() -> None:
    if not METHODS_PARQUET.exists():
        pytest.skip(f"{METHODS_PARQUET} missing; run scripts/run_benchmark.py first")

    posterior = pl.read_parquet(METHODS_PARQUET)
    matrix = assemble_feature_matrix(posterior)
    fold_to_dates = {
        int(fid): set(
            posterior.filter(pl.col("fold_id") == fid)["data_time"].unique().to_list()
        )
        for fid in posterior["fold_id"].unique().to_list()
    }

    p_raw = _walk_forward_raw_predictions(matrix, fold_to_dates)
    finite = ~np.isnan(p_raw)
    obs = finite & (matrix.y != UNOBSERVABLE)
    assert obs.any(), "no rows covered by walk-forward predictions"

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw[obs], matrix.y[obs].astype(np.float64))
    p_calibrated = np.full_like(p_raw, np.nan)
    p_calibrated[finite] = iso.transform(p_raw[finite])

    y_obs = matrix.y[obs].astype(np.int64)
    ens_pr_auc = pr_auc(p_calibrated[obs], y_obs)
    ens_brier = brier_score(p_calibrated[obs], y_obs.astype(np.float64))

    method_metrics = _per_method_pooled_metrics(posterior, matrix.data_times, obs, y_obs)
    best_pr_name, best_pr = max(
        method_metrics.items(), key=lambda kv: kv[1]["pr_auc"]
    )
    best_pr_value = best_pr["pr_auc"]
    best_brier_name, best_brier = min(
        method_metrics.items(), key=lambda kv: kv[1]["brier"]
    )

    assert ens_pr_auc > best_pr_value, (
        f"ensemble walk-forward PR-AUC {ens_pr_auc:.4f} does not strictly "
        f"improve over best single method {best_pr_name} ({best_pr_value:.4f}). "
        f"Brier reported (not asserted): ensemble {ens_brier:.4f} vs "
        f"best single by Brier {best_brier_name} ({best_brier['brier']:.4f}). "
        f"Per-method metrics: {method_metrics}"
    )


def _walk_forward_raw_predictions(
    matrix, fold_to_dates: dict[int, set]
) -> np.ndarray:
    """For each fold, fit CrisisHead on rows strictly before the fold and emit
    the LR raw (uncalibrated) probability on the fold's rows. Returns NaN
    where the fold's training set is too small or has insufficient positives."""
    p_raw = np.full(matrix.X.shape[0], np.nan, dtype=np.float64)
    for fid in sorted(fold_to_dates):
        held_mask = np.array(
            [d in fold_to_dates[fid] for d in matrix.data_times], dtype=bool
        )
        held_dates = [d for d, k in zip(matrix.data_times, held_mask, strict=True) if k]
        if not held_dates:
            continue
        min_held = min(held_dates)
        train_mask = np.array([d < min_held for d in matrix.data_times], dtype=bool)
        if train_mask.sum() < MIN_TRAIN_ROWS:
            continue
        if int((matrix.y[train_mask] == 1).sum()) < MIN_TRAIN_POSITIVES:
            continue
        head = CrisisHead(n_calibration_splits=5, max_iter=1000, random_state=42)
        head.fit(matrix.X[train_mask], matrix.y[train_mask])
        p_raw[held_mask] = head.predict_raw(matrix.X[held_mask])
    return p_raw


def _per_method_pooled_metrics(
    posterior: pl.DataFrame,
    data_times: list,
    obs: np.ndarray,
    y_obs: np.ndarray,
) -> dict[str, dict[str, float]]:
    """For each method, align crisis_score against the same data_times the
    ensemble was scored on; compute pooled Brier + PR-AUC on observable rows."""
    keep_dates = [d for d, m in zip(data_times, obs, strict=True) if m]
    keep_frame = pl.DataFrame({"data_time": keep_dates})
    out: dict[str, dict[str, float]] = {}
    for name in sorted(posterior["method"].unique().to_list()):
        sub = (
            posterior.filter(pl.col("method") == name)
            .select(["data_time", "crisis_score"])
            .sort("data_time")
        )
        joined = keep_frame.join(sub, on="data_time", how="inner").sort("data_time")
        scores = joined["crisis_score"].to_numpy()
        if scores.shape[0] != len(keep_dates):
            raise RuntimeError(
                f"{name}: aligned to {scores.shape[0]} rows, expected {len(keep_dates)}"
            )
        out[name] = {
            "brier": brier_score(scores, y_obs.astype(np.float64)),
            "pr_auc": pr_auc(scores, y_obs),
        }
    return out
