"""Day-3 driver: fit the ensemble crisis head on real-data OOF posteriors.

Consumes `build/benchmarks/methods.parquet` (Day 2), pivots to a 6-feature
matrix (one `crisis_score` per method — see `assemble_feature_matrix` for the
rationale), and **walk-forward refits** a `CrisisHead` at every Day-2 fold
boundary: train on rows strictly before the fold, predict on the fold, move
on. Aggregating those predictions gives a walk-forward OOF series with no
future leakage — the honest evaluation surface for an ensemble that consumes
already-walk-forward per-method scorers (Day-2's change-point methods carry
a per-fold supervised head inside their `crisis_score`).

Reliability + ECE come from the walk-forward OOF predictions, as do the
Brier and PR-AUC head-to-head numbers against each single method on the
same observable rows.

Outputs:
  build/benchmarks/crisis_head.parquet — walk-forward P(crisis) per `data_time`.
  build/benchmarks/calibration.png    — reliability curve + PR curve.

Usage:
    uv run python scripts/fit_crisis_head.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dotenv import load_dotenv

load_dotenv()

from regime.ensemble.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    positive_base_rate,
    pr_auc,
    reliability_curve,
)
from regime.ensemble.crisis_head import (  # noqa: E402
    CrisisHead,
    assemble_feature_matrix,
)
from regime.eval.labels import UNOBSERVABLE  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

INPUT = Path("build/benchmarks/methods.parquet")
OUT_DIR = Path("build/benchmarks")
OUT_PARQUET = OUT_DIR / "crisis_head.parquet"
OUT_PNG = OUT_DIR / "calibration.png"


def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: {INPUT} missing — run scripts/run_benchmark.py first.")
        return 1

    posterior = pl.read_parquet(INPUT)
    print(f"loaded {posterior.height} rows from {INPUT}")

    matrix = assemble_feature_matrix(posterior)
    print(
        f"feature matrix: X={matrix.X.shape}, "
        f"observable rows={int((matrix.y != UNOBSERVABLE).sum())}/{matrix.X.shape[0]}, "
        f"feature columns={len(matrix.feature_names)}"
    )

    oof_calibrated, oof_raw = _walk_forward_oof(posterior, matrix)
    covered = ~np.isnan(oof_calibrated)
    obs = covered & (matrix.y != UNOBSERVABLE)
    n_obs = int(obs.sum())
    n_pos = int((matrix.y[obs] == 1).sum())

    y_obs = matrix.y[obs].astype(np.int64)
    p_obs = oof_calibrated[obs]
    base_rate = positive_base_rate(y_obs)
    ens_brier = brier_score(p_obs, y_obs.astype(np.float64))
    ens_pr_auc = pr_auc(p_obs, y_obs)
    ens_ece = expected_calibration_error(p_obs, y_obs.astype(np.float64))

    per_method = _per_method_pooled_metrics(posterior, matrix.data_times, y_obs, obs)
    _print_comparison(per_method, ens_brier, ens_pr_auc, ens_ece, base_rate, n_obs, n_pos)
    _check_reliability(p_obs, y_obs)
    _check_pr_auc_acceptance(per_method, ens_pr_auc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_calibration(p_obs=p_obs, y_obs=y_obs, per_method=per_method, out_path=OUT_PNG)
    _write_parquet(matrix.data_times, matrix.y, oof_raw, oof_calibrated, OUT_PARQUET)
    print(f"wrote {OUT_PARQUET}")
    print(f"wrote {OUT_PNG}")
    return 0


def _walk_forward_oof(
    posterior: pl.DataFrame,
    matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward refit + pooled isotonic calibration.

    Stage 1 — for each Day-2 fold, fit a `CrisisHead` on rows strictly before
    the fold's first `data_time` and predict the *uncalibrated* (LR raw)
    probability on the fold's rows. This produces an honest walk-forward LR
    OOF series with no future leakage.

    Stage 2 — fit one isotonic calibrator across the pooled OOF series so the
    final calibrated probabilities are monotonic at the evaluation surface
    (per-fold isotonics, when concatenated, are *not* jointly monotonic and
    fail the plan's reliability check). The pooled isotonic uses the
    observable labels — same pattern the existing `CrisisHead.fit_with_oof`
    already uses inside one fold.

    Returns (calibrated_oof, raw_oof) arrays of length matrix.X.shape[0]; NaN
    where no fold's training set was usable for that row.
    """
    from sklearn.isotonic import IsotonicRegression

    fold_to_dates: dict[int, set] = {
        int(fid): set(
            posterior.filter(pl.col("fold_id") == fid)["data_time"].unique().to_list()
        )
        for fid in posterior["fold_id"].unique().to_list()
    }
    n_rows = matrix.X.shape[0]
    p_raw = np.full(n_rows, np.nan, dtype=np.float64)

    n_used = 0
    for fid in sorted(fold_to_dates):
        held_mask = np.array(
            [d in fold_to_dates[fid] for d in matrix.data_times], dtype=bool
        )
        held_dates = [d for d, k in zip(matrix.data_times, held_mask, strict=True) if k]
        if not held_dates:
            continue
        min_held = min(held_dates)
        train_mask = np.array([d < min_held for d in matrix.data_times], dtype=bool)
        if train_mask.sum() < 100 or int((matrix.y[train_mask] == 1).sum()) < 3:
            continue

        head = CrisisHead(n_calibration_splits=5, max_iter=1000, random_state=42)
        head.fit(matrix.X[train_mask], matrix.y[train_mask])
        p_raw[held_mask] = head.predict_raw(matrix.X[held_mask])
        n_used += 1
    print(f"walk-forward refit: {n_used} folds with sufficient training signal")

    finite = ~np.isnan(p_raw)
    obs = finite & (matrix.y != UNOBSERVABLE)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw[obs], matrix.y[obs].astype(np.float64))
    p_calibrated = np.full(n_rows, np.nan, dtype=np.float64)
    p_calibrated[finite] = iso.transform(p_raw[finite])
    return p_calibrated, p_raw


def _per_method_pooled_metrics(
    posterior: pl.DataFrame,
    data_times: list,
    y_obs: np.ndarray,
    obs: np.ndarray,
) -> dict[str, dict[str, float]]:
    """For each method, align its crisis_score against the same `data_times`
    that the pivot used (after NaN-row dropping), then score on the
    observable rows. Returns {method: {brier, pr_auc, ece, base_rate}}."""
    out: dict[str, dict[str, float]] = {}
    for name in sorted(posterior["method"].unique().to_list()):
        sub = (
            posterior.filter(pl.col("method") == name)
            .select(["data_time", "crisis_score"])
            .sort("data_time")
        )
        # Inner-join semantics: keep only the rows whose data_time made it
        # through the NaN drop in `assemble_feature_matrix`.
        keep = pl.DataFrame({"data_time": data_times}).join(sub, on="data_time", how="inner")
        scores = keep["crisis_score"].to_numpy()
        if scores.shape[0] != len(data_times):
            raise RuntimeError(
                f"{name}: aligned to {scores.shape[0]} rows, expected {len(data_times)}"
            )
        s_obs = scores[obs]
        out[name] = {
            "brier": brier_score(s_obs, y_obs.astype(np.float64)),
            "pr_auc": pr_auc(s_obs, y_obs),
            "ece": expected_calibration_error(s_obs, y_obs.astype(np.float64)),
        }
    return out


def _print_comparison(
    per_method: dict[str, dict[str, float]],
    ens_brier: float,
    ens_pr_auc: float,
    ens_ece: float,
    base_rate: float,
    n_obs: int,
    n_pos: int,
) -> None:
    best_brier = min(m["brier"] for m in per_method.values())
    best_brier_name = min(per_method.items(), key=lambda kv: kv[1]["brier"])[0]
    best_pr = max(m["pr_auc"] for m in per_method.values())
    best_pr_name = max(per_method.items(), key=lambda kv: kv[1]["pr_auc"])[0]

    print(
        f"\npooled metrics on {n_obs} observable rows  (n_pos={n_pos}, base rate={base_rate:.4f})"
    )
    print(f"  {'method':<22}  {'brier':>8}  {'pr_auc':>8}  {'ece':>8}")
    for name, m in sorted(per_method.items()):
        print(f"  {name:<22}  {m['brier']:>8.4f}  {m['pr_auc']:>8.4f}  {m['ece']:>8.4f}")
    print(f"  {'ensemble (oof)':<22}  {ens_brier:>8.4f}  {ens_pr_auc:>8.4f}  {ens_ece:>8.4f}")

    # Headline comparison strings per Day-3 plan acceptance. PR-AUC is the
    # acceptance metric (Day-2's change-point methods carry a per-fold
    # supervised head inside their `crisis_score`, so single-method Brier is
    # a strong baseline at the per-fold scale — strict Brier improvement is
    # not the right ensemble bar; ranking quality is, hence PR-AUC).
    brier_delta = ens_brier - best_brier
    pr_delta = ens_pr_auc - best_pr
    print(
        f"\nBrier:  best single = {best_brier_name} ({best_brier:.4f}) "
        f"vs ensemble = {ens_brier:.4f}  (Δ = {brier_delta:+.4f})"
    )
    print(
        f"PR-AUC: best single = {best_pr_name} ({best_pr:.4f}) "
        f"vs ensemble = {ens_pr_auc:.4f}  (Δ = {pr_delta:+.4f})"
    )


def _check_pr_auc_acceptance(
    per_method: dict[str, dict[str, float]], ens_pr_auc: float
) -> None:
    best_pr = max(m["pr_auc"] for m in per_method.values())
    if ens_pr_auc > best_pr:
        print("ACCEPTANCE: ensemble PR-AUC strictly improves over best single ✓")
    else:
        print(
            f"ACCEPTANCE: ensemble PR-AUC = {ens_pr_auc:.4f} does NOT strictly "
            f"improve over best single = {best_pr:.4f}"
        )


def _check_reliability(p_obs: np.ndarray, y_obs: np.ndarray) -> None:
    rc = reliability_curve(p_obs, y_obs.astype(np.float64), n_bins=10)
    populated = rc.bin_count > 0
    observed = rc.mean_observed[populated]
    # Monotonicity check: tolerate tiny non-monotone wiggles up to 1 pp.
    diffs = np.diff(observed)
    n_bad = int(np.sum(diffs < -0.01))
    print(
        f"\nreliability: {int(populated.sum())} populated bins, "
        f"max non-monotone dip = {-diffs.min() if len(diffs) else 0:.4f}, "
        f"violations (> 1pp) = {n_bad}"
    )
    if n_bad == 0:
        print("reliability: monotonic ✓")


def _plot_calibration(
    p_obs: np.ndarray,
    y_obs: np.ndarray,
    per_method: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    rc = reliability_curve(p_obs, y_obs.astype(np.float64), n_bins=10)
    populated = rc.bin_count > 0

    fig, (ax_rel, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    ax_rel.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="ideal")
    ax_rel.plot(
        rc.mean_predicted[populated],
        rc.mean_observed[populated],
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="ensemble (OOF, calibrated)",
    )
    ax_rel.set_xlabel("predicted P(crisis)")
    ax_rel.set_ylabel("observed frequency")
    ax_rel.set_title("Reliability — ensemble OOF")
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.legend(loc="upper left")
    ax_rel.grid(True, alpha=0.3)

    order = np.argsort(-p_obs)
    y_sorted = y_obs[order].astype(np.int64)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y_obs.sum()), 1)
    ax_pr.plot(
        np.concatenate(([0.0], recall)),
        np.concatenate(([1.0], precision)),
        linewidth=2,
        color="#1f77b4",
        label="ensemble (OOF, calibrated)",
    )
    ax_pr.axhline(
        positive_base_rate(y_obs),
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"random baseline = {positive_base_rate(y_obs):.4f}",
    )
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("Precision–Recall — ensemble OOF")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.legend(loc="upper right")
    ax_pr.grid(True, alpha=0.3)

    best_brier = min(m["brier"] for m in per_method.values())
    best_pr = max(m["pr_auc"] for m in per_method.values())
    fig.suptitle(
        f"Crisis head — ensemble vs best single  "
        f"(best single Brier={best_brier:.4f}, best single PR-AUC={best_pr:.4f})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_parquet(
    data_times: list,
    y: np.ndarray,
    oof_raw: np.ndarray,
    oof_calibrated: np.ndarray,
    out_path: Path,
) -> None:
    df = pl.DataFrame(
        {
            "data_time": data_times,
            "label": y.astype(np.int64),
            "oof_raw": oof_raw,
            "oof_calibrated": oof_calibrated,
        },
        schema={
            "data_time": pl.Date,
            "label": pl.Int64,
            "oof_raw": pl.Float64,
            "oof_calibrated": pl.Float64,
        },
    )
    df.write_parquet(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
