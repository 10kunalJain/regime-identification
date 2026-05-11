"""Walk-forward runner for the joint cross-sectional HMM.

Expanding-window: minimum 5y train (≈ 1260 trading days), 1y test (≈ 252),
refit at every fold boundary. At each test bar t the filtered posterior is
computed by running `forward_filter` on `[0, t]` with the model fitted on
`[0, train_end)` — i.e. the filter message is carried continuously across
the fold boundary, not restarted from π. The existing row-based
`walk_forward` in `regime/eval/walkforward.py` uses a per-fold restart and
is kept for its synthetic-HMM property tests; this runner is the
joint-HMM-specific path.

Outputs (per `WalkForwardResult`):
  - `posterior`: long-format DataFrame with the per-test-day filtered
    posterior, the crisis probability, the joint label, and the fold id.
  - `folds`: per-fold metadata (Brier, PR-AUC, base rate, sizes).
  - `crisis_lag`: per-crisis first-fire date and lag in trading days against
    both the m5 and m10 anchors.

Pre-registered (per Q3/Q4 of the design grill and STRATEGY_HYPERPARAMETERS.md):
  - sustained-fire = 3 consecutive trading days at P(crisis) ≥ 0.5;
  - crisis state = the last state index after the joint HMM's
    SPY-mean-descending sort, i.e. `K - 1`;
  - label = forward 21-day -10% drawdown indicator on SPY.

Reproducibility contract (`CLAUDE.md` §Reproducibility):
  - integer lag values are bit-exact under fixed seed + single-threaded BLAS;
  - float metrics (Brier, PR-AUC) carry a tolerance, declared by the caller
    in snapshot tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from regime.ensemble.calibration import brier_score, positive_base_rate, pr_auc
from regime.eval.crises import CRISES, CrisisEvent
from regime.eval.labels import UNOBSERVABLE, forward_drawdown_indicator, observable_mask
from regime.eval.metrics import SUSTAINED_FIRE_DAYS, first_sustained_fire
from regime.models.base import StateRegimeModel

# Pre-registered defaults from STRATEGY_HYPERPARAMETERS.md.
INITIAL_TRAIN_ROWS = 1260  # ~5 years at 252 trading days/year
REFIT_EVERY_ROWS = 252  # ~1 year
DEFAULT_CRISIS_THRESHOLD = 0.5
DEFAULT_LABEL_HORIZON = 21
DEFAULT_LABEL_THRESHOLD = 0.10
SPY_RETURN_COLUMN = "ret_SPY"


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward configuration. Defaults are the pre-registered values."""

    initial_train_rows: int = INITIAL_TRAIN_ROWS
    refit_every_rows: int = REFIT_EVERY_ROWS
    crisis_threshold: float = DEFAULT_CRISIS_THRESHOLD
    sustained_days: int = SUSTAINED_FIRE_DAYS
    label_horizon: int = DEFAULT_LABEL_HORIZON
    label_threshold: float = DEFAULT_LABEL_THRESHOLD


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    n_train: int
    n_test: int
    brier: float | None
    pr_auc: float | None
    base_rate: float | None


@dataclass(frozen=True)
class CrisisLag:
    """First-fire date and detection lag for a single crisis.

    `lag_m5` and `lag_m10` are integer trading-day counts from the anchor to
    the start of the first 3-day sustained fire of P(crisis) ≥ threshold.
    `None` means the crisis was not detected within the evaluation window.
    """

    crisis_name: str
    m5_date: date
    m10_date: date
    first_fire_date: date | None
    lag_m5: int | None
    lag_m10: int | None
    in_eval_window: bool


@dataclass(frozen=True)
class WalkForwardResult:
    posterior: pl.DataFrame
    folds: tuple[FoldResult, ...]
    crisis_lag: tuple[CrisisLag, ...]


def run_joint_hmm_walkforward(
    features: pl.DataFrame,
    model_factory: Callable[[], StateRegimeModel],
    config: WalkForwardConfig | None = None,
    crises: tuple[CrisisEvent, ...] = CRISES,
) -> WalkForwardResult:
    """Run expanding-window walk-forward eval for the joint HMM.

    Args:
        features: Wide DataFrame with `data_time` plus per-asset return columns
            plus FF factor columns. Must include `ret_SPY` (used for the
            forward-drawdown label). Rows sorted by `data_time`, drop_nulls
            applied upstream.
        model_factory: Returns a fresh, unfit `StateRegimeModel` (specifically
            the joint HMM — but anything with the protocol works).
        config: Walk-forward config; defaults to the pre-registered values.
        crises: Crisis registry to score detection lag against.
    """
    cfg = config or WalkForwardConfig()
    if SPY_RETURN_COLUMN not in features.columns:
        raise ValueError(f"features must include {SPY_RETURN_COLUMN!r} for forward-drawdown labels")

    n = features.height
    if n <= cfg.initial_train_rows:
        raise ValueError(
            f"feature rows ({n}) must exceed initial_train_rows ({cfg.initial_train_rows})"
        )

    dates_all: list[date] = features["data_time"].to_list()
    spy_log_returns = features[SPY_RETURN_COLUMN].to_numpy().astype(np.float64)
    spy_price_proxy = np.exp(np.cumsum(spy_log_returns))
    labels_all = forward_drawdown_indicator(
        spy_price_proxy,
        horizon=cfg.label_horizon,
        threshold=cfg.label_threshold,
    )

    fold_results: list[FoldResult] = []
    posterior_rows: list[dict[str, object]] = []
    train_end = cfg.initial_train_rows
    fold_id = 0
    while train_end < n:
        test_end = min(train_end + cfg.refit_every_rows, n)
        train_idx = np.arange(0, train_end, dtype=np.int64)
        full_idx = np.arange(0, test_end, dtype=np.int64)

        model = model_factory()
        model.fit(features, train_idx)
        full_posterior = model.filter(features, full_idx)
        K = full_posterior.shape[1]
        fold_posterior = full_posterior[train_end:test_end]
        crisis_state_idx = K - 1  # joint HMM sorts states by SPY-mean descending

        crisis_prob_fold = fold_posterior[:, crisis_state_idx]
        labels_fold = labels_all[train_end:test_end]
        obs_mask_fold = observable_mask(labels_fold)

        brier_val: float | None = None
        pr_auc_val: float | None = None
        base_rate_val: float | None = None
        if obs_mask_fold.any():
            y = labels_fold[obs_mask_fold].astype(np.int64)
            p = crisis_prob_fold[obs_mask_fold]
            brier_val = brier_score(p, y.astype(np.float64))
            pr_auc_val = pr_auc(p, y)
            base_rate_val = positive_base_rate(y)

        for i, day in enumerate(dates_all[train_end:test_end]):
            row: dict[str, object] = {
                "data_time": day,
                "fold_id": fold_id,
                "crisis_prob": float(crisis_prob_fold[i]),
                "label": int(labels_fold[i]),
            }
            for k in range(K):
                row[f"filtered_{k}"] = float(fold_posterior[i, k])
            posterior_rows.append(row)

        fold_results.append(
            FoldResult(
                fold_id=fold_id,
                train_start=dates_all[0],
                train_end=dates_all[train_end - 1],
                test_start=dates_all[train_end],
                test_end=dates_all[test_end - 1],
                n_train=train_end,
                n_test=test_end - train_end,
                brier=brier_val,
                pr_auc=pr_auc_val,
                base_rate=base_rate_val,
            )
        )

        train_end = test_end
        fold_id += 1

    posterior_df = pl.DataFrame(posterior_rows).sort("data_time")
    full_dates = dates_all
    crisis_lag = _score_crisis_lag(posterior_df, full_dates, crises, cfg)
    return WalkForwardResult(
        posterior=posterior_df,
        folds=tuple(fold_results),
        crisis_lag=crisis_lag,
    )


def _score_crisis_lag(
    posterior_df: pl.DataFrame,
    full_dates: list[date],
    crises: tuple[CrisisEvent, ...],
    cfg: WalkForwardConfig,
) -> tuple[CrisisLag, ...]:
    """Compute first-fire date and m5/m10 lags from the concatenated posterior.

    Eval-window semantics:
      - A crisis is `in_eval_window` if its m5 anchor is on or before the last
        test-window date — i.e. the crisis could potentially be detected within
        the time range we observed. Crises whose anchor falls in the *train*
        portion are still measurable (the model is "warmed up" on them and
        first-fire is searched among test dates).
      - Lag is counted as trading days in the full feature series from the
        anchor to the first-fire date, so cross-window distances are correct
        even when the anchor predates the first test bar.
    """
    test_dates = posterior_df["data_time"].to_list()
    crisis_prob = posterior_df["crisis_prob"].to_numpy()
    test_first = test_dates[0] if test_dates else None
    test_last = test_dates[-1] if test_dates else None

    results: list[CrisisLag] = []
    for c in crises:
        in_window = test_last is not None and c.m5_date <= test_last
        if not in_window or test_first is None:
            results.append(
                CrisisLag(
                    crisis_name=c.name,
                    m5_date=c.m5_date,
                    m10_date=c.m10_date,
                    first_fire_date=None,
                    lag_m5=None,
                    lag_m10=None,
                    in_eval_window=False,
                )
            )
            continue

        # Look for first sustained fire on or after the anchor; if the anchor
        # is in the train portion, clamp to the first test bar.
        effective_after = max(c.m5_date, test_first)
        fire = first_sustained_fire(
            test_dates,
            crisis_prob,
            after=effective_after,
            threshold=cfg.crisis_threshold,
            sustained_days=cfg.sustained_days,
        )
        lag_m5 = _trading_day_lag(full_dates, c.m5_date, fire) if fire is not None else None
        lag_m10 = _trading_day_lag(full_dates, c.m10_date, fire) if fire is not None else None
        results.append(
            CrisisLag(
                crisis_name=c.name,
                m5_date=c.m5_date,
                m10_date=c.m10_date,
                first_fire_date=fire,
                lag_m5=lag_m5,
                lag_m10=lag_m10,
                in_eval_window=True,
            )
        )
    return tuple(results)


def _trading_day_lag(full_dates: list[date], anchor: date, fire: date) -> int | None:
    """Trading-day count from the first date ≥ anchor to `fire` in the full
    feature date series. None if anchor or fire isn't in the series."""
    anchor_idx: int | None = next((i for i, d in enumerate(full_dates) if d >= anchor), None)
    if anchor_idx is None:
        return None
    try:
        fire_idx = full_dates.index(fire)
    except ValueError:
        return None
    return fire_idx - anchor_idx


# ----------------------------------------------------------------------
# Reporting helpers


def format_crisis_lag_table(results: tuple[CrisisLag, ...]) -> str:
    """Render a fixed-width table of per-crisis first-fire / lag. Pure stdout."""
    header = f"  {'crisis':<32}  {'m5_date':>12}  {'m10_date':>12}  {'first_fire':>12}  {'lag_m5':>7}  {'lag_m10':>7}"  # noqa: E501
    lines = [header]
    for r in results:
        fire = r.first_fire_date.isoformat() if r.first_fire_date is not None else "—"
        lag_m5 = f"{r.lag_m5:>7d}" if r.lag_m5 is not None else f"{'—':>7}"
        lag_m10 = f"{r.lag_m10:>7d}" if r.lag_m10 is not None else f"{'—':>7}"
        if not r.in_eval_window:
            fire, lag_m5, lag_m10 = "(out of window)", f"{'—':>7}", f"{'—':>7}"
        lines.append(
            f"  {r.crisis_name:<32}  {r.m5_date.isoformat():>12}  "
            f"{r.m10_date.isoformat():>12}  {fire:>12}  {lag_m5}  {lag_m10}"
        )
    return "\n".join(lines)


def format_fold_summary(folds: tuple[FoldResult, ...]) -> str:
    """Render a fixed-width table of per-fold metrics."""
    header = (
        f"  {'fold':>4}  {'train_end':>12}  {'test_end':>12}  "
        f"{'n_test':>6}  {'brier':>7}  {'pr_auc':>7}  {'base':>6}"
    )
    lines = [header]
    for f in folds:
        b = f"{f.brier:.4f}" if f.brier is not None and math.isfinite(f.brier) else "—"
        p = f"{f.pr_auc:.4f}" if f.pr_auc is not None and math.isfinite(f.pr_auc) else "—"
        br = f"{f.base_rate:.3f}" if f.base_rate is not None else "—"
        lines.append(
            f"  {f.fold_id:>4d}  {f.train_end.isoformat():>12}  "
            f"{f.test_end.isoformat():>12}  {f.n_test:>6d}  {b:>7}  {p:>7}  {br:>6}"
        )
    return "\n".join(lines)


def crisis_lag_to_dataframe(results: tuple[CrisisLag, ...]) -> pl.DataFrame:
    """Convert crisis-lag results to a polars DataFrame for parquet output."""
    return pl.DataFrame(
        {
            "crisis_name": [r.crisis_name for r in results],
            "m5_date": [r.m5_date for r in results],
            "m10_date": [r.m10_date for r in results],
            "first_fire_date": [r.first_fire_date for r in results],
            "lag_m5": [r.lag_m5 for r in results],
            "lag_m10": [r.lag_m10 for r in results],
            "in_eval_window": [r.in_eval_window for r in results],
        },
        schema={
            "crisis_name": pl.Utf8,
            "m5_date": pl.Date,
            "m10_date": pl.Date,
            "first_fire_date": pl.Date,
            "lag_m5": pl.Int64,
            "lag_m10": pl.Int64,
            "in_eval_window": pl.Boolean,
        },
    )


def folds_to_dataframe(folds: tuple[FoldResult, ...]) -> pl.DataFrame:
    """Convert per-fold metadata to a polars DataFrame for parquet output."""
    return pl.DataFrame(
        {
            "fold_id": [f.fold_id for f in folds],
            "train_start": [f.train_start for f in folds],
            "train_end": [f.train_end for f in folds],
            "test_start": [f.test_start for f in folds],
            "test_end": [f.test_end for f in folds],
            "n_train": [f.n_train for f in folds],
            "n_test": [f.n_test for f in folds],
            "brier": [f.brier for f in folds],
            "pr_auc": [f.pr_auc for f in folds],
            "base_rate": [f.base_rate for f in folds],
        },
        schema={
            "fold_id": pl.Int64,
            "train_start": pl.Date,
            "train_end": pl.Date,
            "test_start": pl.Date,
            "test_end": pl.Date,
            "n_train": pl.Int64,
            "n_test": pl.Int64,
            "brier": pl.Float64,
            "pr_auc": pl.Float64,
            "base_rate": pl.Float64,
        },
    )


__all__ = [
    "DEFAULT_CRISIS_THRESHOLD",
    "DEFAULT_LABEL_HORIZON",
    "DEFAULT_LABEL_THRESHOLD",
    "INITIAL_TRAIN_ROWS",
    "REFIT_EVERY_ROWS",
    "SPY_RETURN_COLUMN",
    "UNOBSERVABLE",
    "CrisisLag",
    "FoldResult",
    "WalkForwardConfig",
    "WalkForwardResult",
    "crisis_lag_to_dataframe",
    "folds_to_dataframe",
    "format_crisis_lag_table",
    "format_fold_summary",
    "run_joint_hmm_walkforward",
]
