"""Walk-forward runners.

Two paths share the same expanding-window scaffold (5y train ≈ 1260 trading
days minimum, 1y test ≈ 252, refit at every fold boundary; filter message
carried continuously across folds by running `forward_filter` on `[0, t]`
with the model fitted on `[0, train_end)`):

  - `run_joint_hmm_walkforward` — Day-1 joint-HMM-specific path; assumes
    the joint HMM's SPY-mean-descending state sort puts the crisis state at
    `K - 1`. Kept verbatim for its committed detection-lag snapshot test.
  - `run_cross_method_walkforward` — Day-2 cross-method path; iterates the
    `regime.eval.registry` and dispatches by protocol. `StateRegimeModel`
    methods produce P(state) and the runner picks the crisis state per fold
    by training-label correlation (fallback: lowest training-mean SPY return
    weighted by state posterior). `ChangePointModel` methods produce native
    features and a thin per-fold logistic head turns those into a comparable
    P(crisis) headline scalar — CLAUDE.md's two-protocol design, with no
    forced K=3 wrappers on the change-point methods.

Output schemas:
  - joint-HMM `WalkForwardResult.posterior`: one row per test day with
    `filtered_k`, `crisis_prob`, `label`, `fold_id`.
  - cross-method `CrossMethodResult.posterior`: long-format, one row per
    (method, test-day) with `crisis_score`, `label`, `fold_id`, and a
    `List[Float64]` column `raw_features` whose entries are in the order of
    the registered method's `raw_feature_names`.

Pre-registered (STRATEGY_HYPERPARAMETERS.md):
  - sustained-fire = 3 consecutive trading days at P(crisis) ≥ 0.5;
  - label = forward 21-day -10% drawdown indicator on SPY.

Reproducibility contract (`CLAUDE.md` §Reproducibility):
  - integer lag values are bit-exact under fixed seed + single-threaded BLAS;
  - float metrics (Brier, PR-AUC) carry a tolerance, declared by the caller
    in snapshot tests.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from regime.ensemble.calibration import brier_score, positive_base_rate, pr_auc
from regime.eval.crises import CRISES, CrisisEvent
from regime.eval.labels import UNOBSERVABLE, forward_drawdown_indicator, observable_mask
from regime.eval.metrics import SUSTAINED_FIRE_DAYS, first_sustained_fire
from regime.eval.registry import REGISTRY, MethodKind, RegisteredMethod
from regime.models.base import ChangePointModel, StateRegimeModel

_LOG = logging.getLogger(__name__)

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


# ======================================================================
# Cross-method benchmark (Day 2)
# ======================================================================


@dataclass(frozen=True)
class MethodFoldResult:
    """Per-method per-fold metrics. `crisis_state` is only meaningful for
    `kind == "state"` (the state index picked as crisis on this fold)."""

    method_name: str
    method_kind: MethodKind
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    n_train: int
    n_test: int
    crisis_state: int | None
    brier: float | None
    pr_auc: float | None
    base_rate: float | None


@dataclass(frozen=True)
class MethodCrisisLag:
    """Per-method tuple of CrisisLag entries."""

    method_name: str
    crisis_lag: tuple[CrisisLag, ...]


@dataclass(frozen=True)
class CrossMethodResult:
    """Output of `run_cross_method_walkforward`.

    - `posterior`: long-format with one row per (method, test-day).
    - `folds`: flat tuple over (method, fold).
    - `crisis_lag`: per-method per-crisis lag (one MethodCrisisLag per method).
    - `summary`: 1 row per method with mean lag / Brier / PR-AUC / agreement.
    """

    posterior: pl.DataFrame
    folds: tuple[MethodFoldResult, ...]
    crisis_lag: tuple[MethodCrisisLag, ...]
    summary: pl.DataFrame


def run_cross_method_walkforward(
    features: pl.DataFrame,
    obs_cols: tuple[str, ...],
    factor_cols: tuple[str, ...],
    methods: tuple[RegisteredMethod, ...] = REGISTRY,
    config: WalkForwardConfig | None = None,
    crises: tuple[CrisisEvent, ...] = CRISES,
) -> CrossMethodResult:
    """Run the expanding-window walk-forward for every registered method.

    Folds and labels are computed once and shared so all methods see the
    same evaluation surface — this is what makes the per-method numbers
    comparable. Each fold dispatches to `_run_state_method_fold` or
    `_run_changepoint_method_fold` by `method.kind`.
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

    fold_bounds = _enumerate_folds(n, cfg)
    posterior_rows: list[dict[str, object]] = []
    folds_out: list[MethodFoldResult] = []
    n_folds = len(fold_bounds)
    _LOG.info(
        "cross-method walk-forward: %d methods × %d folds (rows=%d, train≥%d, refit=%d)",
        len(methods),
        n_folds,
        n,
        cfg.initial_train_rows,
        cfg.refit_every_rows,
    )

    for method in methods:
        method_start = time.perf_counter()
        _LOG.info("[%s] start (%d folds)", method.name, n_folds)
        for fold_id, train_end, test_end in fold_bounds:
            fold_start = time.perf_counter()
            train_idx = np.arange(0, train_end, dtype=np.int64)
            full_idx = np.arange(0, test_end, dtype=np.int64)
            labels_train = labels_all[:train_end]
            labels_test = labels_all[train_end:test_end]

            if method.kind == "state":
                fold_rows, fold_metrics, crisis_state = _run_state_method_fold(
                    method=method,
                    features=features,
                    obs_cols=obs_cols,
                    factor_cols=factor_cols,
                    train_idx=train_idx,
                    full_idx=full_idx,
                    train_end=train_end,
                    test_end=test_end,
                    labels_train=labels_train,
                    labels_test=labels_test,
                    dates_all=dates_all,
                    fold_id=fold_id,
                    spy_log_returns=spy_log_returns,
                )
            else:
                fold_rows, fold_metrics, crisis_state = _run_changepoint_method_fold(
                    method=method,
                    features=features,
                    obs_cols=obs_cols,
                    factor_cols=factor_cols,
                    train_idx=train_idx,
                    full_idx=full_idx,
                    train_end=train_end,
                    test_end=test_end,
                    labels_train=labels_train,
                    labels_test=labels_test,
                    dates_all=dates_all,
                    fold_id=fold_id,
                )

            posterior_rows.extend(fold_rows)
            folds_out.append(
                MethodFoldResult(
                    method_name=method.name,
                    method_kind=method.kind,
                    fold_id=fold_id,
                    train_start=dates_all[0],
                    train_end=dates_all[train_end - 1],
                    test_start=dates_all[train_end],
                    test_end=dates_all[test_end - 1],
                    n_train=train_end,
                    n_test=test_end - train_end,
                    crisis_state=crisis_state,
                    brier=fold_metrics["brier"],
                    pr_auc=fold_metrics["pr_auc"],
                    base_rate=fold_metrics["base_rate"],
                )
            )
            _LOG.info(
                "[%s] fold %d/%d done in %.1fs (n_train=%d, n_test=%d, brier=%s, pr_auc=%s)",
                method.name,
                fold_id + 1,
                n_folds,
                time.perf_counter() - fold_start,
                train_end,
                test_end - train_end,
                _fmt_metric(fold_metrics["brier"]),
                _fmt_metric(fold_metrics["pr_auc"]),
            )
        _LOG.info(
            "[%s] complete in %.1fs", method.name, time.perf_counter() - method_start
        )

    posterior_df = _posterior_rows_to_frame(posterior_rows)

    crisis_lag_per_method: list[MethodCrisisLag] = []
    for method in methods:
        sub = posterior_df.filter(pl.col("method") == method.name)
        sub_dates = sub["data_time"].to_list()
        sub_scores = sub["crisis_score"].to_numpy()
        if not sub_dates:
            crisis_lag_per_method.append(
                MethodCrisisLag(method_name=method.name, crisis_lag=tuple())
            )
            continue
        # `_score_crisis_lag` is schema-agnostic — it just consumes the
        # `crisis_prob` column. Make a temporary frame with that column name
        # so we can reuse the Day-1 implementation verbatim.
        score_df = pl.DataFrame({"data_time": sub_dates, "crisis_prob": sub_scores}).sort(
            "data_time"
        )
        crisis_lag_per_method.append(
            MethodCrisisLag(
                method_name=method.name,
                crisis_lag=_score_crisis_lag(score_df, dates_all, crises, cfg),
            )
        )

    summary = _build_summary_table(posterior_df, folds_out, crisis_lag_per_method)
    return CrossMethodResult(
        posterior=posterior_df,
        folds=tuple(folds_out),
        crisis_lag=tuple(crisis_lag_per_method),
        summary=summary,
    )


def _fmt_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None and math.isfinite(value) else "—"


def _enumerate_folds(n_rows: int, cfg: WalkForwardConfig) -> list[tuple[int, int, int]]:
    """Return the (fold_id, train_end, test_end) triples for the expanding window."""
    folds: list[tuple[int, int, int]] = []
    train_end = cfg.initial_train_rows
    fold_id = 0
    while train_end < n_rows:
        test_end = min(train_end + cfg.refit_every_rows, n_rows)
        folds.append((fold_id, train_end, test_end))
        train_end = test_end
        fold_id += 1
    return folds


def _run_state_method_fold(
    *,
    method: RegisteredMethod,
    features: pl.DataFrame,
    obs_cols: tuple[str, ...],
    factor_cols: tuple[str, ...],
    train_idx: np.ndarray,
    full_idx: np.ndarray,
    train_end: int,
    test_end: int,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    dates_all: list[date],
    fold_id: int,
    spy_log_returns: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, float | None], int]:
    """Fit a state model on `train_idx`, filter over `[0, test_end)`, and
    return (posterior rows for test bars, fold metric dict, crisis state idx)."""
    model: StateRegimeModel = method.factory(obs_cols, factor_cols)  # type: ignore[assignment]
    model.fit(features, train_idx)
    full_posterior = model.filter(features, full_idx)
    K = full_posterior.shape[1]

    filtered_train = full_posterior[:train_end]
    filtered_test = full_posterior[train_end:test_end]

    crisis_state = _pick_crisis_state_for_state_model(
        filtered_train=filtered_train,
        labels_train=labels_train,
        spy_returns_train=spy_log_returns[:train_end],
    )

    crisis_score_test = filtered_test[:, crisis_state]
    fold_metrics = _score_fold(crisis_score_test, labels_test)

    rows: list[dict[str, object]] = []
    for i, day in enumerate(dates_all[train_end:test_end]):
        raw = [float(filtered_test[i, k]) for k in range(K)]
        rows.append(
            {
                "method": method.name,
                "kind": method.kind,
                "data_time": day,
                "fold_id": fold_id,
                "crisis_score": float(crisis_score_test[i]),
                "label": int(labels_test[i]),
                "raw_features": raw,
            }
        )
    return rows, fold_metrics, int(crisis_state)


def _run_changepoint_method_fold(
    *,
    method: RegisteredMethod,
    features: pl.DataFrame,
    obs_cols: tuple[str, ...],
    factor_cols: tuple[str, ...],
    train_idx: np.ndarray,
    full_idx: np.ndarray,
    train_end: int,
    test_end: int,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    dates_all: list[date],
    fold_id: int,
) -> tuple[list[dict[str, object]], dict[str, float | None], None]:
    """Fit a change-point model and run native_features over `[0, test_end)`.
    A per-fold logistic head turns native features into a P(crisis) scalar."""
    model: ChangePointModel = method.factory(obs_cols, factor_cols)  # type: ignore[assignment]
    model.fit(features, train_idx)
    full_native = model.native_features(features, full_idx)
    if full_native.shape[1] != len(method.raw_feature_names):
        raise RuntimeError(
            f"{method.name}: native_features returned {full_native.shape[1]} cols, "
            f"registry declared {len(method.raw_feature_names)} ({method.raw_feature_names})"
        )

    native_train = full_native[:train_end]
    native_test = full_native[train_end:test_end]

    scorer_train_mask = _finite_observable_mask(native_train, labels_train)
    scorer_test_mask_finite = np.all(np.isfinite(native_test), axis=1)
    crisis_score_test = _score_changepoint_fold(
        native_train_obs=native_train[scorer_train_mask],
        labels_train_obs=labels_train[scorer_train_mask],
        native_train_all=native_train,
        native_test=native_test,
        test_finite=scorer_test_mask_finite,
        method_name=method.name,
    )

    fold_metrics = _score_fold(crisis_score_test, labels_test)

    rows: list[dict[str, object]] = []
    for i, day in enumerate(dates_all[train_end:test_end]):
        raw = [float(native_test[i, j]) for j in range(full_native.shape[1])]
        rows.append(
            {
                "method": method.name,
                "kind": method.kind,
                "data_time": day,
                "fold_id": fold_id,
                "crisis_score": float(crisis_score_test[i]),
                "label": int(labels_test[i]),
                "raw_features": raw,
            }
        )
    return rows, fold_metrics, None


def _pick_crisis_state_for_state_model(
    *,
    filtered_train: np.ndarray,
    labels_train: np.ndarray,
    spy_returns_train: np.ndarray,
) -> int:
    """Pick the crisis state index for a state model on a fold's training data.

    Primary rule: state with the highest Pearson correlation between filtered
    posterior and the observable forward-drawdown label. Requires at least one
    positive label in the (observable portion of the) training window.

    Fallback (no positive labels in training, e.g. the 2003-2007 first fold):
    state with the lowest training-mean SPY log return, weighted by that
    state's filtered posterior. This is the well-known "lowest-mean-return =
    crisis" heuristic — defensible without a label signal.
    """
    K = filtered_train.shape[1]
    obs_mask = observable_mask(labels_train)
    y = labels_train[obs_mask].astype(np.float64)

    if y.size > 0 and 0 < y.sum() < y.size:
        # Both classes present — correlation pick.
        post = filtered_train[obs_mask]
        corrs = np.zeros(K, dtype=np.float64)
        for k in range(K):
            pk = post[:, k]
            if pk.std() < 1e-12 or y.std() < 1e-12:
                corrs[k] = 0.0
            else:
                corrs[k] = float(np.corrcoef(pk, y)[0, 1])
        return int(np.argmax(corrs))

    # Fallback: weighted mean of SPY log return per state, pick the minimum.
    weighted_returns = np.zeros(K, dtype=np.float64)
    for k in range(K):
        w = filtered_train[:, k]
        w_sum = float(w.sum())
        if w_sum < 1e-12:
            weighted_returns[k] = 0.0
        else:
            weighted_returns[k] = float(np.dot(w, spy_returns_train) / w_sum)
    return int(np.argmin(weighted_returns))


def _score_changepoint_fold(
    *,
    native_train_obs: np.ndarray,
    labels_train_obs: np.ndarray,
    native_train_all: np.ndarray,
    native_test: np.ndarray,
    test_finite: np.ndarray,
    method_name: str,
) -> np.ndarray:
    """Train a thin logistic head on native features → crisis label; predict
    on the test bars. NaN test rows (e.g., Wasserstein's leading window-1 rows)
    receive crisis_score = 0.0.

    Fallback (no positive labels in the observable training portion, or no
    usable rows at all): use the first native feature column as the headline
    scalar, min-max scaled into [0, 1] using **training-portion** min/max only
    — never the test rows, to keep the no-leak invariant intact. For BOCPD
    this is `change_prob` (already in [0, 1] so scaling is a no-op); for
    Wasserstein it is `dist_0`, the distance to the first medoid — a noisy
    but defensible ranking signal when no crisis labels are available yet.
    """
    out = np.zeros(native_test.shape[0], dtype=np.float64)
    if native_train_obs.shape[0] == 0:
        return _min_max_scale_with_ref(
            native_test[:, 0], reference=native_train_all[:, 0]
        )

    y_train = labels_train_obs.astype(np.int64)
    if not (0 < y_train.sum() < y_train.size):
        _ = method_name  # silence unused-variable lint; kept for future logs
        scaled = _min_max_scale_with_ref(
            native_test[:, 0], reference=native_train_all[:, 0]
        )
        out[test_finite] = scaled[test_finite]
        return out

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(native_train_obs, y_train)
    proba_test = np.zeros(native_test.shape[0], dtype=np.float64)
    if test_finite.any():
        proba_test[test_finite] = lr.predict_proba(native_test[test_finite])[:, 1]
    return proba_test


def _finite_observable_mask(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Boolean mask of rows whose features are finite AND label is observable."""
    finite = np.all(np.isfinite(features), axis=1)
    return finite & observable_mask(labels)


def _min_max_scale_with_ref(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Min-max scale `x` using `reference`'s min/max (keeps no-leak: scale
    parameters come from the training portion only). Constant reference maps
    to 0.5; NaN entries in `x` map to 0; out-of-range outputs are clipped."""
    finite_ref = np.isfinite(reference)
    if not finite_ref.any():
        return _min_max_scale(x)
    lo = float(reference[finite_ref].min())
    hi = float(reference[finite_ref].max())
    if hi - lo < 1e-12:
        out = np.full_like(x, 0.5)
        out[~np.isfinite(x)] = 0.0
        return out
    finite_x = np.isfinite(x)
    out = (x - lo) / (hi - lo)
    out[~finite_x] = 0.0
    return np.clip(out, 0.0, 1.0)


def _min_max_scale(x: np.ndarray) -> np.ndarray:
    """Min-max scale `x` into [0, 1]; constant input maps to 0.5. NaNs map to 0."""
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lo = float(x[finite].min())
    hi = float(x[finite].max())
    if hi - lo < 1e-12:
        out = np.full_like(x, 0.5)
        out[~finite] = 0.0
        return out
    out = (x - lo) / (hi - lo)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0)


def _score_fold(crisis_score: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Brier / PR-AUC / base-rate for a single fold's test scores against labels."""
    obs = observable_mask(labels)
    if not obs.any():
        return {"brier": None, "pr_auc": None, "base_rate": None}
    y = labels[obs].astype(np.int64)
    p = crisis_score[obs].astype(np.float64)
    return {
        "brier": brier_score(p, y.astype(np.float64)),
        "pr_auc": pr_auc(p, y),
        "base_rate": positive_base_rate(y),
    }


def _posterior_rows_to_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "method": pl.Utf8,
                "kind": pl.Utf8,
                "data_time": pl.Date,
                "fold_id": pl.Int64,
                "crisis_score": pl.Float64,
                "label": pl.Int64,
                "raw_features": pl.List(pl.Float64),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "method": pl.Utf8,
            "kind": pl.Utf8,
            "data_time": pl.Date,
            "fold_id": pl.Int64,
            "crisis_score": pl.Float64,
            "label": pl.Int64,
            "raw_features": pl.List(pl.Float64),
        },
    ).sort(["method", "data_time"])


def _build_summary_table(
    posterior: pl.DataFrame,
    folds: tuple[MethodFoldResult, ...] | list[MethodFoldResult],
    crisis_lag_per_method: list[MethodCrisisLag],
) -> pl.DataFrame:
    """6×4 cross-method summary: mean detection lag (m5), mean Brier, mean PR-AUC,
    mean pairwise agreement (Pearson correlation of crisis_score across methods).

    Aggregation rules:
      - mean_lag_m5: simple mean over crises that were both `in_eval_window`
        and detected (`lag_m5 is not None`).
      - mean_brier, mean_pr_auc: simple mean over folds whose metric is not
        None. Equal-weighted across folds; not test-row-weighted.
      - agreement: mean Pearson correlation of this method's crisis_score
        vs each other method's crisis_score, over the *overlap* of test
        dates between them.
    """
    method_names = list(dict.fromkeys(f.method_name for f in folds))
    folds_by_method: dict[str, list[MethodFoldResult]] = {n: [] for n in method_names}
    for f in folds:
        folds_by_method[f.method_name].append(f)

    lag_by_method = {m.method_name: m.crisis_lag for m in crisis_lag_per_method}
    score_by_method: dict[str, pl.DataFrame] = {
        name: posterior.filter(pl.col("method") == name).select(["data_time", "crisis_score"])
        for name in method_names
    }

    rows: list[dict[str, object]] = []
    for name in method_names:
        # mean detection lag
        lags = [
            c.lag_m5
            for c in lag_by_method.get(name, ())
            if c.in_eval_window and c.lag_m5 is not None
        ]
        mean_lag = float(np.mean(lags)) if lags else None

        # mean Brier / PR-AUC (over non-None folds)
        b_vals = [f.brier for f in folds_by_method[name] if f.brier is not None]
        p_vals = [f.pr_auc for f in folds_by_method[name] if f.pr_auc is not None]
        mean_brier = float(np.mean(b_vals)) if b_vals else None
        mean_pr_auc = float(np.mean(p_vals)) if p_vals else None

        # agreement: mean pairwise corr with all OTHER methods
        corrs: list[float] = []
        a = score_by_method[name]
        for other in method_names:
            if other == name:
                continue
            b = score_by_method[other]
            joined = a.join(b, on="data_time", how="inner", suffix="_other")
            if joined.height < 2:
                continue
            xs = joined["crisis_score"].to_numpy()
            ys = joined["crisis_score_other"].to_numpy()
            if xs.std() < 1e-12 or ys.std() < 1e-12:
                continue
            corrs.append(float(np.corrcoef(xs, ys)[0, 1]))
        mean_agreement = float(np.mean(corrs)) if corrs else None

        rows.append(
            {
                "method": name,
                "mean_lag_m5": mean_lag,
                "mean_brier": mean_brier,
                "mean_pr_auc": mean_pr_auc,
                "agreement": mean_agreement,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "method": pl.Utf8,
            "mean_lag_m5": pl.Float64,
            "mean_brier": pl.Float64,
            "mean_pr_auc": pl.Float64,
            "agreement": pl.Float64,
        },
    )


def format_cross_method_summary(summary: pl.DataFrame) -> str:
    """Render the 6×4 cross-method summary as a fixed-width table."""
    header = (
        f"  {'method':<22}  {'mean_lag_m5':>11}  "
        f"{'mean_brier':>10}  {'mean_pr_auc':>11}  {'agreement':>9}"
    )
    lines = [header]
    for row in summary.iter_rows(named=True):
        lag = f"{row['mean_lag_m5']:>11.1f}" if row["mean_lag_m5"] is not None else f"{'—':>11}"
        b = f"{row['mean_brier']:>10.4f}" if row["mean_brier"] is not None else f"{'—':>10}"
        p = f"{row['mean_pr_auc']:>11.4f}" if row["mean_pr_auc"] is not None else f"{'—':>11}"
        a = f"{row['agreement']:>9.3f}" if row["agreement"] is not None else f"{'—':>9}"
        lines.append(f"  {row['method']:<22}  {lag}  {b}  {p}  {a}")
    return "\n".join(lines)


def method_folds_to_dataframe(folds: tuple[MethodFoldResult, ...]) -> pl.DataFrame:
    """Convert per-method per-fold results to a parquet-ready DataFrame."""
    return pl.DataFrame(
        {
            "method": [f.method_name for f in folds],
            "kind": [f.method_kind for f in folds],
            "fold_id": [f.fold_id for f in folds],
            "train_start": [f.train_start for f in folds],
            "train_end": [f.train_end for f in folds],
            "test_start": [f.test_start for f in folds],
            "test_end": [f.test_end for f in folds],
            "n_train": [f.n_train for f in folds],
            "n_test": [f.n_test for f in folds],
            "crisis_state": [f.crisis_state for f in folds],
            "brier": [f.brier for f in folds],
            "pr_auc": [f.pr_auc for f in folds],
            "base_rate": [f.base_rate for f in folds],
        },
        schema={
            "method": pl.Utf8,
            "kind": pl.Utf8,
            "fold_id": pl.Int64,
            "train_start": pl.Date,
            "train_end": pl.Date,
            "test_start": pl.Date,
            "test_end": pl.Date,
            "n_train": pl.Int64,
            "n_test": pl.Int64,
            "crisis_state": pl.Int64,
            "brier": pl.Float64,
            "pr_auc": pl.Float64,
            "base_rate": pl.Float64,
        },
    )


def method_crisis_lag_to_dataframe(
    crisis_lag: tuple[MethodCrisisLag, ...],
) -> pl.DataFrame:
    """Concatenate per-method crisis-lag tables with a `method` column."""
    rows: list[dict[str, object]] = []
    for entry in crisis_lag:
        for c in entry.crisis_lag:
            rows.append(
                {
                    "method": entry.method_name,
                    "crisis_name": c.crisis_name,
                    "m5_date": c.m5_date,
                    "m10_date": c.m10_date,
                    "first_fire_date": c.first_fire_date,
                    "lag_m5": c.lag_m5,
                    "lag_m10": c.lag_m10,
                    "in_eval_window": c.in_eval_window,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "method": pl.Utf8,
            "crisis_name": pl.Utf8,
            "m5_date": pl.Date,
            "m10_date": pl.Date,
            "first_fire_date": pl.Date,
            "lag_m5": pl.Int64,
            "lag_m10": pl.Int64,
            "in_eval_window": pl.Boolean,
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
    "CrossMethodResult",
    "FoldResult",
    "MethodCrisisLag",
    "MethodFoldResult",
    "WalkForwardConfig",
    "WalkForwardResult",
    "crisis_lag_to_dataframe",
    "folds_to_dataframe",
    "format_crisis_lag_table",
    "format_cross_method_summary",
    "format_fold_summary",
    "method_crisis_lag_to_dataframe",
    "method_folds_to_dataframe",
    "run_cross_method_walkforward",
    "run_joint_hmm_walkforward",
]
