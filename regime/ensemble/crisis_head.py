"""Crisis-onset early-warning head.

Logistic regression on a feature matrix + per-fold isotonic calibration.
Trained against the observable 21-day forward-drawdown indicator
(`regime/eval/labels.py`); rows whose label is unobservable (the trailing
horizon at the end of training) are masked out automatically.

Per Q10 of the design grill:
  - Loss function = unweighted cross-entropy (preserves calibration).
  - Headline metric = PR-AUC with random baseline reported alongside.
  - Calibration evidence = reliability diagram + Brier score.
  - Decision threshold for *strategy* use = economic-loss-optimal (~0.08 under
    asymmetric loss); detection-lag headline still uses 0.5 unchanged.

The fitted classifier exposes `predict_proba(X)` returning calibrated
P(crisis within 21d).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from regime.eval.labels import UNOBSERVABLE


@dataclass(frozen=True)
class MethodFeatureMatrix:
    """Wide feature matrix assembled from the Day-2 long-format methods parquet.

    Attributes:
        X: shape `(n_rows, sum_of_per_method_raw_feature_widths)`. Columns are
           per-method native features concatenated in registry order (state
           methods' filtered_k posteriors, change-point methods' native
           features).
        y: shape `(n_rows,)` int64. The forward-drawdown indicator at each
           `data_time`. UNOBSERVABLE preserved.
        data_times: list of date, one per row, sorted ascending.
        feature_names: list of "method__raw_feature" strings, one per column
           of X.
    """

    X: np.ndarray
    y: np.ndarray
    data_times: list[date]
    feature_names: list[str]


def assemble_feature_matrix(
    posterior: pl.DataFrame, include_raw: bool = False
) -> MethodFeatureMatrix:
    """Pivot the Day-2 long-format methods parquet into a wide feature matrix.

    Schema expected (long format, one row per (method, data_time)):
      method (Utf8), data_time (Date), label (Int64), crisis_score (Float64),
      raw_features (List[Float64]).

    By default we use one feature per method — its `crisis_score`. This is
    Day-2's per-method headline scalar: a supervised, per-walk-forward-fold
    logistic head for change-point methods, and the crisis-state filtered
    probability for state methods. Combining six scalars keeps the ensemble
    LR low-variance (~210 positives over ~4500 rows → ~35 positives per
    feature, comfortable for an L2-regularized LR). Adding the unsupervised
    native features back in (via `include_raw=True`) typically hurts Brier
    on the recent folds: the LR is fit once on the whole training set and
    cannot match the adaptivity of the per-fold heads inside the change-point
    methods, so the extra unsupervised columns dilute rather than add signal.

    Rows where any feature is NaN/inf are dropped (e.g., Wasserstein k-means'
    first `window-1` rows in each fold). Labels are taken from the first
    method's row at each `data_time`; they must agree across methods by
    construction in `run_cross_method_walkforward`, and we sanity-check.
    """
    required = {"method", "data_time", "label", "raw_features", "crisis_score"}
    missing = required - set(posterior.columns)
    if missing:
        raise ValueError(f"posterior missing required columns: {sorted(missing)}")

    methods_sorted = posterior["method"].unique().sort().to_list()
    per_method: dict[str, pl.DataFrame] = {}
    feature_widths: dict[str, int] = {}
    for name in methods_sorted:
        sub = (
            posterior.filter(pl.col("method") == name)
            .select(["data_time", "label", "crisis_score", "raw_features"])
            .sort("data_time")
        )
        per_method[name] = sub
        widths = sub["raw_features"].list.len().unique().to_list()
        if len(widths) != 1:
            raise ValueError(
                f"method {name!r} has inconsistent raw_features widths: {widths}"
            )
        feature_widths[name] = int(widths[0])

    base_dates = per_method[methods_sorted[0]]["data_time"].to_list()
    base_labels = per_method[methods_sorted[0]]["label"].to_numpy()
    for name in methods_sorted[1:]:
        other_dates = per_method[name]["data_time"].to_list()
        if other_dates != base_dates:
            raise ValueError(
                f"method {name!r} data_time series differs from {methods_sorted[0]!r}"
            )
        other_labels = per_method[name]["label"].to_numpy()
        if not np.array_equal(other_labels, base_labels):
            raise ValueError(
                f"method {name!r} labels disagree with {methods_sorted[0]!r}"
            )

    n = len(base_dates)
    cols: list[np.ndarray] = []
    feature_names: list[str] = []
    from regime.eval.registry import REGISTRY

    name_to_meta = {m.name: m for m in REGISTRY}
    for name in methods_sorted:
        score = (
            per_method[name]["crisis_score"].to_numpy().astype(np.float64).reshape(-1, 1)
        )
        cols.append(score)
        feature_names.append(f"{name}__crisis_score")
        if include_raw:
            raw = np.array(per_method[name]["raw_features"].to_list(), dtype=np.float64)
            if raw.shape != (n, feature_widths[name]):
                raise ValueError(
                    f"method {name!r}: raw_features shape {raw.shape} != "
                    f"expected {(n, feature_widths[name])}"
                )
            meta = name_to_meta.get(name)
            raw_names = (
                list(meta.raw_feature_names)
                if meta is not None
                else [f"raw_{i}" for i in range(feature_widths[name])]
            )
            cols.append(raw)
            feature_names.extend(f"{name}__{fn}" for fn in raw_names)

    X = np.hstack(cols)
    finite_mask = np.all(np.isfinite(X), axis=1)
    if not finite_mask.all():
        X = X[finite_mask]
        base_labels = base_labels[finite_mask]
        base_dates = [d for d, keep in zip(base_dates, finite_mask, strict=True) if keep]

    return MethodFeatureMatrix(
        X=X,
        y=base_labels.astype(np.int64),
        data_times=base_dates,
        feature_names=feature_names,
    )


class CrisisHead:
    """Logistic regression with isotonic calibration via cross-validated OOF predictions."""

    def __init__(
        self,
        n_calibration_splits: int = 5,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.n_calibration_splits = int(n_calibration_splits)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self._lr: LogisticRegression | None = None
        self._iso: IsotonicRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit logistic regression + isotonic calibrator.

        Rows where y == UNOBSERVABLE are dropped before fitting. The isotonic
        calibrator is fit on cross-validated out-of-fold logistic-regression
        outputs to avoid the trivial in-sample calibration tautology.
        """
        self.fit_with_oof(X, y)

    def fit_with_oof(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Same as `fit`, but also returns the OOF-predicted probabilities
        aligned to the input rows.

        Shape: same as `y`. UNOBSERVABLE input rows receive `NaN` in the
        returned array (they were excluded from training and never received
        an OOF prediction). The returned probabilities are the *uncalibrated*
        logistic-regression OOF outputs — the same ones the isotonic
        calibrator fits on. Pass them through `self._iso.transform(...)` if
        you want the calibrated OOF series instead.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        mask = y != UNOBSERVABLE
        Xm, ym = X[mask], y[mask]
        if len(np.unique(ym)) < 2:
            raise ValueError("crisis head requires both positive and negative labels")

        kf = KFold(
            n_splits=min(self.n_calibration_splits, len(Xm)),
            shuffle=True,
            random_state=self.random_state,
        )
        oof = np.zeros_like(ym, dtype=np.float64)
        for train_i, test_i in kf.split(Xm):
            if len(np.unique(ym[train_i])) < 2:
                continue
            lr_fold = LogisticRegression(max_iter=self.max_iter)
            lr_fold.fit(Xm[train_i], ym[train_i])
            oof[test_i] = lr_fold.predict_proba(Xm[test_i])[:, 1]

        # Final base classifier on full data.
        self._lr = LogisticRegression(max_iter=self.max_iter)
        self._lr.fit(Xm, ym)

        # Isotonic calibrator on OOF predictions.
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(oof, ym)

        oof_full = np.full(len(y), np.nan, dtype=np.float64)
        oof_full[mask] = oof
        return oof_full

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated P(crisis within 21d) for each row."""
        if self._lr is None or self._iso is None:
            raise RuntimeError("crisis head not fit")
        raw = self._lr.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]
        return self._iso.transform(raw)

    def calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply the fitted isotonic calibrator to already-LR raw probabilities.

        Use this to calibrate `fit_with_oof`'s return value (the OOF LR
        probabilities) without re-running the LR. NaN entries are preserved.
        """
        if self._iso is None:
            raise RuntimeError("crisis head not fit")
        raw = np.asarray(raw_proba, dtype=np.float64)
        out = np.full_like(raw, np.nan, dtype=np.float64)
        finite = np.isfinite(raw)
        out[finite] = self._iso.transform(raw[finite])
        return out

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated logistic-regression P(positive). Provided for diagnostics."""
        if self._lr is None:
            raise RuntimeError("crisis head not fit")
        return self._lr.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]

    def state_dict(self) -> dict:
        if self._lr is None or self._iso is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "lr_coef": np.asarray(self._lr.coef_).tolist(),
            "lr_intercept": np.asarray(self._lr.intercept_).tolist(),
            "iso_x": np.asarray(self._iso.X_thresholds_).tolist(),
            "iso_y": np.asarray(self._iso.y_thresholds_).tolist(),
            "n_features": int(np.asarray(self._lr.coef_).shape[1]),
        }

    def load_state_dict(self, state: dict) -> None:
        if not state.get("fitted"):
            self._lr = None
            self._iso = None
            return
        n_features = int(state["n_features"])
        # Rebuild a minimal LogisticRegression with stored coefficients.
        self._lr = LogisticRegression(max_iter=self.max_iter)
        # sklearn's API doesn't expose a direct "set fitted parameters" path;
        # we re-fit on a tiny dummy to mark the estimator fitted, then overwrite
        # the coefficients with the stored ones.
        rng = np.random.default_rng(0)
        Xd = rng.normal(size=(20, n_features))
        yd = rng.integers(0, 2, size=20)
        if len(np.unique(yd)) < 2:
            yd[0], yd[1] = 0, 1
        self._lr.fit(Xd, yd)
        self._lr.coef_ = np.asarray(state["lr_coef"], dtype=np.float64)
        self._lr.intercept_ = np.asarray(state["lr_intercept"], dtype=np.float64)

        # Refit isotonic on the stored (monotonic) threshold pairs — fitting on
        # already-isotone data reproduces the same interpolation function and
        # rebuilds sklearn's lazy `f_` callable cleanly.
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        Xt = np.asarray(state["iso_x"], dtype=np.float64)
        Yt = np.asarray(state["iso_y"], dtype=np.float64)
        self._iso.fit(Xt, Yt)
