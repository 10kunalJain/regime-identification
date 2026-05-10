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

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from regime.eval.labels import UNOBSERVABLE


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
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        mask = y != UNOBSERVABLE
        Xm, ym = X[mask], y[mask]
        if len(np.unique(ym)) < 2:
            raise ValueError("crisis head requires both positive and negative labels")

        # OOF predictions for calibration target.
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated P(crisis within 21d) for each row."""
        if self._lr is None or self._iso is None:
            raise RuntimeError("crisis head not fit")
        raw = self._lr.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]
        return self._iso.transform(raw)

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
