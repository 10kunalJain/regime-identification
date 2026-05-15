"""Tests for the crisis-onset early-warning head."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regime.ensemble.crisis_head import CrisisHead, assemble_feature_matrix
from regime.eval.labels import UNOBSERVABLE


def _separable_data(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Linearly separable 5-feature classification problem with ~10% positive rate."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 5))
    # True logit: positive when first two features sum positive
    score = X[:, 0] + 0.5 * X[:, 1] - 1.5
    proba_true = 1.0 / (1.0 + np.exp(-score))
    y = (rng.uniform(0, 1, size=n) < proba_true).astype(np.int64)
    return X, y


def test_crisis_head_predicts_in_zero_one():
    X, y = _separable_data(n=600, seed=1)
    head = CrisisHead()
    head.fit(X, y)
    p = head.predict_proba(X)
    assert (p >= 0.0).all() and (p <= 1.0).all()


def test_crisis_head_better_than_baserate_on_separable():
    """On linearly-separable data the head should clearly beat a constant baserate predictor."""
    X, y = _separable_data(n=2000, seed=2)
    head = CrisisHead()
    head.fit(X, y)
    p = head.predict_proba(X)
    base = float(y.mean())
    base_brier = float(np.mean((np.full_like(p, base) - y) ** 2))
    head_brier = float(np.mean((p - y) ** 2))
    assert head_brier < base_brier


def test_crisis_head_drops_unobservable_rows():
    """UNOBSERVABLE-labelled rows must be excluded from training."""
    X, y_obs = _separable_data(n=400, seed=3)
    # Mark the trailing 50 rows as unobservable.
    y = y_obs.copy()
    y[-50:] = UNOBSERVABLE
    head = CrisisHead()
    head.fit(X, y)
    # Predicting on the unobservable rows should still work — the head only
    # rejected them at training time.
    p = head.predict_proba(X[-50:])
    assert (p >= 0.0).all() and (p <= 1.0).all()


def test_crisis_head_state_dict_roundtrip():
    X, y = _separable_data(n=400, seed=4)
    head = CrisisHead()
    head.fit(X, y)
    p_before = head.predict_proba(X)

    state = head.state_dict()
    restored = CrisisHead()
    restored.load_state_dict(state)
    p_after = restored.predict_proba(X)

    np.testing.assert_allclose(p_before, p_after, atol=1e-9)


def test_crisis_head_requires_both_classes():
    """All-zero labels should raise."""
    X = np.random.RandomState(0).normal(size=(20, 3))
    y = np.zeros(20, dtype=np.int64)
    head = CrisisHead()
    with pytest.raises(ValueError, match="positive and negative"):
        head.fit(X, y)


def test_crisis_head_unfitted_predict_raises():
    head = CrisisHead()
    with pytest.raises(RuntimeError, match="not fit"):
        head.predict_proba(np.zeros((1, 3)))


def test_crisis_head_fit_with_oof_returns_full_length_oof(seed: int = 5):
    X, y_obs = _separable_data(n=300, seed=seed)
    y = y_obs.copy()
    y[-30:] = UNOBSERVABLE
    head = CrisisHead()
    oof = head.fit_with_oof(X, y)

    assert oof.shape == y.shape
    # Unobservable rows are never assigned an OOF prediction.
    assert np.isnan(oof[-30:]).all()
    observed = oof[:-30]
    assert np.isfinite(observed).all()
    assert (observed >= 0.0).all() and (observed <= 1.0).all()


def test_crisis_head_calibrate_preserves_nans_and_matches_predict_proba():
    X, y = _separable_data(n=400, seed=6)
    head = CrisisHead()
    head.fit(X, y)

    raw = head.predict_raw(X)
    raw_with_nan = raw.copy()
    raw_with_nan[0] = np.nan
    calibrated = head.calibrate(raw_with_nan)

    assert np.isnan(calibrated[0])
    # `predict_proba` is `iso.transform(predict_raw(...))`, so calibrating
    # `predict_raw`'s output must equal `predict_proba`'s output row-for-row.
    np.testing.assert_allclose(calibrated[1:], head.predict_proba(X)[1:], atol=1e-12)


def test_crisis_head_predict_raw_in_zero_one():
    X, y = _separable_data(n=200, seed=7)
    head = CrisisHead()
    head.fit(X, y)
    raw = head.predict_raw(X)
    assert (raw >= 0.0).all() and (raw <= 1.0).all()


def test_crisis_head_predict_raw_unfit_raises():
    head = CrisisHead()
    with pytest.raises(RuntimeError, match="not fit"):
        head.predict_raw(np.zeros((1, 3)))


def test_crisis_head_calibrate_unfit_raises():
    head = CrisisHead()
    with pytest.raises(RuntimeError, match="not fit"):
        head.calibrate(np.array([0.1, 0.5]))


def test_crisis_head_unfitted_state_dict_roundtrip():
    head = CrisisHead()
    state = head.state_dict()
    assert state == {"fitted": False}

    restored = CrisisHead()
    restored.load_state_dict(state)
    with pytest.raises(RuntimeError, match="not fit"):
        restored.predict_proba(np.zeros((1, 3)))


# ---------------------------------------------------------------------------
# assemble_feature_matrix
# ---------------------------------------------------------------------------


def _posterior_row(
    method: str,
    data_time: date,
    label: int,
    crisis_score: float,
    raw_features: list[float],
) -> dict[str, object]:
    return {
        "method": method,
        "data_time": data_time,
        "label": label,
        "crisis_score": crisis_score,
        "raw_features": raw_features,
    }


def _toy_posterior(
    methods: tuple[str, ...] = ("bocpd", "hmm_gaussian"),
    n: int = 5,
    start: date = date(2020, 1, 1),
    width: int = 3,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n):
        d = start + timedelta(days=i)
        label = 1 if i % 2 == 0 else 0
        for m in methods:
            rows.append(
                _posterior_row(
                    method=m,
                    data_time=d,
                    label=label,
                    crisis_score=0.1 * (i + 1),
                    raw_features=[float(j) for j in range(width)],
                )
            )
    return pl.DataFrame(rows)


def test_assemble_feature_matrix_smoke():
    df = _toy_posterior()
    fm = assemble_feature_matrix(df)

    assert fm.X.shape == (5, 2)  # 5 rows, one crisis_score per method
    assert fm.y.shape == (5,)
    assert fm.feature_names == ["bocpd__crisis_score", "hmm_gaussian__crisis_score"]
    assert len(fm.data_times) == 5


def test_assemble_feature_matrix_missing_columns_raises():
    df = _toy_posterior().drop("raw_features")
    with pytest.raises(ValueError, match="missing required columns"):
        assemble_feature_matrix(df)


def test_assemble_feature_matrix_mismatched_dates_raises():
    df = _toy_posterior()
    # Shift one method's dates by one day.
    shifted = df.with_columns(
        pl.when(pl.col("method") == "hmm_gaussian")
        .then(pl.col("data_time").dt.offset_by("1d"))
        .otherwise(pl.col("data_time"))
        .alias("data_time")
    )
    with pytest.raises(ValueError, match="data_time series differs"):
        assemble_feature_matrix(shifted)


def test_assemble_feature_matrix_mismatched_labels_raises():
    df = _toy_posterior()
    flipped = df.with_columns(
        pl.when(pl.col("method") == "hmm_gaussian")
        .then(1 - pl.col("label"))
        .otherwise(pl.col("label"))
        .alias("label")
    )
    with pytest.raises(ValueError, match="labels disagree"):
        assemble_feature_matrix(flipped)


def test_assemble_feature_matrix_inconsistent_widths_raises():
    df = _toy_posterior()
    # Replace the first bocpd row's raw_features with a longer list — polars
    # lists are variable-length, so the width check should trigger.
    base_dates = (
        df.filter(pl.col("method") == "bocpd").sort("data_time")["data_time"].to_list()
    )
    target = base_dates[0]
    ragged = df.with_columns(
        pl.when((pl.col("method") == "bocpd") & (pl.col("data_time") == target))
        .then(pl.lit([1.0, 2.0, 3.0, 4.0]))
        .otherwise(pl.col("raw_features"))
        .alias("raw_features")
    )
    with pytest.raises(ValueError, match="inconsistent raw_features widths"):
        assemble_feature_matrix(ragged)


def test_assemble_feature_matrix_include_raw_uses_registry_names():
    # `bocpd` is in REGISTRY with raw_feature_names
    # = ("change_prob", "expected_run_length", "run_length_entropy").
    df = _toy_posterior(methods=("bocpd",), width=3)
    fm = assemble_feature_matrix(df, include_raw=True)

    assert fm.feature_names == [
        "bocpd__crisis_score",
        "bocpd__change_prob",
        "bocpd__expected_run_length",
        "bocpd__run_length_entropy",
    ]
    assert fm.X.shape == (5, 4)


def test_assemble_feature_matrix_include_raw_unknown_method_fallback():
    # An unregistered method name forces the `raw_<i>` fallback path.
    df = _toy_posterior(methods=("not_in_registry",), width=2)
    fm = assemble_feature_matrix(df, include_raw=True)

    assert fm.feature_names == [
        "not_in_registry__crisis_score",
        "not_in_registry__raw_0",
        "not_in_registry__raw_1",
    ]


def test_assemble_feature_matrix_drops_nonfinite_rows():
    df = _toy_posterior(methods=("bocpd",), n=5)
    # Inject NaN into the third row's crisis_score.
    third = df["data_time"].unique().sort().to_list()[2]
    with_nan = df.with_columns(
        pl.when(pl.col("data_time") == third)
        .then(pl.lit(float("nan")))
        .otherwise(pl.col("crisis_score"))
        .alias("crisis_score")
    )
    fm = assemble_feature_matrix(with_nan)
    assert fm.X.shape[0] == 4
    assert third not in fm.data_times
