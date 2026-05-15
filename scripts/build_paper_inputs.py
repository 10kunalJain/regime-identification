"""Materialize the per-figure input parquets the paper figures read.

Two artefacts are produced; both are derived from the fitted joint HMM cached
in `build/joint_hmm_real.json` (the same cache `plot_regime_path.py` uses):

  - `build/paper/posterior_covid.parquet` — filtered + smoothed crisis-state
    posteriors over the COVID window. Feeds `figure_smoothed_vs_filtered.py`.
  - `build/paper/regime_path.parquet` — full-history (data_time, state,
    crisis_prob, spy_close). Feeds `figure_regime_path.py`.

Subsampled copies are written under `tests/fixtures/paper/` so the
`paper/figures/figure_*.py` scripts can run in CI against committed
fixtures without re-fitting the model or touching the PIT data store.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

load_dotenv()

from regime.data.joint_dataset import (  # noqa: E402
    FF_COLUMNS,
    OBSERVATION_TICKERS,
    RENAMED_FF,
    build_wide_dataframe,
)
from regime.data.query import as_of  # noqa: E402
from regime.models.joint_hmm import JointHmm  # noqa: E402

CACHE_PATH = Path("build/joint_hmm_real.json")
POSTERIOR_COVID_PATH = Path("build/paper/posterior_covid.parquet")
REGIME_PATH_PATH = Path("build/paper/regime_path.parquet")
FIXTURE_DIR = Path("tests/fixtures/paper")

COVID_START = date(2020, 1, 2)
COVID_END = date(2020, 6, 30)
# Subsample stride for the regime-path fixture — keeps the committed parquet
# under ~30 KB while preserving the visual shape of the price + state path.
REGIME_PATH_FIXTURE_STRIDE = 6


def _fit_or_load(
    df: pl.DataFrame, obs_cols: tuple[str, ...], fact_cols: tuple[str, ...]
) -> JointHmm:
    model = JointHmm(
        K=3,
        observation_columns=obs_cols,
        factor_columns=fact_cols,
        latent_factor_rank=3,
        n_restarts=3,
        max_iter=50,
        random_state=42,
    )
    if CACHE_PATH.exists():
        print(f"loading cached params from {CACHE_PATH}")
        with CACHE_PATH.open() as f:
            model.load_state_dict(json.load(f))
        return model

    print("fitting joint HMM on real data (no cache found) ...")
    model.fit(df, np.arange(df.height, dtype=np.int64))
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w") as f:
        json.dump(model.state_dict(), f)
    return model


def _spy_close_by_date(t: date) -> dict[date, float]:
    spy = as_of("SPY", t)
    dates = spy["data_time"].to_list()
    closes = spy["close"].to_numpy()
    return dict(zip(dates, closes, strict=True))


def main() -> int:
    today = date.today()
    df = build_wide_dataframe(today)
    obs_cols = tuple(f"ret_{x}" for x in OBSERVATION_TICKERS)
    fact_cols = tuple(RENAMED_FF[f] for f in FF_COLUMNS)
    model = _fit_or_load(df, obs_cols, fact_cols)

    idx = np.arange(df.height, dtype=np.int64)
    print("computing filtered + smoothed posteriors ...")
    filt = model.filter(df, idx)
    smooth = model.smooth(df, idx)

    # State 2 is crisis (SPY-mean-descending sort applied during fit alignment).
    crisis_idx = filt.shape[1] - 1
    df_dates = df["data_time"].to_list()
    state_seq = filt.argmax(axis=1)

    spy_close_map = _spy_close_by_date(today)
    spy_close = np.array([spy_close_map.get(d, float("nan")) for d in df_dates])

    full = pl.DataFrame(
        {
            "data_time": df_dates,
            "state": state_seq.astype(np.int64),
            "crisis_prob": filt[:, crisis_idx],
            "spy_close": spy_close,
        }
    ).drop_nulls()
    REGIME_PATH_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.write_parquet(REGIME_PATH_PATH, compression="zstd")
    print(f"wrote {REGIME_PATH_PATH} ({REGIME_PATH_PATH.stat().st_size} bytes)")

    # COVID-window posterior pair, restricted to the user-facing date range.
    keep = [(d >= COVID_START) and (d <= COVID_END) for d in df_dates]
    keep_arr = np.array(keep)
    covid = pl.DataFrame(
        {
            "data_time": [d for d, k in zip(df_dates, keep, strict=True) if k],
            "filtered_crisis": filt[keep_arr, crisis_idx],
            "smoothed_crisis": smooth[keep_arr, crisis_idx],
            "ret_SPY": df["ret_SPY"].to_numpy()[keep_arr],
        }
    )
    POSTERIOR_COVID_PATH.parent.mkdir(parents=True, exist_ok=True)
    covid.write_parquet(POSTERIOR_COVID_PATH, compression="zstd")
    print(f"wrote {POSTERIOR_COVID_PATH} ({POSTERIOR_COVID_PATH.stat().st_size} bytes)")

    # Fixtures: COVID parquet committed at full size; regime path subsampled
    # to keep the committed file small enough for git.
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    covid.write_parquet(FIXTURE_DIR / "posterior_covid.parquet", compression="zstd")
    full.gather_every(REGIME_PATH_FIXTURE_STRIDE).write_parquet(
        FIXTURE_DIR / "regime_path.parquet", compression="zstd"
    )

    # Crisis-head OOF fixture: drop NaN rows (unobservable horizon + leading
    # holdout) and sample down to ~500 rows for the reliability fixture.
    crisis_head_src = Path("build/benchmarks/crisis_head.parquet")
    if crisis_head_src.exists():
        ch = pl.read_parquet(crisis_head_src).drop_nulls(
            subset=["oof_calibrated", "oof_raw", "label"]
        )
        # Deterministic sample by stride.
        stride = max(1, ch.height // 500)
        ch.gather_every(stride).write_parquet(
            FIXTURE_DIR / "crisis_head.parquet", compression="zstd"
        )

    lag_src = Path("build/benchmarks/methods_crisis_lag.parquet")
    if lag_src.exists():
        # Already tiny (~48 rows). Copy verbatim.
        pl.read_parquet(lag_src).write_parquet(
            FIXTURE_DIR / "methods_crisis_lag.parquet", compression="zstd"
        )

    for p in sorted(FIXTURE_DIR.glob("*.parquet")):
        print(f"  fixture: {p}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
