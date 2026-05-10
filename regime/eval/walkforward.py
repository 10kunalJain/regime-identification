"""Walk-forward evaluation harness.

Expanding-window walk-forward refitting: at each fold boundary, fit a fresh
model on `[0, train_end)` and produce filtered + smoothed posteriors over
`[train_end, train_end + refit_every_days)`. Refits cascade until the entire
feature DataFrame is covered.

The `model_factory` callable returns a fresh, unfit `StateRegimeModel` for
each fold. This guarantees each fold's parameters are independent — the
foundation of out-of-sample evaluation.

Output schema:
  data_time: Date
  fold_id: Int64
  filtered_*: Float64 (one column per regime)
  smoothed_*: Float64 (one column per regime)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from regime.models.base import StateRegimeModel


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train_rows: int = 1000
    refit_every_rows: int = 63


def walk_forward(
    features: pl.DataFrame,
    model_factory: Callable[[], StateRegimeModel],
    config: WalkForwardConfig | None = None,
) -> pl.DataFrame:
    """Run expanding-window walk-forward on `features`.

    `features` must contain a `data_time` column and whatever feature columns
    the model_factory's model uses. Rows must be sorted by `data_time`.
    """
    cfg = config or WalkForwardConfig()
    n = features.height
    if n <= cfg.initial_train_rows:
        return _empty_output()

    rows: list[dict[str, object]] = []
    fold_id = 0
    train_end = cfg.initial_train_rows
    while train_end < n:
        train_idx = np.arange(0, train_end, dtype=np.int64)
        test_end = min(train_end + cfg.refit_every_rows, n)
        test_idx = np.arange(train_end, test_end, dtype=np.int64)

        model = model_factory()
        model.fit(features, train_idx)
        filtered = model.filter(features, test_idx)
        smoothed = model.smooth(features, test_idx)

        K = filtered.shape[1]
        # Some test rows can have nulls in their feature columns and be dropped
        # by the model's _extract; len(filtered) may be < len(test_idx). We
        # align by trailing length.
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True
        test_dates = (
            features.select("data_time").filter(pl.Series(test_mask))["data_time"].to_list()
        )
        kept = len(filtered)
        kept_dates = test_dates[-kept:] if kept > 0 else []
        for i, d in enumerate(kept_dates):
            row: dict[str, object] = {"data_time": d, "fold_id": fold_id}
            for k in range(K):
                row[f"filtered_{k}"] = float(filtered[i, k])
                row[f"smoothed_{k}"] = float(smoothed[i, k])
            rows.append(row)

        train_end = test_end
        fold_id += 1

    if not rows:
        return _empty_output()
    return pl.DataFrame(rows).sort("data_time")


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(schema={"data_time": pl.Date, "fold_id": pl.Int64})
