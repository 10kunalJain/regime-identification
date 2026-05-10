"""Crisis-label generation from observable forward outcomes.

Headline label: `forward_drawdown_indicator` — 1 if SPY's max drawdown over
the next `horizon` trading days exceeds `threshold`, else 0. The last
`horizon` rows are marked `-1` (unobservable) so calibration / training code
can mask them out cleanly.

Per Q5/Q10 of the design grill: this is what the calibrator and the crisis
head are trained to predict. The label is computed entirely from price data
the model does not use as a feature, so there's no regime-label tautology.
"""

from __future__ import annotations

import numpy as np

UNOBSERVABLE = -1
DEFAULT_HORIZON = 21
DEFAULT_THRESHOLD = 0.10


def forward_drawdown_indicator(
    close: np.ndarray,
    horizon: int = DEFAULT_HORIZON,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Return a length-N array of {0, 1, UNOBSERVABLE}.

    `y[t] == 1` iff `(close[t] - min(close[t+1 : t+horizon+1])) / close[t] > threshold`.
    `y[t] == 0` otherwise, while still observable.
    `y[t] == UNOBSERVABLE` for the last `horizon` rows where the future window
    has fewer than `horizon` trading days available.
    """
    close = np.asarray(close, dtype=np.float64).reshape(-1)
    n = len(close)
    y = np.full(n, UNOBSERVABLE, dtype=np.int64)
    if n <= horizon:
        return y

    for t in range(n - horizon):
        window = close[t + 1 : t + horizon + 1]
        if len(window) < horizon:
            continue
        max_dd = (close[t] - window.min()) / close[t]
        y[t] = 1 if max_dd > threshold else 0
    return y


def observable_mask(y: np.ndarray) -> np.ndarray:
    """Boolean mask of rows whose label is observable (not UNOBSERVABLE)."""
    return np.asarray(y) != UNOBSERVABLE
