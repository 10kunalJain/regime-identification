"""Headline figure: filtered vs smoothed posterior on a synthetic 3-state HMM.

Illustrates the central wedge of the paper — that smoothed posteriors use
information from the entire fold and assign high crisis probability several
days before any real-time observer could. The synthetic data here lets the
figure run with no external data fetch; the real-data analogue (replacing
`sample_gaussian_hmm` with `as_of` queries on COVID-window SPY returns) is a
single-line substitution and is what the published paper will use.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python paper/figures/figure_smoothed_vs_filtered.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _common import (
    FIGURE_HEIGHT_INCHES,
    FIGURE_WIDTH_INCHES,
    configure_style,
    parse_output_path,
    save_and_close,
)

from regime.models._hmm_core import (
    forward_backward,
    forward_filter,
    gaussian_log_emissions,
    sample_gaussian_hmm,
)


def main() -> int:
    output = parse_output_path()
    configure_style()

    rng = np.random.default_rng(20200319)
    pi = np.array([0.7, 0.25, 0.05])
    A = np.array(
        [
            [0.97, 0.025, 0.005],
            [0.05, 0.90, 0.05],
            [0.05, 0.20, 0.75],
        ]
    )
    means = np.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 1.5]])
    covs = np.tile(np.eye(2) * 0.1, (3, 1, 1))
    states, X = sample_gaussian_hmm(pi, A, means, covs, T=300, rng=rng)

    log_emissions = gaussian_log_emissions(X, means, covs)
    filtered = forward_filter(log_emissions, np.log(pi), np.log(A))
    smoothed = forward_backward(log_emissions, np.log(pi), np.log(A))

    crisis_filtered = filtered[:, 2]
    crisis_smoothed = smoothed[:, 2]
    is_crisis = states == 2

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    t = np.arange(len(filtered))
    ax.fill_between(
        t,
        0,
        is_crisis.astype(float),
        color="0.85",
        step="mid",
        label="True crisis state",
    )
    ax.plot(
        t,
        crisis_smoothed,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"Smoothed $P(s_t = \text{crisis} \mid y_{1:T})$",
    )
    ax.plot(
        t,
        crisis_filtered,
        color="black",
        linestyle="-",
        linewidth=1.0,
        label=r"Filtered $P(s_t = \text{crisis} \mid y_{1:t})$",
    )
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Posterior probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(t.min(), t.max())
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Smoothed vs filtered crisis-state posterior (synthetic 3-state HMM)")

    save_and_close(fig, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
