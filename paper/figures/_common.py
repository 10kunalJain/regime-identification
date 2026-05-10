"""Shared matplotlib styling for paper figures.

Black-and-white-friendly, monospaced numbers, no decorative chartjunk —
matches the dashboard aesthetic and the writeup's quant-paper convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

FIGURE_WIDTH_INCHES = 5.5
FIGURE_HEIGHT_INCHES = 3.5
DPI = 200


def configure_style() -> None:
    """Apply the paper's matplotlib style. Idempotent — safe to call repeatedly."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "legend.fontsize": 8,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_output_path() -> Path:
    """Standard CLI: every figure script accepts `--output <path>` from the Makefile."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the figure (.png by convention).",
    )
    args = p.parse_args()
    out: Path = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def save_and_close(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output)
    plt.close(fig)
