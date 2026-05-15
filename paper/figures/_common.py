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
from matplotlib.figure import Figure

# Repo root, anchoring default --input paths so `make figures` (which chdirs
# to paper/) and ad-hoc `python paper/figures/foo.py` invocations both work.
REPO_ROOT = Path(__file__).resolve().parents[2]

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


def parse_io_paths(default_input: Path) -> tuple[Path, Path]:
    """CLI helper for real-data figures: `--input <parquet>` + `--output <png>`.

    `default_input` is interpreted relative to the repo root (`REPO_ROOT`)
    when it is a relative path, so the paper Makefile (which chdirs to
    `paper/` before invoking each script) and ad-hoc invocations from the
    repo root both resolve the same file. Tests pass `--input <fixture>`
    explicitly and are unaffected.
    """
    anchored_default = default_input if default_input.is_absolute() else REPO_ROOT / default_input
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=anchored_default,
        help=f"Path to the input parquet (default: {anchored_default}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the figure (.png by convention).",
    )
    args = p.parse_args()
    inp: Path = args.input
    out: Path = args.output
    if not inp.exists():
        raise FileNotFoundError(
            f"input parquet not found: {inp}. Run `uv run python scripts/build_paper_inputs.py`."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    return inp, out


def save_and_close(fig: Figure, output: Path) -> None:
    fig.savefig(output)
    plt.close(fig)
