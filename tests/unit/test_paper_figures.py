"""Smoke tests for the paper-figure scripts.

Each script must run end-to-end and produce a PNG file at the requested path.
Tests run the scripts as subprocesses to mirror what the Makefile does.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper" / "figures"


def _run_script(script_path: Path, output_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script_path), "--output", str(output_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{script_path.name} failed with exit code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


@pytest.mark.parametrize(
    "script_name",
    ["figure_smoothed_vs_filtered.py", "figure_reliability.py"],
)
def test_figure_script_produces_png(tmp_path: Path, script_name: str):
    script = FIGURES / script_name
    assert script.exists(), f"{script_name} missing"
    output = tmp_path / f"{script.stem}.png"
    _run_script(script, output)
    assert output.exists()
    # PNG signature: 8 bytes 89 50 4E 47 0D 0A 1A 0A
    sig = output.read_bytes()[:8]
    assert sig == b"\x89PNG\r\n\x1a\n", "output is not a valid PNG"
    # Some non-trivial size (>2 KB) — empty plots are smaller.
    assert output.stat().st_size > 2048
