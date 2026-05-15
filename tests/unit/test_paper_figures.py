"""Smoke tests for the paper-figure scripts.

Each script must run end-to-end and produce a valid PNG at the requested
path. Tests run the scripts as subprocesses to mirror what the Makefile
does, and pass `--input <fixture>` so the smoke test never depends on the
locally-built `build/paper/...` artefacts (which are gitignored). The
fixtures live under `tests/fixtures/paper/` and are regenerable via
`uv run python scripts/build_paper_inputs.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper" / "figures"
FIXTURES = ROOT / "tests" / "fixtures" / "paper"


FIGURE_SPECS: tuple[tuple[str, str], ...] = (
    ("figure_smoothed_vs_filtered.py", "posterior_covid.parquet"),
    ("figure_reliability.py", "crisis_head.parquet"),
    ("figure_detection_lag.py", "methods_crisis_lag.parquet"),
    ("figure_regime_path.py", "regime_path.parquet"),
)


def _run_script(script_path: Path, input_path: Path, output_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{script_path.name} failed with exit code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


@pytest.mark.parametrize(("script_name", "fixture_name"), FIGURE_SPECS)
def test_figure_script_produces_png(tmp_path: Path, script_name: str, fixture_name: str):
    script = FIGURES / script_name
    fixture = FIXTURES / fixture_name
    assert script.exists(), f"{script_name} missing"
    assert fixture.exists(), (
        f"{fixture_name} missing — regenerate via scripts/build_paper_inputs.py"
    )

    output = tmp_path / f"{script.stem}.png"
    _run_script(script, fixture, output)
    assert output.exists()
    sig = output.read_bytes()[:8]
    assert sig == b"\x89PNG\r\n\x1a\n", "output is not a valid PNG"
    assert output.stat().st_size > 2048
