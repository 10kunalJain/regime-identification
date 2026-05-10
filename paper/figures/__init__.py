"""Reproducible figure-generation scripts.

Convention:
  - Each script is `figure_<name>.py` and produces `build/<name>.png` when
    invoked with `--output <path>`.
  - Scripts import only from `regime.*` and the standard library; data is
    either pulled from the project's PIT layer or generated synthetically with
    a fixed seed (the latter is the default for figures that illustrate
    methodology rather than report empirical results).
  - All scripts share `paper/figures/_common.py` for matplotlib styling so
    the paper's figures have consistent typography, color, and dimensions.
  - The `paper/Makefile` `figures` target runs every script.
"""
