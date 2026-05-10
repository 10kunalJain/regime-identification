default:
    @just --list

# Install dev dependencies
install:
    uv sync --all-extras

# Refresh all data sources (network)
refresh:
    uv run regime data refresh

# Verify data.lock matches local state
verify:
    uv run regime data verify

# Generate / update data.lock
lock:
    uv run regime data lock

# Run all tests with single-threaded BLAS for determinism
test:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 uv run pytest

# Property tests only (the ones that gate CI)
test-prop:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 uv run pytest tests/property -v

# Lint + format check
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint and format
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Type check
types:
    uv run pyright

# Pre-commit on all files
pre-commit:
    uv run pre-commit run --all-files

# Quick: lint + types + tests
check: lint types test
