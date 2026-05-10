"""data.lock generation and verification.

A SHA256 manifest of every Parquet partition under the data root. Committed to the
repo so that any divergence between a local rebuild and the committed manifest fails
CI — the engineering signal that "everyone sees the same numbers."
"""

from __future__ import annotations

import hashlib
from pathlib import Path

LOCK_FILENAME = "data.lock"


def lock_path() -> Path:
    return Path.cwd() / LOCK_FILENAME


def write_lock(root: Path) -> Path:
    """Write data.lock with SHA256 of every Parquet partition under root."""
    entries = _scan(root)
    p = lock_path()
    with p.open("w") as f:
        f.write("# data.lock — sha256 manifest of every Parquet partition\n")
        for rel, sha in entries:
            f.write(f"{rel}  sha256:{sha}\n")
    return p


def verify_lock(root: Path) -> bool:
    """Return True iff data.lock matches scanned state of root."""
    p = lock_path()
    if not p.exists():
        return False
    expected: dict[str, str] = {}
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        rel, sha_token = line.rsplit("  ", 1)
        expected[rel.strip()] = sha_token.strip().removeprefix("sha256:")
    actual = dict(_scan(root))
    return expected == actual


def _scan(root: Path) -> list[tuple[str, str]]:
    if not root.exists():
        return []
    parts = sorted(root.rglob("*.parquet"))
    return [(str(p.relative_to(root)), _sha256(p)) for p in parts]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
