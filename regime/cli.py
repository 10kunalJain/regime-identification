"""Command-line interface."""

from __future__ import annotations

import logging
import sys

import typer
from dotenv import load_dotenv

from regime.data import manifest, store

# Load .env at CLI startup so commands inherit FRED_API_KEY / REGIME_DATA_ROOT
# / etc. without the user having to `export` them in every shell. `.env` is
# gitignored; it should never make it into version control.
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True, help="Data layer commands.")
app.add_typer(data_app, name="data")


@data_app.command("refresh")
def data_refresh(
    universe: bool = typer.Option(True, help="Refresh ETF universe via yfinance."),
    factors: bool = typer.Option(True, help="Refresh Fama-French factors."),
    macro: bool = typer.Option(True, help="Refresh FRED macro series."),
    dry_run: bool = typer.Option(False, help="Plan only; do not write."),
) -> None:
    """Refresh all configured data sources into the PIT store."""
    root = store.data_root()
    typer.echo(f"data root: {root}")
    if dry_run:
        typer.echo("dry run — no writes")
        return

    # Lazy imports keep --help fast and avoid network deps at import time.
    if universe:
        from regime.data.fetchers import yfinance_fetcher

        yfinance_fetcher.refresh_all()
    if factors:
        from regime.data.fetchers import fama_french

        fama_french.refresh()
    if macro:
        from regime.data.fetchers import fred

        fred.refresh_all()
    manifest.write_lock(root)
    typer.echo("done")


@data_app.command("verify")
def data_verify() -> None:
    """Verify data.lock matches the local Parquet state."""
    ok = manifest.verify_lock(store.data_root())
    if not ok:
        typer.secho("data.lock mismatch", fg=typer.colors.RED)
        sys.exit(1)
    typer.secho("data.lock OK", fg=typer.colors.GREEN)


@data_app.command("lock")
def data_lock() -> None:
    """Write data.lock with SHA256 of every Parquet partition."""
    p = manifest.write_lock(store.data_root())
    typer.echo(f"wrote {p}")


if __name__ == "__main__":
    app()
