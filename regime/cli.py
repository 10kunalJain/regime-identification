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
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation commands.")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")


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


@eval_app.command("walkforward")
def eval_walkforward(
    model: str = typer.Option("joint_hmm", help="Model id. Only 'joint_hmm' is wired today."),
    start: str = typer.Option("2003-01-01", help="ISO date; inclusive."),
    end: str | None = typer.Option(None, help="ISO date; inclusive. Default = today."),
    output: str | None = typer.Option(
        "build/eval/walkforward_joint_hmm.parquet",
        help="Output parquet path. Set to '' to skip writing.",
    ),
    initial_train_rows: int = typer.Option(
        1260, help="Rows in the first training fold (~5y = 1260)."
    ),
    refit_every_rows: int = typer.Option(
        252, help="Rows per test window; refit at each boundary (~1y = 252)."
    ),
) -> None:
    """Walk-forward eval: fit on expanding windows, score detection lag against
    the canonical crisis registry. Prints fold + crisis tables; optionally
    writes a long-format parquet."""
    from datetime import date as _date

    from regime.eval.runner import (
        WalkForwardConfig,
        crisis_lag_to_dataframe,
        folds_to_dataframe,
        format_crisis_lag_table,
        format_fold_summary,
        run_joint_hmm_walkforward,
    )

    if model != "joint_hmm":
        typer.secho(
            f"model={model!r} not wired; only 'joint_hmm' is supported today.",
            fg=typer.colors.RED,
        )
        sys.exit(2)

    start_date = _date.fromisoformat(start)
    end_date = _date.fromisoformat(end) if end else _date.today()
    cfg = WalkForwardConfig(
        initial_train_rows=initial_train_rows,
        refit_every_rows=refit_every_rows,
    )
    result = _run_joint_hmm_eval(start_date, end_date, cfg, run_joint_hmm_walkforward)

    typer.echo("\nPer-fold metrics:")
    typer.echo(format_fold_summary(result.folds))
    typer.echo("\nPer-crisis detection lag:")
    typer.echo(format_crisis_lag_table(result.crisis_lag))

    if output:
        from pathlib import Path

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.posterior.write_parquet(out_path)
        crisis_path = out_path.with_name(out_path.stem + "_crisis_lag.parquet")
        folds_path = out_path.with_name(out_path.stem + "_folds.parquet")
        crisis_lag_to_dataframe(result.crisis_lag).write_parquet(crisis_path)
        folds_to_dataframe(result.folds).write_parquet(folds_path)
        typer.echo(f"\nwrote {out_path}")
        typer.echo(f"wrote {crisis_path}")
        typer.echo(f"wrote {folds_path}")


def _run_joint_hmm_eval(start, end, cfg, runner):
    """Tiny wrapper isolating the data-fetch + model-factory from the CLI body.

    Kept module-level (not nested) so a future refactor can re-use it directly.
    """
    from datetime import date as _date

    import polars as pl

    from regime.data.joint_dataset import (
        FF_COLUMNS,
        OBSERVATION_TICKERS,
        RENAMED_FF,
        build_wide_dataframe,
    )
    from regime.models.joint_hmm import JointHmm

    assert isinstance(end, _date)
    df = build_wide_dataframe(end)
    df = df.filter(pl.col("data_time") >= start)
    obs_cols = tuple(f"ret_{t}" for t in OBSERVATION_TICKERS)
    fact_cols = tuple(RENAMED_FF[f] for f in FF_COLUMNS)
    typer.echo(
        f"loaded {df.height} rows × {df.width} cols  "
        f"({df['data_time'].min()} → {df['data_time'].max()})"
    )

    def factory():
        return JointHmm(
            K=3,
            observation_columns=obs_cols,
            factor_columns=fact_cols,
            latent_factor_rank=3,
            n_restarts=3,
            max_iter=50,
            random_state=42,
        )

    return runner(df, factory, cfg)


if __name__ == "__main__":
    app()
