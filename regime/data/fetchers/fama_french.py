"""Fama-French daily 5+Mom factor downloader.

Pulls the daily research factors (Mkt-RF, SMB, HML, RMW, CMA) and the Momentum factor
from Ken French's data library. Free, no API key required.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from datetime import date

import polars as pl

from regime.data import store

_LOG = logging.getLogger(__name__)

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
)


def refresh() -> None:
    ff5 = _download_csv_from_zip(FF5_URL)
    mom = _download_csv_from_zip(MOM_URL)
    df = ff5.join(mom, on="data_time", how="inner")
    df = df.with_columns(pl.col("data_time").alias("knowledge_time"))
    store.write_parquet(df, store.fama_french_path())


def _download_csv_from_zip(url: str) -> pl.DataFrame:
    _LOG.info("downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "regime/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        zip_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as f:
            text = f.read().decode("latin-1")
    return _parse_ff_csv(text)


def _parse_ff_csv(text: str) -> pl.DataFrame:
    """Parse a Ken French daily-factor CSV.

    The format prepends a multi-line preamble (descriptive text), then a single
    header line (factor names, with an empty leading field for the date), then
    daily data rows of `YYYYMMDD,val,val,...`, then sometimes a blank line and
    annual data after that.

    We locate the header as: the line *immediately preceding* the first row
    starting with an 8-digit YYYYMMDD date — this avoids matching factor names
    embedded in the preamble's prose (e.g., "MOM is the average ..." in the
    momentum file, which our naive substring-match would have hit first).
    """
    lines = text.splitlines()

    # Find the index of the first 8-digit-YYYYMMDD row.
    first_data_idx: int | None = None
    for i, ln in enumerate(lines):
        first_token = ln.strip().split(",")[0].strip()
        if first_token.isdigit() and len(first_token) == 8:
            first_data_idx = i
            break
    if first_data_idx is None or first_data_idx == 0:
        return pl.DataFrame()

    # Scan upward from the first data row for the header — the closest
    # non-empty line above. This naturally skips blank separator lines.
    header_idx: int | None = None
    for j in range(first_data_idx - 1, -1, -1):
        if lines[j].strip():
            header_idx = j
            break
    if header_idx is None:
        return pl.DataFrame()

    header = [h.strip() for h in lines[header_idx].strip().split(",")]
    header[0] = "data_time"

    rows: list[list[float]] = []
    dates: list[date] = []
    for ln in lines[first_data_idx:]:
        s = ln.strip()
        if not s:
            break
        first = s.split(",")[0].strip()
        if not (first.isdigit() and len(first) == 8):
            break
        parts = [p.strip() for p in s.split(",")]
        dates.append(_parse_yyyymmdd(parts[0]))
        rows.append([float(p) / 100.0 for p in parts[1:]])

    data: dict[str, list[date] | list[float]] = {header[0]: dates}
    for i, name in enumerate(header[1:], start=1):
        data[name] = [r[i - 1] for r in rows]
    schema = {h: (pl.Date if i == 0 else pl.Float64) for i, h in enumerate(header)}
    return pl.DataFrame(data, schema=schema)


def _parse_yyyymmdd(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))
