"""Load SMARD (Bundesnetzagentur | SMARD.de, CC BY 4.0) CSV exports.

SMARD's download centre exports semicolon-separated files with:
  * a UTF-8 byte-order mark and Windows line endings,
  * timestamps in German local time, written like "Sep 4, 2025 8:30 PM",
  * thousands separators ("1,145.44") and "-" for missing values,
  * column names like "Germany/Luxembourg [€/MWh] Original resolutions".

Before 1 October 2025 the German day-ahead auction cleared hourly; a quarter-hour export
repeats each hourly price in all four quarters. From 1 October 2025 prices are 15-minute.

load_smard() returns a DataFrame indexed by the interval start as a tz-aware Europe/Berlin
timestamp, with short column names and float values.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"
QH_SWITCH = pd.Timestamp("2025-10-01", tz=TZ)  # first delivery day with 15-minute day-ahead prices


def _short(col: str) -> str:
    col = re.sub(r"\s*(Original|Calculated) resolutions\s*$", "", col)
    col = re.sub(r"\s*\[[^\]]*\]\s*$", "", col)  # drop the unit
    return col.strip()


def load_smard(path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, na_values=["-"], keep_default_na=False)
    start = pd.to_datetime(raw["Start date"], format="%b %d, %Y %I:%M %p")
    # Localise German wall-clock time. In the autumn change the hour 02:00-03:00 appears twice;
    # 'infer' uses the order of rows to tell the two apart. Spring-forward hours do not exist.
    idx = pd.DatetimeIndex(start).tz_localize(TZ, ambiguous="infer", nonexistent="raise")
    values = raw.drop(columns=["Start date", "End date"])
    values = values.apply(lambda s: pd.to_numeric(s.str.replace(",", "", regex=False), errors="raise"))
    values.columns = [_short(c) for c in values.columns]
    values.index = idx
    values.index.name = "start"
    return values.astype(float)


def whole_days(df: pd.DataFrame, step_seconds: int = 900) -> pd.DataFrame:
    """Keep only German calendar days that are complete (e.g. 92/96/100 quarter-hours or 23/24/25 hours)."""
    day = df.index.tz_convert(TZ).normalize()
    counts = pd.Series(1, index=df.index).groupby(day).transform("size")
    expected = pd.Series(
        # wall-clock next midnight, so DST days count 23 or 25 hours
        [((d.tz_localize(None) + pd.Timedelta(days=1)).tz_localize(TZ) - d).total_seconds() / step_seconds for d in day],
        index=df.index,
    )
    return df[counts.values == expected.values]


def load_many(paths) -> pd.DataFrame:
    """Load several SMARD exports of the same series (e.g. year blocks), join them and drop overlaps."""
    df = pd.concat([load_smard(p) for p in paths]).sort_index()
    return df[~df.index.duplicated(keep="first")]


def check_grid(df: pd.DataFrame, step_seconds: int = 900) -> dict:
    """Basic integrity checks; returns a summary dict (raises on a broken time grid)."""
    utc = df.index.tz_convert("UTC")
    steps = (utc[1:] - utc[:-1]).total_seconds().to_numpy()
    if not (steps == step_seconds).all():
        bad = utc[1:][steps != 900][:5]
        raise ValueError(f"time grid is not regular 15-minute; first gaps at {list(bad)}")
    if utc.has_duplicates:
        raise ValueError("duplicate timestamps")
    return {"rows": len(df), "first": df.index[0], "last": df.index[-1], "missing": int(df.isna().sum().sum())}


def hourly_repeat_share(price: pd.Series) -> float:
    """Share of clock hours in which all four quarter-hour prices are identical."""
    g = price.groupby(price.index.floor("h"))
    full = g.size() == 4
    same = g.nunique() == 1
    return float((same & full).sum() / full.sum()) if full.sum() else float("nan")
