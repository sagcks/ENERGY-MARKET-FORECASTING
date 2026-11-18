"""Parse an IEX 'DAM Market Snapshot' Excel export into a tidy DataFrame.

File layout (verified on the 1-7 April 2022 export):
  row 2:  'Market Snapshot'
  row 3:  'Date: 01-04-2022 to 07-04-2022'   <- the range that was requested
  row 5:  header: Date, Hour, Time Block, Purchase Bid (MW), Sell Bid (MW), MCV (MW),
          Final Scheduled Volume (MW), MCP (Rs/MWh) *
  then 96 rows per day (15-minute blocks), followed by per-day Total/Max/Min/Avg rows,
  which we drop.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BLOCK_RE = r"^\d{2}:\d{2} - \d{2}:\d{2}$"
RANGE_RE = re.compile(r"Date:\s*(\d{2}-\d{2}-\d{4})\s*to\s*(\d{2}-\d{2}-\d{4})")
NUMERIC = ["purchase_bid_mw", "sell_bid_mw", "mcv_mw", "scheduled_mw", "mcp_rs_mwh"]
COLUMNS = ["date", "hour", "block", *NUMERIC]


def parse_export(path: Path) -> pd.DataFrame:
    """Return one row per 15-minute block. Raises ValueError if the layout is not recognised.

    df.attrs['requested'] holds the (start, end) range written in the file's title line.
    """
    raw = pd.read_excel(path, header=None)

    requested = None
    for cell in raw.iloc[:8, 0].fillna("").astype(str):
        m = RANGE_RE.search(cell)
        if m:
            requested = (pd.to_datetime(m.group(1), format="%d-%m-%Y"),
                         pd.to_datetime(m.group(2), format="%d-%m-%Y"))
            break
    if requested is None:
        raise ValueError(f"{path.name}: no 'Date: dd-mm-yyyy to dd-mm-yyyy' line found")

    header_rows = raw.index[raw[0].fillna("").astype(str).str.strip() == "Date"]
    if len(header_rows) == 0:
        raise ValueError(f"{path.name}: no table header found")

    body = raw.iloc[header_rows[0] + 1:, :8].copy()
    body.columns = COLUMNS
    body = body[body["block"].fillna("").astype(str).str.match(BLOCK_RE)].copy()
    if body.empty:
        raise ValueError(f"{path.name}: no 15-minute data rows found")

    body["date"] = pd.to_datetime(body["date"], format="%d-%m-%Y")
    for col in NUMERIC:
        body[col] = pd.to_numeric(body[col], errors="raise")
    body["hour"] = body["hour"].astype(int)
    body = body.sort_values(["date", "block"]).reset_index(drop=True)
    body["block_no"] = body.groupby("date").cumcount() + 1
    body["start"] = body["block"].str[:5]
    body.attrs["requested"] = requested
    return body[["date", "block_no", "start", "hour", *NUMERIC]]


def problems(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Human-readable data problems: missing days, days without exactly 96 blocks, duplicates."""
    out = []
    expected = pd.date_range(start, end, freq="D")
    present = pd.DatetimeIndex(df["date"].unique())
    missing = expected.difference(present)
    if len(missing):
        out.append(f"{len(missing)} missing day(s): {[d.strftime('%Y-%m-%d') for d in missing[:5]]}")
    counts = df.groupby("date").size()
    odd = counts[counts != 96]
    if len(odd):
        out.append(f"{len(odd)} day(s) without 96 blocks: {odd.head(5).to_dict()}")
    if df.duplicated(["date", "block_no"]).any():
        out.append("duplicate date/block rows")
    return out
