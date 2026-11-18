"""Combine the downloaded Excel chunks (data/raw_xlsx/dam_<start>_<end>.xlsx) into one dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .xlsx import parse_export, problems


def load_dataset(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("dam_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No dam_*.xlsx files in {raw_dir}")
    df = pd.concat((parse_export(f) for f in files), ignore_index=True)
    df = df.drop_duplicates(["date", "block_no"], keep="last")
    return df.sort_values(["date", "block_no"]).reset_index(drop=True)


def check(df: pd.DataFrame) -> list[str]:
    """Problems over the full span covered by the dataset."""
    return problems(df, df["date"].min(), df["date"].max())
