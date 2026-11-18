"""Tests for the SMARD loader on the 2025-09-04 to 2025-10-07 sample (set SMARD_SAMPLE_DIR)."""
import os
from pathlib import Path

import pandas as pd
import pytest

from iexfc.smard import QH_SWITCH, check_grid, hourly_repeat_share, load_smard, whole_days

SAMPLE = Path(os.environ.get("SMARD_SAMPLE_DIR", "data/external/smard_sample"))
FILES = {k: next(iter(sorted(SAMPLE.glob(f"*{k}*Quarterhour.csv"))), None)
         for k in ("Day-ahead_prices", "Actual_consumption", "Actual_generation")}
pytestmark = pytest.mark.skipif(any(v is None for v in FILES.values()), reason="SMARD sample not present")


@pytest.fixture(scope="module")
def data():
    return {k: load_smard(p) for k, p in FILES.items()}


def test_regular_grid_and_alignment(data):
    for df in data.values():
        check_grid(df)
    idx = [df.index for df in data.values()]
    assert all(i.equals(idx[0]) for i in idx), "the three files must share one time grid"


def test_parsing(data):
    p = data["Day-ahead_prices"]
    assert p.index.tz is not None and str(p.index.tz) == "Europe/Berlin"
    assert "Germany/Luxembourg" in p.columns
    first = p.index[0]
    assert (first.month, first.day, first.hour, first.minute) == (9, 4, 20, 30)
    assert p["Germany/Luxembourg"].iloc[0] == pytest.approx(218.82)
    g = data["Actual_generation"]
    assert g["Biomass"].iloc[0] == pytest.approx(1145.44)  # thousands separator removed
    assert g["Nuclear"].isna().all()  # '-' read as missing (no German nuclear since 2023)


def test_hourly_before_switch_quarterhour_after(data):
    de = data["Day-ahead_prices"]["Germany/Luxembourg"]
    before = de[de.index < QH_SWITCH]
    after = de[de.index >= QH_SWITCH]
    assert hourly_repeat_share(before) == 1.0
    assert hourly_repeat_share(after) < 0.5


def test_whole_days(data):
    w = whole_days(data["Day-ahead_prices"])
    days = w.index.normalize().unique()
    assert len(w) == 96 * len(days)
    assert w.index[0].hour == 0 and w.index[-1].hour == 23 and w.index[-1].minute == 45


HOURLY = sorted(Path(os.environ.get("SMARD_DIR", "data/external/smard")).glob("*Day-ahead_prices*_Hour.csv"))


@pytest.mark.skipif(not HOURLY, reason="hourly SMARD price files not present")
def test_hourly_prices_join_without_gaps():
    from iexfc.smard import load_many
    p = whole_days(load_many(HOURLY), step_seconds=3600)
    check_grid(p, step_seconds=3600)
    assert p["Germany/Luxembourg"].isna().sum() == 0
    assert p["Germany/Luxembourg"].min() >= -500  # harmonised minimum clearing price
