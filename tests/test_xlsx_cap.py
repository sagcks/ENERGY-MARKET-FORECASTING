import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iexfc.cap import documented_cap, monthly_cap_summary  # noqa: E402
from iexfc.xlsx import parse_export, problems  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "dam_2022-04-01_2022-04-07.xlsx"
# The IEX fixture is not published with the code; download that week to run these tests.
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="IEX fixture not present")


@pytest.fixture(scope="module")
def df():
    return parse_export(FIXTURE)


def test_parse_shape_and_range(df):
    assert len(df) == 7 * 96
    assert df["date"].nunique() == 7
    assert df.attrs["requested"] == (pd.Timestamp("2022-04-01"), pd.Timestamp("2022-04-07"))
    assert df["block_no"].min() == 1 and df["block_no"].max() == 96
    assert df.isna().sum().sum() == 0


def test_parse_values_match_the_file(df):
    first = df.iloc[0]
    assert first["start"] == "00:00" and first["mcp_rs_mwh"] == 20000.0
    assert df.groupby("date")["mcp_rs_mwh"].max().iloc[0] == 20000.0   # 1 April 2022
    assert df.groupby("date")["mcp_rs_mwh"].max().iloc[-1] == 12000.0  # 7 April 2022


def test_no_problems_on_clean_file(df):
    assert problems(df, pd.Timestamp("2022-04-01"), pd.Timestamp("2022-04-07")) == []


def test_problems_detects_a_missing_day_and_short_day(df):
    broken = df[df["date"] != pd.Timestamp("2022-04-03")]
    broken = broken.drop(broken.index[:5])  # 1 April loses 5 blocks
    msgs = problems(broken, pd.Timestamp("2022-04-01"), pd.Timestamp("2022-04-07"))
    assert any("missing" in m for m in msgs)
    assert any("96" in m for m in msgs)


def test_parse_fails_loudly_on_unknown_layout(tmp_path):
    p = tmp_path / "bad.xlsx"
    pd.DataFrame([["nothing", "here"]]).to_excel(p, header=False, index=False)
    with pytest.raises(ValueError):
        parse_export(p)


def test_documented_cap_regimes():
    assert documented_cap(pd.Timestamp("2022-04-01")) == 20_000.0
    assert documented_cap(pd.Timestamp("2022-04-10")) == 12_000.0
    assert documented_cap(pd.Timestamp("2023-04-03")) == 12_000.0
    assert documented_cap(pd.Timestamp("2023-04-04")) == 10_000.0


def test_monthly_cap_summary(df):
    row = monthly_cap_summary(df).iloc[0]
    assert row["max_mcp"] == 20000.0
    assert 0 < row["share_at_cap"] < 0.1
