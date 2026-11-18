"""Price-cap (ceiling) analysis for IEX day-ahead prices, in Rs/MWh.

Regimes below are what the data and news reports suggest. The dates marked VERIFY must be
checked against the CERC orders before they are quoted in a paper:

  * 2022-04-01: ceiling Rs 20/kWh. Seen in the data: MCP = 20,000.00 on 1 April 2022.
  * ~2022-04-07: ceiling Rs 12/kWh. News reports say CERC directed a Rs 12 cap in April 2022;
    in the data, 2 April still reaches 15,000 and 7 April tops out at exactly 12,000.00.  VERIFY the date.
  * 2023-04-04: ceiling Rs 10/kWh (CERC order 04/SM/2023).

`monthly_cap_summary` also reports the empirical monthly maximum, so the regime table can be
checked against the data instead of trusted.
"""
from __future__ import annotations

import pandas as pd

# (effective_from, ceiling in Rs/MWh), in date order.
CAP_REGIMES = [
    (pd.Timestamp("2022-04-01"), 20_000.0),
    (pd.Timestamp("2022-04-07"), 12_000.0),  # VERIFY exact start date
    (pd.Timestamp("2023-04-04"), 10_000.0),
]


def documented_cap(ts: pd.Timestamp) -> float:
    ts = pd.Timestamp(ts)
    cap = CAP_REGIMES[0][1]
    for start, value in CAP_REGIMES:
        if ts >= start:
            cap = value
    return cap


def monthly_cap_summary(df: pd.DataFrame, tol: float = 0.005) -> pd.DataFrame:
    """Per month: empirical max MCP, documented cap, share of blocks at the cap.

    A block counts as 'at cap' if MCP >= (1 - tol) * documented cap, because prices can sit
    just under the ceiling (e.g. 9,999.84).
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["cap"] = d["date"].map(documented_cap)
    d["at_cap"] = d["mcp_rs_mwh"] >= (1 - tol) * d["cap"]
    d["month"] = d["date"].dt.to_period("M")
    return d.groupby("month").agg(
        blocks=("mcp_rs_mwh", "size"),
        max_mcp=("mcp_rs_mwh", "max"),
        documented_cap=("cap", "max"),
        share_at_cap=("at_cap", "mean"),
    )
