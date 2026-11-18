"""Feature table for day-ahead price forecasting (one row per delivery day x 15-minute block).

Information rule (what is known when forecasting delivery day D, on the morning of D-1):
  * IEX results (prices, bids, volumes) for delivery days <= D-1. They were cleared on D-2.
  * Grid-India daily report for day X is published on X+1, so only days <= D-2 are known.
    Demand and generation features are therefore lagged by 2 days (conservative).
  * Calendar fields and the cap level are known in advance.
Nothing from day D itself is used as an input.

Target days start on 2023-04-15: the ceiling has been Rs 10,000/MWh since early April 2023, so
one cap regime applies to every target and to all the lags (max 7 days) behind it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .cap import documented_cap

QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CAP_TOL = 0.995
POSOCO_COLS = {
    "India: MaximumDemand": "max_demand_mw",
    "India: EnergyMet": "energy_gwh",
    "India: SolarGen": "solar_gwh",
    "India: WindGen": "wind_gwh",
    "India: HydroGen": "hydro_gwh",
    "India: Coal": "coal_gwh",
}


def load_posoco(path) -> pd.DataFrame:
    r = pd.read_csv(path)
    r["date"] = pd.to_datetime(r["yyyymmdd"].astype(str), format="%Y%m%d")
    return r.set_index("date")[list(POSOCO_COLS)].rename(columns=POSOCO_COLS)


def _stack_stat(frame: pd.DataFrame, lags, fn) -> pd.DataFrame:
    arr = np.stack([frame.shift(k).to_numpy() for k in lags])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fn(arr, axis=0)
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def build_features(prices: pd.DataFrame, posoco: pd.DataFrame, first_target: str = "2023-04-15") -> pd.DataFrame:
    idx = pd.date_range(prices["date"].min(), prices["date"].max(), freq="D")
    cap = pd.Series([documented_cap(d) for d in idx], index=idx)

    def wide(col):
        return prices.pivot(index="date", columns="block_no", values=col).reindex(idx)

    P, BUY, SELL, MCV = (wide(c) for c in ("mcp_rs_mwh", "purchase_bid_mw", "sell_bid_mw", "mcv_mw"))
    AT = P.ge(CAP_TOL * cap, axis=0).astype(float).where(P.notna())
    G = posoco.reindex(idx)

    tdays = idx[idx >= pd.Timestamp(first_target)]
    n = len(tdays)
    cols: dict[str, np.ndarray] = {}

    def put(name, frame):
        cols[name] = frame.loc[tdays].to_numpy().ravel()

    def put_day(name, series):
        cols[name] = np.repeat(series.loc[tdays].to_numpy(), 96)

    # --- price history for the same block ---
    put("p1", P.shift(1)); put("p2", P.shift(2)); put("p7", P.shift(7))
    put("p1_prev_block", P.shift(1).shift(1, axis=1)); put("p1_next_block", P.shift(1).shift(-1, axis=1))
    put("p_mean7", _stack_stat(P, range(1, 8), np.nanmean))
    put("p_max7", _stack_stat(P, range(1, 8), np.nanmax))
    put("p_min7", _stack_stat(P, range(1, 8), np.nanmin))
    # --- cap-hit history ---
    put("at1", AT.shift(1)); put("at7", AT.shift(7)); put("at_mean7", _stack_stat(AT, range(1, 8), np.nanmean))
    # --- exchange bids and volume (yesterday, same block) ---
    put("buy1", BUY.shift(1)); put("sell1", SELL.shift(1)); put("mcv1", MCV.shift(1))
    put("buy7", BUY.shift(7)); put("sell7", SELL.shift(7))
    put("buy_sell_ratio1", BUY.shift(1) / SELL.shift(1))
    # --- yesterday's whole-day summary ---
    dmean = P.mean(axis=1)
    put_day("d1_mean", dmean.shift(1))
    put_day("d1_max", P.max(axis=1).shift(1)); put_day("d1_min", P.min(axis=1).shift(1))
    put_day("d1_cap_share", AT.mean(axis=1).shift(1))
    put_day("d1_evening_mean", P.iloc[:, 72:88].mean(axis=1).shift(1))  # blocks 73-88 = 18:00-22:00
    put_day("d1_midday_mean", P.iloc[:, 40:60].mean(axis=1).shift(1))   # blocks 41-60 = 10:00-15:00
    put_day("d1_buy_mean", BUY.mean(axis=1).shift(1)); put_day("d1_sell_mean", SELL.mean(axis=1).shift(1))
    put_day("mean_price_7d", dmean.shift(1).rolling(7, min_periods=4).mean())
    # --- Grid-India daily data, lagged 2 days ---
    for c in POSOCO_COLS.values():
        put_day(f"{c}_lag2", G[c].shift(2))
    put_day("max_demand_chg_wk", G["max_demand_mw"].shift(2) - G["max_demand_mw"].shift(9))
    put_day("energy_chg_wk", G["energy_gwh"].shift(2) - G["energy_gwh"].shift(9))
    # --- calendar (known in advance) ---
    cols["block"] = np.tile(np.arange(1, 97), n)
    cols["dow"] = np.repeat(tdays.dayofweek.to_numpy(), 96)
    cols["month"] = np.repeat(tdays.month.to_numpy(), 96)
    cols["doy"] = np.repeat(tdays.dayofyear.to_numpy(), 96)

    feats = pd.DataFrame(cols)
    # --- baselines and target (never used as inputs) ---
    out = feats.copy()
    out.insert(0, "date", np.repeat(tdays.to_numpy(), 96))
    out["y"] = P.loc[tdays].to_numpy().ravel()
    out["y_cap"] = (out["y"] >= CAP_TOL * np.repeat(cap.loc[tdays].to_numpy(), 96)).astype(float).where(out["y"].notna())
    Pshift = P.shift(1)
    for q in QS:
        bq = Pshift.rolling(28, min_periods=20).quantile(q)
        out[f"base_q{int(q * 100)}"] = bq.loc[tdays].to_numpy().ravel()
    out = out[out["y"].notna()].reset_index(drop=True)
    return out


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in ("date", "y", "y_cap") and not c.startswith("base_")]
