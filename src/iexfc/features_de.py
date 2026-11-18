"""Feature table for German (DE-LU) day-ahead price forecasting, one row per delivery day x hour.

Mirrors the India design in features.py so results are comparable.

Information rule (forecast for delivery day D is made before the auction closes at 12:00 on D-1):
  * Day-ahead prices (Germany and neighbours) for days <= D-1: published around 13:00 on D-2.
  * Actual consumption and generation: only complete for days <= D-2, so they are lagged 2 days
    (the same conservative lag as Grid-India data for the Indian model).
  * Calendar fields are known in advance.
Nothing from day D is used as an input.

Daylight-saving days are put on a 24-slot grid: on the 23-hour spring day the missing 02:00 slot
is the mean of 01:00 and 03:00; on the 25-hour autumn day the two 02:00 values are averaged.

The "extreme regime" in Germany is the low end: prices below zero (the harmonised floor is
-500 EUR/MWh). y_ext marks hours with a negative price; it plays the role of y_cap in India.
"""
from __future__ import annotations

import glob
import warnings

import numpy as np
import pandas as pd

from .smard import TZ, load_many, whole_days

QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
H = 24
NEIGHBOURS = {"France": "fr", "Netherlands": "nl", "Austria": "at", "Denmark 1": "dk1"}
GEN = {"Wind onshore": "wind_on", "Wind offshore": "wind_off", "Photovoltaics": "solar",
       "Lignite": "lignite", "Hard coal": "coal", "Fossil gas": "gas"}


def load_smard_dir(folder: str) -> dict[str, pd.DataFrame]:
    out = {}
    for key in ("Day-ahead_prices", "Actual_consumption", "Actual_generation"):
        files = sorted(glob.glob(f"{folder}/*{key}*_Hour.csv"))
        if not files:
            raise FileNotFoundError(f"no {key} files in {folder}")
        out[key] = whole_days(load_many(files), step_seconds=3600)
    return out


def to_day_hour(s: pd.Series) -> pd.DataFrame:
    """Series on a tz-aware hourly index -> DataFrame (index = local date, columns = hour 0..23)."""
    loc = s.index.tz_convert(TZ)
    df = pd.DataFrame({"v": s.to_numpy(), "day": loc.tz_localize(None).normalize(), "hour": loc.hour})
    wide = df.groupby(["day", "hour"])["v"].mean().unstack("hour")  # autumn: two 02:00 values averaged
    wide = wide.reindex(columns=range(H))
    gap = wide[2].isna() & wide[1].notna() & wide[3].notna()        # spring: 02:00 missing
    wide.loc[gap, 2] = (wide.loc[gap, 1] + wide.loc[gap, 3]) / 2
    return wide.asfreq("D")


def _stack_stat(frame, lags, fn):
    arr = np.stack([frame.shift(k).to_numpy() for k in lags])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.DataFrame(fn(arr, axis=0), index=frame.index, columns=frame.columns)


def build_features_de(data: dict[str, pd.DataFrame], first_target: str = "2019-02-01") -> pd.DataFrame:
    prices, cons, gen = data["Day-ahead_prices"], data["Actual_consumption"], data["Actual_generation"]
    P = to_day_hour(prices["Germany/Luxembourg"])
    idx = P.index
    EXT = P.lt(0).astype(float).where(P.notna())
    LOAD = to_day_hour(cons["grid load"]).reindex(idx)
    RES = to_day_hour(cons["Residual load"]).reindex(idx)
    G = {v: to_day_hour(gen[k].fillna(0)).reindex(idx) for k, v in GEN.items()}
    NB = {v: to_day_hour(prices[k]).reindex(idx) for k, v in NEIGHBOURS.items()}

    tdays = idx[idx >= pd.Timestamp(first_target)]
    n = len(tdays)
    cols: dict[str, np.ndarray] = {}

    def put(name, frame):
        cols[name] = frame.loc[tdays].to_numpy().ravel()

    def put_day(name, series):
        cols[name] = np.repeat(series.loc[tdays].to_numpy(), H)

    # --- price history, same hour ---
    put("p1", P.shift(1)); put("p2", P.shift(2)); put("p7", P.shift(7))
    put("p1_prev_hour", P.shift(1).shift(1, axis=1)); put("p1_next_hour", P.shift(1).shift(-1, axis=1))
    put("p_mean7", _stack_stat(P, range(1, 8), np.nanmean))
    put("p_max7", _stack_stat(P, range(1, 8), np.nanmax))
    put("p_min7", _stack_stat(P, range(1, 8), np.nanmin))
    # --- extreme-regime (negative price) history ---
    put("ext1", EXT.shift(1)); put("ext7", EXT.shift(7)); put("ext_mean7", _stack_stat(EXT, range(1, 8), np.nanmean))
    # --- neighbour prices yesterday, same hour (known: published on D-2) ---
    for v, F in NB.items():
        put(f"{v}_p1", F.shift(1))
    # --- yesterday's whole-day summary ---
    dmean = P.mean(axis=1)
    put_day("d1_mean", dmean.shift(1)); put_day("d1_max", P.max(axis=1).shift(1)); put_day("d1_min", P.min(axis=1).shift(1))
    put_day("d1_ext_share", EXT.mean(axis=1).shift(1))
    put_day("d1_evening_mean", P.iloc[:, 17:21].mean(axis=1).shift(1))  # 17:00-21:00
    put_day("d1_midday_mean", P.iloc[:, 10:15].mean(axis=1).shift(1))   # 10:00-15:00
    put_day("mean_price_7d", dmean.shift(1).rolling(7, min_periods=4).mean())
    # --- load and generation, lagged 2 days (same hour and daily totals) ---
    put("load_lag2", LOAD.shift(2)); put("resload_lag2", RES.shift(2))
    put_day("load_day_lag2", LOAD.sum(axis=1, min_count=20).shift(2))
    put_day("resload_day_lag2", RES.sum(axis=1, min_count=20).shift(2))
    for v, F in G.items():
        put(f"{v}_lag2", F.shift(2))
        put_day(f"{v}_day_lag2", F.sum(axis=1, min_count=20).shift(2))
    put_day("load_chg_wk", LOAD.sum(axis=1).shift(2) - LOAD.sum(axis=1).shift(9))
    # --- calendar ---
    cols["hour"] = np.tile(np.arange(H), n)
    cols["dow"] = np.repeat(tdays.dayofweek.to_numpy(), H)
    cols["month"] = np.repeat(tdays.month.to_numpy(), H)
    cols["doy"] = np.repeat(tdays.dayofyear.to_numpy(), H)

    out = pd.DataFrame(cols)
    out.insert(0, "date", np.repeat(tdays.to_numpy(), H))
    out["y"] = P.loc[tdays].to_numpy().ravel()
    out["y_ext"] = (out["y"] < 0).astype(float).where(out["y"].notna())
    Pshift = P.shift(1)
    for q in QS:
        out[f"base_q{int(q * 100)}"] = Pshift.rolling(28, min_periods=20).quantile(q).loc[tdays].to_numpy().ravel()
    return out[out["y"].notna()].reset_index(drop=True)


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in ("date", "y", "y_ext") and not c.startswith("base_")]
