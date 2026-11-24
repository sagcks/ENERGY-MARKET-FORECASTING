"""Sensitivity of the main results to design choices made after seeing early results.

Usage:
    python scripts/robustness.py

1. Recalibration window: 30, 60, 90, 120 and 180 days (60 is the setting used in the paper),
   for the one-part and two-part models in India and Germany. All windows are evaluated on the
   same sample (from the first day the 180-day window is available), so the numbers are comparable.
2. Two-part against one-part by calendar half-year, to check that the gain is not driven by
   one period.
Writes data/results/robustness.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data/results"
QS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
WINDOWS = [30, 60, 90, 120, 180]


def pinball_mat(y, Q):
    d = y[:, None] - Q
    t = np.array(QS)[None, :] / 100
    return np.maximum(t * d, (t - 1) * d).mean(axis=1)


def recal(R, prefix, window, lo, hi):
    """Shift each quantile by the error quantile of the previous `window` days ending D-2."""
    R = R.sort_values("date")
    days = np.array(sorted(R["date"].unique()))
    err = {q: (R["y"] - R[f"{prefix}{q}"]).to_numpy() for q in QS}
    dvals = R["date"].to_numpy()
    out = np.full((len(R), len(QS)), np.nan)
    for d in days:
        cur = dvals == d
        past = (dvals >= d - np.timedelta64(window + 1, "D")) & (dvals <= d - np.timedelta64(2, "D"))
        if past.sum() == 0:
            continue
        for i, q in enumerate(QS):
            out[cur, i] = R.loc[cur, f"{prefix}{q}"].to_numpy() + np.quantile(err[q][past], q / 100)
    out = np.sort(out, axis=1)
    if lo is not None or hi is not None:
        out = np.clip(out, lo, hi)
    return out, R


def main():
    rows = []
    a = pd.read_csv(RES / "predictions.csv", parse_dates=["date"])
    b = pd.read_csv(RES / "predictions_capaware.csv", parse_dates=["date"])
    india = a.merge(b[["date", "block"] + [f"k{q}" for q in QS]], on=["date", "block"])
    de = pd.read_csv(RES / "predictions_de.csv", parse_dates=["date"])
    for market, R, clip in [("India", india, (0, 10000)), ("Germany", de, (None, None))]:
        start = R["date"].min() + pd.Timedelta(days=max(WINDOWS) + 2)
        for model, pre in [("one-part", "q"), ("two-part", "k")]:
            raw_mask = R["date"] >= start
            y = R.loc[raw_mask, "y"].to_numpy()
            raw = R.loc[raw_mask, [f"{pre}{q}" for q in QS]].to_numpy()
            rows.append(dict(market=market, model=model, window="raw",
                             pinball=pinball_mat(y, raw).mean(),
                             cov80=np.mean((y >= raw[:, 0]) & (y <= raw[:, -1]))))
            for w in WINDOWS:
                Q, Rs = recal(R, pre, w, *clip)
                m = (Rs["date"] >= start).to_numpy()
                yy = Rs["y"].to_numpy()[m]
                rows.append(dict(market=market, model=model, window=w,
                                 pinball=pinball_mat(yy, Q[m]).mean(),
                                 cov80=np.mean((yy >= Q[m, 0]) & (yy <= Q[m, -1]))))
            print(market, model, "done", flush=True)
        # two-part vs one-part by half-year (raw quantiles)
        R = R.copy()
        R["half"] = R["date"].dt.year.astype(str) + "H" + np.where(R["date"].dt.month <= 6, "1", "2")
        for h, g in R.groupby("half"):
            y = g["y"].to_numpy()
            p1 = pinball_mat(y, g[[f"q{q}" for q in QS]].to_numpy()).mean()
            p2 = pinball_mat(y, g[[f"k{q}" for q in QS]].to_numpy()).mean()
            rows.append(dict(market=market, model="two vs one", window=h, pinball=100 * (p2 / p1 - 1), cov80=np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(RES / "robustness.csv", index=False)
    with pd.option_context("display.width", 140, "display.max_rows", 200):
        print(T.round(3))


if __name__ == "__main__":
    main()
