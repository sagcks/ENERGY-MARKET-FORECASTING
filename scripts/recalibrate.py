"""Recalibrate quantile forecasts using only past errors (no retraining).

Usage: python scripts/recalibrate.py            # reads data/results/predictions.csv
For each target day D and each quantile level tau, shift the forecast by the amount that would
have made the last `--window` days (ending D-2, i.e. outcomes already published) hit tau exactly.
Writes data/results/predictions_cal.csv and prints before/after coverage and pinball loss.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

QS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
CAP = 10000.0

def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))

ap = argparse.ArgumentParser()
ap.add_argument("--pred", type=Path, default=Path("data/results/predictions.csv"))
ap.add_argument("--window", type=int, default=60)
ap.add_argument("--prefix", default="q", help="quantile column prefix: q (model_gbm) or k (model_capaware)")
ap.add_argument("--out", default=None, help="output file name")
a = ap.parse_args()
R = pd.read_csv(a.pred, parse_dates=["date"])
X = a.prefix
days = sorted(R["date"].unique())
first_ok = days[0] + pd.Timedelta(days=a.window + 2)
out = []
for d in days:
    if d < first_ok:
        continue
    cur = R[R["date"] == d].copy()
    past = R[(R["date"] >= d - pd.Timedelta(days=a.window + 1)) & (R["date"] <= d - pd.Timedelta(days=2))]
    for q in QS:
        shift = np.quantile(past["y"] - past[f"{X}{q}"], q / 100)
        cur[f"c{q}"] = cur[f"{X}{q}"] + shift
    c = np.sort(cur[[f"c{q}" for q in QS]].to_numpy(), axis=1).clip(0, CAP)
    for i, q in enumerate(QS):
        cur[f"c{q}"] = c[:, i]
    out.append(cur)
C = pd.concat(out, ignore_index=True)
C.to_csv(a.pred.with_name(a.out or "predictions_cal.csv"), index=False)
y = C["y"].to_numpy()
def report(name, g, pre):
    yy = g["y"].to_numpy()
    pb = np.mean([pinball(yy, g[f"{pre}{q}"].to_numpy(), q / 100) for q in QS])
    cov = np.mean((yy >= g[f"{pre}10"]) & (yy <= g[f"{pre}90"]))
    print(f"{name:14s} pinball {pb:6.1f}   80% coverage {cov:.3f}")
print(f"Evaluated {C['date'].min():%Y-%m-%d} to {C['date'].max():%Y-%m-%d}, {len(C):,} blocks")
for nm, g in [("all", C), ("2026", C[C.date.dt.year == 2026]),
              ("Sep-Oct 2026", C[(C.date >= "2026-09-01") & (C.date <= "2026-10-31")])]:
    print(nm); report("  raw", g, X); report("  recalibrated", g, "c")
print("Calibration by level (raw -> recalibrated):")
print({q: (round((C.y <= C[f'{X}{q}']).mean(), 2), round((C.y <= C[f'c{q}']).mean(), 2)) for q in QS})
