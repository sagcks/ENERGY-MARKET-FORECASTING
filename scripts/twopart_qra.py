"""Two-part QRA for India: the cap structure on top of linear forecast averaging.

Usage:
    python scripts/twopart_qra.py

The distribution below the cap is modelled by quantile regression of the price on four point
forecasts (LEAR, LightGBM L1, same block yesterday, same block last week), fitted only on
uncapped blocks of all 96 blocks in the 60 days ending D-2 and re-estimated weekly, at 19 levels
(0.05, 0.10, ..., 0.95). The probability of a cap hit p comes from the LightGBM classifier of the
two-part model. The forecast quantile at level tau is the cap if tau > 1 - p and otherwise the
conditional quantile at tau / (1 - p), interpolated linearly between the fitted levels.
Writes data/results/twopart_qra_india.csv with columns t10..t90.
"""
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data/results"
CAP = 10000.0
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
REGS = ["lear", "pred_point", "p1", "p7"]


def main():
    D = pd.read_csv(RES / "benchmarks2_india.csv", parse_dates=["date"])
    P = pd.read_csv(RES / "predictions_capaware.csv", parse_dates=["date"], usecols=["date", "block", "p_cap"])
    D = D.merge(P, on=["date", "block"], how="left").sort_values(["date", "block"]).reset_index(drop=True)
    D["capped"] = D["y"] >= 0.995 * CAP
    weeks = pd.date_range("2024-01-01", D["date"].max() + pd.Timedelta(days=1), freq="7D")
    out = np.full((len(D), len(QS)), np.nan)
    t0 = time.time()
    for a, b in zip(weeks[:-1].tolist() + [weeks[-1]], weeks[1:].tolist() + [D["date"].max() + pd.Timedelta(days=1)]):
        te = D.index[(D["date"] >= a) & (D["date"] < b)]
        tr = D[(D["date"] <= a - pd.Timedelta(days=2)) & (D["date"] > a - pd.Timedelta(days=62)) & ~D["capped"]]
        tr = tr.dropna(subset=REGS + ["y"])
        g = D.loc[te].dropna(subset=REGS + ["p_cap"])
        if len(tr) < 500 or g.empty:
            continue
        Xtr = np.column_stack([np.ones(len(tr)), tr[REGS].to_numpy()])
        Xte = np.column_stack([np.ones(len(g)), g[REGS].to_numpy()])
        G = np.column_stack([Xte @ QuantReg(tr["y"].to_numpy(), Xtr).fit(q=u, max_iter=3000).params for u in GRID])
        G = np.clip(np.sort(G, axis=1), 0, CAP)
        p = g["p_cap"].to_numpy()
        res = np.empty((len(g), len(QS)))
        for i, t in enumerate(QS):
            u = np.clip(t / np.maximum(1 - p, 1e-6), GRID[0], GRID[-1])
            cond = np.array([np.interp(uu, GRID, row) for uu, row in zip(u, G)])
            res[:, i] = np.where(t > 1 - p, CAP, cond)
        out[g.index] = np.sort(res, axis=1)
        print(f"{a:%Y-%m-%d} ({time.time()-t0:.0f}s)", flush=True)
    for i, t in enumerate(QS):
        D[f"t{int(t*100)}"] = out[:, i]
    D[["date", "block"] + [f"t{int(t*100)}" for t in QS]].to_csv(RES / "twopart_qra_india.csv", index=False)
    print("written")


if __name__ == "__main__":
    main()
