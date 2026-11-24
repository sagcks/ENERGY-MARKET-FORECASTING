"""Compare all probabilistic benchmarks on one common sample (India and Germany).

Usage:
    python scripts/eval_benchmarks2.py --market india
    python scripts/eval_benchmarks2.py --market germany

Reads data/results/benchmarks_<market>.csv (empirical, QR-X, pooled QRA, one-part, two-part)
and data/results/benchmarks2_<market>.csv (LEAR point, LEAR-QRA, per-block QRA, QRA with
LEAR + LightGBM). Keeps the blocks/hours where every model has a forecast, then prints point
accuracy, mean pinball loss, 80% coverage and Diebold-Mariano tests against the two-part model
(daily mean pinball loss, Newey-West variance with 7 lags). Writes data/results/table_<market>.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QS = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def pinball_rows(y, Q):
    d = y[:, None] - Q
    t = np.array(QS)[None, :] / 100
    return np.maximum(t * d, (t - 1) * d).mean(axis=1)


def dm(a, b, dates, lags=7):
    d = pd.Series(a - b).groupby(np.asarray(dates)).mean().to_numpy()
    u = d - d.mean()
    n = len(d)
    v = u @ u / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return d.mean() / np.sqrt(v / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["india", "germany"], required=True)
    ap.add_argument("--forecasts", action="store_true", help="Germany with day-ahead forecasts (files with _fc)")
    a = ap.parse_args()
    tag = "_fc" if a.forecasts else ""
    res = ROOT / "data/results"
    B = pd.read_csv(res / f"benchmarks_{a.market}{tag}.csv", parse_dates=["date"])
    C = pd.read_csv(res / f"benchmarks2_{a.market}{tag}.csv", parse_dates=["date"])
    hc = "block" if "block" in C.columns else "hour"
    C = C.rename(columns={hc: "slot"})
    keep = ["date", "slot", "lear"] + [c for c in C.columns if c.startswith(("learqra_", "qrab_", "qra2_"))]
    D = B.merge(C[keep], on=["date", "slot"], how="inner")
    extra = res / f"twopart_qra_{a.market}.csv"
    if extra.exists():
        E = pd.read_csv(extra, parse_dates=["date"]).rename(columns={hc: "slot"})
        D = D.merge(E, on=["date", "slot"], how="inner")
    models = {
        "Empirical 28-day quantiles": "base_q",
        "QR-X (pooled)": "x",
        "QRA (pooled)": "a",
        "QRA (per block)": "qrab_q",
        "LEAR-QRA (per block)": "learqra_q",
        "QRA, LEAR + LightGBM (per block)": "qra2_q",
        "One-part LightGBM": "q",
        "Two-part LightGBM": "k",
    }
    if extra.exists():
        models["Two-part QRA (LEAR + LightGBM)"] = "t"
    cols = {m: [f"{p}{q}" for q in QS] for m, p in models.items()}
    allc = sum(cols.values(), [])
    D = D.dropna(subset=allc + ["y", "lear", "pred_point"]).reset_index(drop=True)
    y = D["y"].to_numpy()
    print(f"{a.market}{tag}: common sample {D['date'].min():%Y-%m-%d} to {D['date'].max():%Y-%m-%d}, "
          f"{len(D):,} rows, {D['date'].nunique()} days")
    print("Point forecasts:")
    for nm, c in [("Yesterday", "p1"), ("LEAR", "lear"), ("LightGBM L1", "pred_point")]:
        e = y - D[c].to_numpy()
        print(f"  {nm:14s} MAE {np.mean(np.abs(e)):9.2f}  RMSE {np.sqrt(np.mean(e**2)):9.2f}")
    L = {m: pinball_rows(y, D[c].to_numpy()) for m, c in cols.items()}
    ref = L["Two-part LightGBM"]
    rows = []
    for m, l in L.items():
        Q = D[cols[m]].to_numpy()
        rows.append(dict(model=m, pinball=l.mean(), cov80=np.mean((y >= Q[:, 0]) & (y <= Q[:, -1])),
                         two_part_change_pct=100 * (ref.mean() / l.mean() - 1),
                         dm_t_two_vs_model=np.nan if m == "Two-part LightGBM" else dm(ref, l, D["date"])))
    T = pd.DataFrame(rows)
    # pairwise: best linear benchmark vs one-part
    best_lin = T.iloc[:6].sort_values("pinball").iloc[0]["model"]
    t_one = dm(L["One-part LightGBM"], L[best_lin], D["date"])
    with pd.option_context("display.width", 160):
        print(T.round(3).to_string(index=False))
    print(f"Best linear benchmark: {best_lin}; one-part vs it: DM t = {t_one:.2f}")
    T.to_csv(res / f"table_{a.market}{tag}.csv", index=False)


if __name__ == "__main__":
    main()
