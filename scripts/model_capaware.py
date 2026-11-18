"""Two-part (cap-aware) probabilistic forecast, rolling quarterly retraining.

Part 1: classifier  P(block clears at the cap)            -> p
Part 2: quantile models trained ONLY on blocks below the cap -> conditional quantile function G
Forecast quantile for level tau:  if tau > 1-p -> cap price (10,000);  else G(tau/(1-p)).
Writes data/results/predictions_capaware.csv and prints pinball loss and coverage next to the
one-part model (data/results/predictions.csv, from model_gbm.py).
Usage: python scripts/model_capaware.py --prices data/dam_all.csv
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, lightgbm as lgb
from iexfc.features import QS, build_features, feature_columns, load_posoco

CAP = 10000.0
GRID = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
P = dict(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=100,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2)

def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))

ap = argparse.ArgumentParser()
ap.add_argument("--prices", type=Path, default=Path("data/dam_all.csv"))
ap.add_argument("--posoco", type=Path, default=Path("data/external/POSOCO_data.csv"))
ap.add_argument("--out", type=Path, default=Path("data/results"))
ap.add_argument("--test-start", default="2024-01-01")
a = ap.parse_args()

T = build_features(pd.read_csv(a.prices, parse_dates=["date"]), load_posoco(a.posoco))
F = feature_columns(T)
T = T.dropna(subset=["p1"] + [f"base_q{int(q*100)}" for q in QS]).reset_index(drop=True)
edges = list(pd.date_range(a.test_start, T["date"].max() + pd.Timedelta(days=1), freq="QS")) + [T["date"].max() + pd.Timedelta(days=1)]
t0, parts = time.time(), []
for lo, hi in zip(edges[:-1], edges[1:]):
    tr, te = T[T["date"] < lo], T[(T["date"] >= lo) & (T["date"] < hi)]
    if te.empty: continue
    below = tr[tr["y_cap"] == 0]
    p = lgb.LGBMClassifier(**P).fit(tr[F], tr["y_cap"].astype(int)).predict_proba(te[F])[:, 1]
    G = np.sort(np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=g, **P).fit(below[F], below["y"]).predict(te[F]) for g in GRID]), axis=1)
    G = np.clip(G, 0, CAP - 1)
    o = te[["date", "block", "y", "y_cap"]].copy(); o["p_cap"] = p
    grid = np.array(GRID)
    for tau in QS:
        lvl = np.clip(tau / np.maximum(1 - p, 1e-6), grid[0], grid[-1])
        val = np.array([np.interp(l, grid, g) for l, g in zip(lvl, G)])
        o[f"k{int(tau*100)}"] = np.where(tau > 1 - p, CAP, val)
    parts.append(o); print(f"{lo:%Y-%m-%d} done ({time.time()-t0:.0f}s)", flush=True)
R = pd.concat(parts, ignore_index=True)
a.out.mkdir(parents=True, exist_ok=True)
R.to_csv(a.out / "predictions_capaware.csv", index=False)

def rep(name, g, pre):
    y = g["y"].to_numpy()
    pb = np.mean([pinball(y, g[f"{pre}{int(q*100)}"].to_numpy(), q) for q in QS])
    cov = np.mean((y >= g[f"{pre}10"]) & (y <= g[f"{pre}90"]))
    print(f"  {name:22s} pinball {pb:6.1f}  80% coverage {cov:.3f}")
old = pd.read_csv(a.out / "predictions.csv", parse_dates=["date"]) if (a.out / "predictions.csv").exists() else None
M = R.merge(old[["date", "block"] + [f"q{int(q*100)}" for q in QS]], on=["date", "block"]) if old is not None else R
for nm, g in [("all", M), ("2026", M[M.date.dt.year == 2026]), ("Sep-Oct 2026", M[(M.date >= "2026-09-01") & (M.date <= "2026-10-31")])]:
    print(nm); rep("two-part (cap-aware)", g, "k")
    if old is not None: rep("one-part (model_gbm)", g, "q")
print("Calibration by level (two-part):", {int(q*100): round((M.y <= M[f'k{int(q*100)}']).mean(), 2) for q in QS})
