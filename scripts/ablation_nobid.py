"""India without exchange bid and volume data: the fair comparison with Germany.

Germany's SMARD data has no bid or traded-volume series, so the German model uses prices,
fundamentals and calendar only. This script reruns the India models (one-part and two-part,
same settings and rolling quarterly retraining as model_gbm.py and model_capaware.py) with the
bid and volume features removed, then compares them with the full models.

Usage:
    python scripts/ablation_nobid.py --prices data/dam_all.csv
Reads data/results/predictions.csv and predictions_capaware.csv (the full models) if present.
Writes data/results/predictions_nobid.csv.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from iexfc.features import QS, build_features, feature_columns, load_posoco  # noqa: E402

CAP = 10000.0
BID_COLS = ["buy1", "sell1", "mcv1", "buy7", "sell7", "buy_sell_ratio1", "d1_buy_mean", "d1_sell_mean"]
ONE = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2)
TWO = dict(ONE, n_estimators=250)
GRID = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
L = [int(q * 100) for q in QS]


def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", type=Path, default=Path("data/dam_all.csv"))
    ap.add_argument("--posoco", type=Path, default=Path("data/external/POSOCO_data.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/results"))
    ap.add_argument("--test-start", default="2024-01-01")
    a = ap.parse_args()

    T = build_features(pd.read_csv(a.prices, parse_dates=["date"]), load_posoco(a.posoco))
    F_all = feature_columns(T)
    F = [c for c in F_all if c not in BID_COLS]
    print(f"{len(F_all)} features in the full model; {len(F)} without bids and volumes "
          f"(removed: {', '.join(c for c in BID_COLS if c in F_all)})")
    T = T.dropna(subset=["p1"] + [f"base_q{q}" for q in L]).reset_index(drop=True)
    end = T["date"].max() + pd.Timedelta(days=1)
    edges = list(pd.date_range(a.test_start, end, freq="QS")) + [end]
    t0, parts = time.time(), []
    for lo, hi in zip(edges[:-1], edges[1:]):
        tr, te = T[T["date"] < lo], T[(T["date"] >= lo) & (T["date"] < hi)]
        if te.empty:
            continue
        o = te[["date", "block", "y", "y_cap", "p1"] + [f"base_q{q}" for q in L]].copy()
        # one-part, no bids
        o["nb_point"] = lgb.LGBMRegressor(objective="l1", **ONE).fit(tr[F], tr["y"]).predict(te[F])
        Q = np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=q, **ONE).fit(tr[F], tr["y"]).predict(te[F])
                             for q in QS])
        Q = np.clip(np.sort(Q, axis=1), 0, CAP)
        for i, q in enumerate(L):
            o[f"nq{q}"] = Q[:, i]
        o["nb_pcap"] = lgb.LGBMClassifier(**ONE).fit(tr[F], tr["y_cap"].astype(int)).predict_proba(te[F])[:, 1]
        # two-part, no bids
        below = tr[tr["y_cap"] == 0]
        p = lgb.LGBMClassifier(**TWO).fit(tr[F], tr["y_cap"].astype(int)).predict_proba(te[F])[:, 1]
        G = np.sort(np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=g, **TWO).fit(below[F], below["y"]).predict(te[F])
                                     for g in GRID]), axis=1).clip(0, CAP - 1)
        grid = np.array(GRID)
        for tau, q in zip(QS, L):
            lvl = np.clip(tau / np.maximum(1 - p, 1e-6), grid[0], grid[-1])
            val = np.array([np.interp(x, grid, g) for x, g in zip(lvl, G)])
            o[f"nk{q}"] = np.where(tau > 1 - p, CAP, val)
        parts.append(o)
        print(f"{lo:%Y-%m-%d}: trained on {len(tr):,} blocks, forecast {len(te):,} ({time.time() - t0:.0f}s)", flush=True)

    R = pd.concat(parts, ignore_index=True)
    a.out.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.out / "predictions_nobid.csv", index=False)

    # merge the full (with-bid) models, if their prediction files exist
    full1, full2 = a.out / "predictions.csv", a.out / "predictions_capaware.csv"
    if full1.exists():
        R = R.merge(pd.read_csv(full1, parse_dates=["date"])[["date", "block", "pred_point", "p_cap"] + [f"q{q}" for q in L]],
                    on=["date", "block"], how="left")
    if full2.exists():
        R = R.merge(pd.read_csv(full2, parse_dates=["date"])[["date", "block"] + [f"k{q}" for q in L]],
                    on=["date", "block"], how="left")

    y = R["y"].to_numpy()
    emp = np.mean([pinball(y, R[f"base_q{q}"].to_numpy(), q / 100) for q in L])
    rows = []
    for name, pre in [("empirical 28-day", "base_q"), ("one-part, with bids", "q"), ("one-part, no bids", "nq"),
                      ("two-part, with bids", "k"), ("two-part, no bids", "nk")]:
        if f"{pre}10" not in R:
            continue
        pb = np.mean([pinball(y, R[f"{pre}{q}"].to_numpy(), q / 100) for q in L])
        cov = np.mean((y >= R[f"{pre}10"]) & (y <= R[f"{pre}90"]))
        rows.append(dict(model=name, pinball=round(pb, 1), vs_empirical=f"{100 * (pb / emp - 1):+.1f}%", cov80=round(cov, 3)))
    print(f"\n=== India, test {R.date.min():%Y-%m-%d} to {R.date.max():%Y-%m-%d}: {len(R):,} blocks ===")
    print(pd.DataFrame(rows).to_string(index=False))

    pts = [("yesterday", R["p1"]), ("LightGBM L1, no bids", R["nb_point"])]
    if "pred_point" in R:
        pts.insert(1, ("LightGBM L1, with bids", R["pred_point"]))
    for name, pr in pts:
        pr = pr.to_numpy()
        ok = ~np.isnan(pr)
        print(f"{name:24s} MAE {np.mean(np.abs(y[ok] - pr[ok])):6.0f}   RMSE {np.sqrt(np.mean((y[ok] - pr[ok]) ** 2)):6.0f}")
    yc = R["y_cap"].to_numpy()
    briers = {"no bids": np.mean((R["nb_pcap"].clip(1e-4, 1 - 1e-4) - yc) ** 2),
              "persistence": np.mean((R["p1"].ge(0.995 * CAP).astype(float).to_numpy() - yc) ** 2)}
    if "p_cap" in R:
        briers["with bids"] = np.mean((R["p_cap"].clip(1e-4, 1 - 1e-4) - yc) ** 2)
    print("Cap-hit Brier:", {k: round(v, 4) for k, v in briers.items()})


if __name__ == "__main__":
    main()
