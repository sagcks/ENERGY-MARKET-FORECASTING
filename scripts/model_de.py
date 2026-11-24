"""German (DE-LU) transfer test: same pipeline as India, hourly day-ahead prices from SMARD.

Usage:
    python scripts/model_de.py --smard data/external/smard

Rolling-origin evaluation: for each quarter from 2024-01-01, all models are trained only on days
before that quarter and forecast every hour in it (the India scheme). Models and baselines:
  * naive: same hour yesterday, same hour last week, their average
  * empirical: 28-day same-hour quantiles
  * one-part LightGBM: L1 point, 9 quantiles, classifier for a negative price
  * two-part (regime) LightGBM: classifier p = P(price < 0), quantile models trained separately
    on negative and non-negative hours; quantile tau is taken from the negative part if tau < p,
    else from the non-negative part. This is the India cap-aware design with the regime boundary
    at 0 EUR/MWh instead of at the cap.
  * both LightGBM models also recalibrated from past errors (60 days ending D-2), as in India.
Writes data/results/predictions_de.csv and prints the summary.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from iexfc.features_de import QS, build_features_de, feature_columns, load_smard_dir  # noqa: E402

COMMON = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2)
TWO = dict(COMMON, n_estimators=250)
GRID = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]
FLOOR, CEIL = -500.0, 4000.0  # clip forecasts to the harmonised clearing-price range
L = [int(q * 100) for q in QS]


def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def qmodels(X, y, levels, Xte, params):
    if len(y) < 200:  # too few hours in a regime to fit: fall back to empirical quantiles
        return np.tile(np.quantile(y, levels), (len(Xte), 1)) if len(y) else np.zeros((len(Xte), len(levels)))
    out = np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=a, **params).fit(X, y).predict(Xte)
                           for a in levels])
    return np.sort(out, axis=1)


def recalibrate(R, pre, out_pre, window=60):
    R = R.sort_values(["date", "hour"]).reset_index(drop=True)
    days = np.sort(R["date"].unique())
    res = {f"{out_pre}{q}": np.full(len(R), np.nan) for q in L}
    for d in days:
        d = pd.Timestamp(d)
        past = R[(R["date"] >= d - pd.Timedelta(days=window + 1)) & (R["date"] <= d - pd.Timedelta(days=2))]
        if past["date"].nunique() < window:
            continue
        rows = (R["date"] == d).to_numpy()
        vals = np.column_stack([R.loc[rows, f"{pre}{q}"] + np.quantile(past["y"] - past[f"{pre}{q}"], q / 100) for q in L])
        vals = np.sort(vals, axis=1).clip(FLOOR, CEIL)
        for i, q in enumerate(L):
            res[f"{out_pre}{q}"][rows] = vals[:, i]
    for k, v in res.items():
        R[k] = v
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smard", default="data/external/smard")
    ap.add_argument("--out", type=Path, default=Path("data/results"))
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--forecasts", action="store_true",
                    help="add SMARD day-ahead load, wind and solar forecasts for day D (writes predictions_de_fc.csv)")
    a = ap.parse_args()

    T = build_features_de(load_smard_dir(a.smard, a.forecasts), forecasts=a.forecasts)
    F = feature_columns(T)
    T = T.dropna(subset=["p1", "p7"] + [f"base_q{q}" for q in L]).reset_index(drop=True)
    end = T["date"].max() + pd.Timedelta(days=1)
    edges = list(pd.date_range(a.test_start, end, freq="QS")) + [end]
    t0, parts = time.time(), []
    for lo, hi in zip(edges[:-1], edges[1:]):
        tr, te = T[T["date"] < lo], T[(T["date"] >= lo) & (T["date"] < hi)]
        if te.empty:
            continue
        o = te[["date", "hour", "y", "y_ext", "p1", "p7"] + [f"base_q{q}" for q in L]].copy()
        # one-part
        o["pred_point"] = lgb.LGBMRegressor(objective="l1", **COMMON).fit(tr[F], tr["y"]).predict(te[F])
        Q = qmodels(tr[F], tr["y"], QS, te[F], COMMON).clip(FLOOR, CEIL)
        for i, q in enumerate(L):
            o[f"q{q}"] = Q[:, i]
        o["p_ext"] = lgb.LGBMClassifier(**COMMON).fit(tr[F], tr["y_ext"].astype(int)).predict_proba(te[F])[:, 1]
        # two-part (regime split at zero)
        p = lgb.LGBMClassifier(**TWO).fit(tr[F], tr["y_ext"].astype(int)).predict_proba(te[F])[:, 1]
        neg, pos = tr[tr["y_ext"] == 1], tr[tr["y_ext"] == 0]
        Gn = qmodels(neg[F], neg["y"], GRID, te[F], TWO).clip(FLOOR, 0)
        Gp = qmodels(pos[F], pos["y"], GRID, te[F], TWO).clip(0, CEIL)
        grid = np.array(GRID)
        for tau, q in zip(QS, L):
            ln = np.clip(tau / np.maximum(p, 1e-6), grid[0], grid[-1])
            lp = np.clip((tau - p) / np.maximum(1 - p, 1e-6), grid[0], grid[-1])
            vn = np.array([np.interp(x, grid, g) for x, g in zip(ln, Gn)])
            vp = np.array([np.interp(x, grid, g) for x, g in zip(lp, Gp)])
            o[f"k{q}"] = np.where(tau < p, vn, vp)
        o["p_ext2"] = p
        parts.append(o)
        print(f"{lo:%Y-%m-%d}: trained on {len(tr):,} hours, forecast {len(te):,} ({time.time() - t0:.0f}s)", flush=True)

    R = pd.concat(parts, ignore_index=True)
    R = recalibrate(R, "q", "qc")
    R = recalibrate(R, "k", "kc")
    a.out.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.out / ("predictions_de_fc.csv" if a.forecasts else "predictions_de.csv"), index=False)

    def probs(g):
        y = g["y"].to_numpy()
        res = {}
        for name, pre in [("empirical 28-day", "base_q"), ("one-part LightGBM", "q"), ("two-part (regime)", "k"),
                          ("one-part recalibrated", "qc"), ("two-part recalibrated", "kc")]:
            ok = g[f"{pre}10"].notna().to_numpy()
            yy = y[ok]
            pb = np.mean([pinball(yy, g[f"{pre}{q}"].to_numpy()[ok], q / 100) for q in L])
            cov = np.mean((yy >= g[f"{pre}10"].to_numpy()[ok]) & (yy <= g[f"{pre}90"].to_numpy()[ok]))
            res[name] = dict(pinball=pb, cov80=cov, n=int(ok.sum()))
        return pd.DataFrame(res).T

    def points(g):
        y = g["y"].to_numpy()
        res = {}
        for name, pr in [("yesterday", g["p1"]), ("last week", g["p7"]), ("avg(y'day, wk)", (g["p1"] + g["p7"]) / 2),
                         ("LightGBM L1", g["pred_point"])]:
            pr = pr.to_numpy()
            ok = ~np.isnan(pr)
            res[name] = dict(MAE=np.mean(np.abs(y[ok] - pr[ok])), RMSE=np.sqrt(np.mean((y[ok] - pr[ok]) ** 2)))
        return pd.DataFrame(res).T

    def brier(g):
        yc = g["y_ext"].to_numpy()
        return dict(model=np.mean((g["p_ext"].clip(1e-4, 1 - 1e-4) - yc) ** 2),
                    persistence=np.mean((g["p1"].lt(0).astype(float) - yc) ** 2), neg_share=yc.mean())

    print(f"\n=== Germany, test {R.date.min():%Y-%m-%d} to {R.date.max():%Y-%m-%d}: {len(R):,} hours ===")
    print(points(R).round(1).to_string())
    print(probs(R).round(3).to_string())
    print("negative-price Brier:", {k: round(v, 4) for k, v in brier(R).items()})
    print("\nBy year (pinball: empirical | one-part | two-part | two-part recal; negative share):")
    for yr, g in R.groupby(R.date.dt.year):
        pr = probs(g)
        print(yr, " | ".join(f"{pr.loc[n, 'pinball']:.2f}" for n in
                             ["empirical 28-day", "one-part LightGBM", "two-part (regime)", "two-part recalibrated"]),
              f"; neg {g.y_ext.mean():.3f}")
    low = R[R["y"] < 0]
    print(f"\nNegative hours only ({len(low):,}):")
    print(probs(low)[["pinball"]].round(2).T.to_string())
    print("Calibration by level, raw two-part:", {q: round((R.y <= R[f"k{q}"]).mean(), 2) for q in L})
    print("Calibration by level, recal two-part:", {q: round((R.y <= R[f"kc{q}"]).dropna().mean(), 2) for q in L})


if __name__ == "__main__":
    main()
