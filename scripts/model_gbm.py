"""Rolling-origin test of gradient-boosting forecasts of IEX day-ahead prices.

Usage:
    python scripts/model_gbm.py --prices data/dam_all.csv --posoco data/external/POSOCO_data.csv

For each quarter from 2024-01-01, models are trained only on target days BEFORE that quarter and
used to forecast every block in the quarter (a fixed, honest "retrain every quarter" scheme).
Models: LightGBM point (L1), 9 quantiles (0.1..0.9), and a classifier for "block clears at the cap".
Compared with: yesterday, same block last week, empirical 28-day quantiles, cap persistence.
Writes data/results/predictions.csv and prints the summary.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402

from iexfc.features import QS, build_features, feature_columns, load_posoco  # noqa: E402

CAP = 10000.0
COMMON = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2)


def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", type=Path, default=Path("data/dam_all.csv"))
    ap.add_argument("--posoco", type=Path, default=Path("data/external/POSOCO_data.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/results"))
    ap.add_argument("--test-start", default="2024-01-01")
    args = ap.parse_args()

    prices = pd.read_csv(args.prices, parse_dates=["date"])
    T = build_features(prices, load_posoco(args.posoco))
    F = feature_columns(T)
    T = T.dropna(subset=[f"base_q{int(q*100)}" for q in QS] + ["p1"]).reset_index(drop=True)
    starts = pd.date_range(args.test_start, T["date"].max() + pd.Timedelta(days=1), freq="QS")
    t0 = time.time()
    parts = []
    for a, b in zip(starts[:-1].tolist() + [starts[-1]], starts[1:].tolist() + [T["date"].max() + pd.Timedelta(days=1)]):
        tr, te = T[T["date"] < a], T[(T["date"] >= a) & (T["date"] < b)]
        if te.empty:
            continue
        o = te[["date", "block", "y", "y_cap", "p1", "p7"] + [f"base_q{int(q*100)}" for q in QS]].copy()
        o["pred_point"] = lgb.LGBMRegressor(objective="l1", **COMMON).fit(tr[F], tr["y"]).predict(te[F])
        raw = np.column_stack([
            lgb.LGBMRegressor(objective="quantile", alpha=q, **COMMON).fit(tr[F], tr["y"]).predict(te[F]) for q in QS])
        raw = np.clip(np.sort(raw, axis=1), 0, CAP)
        for i, q in enumerate(QS):
            o[f"q{int(q*100)}"] = raw[:, i]
        o["p_cap"] = lgb.LGBMClassifier(**COMMON).fit(tr[F], tr["y_cap"].astype(int)).predict_proba(te[F])[:, 1]
        parts.append(o)
        print(f"{a:%Y-%m-%d}: trained on {len(tr):,} rows, forecast {len(te):,}  ({time.time()-t0:.0f}s)", flush=True)

    R = pd.concat(parts, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    R.to_csv(args.out / "predictions.csv", index=False)

    def summarize(r):
        y = r["y"].to_numpy()
        res = {}
        for name, p in [("yesterday", r["p1"]), ("last week", r["p7"]), ("avg(y'day, wk)", (r["p1"] + r["p7"]) / 2),
                        ("LightGBM L1", r["pred_point"])]:
            p = p.to_numpy()
            ok = ~np.isnan(p)  # blocks whose lag day is missing from the data are skipped
            res[name] = dict(MAE=np.mean(np.abs(y[ok] - p[ok])), RMSE=np.sqrt(np.mean((y[ok] - p[ok]) ** 2)))
        pb_g = np.mean([pinball(y, r[f"q{int(q*100)}"].to_numpy(), q) for q in QS])
        pb_b = np.mean([pinball(y, r[f"base_q{int(q*100)}"].to_numpy(), q) for q in QS])
        cov_g = np.mean((y >= r["q10"]) & (y <= r["q90"]))
        cov_b = np.mean((y >= r["base_q10"]) & (y <= r["base_q90"]))
        yc = r["y_cap"].to_numpy()
        pc = r["p_cap"].to_numpy().clip(1e-4, 1 - 1e-4)
        brier_g = np.mean((pc - yc) ** 2)
        brier_p = np.mean((r["p1"].ge(0.995 * CAP).astype(float).to_numpy() - yc) ** 2)
        return res, dict(pinball_gbm=pb_g, pinball_emp=pb_b, cov80_gbm=cov_g, cov80_emp=cov_b,
                         brier_gbm=brier_g, brier_persist=brier_p, cap_rate=yc.mean())

    print(f"\n=== Test {R['date'].min():%Y-%m-%d} to {R['date'].max():%Y-%m-%d}: {len(R):,} blocks ===")
    pt, pr = summarize(R)
    print(pd.DataFrame(pt).T.round(0).to_string())
    print({k: round(v, 3) for k, v in pr.items()})
    print("\nBy year (point RMSE: yesterday vs LightGBM | pinball: empirical vs LightGBM | 80% coverage gbm):")
    for yr, g in R.groupby(R["date"].dt.year):
        pt, pr = summarize(g)
        print(yr, round(pt["yesterday"]["RMSE"]), round(pt["LightGBM L1"]["RMSE"]), "|",
              round(pr["pinball_emp"]), round(pr["pinball_gbm"]), "|", round(pr["cov80_gbm"], 3))
    print("\nSep-Oct 2026 only:")
    g = R[(R["date"] >= "2026-09-01") & (R["date"] <= "2026-10-31")]
    if len(g):
        pt, pr = summarize(g)
        print(pd.DataFrame(pt).T.round(0).to_string()); print({k: round(v, 3) for k, v in pr.items()})


if __name__ == "__main__":
    main()
