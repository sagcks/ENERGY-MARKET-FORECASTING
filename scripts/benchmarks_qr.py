"""Linear quantile-regression benchmarks for India (IEX) and Germany (DE-LU).

Usage:
    python scripts/benchmarks_qr.py --market india
    python scripts/benchmarks_qr.py --market germany --smard data/external/smard

Two standard benchmarks from the probabilistic electricity price forecasting literature:

  * QR-X: linear quantile regression on lagged prices, yesterday's daily summary, the
    extreme-regime indicator (cap hit in India, negative price in Germany), lagged load and
    generation, and hour-of-day and weekday dummies. Same information set and same rolling
    scheme as the LightGBM models: refitted at the start of each quarter from 2024-01-01 on the
    365 days before that quarter (a fixed random subsample of 15,000 rows when the window is larger).
  * QRA (quantile regression averaging): for each target day D, quantile regression of the price
    on three point forecasts (LightGBM L1, same hour yesterday, same hour last week), fitted on
    the 60 days ending D-2, whose outcomes are already published. Needs 60 days of out-of-sample
    point forecasts, so it starts in March 2024.

Reads the LightGBM predictions written by model_gbm.py / model_capaware.py (India) or
model_de.py (Germany), adds the two benchmarks, and prints pinball loss, coverage and
Diebold-Mariano tests on daily mean pinball loss (Newey-West variance, 7 lags).
Writes data/results/benchmarks_<market>.csv.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.regression.quantile_regression import QuantReg  # noqa: E402

QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
L = [int(q * 100) for q in QS]
warnings.filterwarnings("ignore")


def pinball_rows(y, Q):
    d = y[:, None] - Q
    t = np.array(QS)[None, :]
    return np.maximum(t * d, (t - 1) * d).mean(axis=1)


def dm_test(loss_a, loss_b, dates, lags=7):
    """DM test on daily mean loss differences a - b. Negative t: model a is better."""
    d = pd.Series(loss_a - loss_b).groupby(np.asarray(dates)).mean().to_numpy()
    n = len(d)
    u = d - d.mean()
    lrv = u @ u / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return d.mean() / np.sqrt(lrv / n)


def fit_qr(X, y, Xte):
    out = []
    for q in QS:
        try:
            res = QuantReg(y, X).fit(q=q, max_iter=1000, p_tol=1e-6)
            out.append(Xte @ res.params)
        except Exception:  # singular design in a short window: fall back to the empirical quantile
            out.append(np.full(len(Xte), np.quantile(y, q)))
    return np.sort(np.column_stack(out), axis=1)


def design(df, cols, hour_col, n_hours):
    X = df[cols].to_numpy(dtype=float)
    hrs = df[hour_col].to_numpy()
    H = np.column_stack([(hrs == h).astype(float) for h in range(1, n_hours)])
    D = np.column_stack([(df["dow"].to_numpy() == k).astype(float) for k in range(1, 7)])
    return np.column_stack([np.ones(len(df)), X, H, D])


def load_market(market, smard):
    if market == "india":
        from iexfc.features import build_features, load_posoco
        T = build_features(pd.read_csv("data/dam_all.csv", parse_dates=["date"]),
                           load_posoco("data/external/POSOCO_data.csv"))
        T["hour"] = (T["block"] - 1) // 4
        cols = ["p1", "p2", "p7", "p1_prev_block", "p1_next_block", "p_min7", "p_max7", "d1_mean", "d1_max",
                "d1_min", "at1", "d1_cap_share", "max_demand_mw_lag2", "solar_gwh_lag2", "wind_gwh_lag2",
                "hydro_gwh_lag2"]
        one = pd.read_csv("data/results/predictions.csv", parse_dates=["date"])
        two = pd.read_csv("data/results/predictions_capaware.csv", parse_dates=["date"])
        R = one[["date", "block", "y", "p1", "p7", "pred_point"] + [f"base_q{q}" for q in L] + [f"q{q}" for q in L]]
        R = R.merge(two[["date", "block"] + [f"k{q}" for q in L]], on=["date", "block"])
        R = R.rename(columns={"block": "slot"})
        T = T.rename(columns={"block": "slot"})
        return T, R, cols, (0.0, 10000.0)
    from iexfc.features_de import build_features_de, load_smard_dir
    T = build_features_de(load_smard_dir(smard))
    cols = ["p1", "p2", "p7", "p1_prev_hour", "p1_next_hour", "p_min7", "p_max7", "d1_mean", "d1_max", "d1_min",
            "ext1", "d1_ext_share", "fr_p1", "nl_p1", "load_lag2", "resload_lag2", "wind_on_lag2", "solar_lag2"]
    R = pd.read_csv("data/results/predictions_de.csv", parse_dates=["date"])
    R = R[["date", "hour", "y", "p1", "p7", "pred_point"] + [f"base_q{q}" for q in L] + [f"q{q}" for q in L]
          + [f"k{q}" for q in L]].rename(columns={"hour": "slot"})
    T = T.rename(columns={"hour": "slot"})
    T["hour"] = T["slot"]
    return T, R, cols, (-500.0, 4000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["india", "germany"], required=True)
    ap.add_argument("--smard", default="data/external/smard")
    a = ap.parse_args()
    lo, hi = None, None
    T, R, cols, (lo, hi) = load_market(a.market, a.smard)
    t0 = time.time()

    # ---------------- QR-X: quarterly refit on the previous 365 days ----------------
    T = T.dropna(subset=["y"]).reset_index(drop=True)
    test = T.merge(R[["date", "slot"]], on=["date", "slot"]).sort_values(["date", "slot"]).reset_index(drop=True)
    starts = pd.date_range("2024-01-01", test["date"].max() + pd.Timedelta(days=1), freq="QS").tolist()
    ends = starts[1:] + [test["date"].max() + pd.Timedelta(days=1)]
    qrx = np.full((len(test), len(QS)), np.nan)
    for s, e in zip(starts, ends):
        tr = T[(T["date"] < s) & (T["date"] >= s - pd.Timedelta(days=365))].dropna(subset=cols)
        if len(tr) > 15000:  # a fixed random subsample keeps the linear fits fast; 15,000 rows for ~45 parameters
            tr = tr.sample(15000, random_state=0)
        rows = ((test["date"] >= s) & (test["date"] < e)).to_numpy()
        if not rows.any():
            continue
        te = test.loc[rows].copy()
        te[cols] = te[cols].fillna(tr[cols].median())  # a missing lag day: use the training median
        Xtr = design(tr, cols, "hour", 24)
        mu, sd = Xtr[:, 1:].mean(0), Xtr[:, 1:].std(0) + 1e-9
        Xtr[:, 1:] = (Xtr[:, 1:] - mu) / sd
        Xte = design(te, cols, "hour", 24)
        Xte[:, 1:] = (Xte[:, 1:] - mu) / sd
        qrx[rows] = fit_qr(Xtr, tr["y"].to_numpy(), Xte).clip(lo, hi)
        print(f"QR-X {s:%Y-%m-%d}: {len(tr):,} training rows ({time.time()-t0:.0f}s)", flush=True)
    test[[f"x{q}" for q in L]] = qrx
    R = R.merge(test[["date", "slot"] + [f"x{q}" for q in L]], on=["date", "slot"], how="left")

    # ---------------- QRA: daily refit on the 60 days ending D-2 ----------------
    R = R.sort_values(["date", "slot"]).reset_index(drop=True)
    days = np.sort(R["date"].unique())
    qra = np.full((len(R), len(QS)), np.nan)
    for i, d in enumerate(days):
        d = pd.Timestamp(d)
        past = R[(R["date"] >= d - pd.Timedelta(days=61)) & (R["date"] <= d - pd.Timedelta(days=2))]
        past = past.dropna(subset=["pred_point", "p1", "p7"])
        if past["date"].nunique() < 60:
            continue
        rows = (R["date"] == d).to_numpy()
        te = R.loc[rows, ["pred_point", "p1", "p7"]].copy()
        te["p1"] = te["p1"].fillna(te["pred_point"]); te["p7"] = te["p7"].fillna(te["pred_point"])
        X = np.column_stack([np.ones(len(past)), past[["pred_point", "p1", "p7"]].to_numpy()])
        Xte = np.column_stack([np.ones(len(te)), te.to_numpy()])
        qra[rows] = fit_qr(X, past["y"].to_numpy(), Xte).clip(lo, hi)
        if i % 100 == 0:
            print(f"QRA {d:%Y-%m-%d} ({time.time()-t0:.0f}s)", flush=True)
    R[[f"a{q}" for q in L]] = qra
    Path("data/results").mkdir(parents=True, exist_ok=True)
    R.to_csv(f"data/results/benchmarks_{a.market}.csv", index=False)

    # ---------------- evaluation ----------------
    models = {"Empirical 28-day": "base_q", "QR-X (linear)": "x", "QRA (linear)": "a",
              "LightGBM one-part": "q", "LightGBM two-part": "k"}

    def report(S, label):
        y = S["y"].to_numpy()
        print(f"\n=== {label}: {S['date'].min():%Y-%m-%d} to {S['date'].max():%Y-%m-%d}, {len(S):,} rows ===")
        loss = {}
        for name, p in models.items():
            Q = S[[f"{p}{q}" for q in L]].to_numpy()
            loss[name] = pinball_rows(y, Q)
            cov = np.mean((y >= Q[:, 0]) & (y <= Q[:, -1]))
            print(f"{name:20s} pinball {loss[name].mean():9.3f}   80% coverage {cov:.3f}")
        print("DM t (row vs column; negative = row better):")
        names = list(models)
        print(" " * 20 + "".join(f"{n[:12]:>14s}" for n in names))
        for r_ in names:
            print(f"{r_:20s}" + "".join(f"{'':>14s}" if r_ == c else f"{dm_test(loss[r_], loss[c], S['date']):14.2f}"
                                         for c in names))

    common = R.dropna(subset=[f"x{q}" for q in L] + [f"a{q}" for q in L])
    report(common, "Common sample (QRA available)")
    if a.market == "india":
        report(common[(common["date"] >= "2026-09-01") & (common["date"] <= "2026-10-31")], "Sep-Oct 2026")


if __name__ == "__main__":
    main()
