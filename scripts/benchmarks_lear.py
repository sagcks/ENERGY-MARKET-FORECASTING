"""LEAR and per-hour QRA benchmarks for India (IEX) and Germany (DE-LU).

Usage:
    python scripts/benchmarks_lear.py --market india
    python scripts/benchmarks_lear.py --market germany --smard data/external/smard

LEAR (Lago et al., 2021): one LASSO-estimated autoregressive model per delivery block (India, 96)
or hour (Germany, 24). Inputs: the full price curves of days D-1, D-2, D-3 and D-7, the same lagged
drivers as the LightGBM models (India: daily demand, solar, wind and hydro, lagged two days;
Germany: hourly load, onshore and offshore wind and solar, lagged two days) and weekday dummies.
Prices are standardised with the median and MAD of the calibration window and passed through
asinh (the variance-stabilising transform used in the LEAR benchmark); drivers are z-scored. The
LASSO penalty is chosen by AIC with LARS, then the model is refitted with coordinate descent.
Calibration window: the 728 days before the refit date (fewer only at the very start); refitted
at the start of every month, i.e. more often than the LightGBM models (every quarter).

Probabilistic benchmarks, all with nine quantiles and re-estimated every week on the 60 days
whose outcomes are published (ending D-2):
  * LEAR-QRA (per block): quantile regression of the price on [LEAR, same block yesterday,
    same block last week], one regression per block/hour.
  * QRA (per block): the same with the LightGBM L1 point forecast in place of LEAR.
  * QRA (LEAR + LightGBM, per block): [LEAR, LightGBM, yesterday, last week].

Writes data/results/lear_<market>.csv (LEAR point forecasts) and
data/results/benchmarks2_<market>.csv (all quantile forecasts on the common sample).
"""
import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import os
WORKERS = max(1, (os.cpu_count() or 2) - 1)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Lasso, LassoLarsIC  # noqa: E402
from statsmodels.regression.quantile_regression import QuantReg  # noqa: E402

QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- data panels
def panels_india():
    from iexfc.features import load_posoco
    pr = pd.read_csv(ROOT / "data/dam_all.csv", parse_dates=["date"])
    idx = pd.date_range(pr["date"].min(), pr["date"].max(), freq="D")
    P = pr.pivot(index="date", columns="block_no", values="mcp_rs_mwh").reindex(idx)
    G = load_posoco(ROOT / "data/external/POSOCO_data.csv").reindex(idx)
    X = G[["max_demand_mw", "solar_gwh", "wind_gwh", "hydro_gwh"]].shift(2)
    X.columns = [f"{c}_lag2" for c in X.columns]
    return P, X


def panels_germany(smard, forecasts=False):
    from iexfc.features_de import forecast_panels, load_smard_dir, to_day_hour
    d = load_smard_dir(str(smard), forecasts)
    P = to_day_hour(d["Day-ahead_prices"]["Germany/Luxembourg"])
    idx = P.index
    if forecasts:  # day-ahead forecasts for day D, as in Lago et al. (2021)
        F = forecast_panels(d, idx)
        parts = []
        for name in ("load_fc", "wind_on_fc", "wind_off_fc", "solar_fc"):
            w = F[name].copy()
            w.columns = [f"{name}_h{h}" for h in w.columns]
            parts.append(w)
        return P, pd.concat(parts, axis=1)
    parts = []
    for name, s in [("load", d["Actual_consumption"]["grid load"]),
                    ("wind_on", d["Actual_generation"]["Wind onshore"].fillna(0)),
                    ("wind_off", d["Actual_generation"]["Wind offshore"].fillna(0)),
                    ("solar", d["Actual_generation"]["Photovoltaics"].fillna(0))]:
        w = to_day_hour(s).reindex(idx).shift(2)
        w.columns = [f"{name}_lag2_h{h}" for h in w.columns]
        parts.append(w)
    return P, pd.concat(parts, axis=1)


def design(P, X):
    """Row = target day D. Columns: price curves of D-1, D-2, D-3, D-7, drivers, weekday dummies."""
    blocks = []
    for k in (1, 2, 3, 7):
        s = P.shift(k)
        s.columns = [f"p{k}_{c}" for c in P.columns]
        blocks.append(s)
    dow = pd.get_dummies(P.index.dayofweek, prefix="dow", drop_first=True).astype(float)
    dow.index = P.index
    return pd.concat(blocks, axis=1), X, dow


# ---------------------------------------------------------------- LEAR
def fit_predict_block(args):
    Ytr, PRtr, EXtr, DWtr, PRte, EXte, DWte = args
    med, mad = np.nanmedian(PRtr), np.nanmedian(np.abs(PRtr - np.nanmedian(PRtr))) * 1.4826
    mad = mad if mad > 0 else 1.0
    tr = lambda a: np.arcsinh((a - med) / mad)
    mu, sd = EXtr.mean(axis=0), EXtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr = np.hstack([tr(PRtr), (EXtr - mu) / sd, DWtr])
    Xte = np.hstack([tr(PRte), (EXte - mu) / sd, DWte])
    ytr = tr(Ytr)
    alpha = LassoLarsIC(criterion="aic", max_iter=2500).fit(Xtr, ytr).alpha_
    m = Lasso(alpha=alpha, max_iter=5000).fit(Xtr, ytr)
    return np.sinh(m.predict(Xte)) * mad + med


def run_lear(P, X, start, window=728):
    PR, EX, DW = design(P, X)
    ok_rows = PR.notna().all(axis=1) & EX.notna().all(axis=1)
    refits = pd.date_range(start, P.index.max() + pd.Timedelta(days=1), freq="MS")
    if refits[0] > pd.Timestamp(start):
        refits = refits.insert(0, pd.Timestamp(start))
    out = []
    t0 = time.time()
    for a, b in zip(refits[:-1], refits[1:]):
        trd = P.index[(P.index < a - pd.Timedelta(days=1)) & (P.index >= a - pd.Timedelta(days=window + 1))]
        ted = P.index[(P.index >= a) & (P.index < b)]
        trd = trd[ok_rows.loc[trd].to_numpy()]
        ted = ted[ok_rows.loc[ted].to_numpy()]
        if len(ted) == 0:
            continue
        jobs = []
        for c in P.columns:
            yv = P.loc[trd, c].to_numpy()
            keep = ~np.isnan(yv)
            jobs.append((yv[keep], PR.loc[trd].to_numpy()[keep], EX.loc[trd].to_numpy()[keep], DW.loc[trd].to_numpy()[keep],
                         PR.loc[ted].to_numpy(), EX.loc[ted].to_numpy(), DW.loc[ted].to_numpy()))
        with ProcessPoolExecutor(WORKERS) as ex:
            preds = list(ex.map(fit_predict_block, jobs))
        f = pd.DataFrame(np.column_stack(preds), index=ted, columns=P.columns)
        out.append(f)
        print(f"LEAR {a:%Y-%m}: window {len(trd)} days, {len(ted)} target days ({time.time()-t0:.0f}s)", flush=True)
    L = pd.concat(out)
    L.index.name = "date"
    return L


# ---------------------------------------------------------------- per-block QRA
def qr_block(args):
    Xtr, ytr, Xte = args
    Xtr = np.column_stack([np.ones(len(Xtr)), Xtr])
    Xte = np.column_stack([np.ones(len(Xte)), Xte])
    res = []
    for q in QS:
        try:
            b = QuantReg(ytr, Xtr).fit(q=q, max_iter=2000).params
            res.append(Xte @ b)
        except Exception:
            res.append(np.full(len(Xte), np.quantile(ytr, q)))
    return np.sort(np.column_stack(res), axis=1)


def run_qra(D, regs, start, win=60, hcol="block"):
    """D: long table (date, hcol, y, regs...). Weekly re-estimation, per block, window ending D-2."""
    D = D.sort_values(["date", hcol]).reset_index(drop=True)
    weeks = pd.date_range(start, D["date"].max() + pd.Timedelta(days=1), freq="7D")
    Q = np.full((len(D), len(QS)), np.nan)
    t0 = time.time()
    for a, b in zip(weeks[:-1].tolist() + [weeks[-1]], weeks[1:].tolist() + [D["date"].max() + pd.Timedelta(days=1)]):
        te = D.index[(D["date"] >= a) & (D["date"] < b)]
        if len(te) == 0:
            continue
        trmask = (D["date"] <= a - pd.Timedelta(days=2)) & (D["date"] > a - pd.Timedelta(days=2 + win))
        jobs, idxs = [], []
        for h, g in D.loc[te].groupby(hcol):
            tr = D[trmask & (D[hcol] == h)].dropna(subset=regs + ["y"])
            gte = g.dropna(subset=regs)
            if len(tr) < 30 or gte.empty:
                continue
            jobs.append((tr[regs].to_numpy(), tr["y"].to_numpy(), gte[regs].to_numpy()))
            idxs.append(gte.index)
        with ProcessPoolExecutor(WORKERS) as ex:
            outs = list(ex.map(qr_block, jobs, chunksize=8))
        for ix, o in zip(idxs, outs):
            Q[ix] = o
    print(f"QRA {regs}: done ({time.time()-t0:.0f}s)", flush=True)
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["india", "germany"], required=True)
    ap.add_argument("--smard", default="data/external/smard")
    ap.add_argument("--skip-lear", action="store_true")
    ap.add_argument("--forecasts", action="store_true", help="Germany with day-ahead forecasts (suffix _fc)")
    args = ap.parse_args()
    tag = "_fc" if args.forecasts else ""
    res = ROOT / "data/results"
    if args.market == "india":
        P, X = panels_india()
        hcol, pred_file = "block", "predictions.csv"
    else:
        P, X = panels_germany(ROOT / args.smard, args.forecasts)
        hcol, pred_file = "hour", f"predictions_de{tag}.csv"
    lear_file = res / f"lear_{args.market}{tag}.csv"
    if args.skip_lear and lear_file.exists():
        L = pd.read_csv(lear_file, parse_dates=["date"]).set_index("date")
        L.columns = [type(P.columns[0])(c) for c in L.columns]
    else:
        L = run_lear(P, X, start="2023-11-01")
        L.to_csv(lear_file)
    # long table with LEAR, LightGBM point, naive lags
    Lg = L.stack(future_stack=True).rename("lear").reset_index().rename(columns={"level_1": hcol, "block_no": hcol})
    Lg.columns = ["date", hcol, "lear"]
    Pl = P.stack(future_stack=True).rename("y").reset_index()
    Pl.columns = ["date", hcol, "y"]
    p1 = P.shift(1).stack(future_stack=True).rename("p1").reset_index(); p1.columns = ["date", hcol, "p1"]
    p7 = P.shift(7).stack(future_stack=True).rename("p7").reset_index(); p7.columns = ["date", hcol, "p7"]
    D = Lg.merge(Pl, on=["date", hcol]).merge(p1, on=["date", hcol]).merge(p7, on=["date", hcol])
    G = pd.read_csv(res / pred_file, parse_dates=["date"], usecols=["date", hcol, "pred_point"])
    D = D.merge(G, on=["date", hcol], how="left")
    D = D[D["y"].notna()].reset_index(drop=True)
    for name, regs in [("learqra", ["lear", "p1", "p7"]), ("qrab", ["pred_point", "p1", "p7"]),
                       ("qra2", ["lear", "pred_point", "p1", "p7"])]:
        Q = run_qra(D, regs, start="2024-01-01", hcol=hcol)
        for i, q in enumerate(QS):
            D[f"{name}_q{int(q*100)}"] = Q[:, i]
    D.to_csv(res / f"benchmarks2_{args.market}{tag}.csv", index=False)
    print("written", res / f"benchmarks2_{args.market}{tag}.csv")


if __name__ == "__main__":
    main()
