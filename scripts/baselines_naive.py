"""Naive and empirical-quantile benchmarks, rolling evaluation on 2024-2026.  Usage: python scripts/baselines_naive.py [data/dam_all.csv]"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from iexfc.cap import documented_cap
df=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else "data/dam_all.csv",parse_dates=["date"])
W=df.pivot(index="date",columns="block_no",values="mcp_rs_mwh")
W=W.reindex(pd.date_range(W.index.min(),W.index.max()))
cap=pd.Series([documented_cap(d) for d in W.index],index=W.index)
AT=(W.ge(0.995*cap,axis=0))&W.notna()

pred={
 "yesterday same block":W.shift(1),
 "last week same block":W.shift(7),
 "avg(yesterday,last week)":(W.shift(1)+W.shift(7))/2,
 "mean of last 7 days":sum(W.shift(k) for k in range(1,8))/7,
 "median of last 7 days":pd.DataFrame(np.nanmedian(np.stack([W.shift(k).values for k in range(1,8)]),axis=0),index=W.index,columns=W.columns),
}
test=(W.index>="2024-01-01")
def score(P,mask):
    a=W[mask].values.ravel(); p=P[mask].values.ravel(); ok=~(np.isnan(a)|np.isnan(p))
    e=p[ok]-a[ok]; return pd.Series({"MAE":np.abs(e).mean(),"RMSE":np.sqrt((e**2).mean()),"bias":e.mean(),"n":ok.sum()})
rows={k:score(P,test) for k,P in pred.items()}
print("== Point-forecast benchmarks, test period 2024-01-01 to 2026-11-09 (Rs/MWh) ==")
print(pd.DataFrame(rows).T.round(0).astype({"n":int}).to_string())
print("\nMAE by year:")
yrs={}
for k,P in pred.items():
    yrs[k]={y:score(P,(W.index.year==y)&test)["MAE"] for y in (2024,2025,2026)}
print(pd.DataFrame(yrs).T.round(0).to_string())

# cap-hit persistence
print("\n== Does yesterday/last week predict a cap hit? (test period) ==")
A=AT[test].values.ravel()
for name,P in {"cap yesterday same block":AT.shift(1,fill_value=False),"cap last week same block":AT.shift(7,fill_value=False)}.items():
    p=P[test].values.ravel().astype(bool); ok=~np.isnan(W[test].values.ravel())
    a=A[ok]; p=p[ok]
    tp=(a&p).sum(); print(f"{name}: base rate {a.mean():.3f} | precision {tp/max(p.sum(),1):.3f} | recall {tp/a.sum():.3f} | P(cap|pred cap)={tp/max(p.sum(),1):.3f} vs P(cap|pred no cap)={(a&~p).sum()/max((~p).sum(),1):.3f}")

# probabilistic baseline: empirical quantiles of the last 28 days, same block
qs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
Q={q:W.shift(1).rolling(28,min_periods=20).quantile(q) for q in qs}
def pinball(q,y,f): d=y-f; return np.maximum(q*d,(q-1)*d)
a=W[test].values
loss=[];
for q in qs:
    f=Q[q][test].values; ok=~(np.isnan(a)|np.isnan(f)); loss.append(pinball(q,a[ok],f[ok]).mean())
lo=Q[0.1][test].values; hi=Q[0.9][test].values; ok=~(np.isnan(a)|np.isnan(lo)|np.isnan(hi))
print("\n== Probabilistic baseline: empirical quantiles, last 28 days same block ==")
print("mean pinball loss over q=0.1..0.9:", round(float(np.mean(loss)),0))
print("coverage of the 10-90% band (target 80%):", round(float(((a[ok]>=lo[ok])&(a[ok]<=hi[ok])).mean()),3))
# the point of the research: how does error concentrate at the cap?
e=(pred["yesterday same block"]-W)[test].values.ravel(); aa=W[test].values.ravel(); at=AT[test].values.ravel(); ok=~np.isnan(e)
print("\nMAE of 'yesterday' when actual is at cap:", round(np.abs(e[ok&at]).mean()), "| when not at cap:", round(np.abs(e[ok&~at]).mean()))
print("share of all absolute error that comes from capped blocks:", round(np.abs(e[ok&at]).sum()/np.abs(e[ok]).sum(),3), "(capped blocks are", round(at[ok].mean(),3),"of all)")
