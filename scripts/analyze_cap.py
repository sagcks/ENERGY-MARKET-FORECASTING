"""Cap regimes, missing days and cap-hit profile by hour/year.  Usage: python scripts/analyze_cap.py [data/dam_all.csv]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from iexfc.cap import documented_cap
df=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else "data/dam_all.csv",parse_dates=["date"])
dm=df.groupby("date").mcp_rs_mwh.max()
full=pd.date_range(dm.index.min(),dm.index.max())
miss=full.difference(dm.index)
print("MISSING DAYS:",[d.strftime("%Y-%m-%d") for d in miss])
print("\n-- cap change dates from daily maxima --")
for thr in (12000,10000):
    sel=dm[dm>thr]
    print(f"last day with max > {thr}:", (f"{sel.index.max():%Y-%m-%d} (max {sel.iloc[-1]:.2f})" if len(sel) else "none"))
print("days with max exactly 20000:", (dm==20000).sum(), "| first:", dm[dm==20000].index.min().strftime("%Y-%m-%d"), "| last:", dm[dm==20000].index.max().strftime("%Y-%m-%d"))
print("days with max exactly 12000:", (dm==12000).sum(), "| first:", dm[dm==12000].index.min().strftime("%Y-%m-%d"), "| last:", dm[dm==12000].index.max().strftime("%Y-%m-%d"))
print("days with max exactly 10000:", (dm==10000).sum(), "| first:", dm[dm==10000].index.min().strftime("%Y-%m-%d"))
print("max after 2023-04-10:", dm["2023-04-10":].max())
print("\ndaily max around the 2022 change:"); print(dm["2022-04-01":"2022-04-12"].to_string())
print("\ndaily max around the 2023 change:"); print(dm["2023-03-30":"2023-04-08"].to_string())

d=df.copy(); d["cap"]=d.date.map(documented_cap); d["at_cap"]=d.mcp_rs_mwh>=0.995*d.cap
print("\nshare of blocks at cap by calendar year:"); print(d.groupby(d.date.dt.year).at_cap.mean().round(3).to_string())
print("\nshare at cap by hour of day (hour 1 = 00:00-01:00), by year:")
d["yr"]=d.date.dt.year
t=d.pivot_table(index="hour",columns="yr",values="at_cap",aggfunc="mean").round(2)
print(t.to_string())
print("\nSep-Oct 2026 share at cap by hour:")
x=d[(d.date>="2026-09-01")&(d.date<="2026-10-31")]
print(x.groupby("hour").at_cap.mean().round(2).to_string())
print("\nmean MCP by year:", d.groupby("yr").mcp_rs_mwh.mean().round(0).to_dict())
print("mean MCP Sep-Oct 2026:", round(x.mcp_rs_mwh.mean()), "| Sep-Oct 2025:", round(d[(d.date>='2025-09-01')&(d.date<='2025-10-31')].mcp_rs_mwh.mean()))
