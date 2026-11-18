"""Combine all downloaded weekly Excel files into ONE dataset and print a quality + cap report.

Usage:
    python scripts/build_dataset.py            # writes data/dam_all.csv
    python scripts/build_dataset.py --excel    # also writes data/dam_all.xlsx (one sheet, takes ~1 minute)

Reads data/raw_xlsx/dam_*.xlsx. Overlapping days are de-duplicated. The weekly files are left
untouched, so you can re-run this any time after collecting more data.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from iexfc.cap import monthly_cap_summary  # noqa: E402
from iexfc.store import check, load_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", type=Path, default=Path("data/raw_xlsx"))
ap.add_argument("--out", type=Path, default=Path("data"))
ap.add_argument("--excel", action="store_true", help="also write one combined .xlsx")
args = ap.parse_args()

df = load_dataset(args.raw)
args.out.mkdir(parents=True, exist_ok=True)

csv_path = args.out / "dam_all.csv"
df.to_csv(csv_path, index=False)
print(f"{df['date'].nunique()} days, {len(df)} blocks, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
print(f"written: {csv_path}")

if args.excel:
    xlsx_path = args.out / "dam_all.xlsx"
    df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_excel(xlsx_path, index=False, sheet_name="DAM_15min")
    print(f"written: {xlsx_path}")

pd.set_option("display.width", 120)
issues = check(df)
print("\nData checks:", "all good" if not issues else "")
for i in issues:
    print("  -", i)
print("\nMonthly price-cap summary:")
print(monthly_cap_summary(df).round(3).to_string())
