"""Download IEX day-ahead data as Excel files, one date range at a time.

Usage (dates are YYYY-MM-DD):
    python scripts/collect_ranges.py --start 2022-04-01 --end 2022-06-30 --days 7
    python scripts/collect_ranges.py --start 2022-04-01 --end 2026-11-09 --days 7 --headless

Each chunk opens the market-snapshot page, picks '-Select Range-', types the two dates,
clicks Update Report, then Export > Export Excel, and saves data/raw_xlsx/dam_<start>_<end>.xlsx.
Safe to stop and re-run: chunks already saved are skipped. A pause between chunks keeps the
load on IEX's site small. Check IEX's terms of use before collecting at scale or sharing data.

If a chunk fails, start with a smaller --days value (the site allows at most 7 days).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selenium import webdriver  # noqa: E402
from selenium.webdriver.chrome.options import Options  # noqa: E402
from selenium.webdriver.chrome.service import Service  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402
from webdriver_manager.chrome import ChromeDriverManager  # noqa: E402

from iexfc.xlsx import parse_export, problems  # noqa: E402

URL = "https://www.iexindia.com/market-data/day-ahead-market/market-snapshot"
SELECT_ALL = Keys.COMMAND if sys.platform == "darwin" else Keys.CONTROL


def make_driver(download_dir: Path, headless: bool):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_experimental_option("prefs", {
        "download.default_directory": str(download_dir.resolve()),
        "download.prompt_for_download": False,
    })
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)


def js_click(driver, el):
    driver.execute_script("arguments[0].click()", el)


def button(driver, text):
    return driver.find_element(By.XPATH, f"//button[normalize-space()='{text}']")


def set_date(el, value: str) -> bool:
    for attempt in (value.replace("-", ""), value):
        el.click()
        el.send_keys(SELECT_ALL, "a")
        el.send_keys(Keys.BACKSPACE)
        el.send_keys(attempt)
        time.sleep(0.5)
        if el.get_attribute("value") == value:
            return True
    return False


def download_chunk(driver, start: date, end: date, tmp_dir: Path, wait_s: int = 60) -> Path:
    driver.get(URL)
    time.sleep(10)
    selects = driver.find_elements(By.CSS_SELECTOR, ".MuiSelect-select")
    js_click(driver, selects[1])
    time.sleep(2)
    for o in driver.find_elements(By.CSS_SELECTOR, "li[role='option']"):
        if "select range" in o.text.strip().lower():
            js_click(driver, o)
            break
    time.sleep(3)
    boxes = driver.find_elements(By.CSS_SELECTOR, "input[placeholder='DD-MM-YYYY']")
    if len(boxes) < 2:
        raise RuntimeError("date boxes not found")
    if not (set_date(boxes[0], f"{start:%d-%m-%Y}") and set_date(boxes[1], f"{end:%d-%m-%Y}")):
        raise RuntimeError("could not type the dates")
    js_click(driver, button(driver, "Update Report"))
    time.sleep(8)

    before = {p.name for p in tmp_dir.iterdir()}
    js_click(driver, button(driver, "Export"))
    time.sleep(1.5)
    js_click(driver, button(driver, "Export Excel"))
    for _ in range(wait_s):
        time.sleep(1)
        fresh = [p for p in tmp_dir.iterdir() if p.name not in before and not p.name.endswith(".crdownload")]
        if fresh:
            time.sleep(1)  # let the file finish writing
            return fresh[0]
    raise RuntimeError(f"no download after {wait_s}s")


def chunks(start: date, end: date, days: int):
    d = start
    while d <= end:
        e = min(d + timedelta(days=days - 1), end)
        yield d, e
        d = e + timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--days", type=int, default=7, help="days per download (the site allows at most 7)")
    ap.add_argument("--out", type=Path, default=Path("data/raw_xlsx"))
    ap.add_argument("--pause", type=float, default=10.0, help="seconds between chunks")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / "_incoming"
    tmp.mkdir(exist_ok=True)
    driver = make_driver(tmp, args.headless)
    failed = []
    try:
        for s, e in chunks(args.start, args.end, args.days):
            target = args.out / f"dam_{s:%Y-%m-%d}_{e:%Y-%m-%d}.xlsx"
            if target.exists():
                print(f"skip   {s} to {e} (already saved)")
                continue
            try:
                got = download_chunk(driver, s, e, tmp)
                df = parse_export(got)
                req = df.attrs["requested"]
                if (req[0].date(), req[1].date()) != (s, e):
                    raise RuntimeError(f"file covers {req[0].date()} to {req[1].date()}, not {s} to {e}")
                got.replace(target)
                warn = problems(df, req[0], req[1])
                print(f"saved  {s} to {e}: {df['date'].nunique()} days, {len(df)} blocks"
                      + (f"  WARNING: {'; '.join(warn)}" if warn else ""))
            except Exception as exc:
                failed.append((s, e))
                print(f"FAILED {s} to {e}: {exc}")
            time.sleep(args.pause)
    finally:
        driver.quit()
    print(f"\nDone. {len(failed)} chunk(s) failed" + (": " + ", ".join(f"{a}..{b}" for a, b in failed) if failed else "."))


if __name__ == "__main__":
    main()
