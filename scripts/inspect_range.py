"""Diagnostic 2 (fixed): test Export, then open "-Select Range-" and record the controls.

Run:  python scripts/inspect_range.py
Chrome opens and does three things:
  1. clicks Export (with "Today" selected) and saves whatever it downloads into data/downloads
  2. picks "-Select Range-" in the Delivery Period dropdown
  3. lists the date inputs/buttons that appear
It prints what it finds, and every data request the page made. It also saves
data/page_dump_range.html and data/page_range.png.
Paste the printed output back to Claude, and send the file(s) in data/downloads too.
"""
import json
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://www.iexindia.com/market-data/day-ahead-market/market-snapshot"
STATIC = (".js", ".css", ".png", ".jpg", ".jpeg", ".svg", ".gif", ".woff", ".woff2", ".ico", ".ttf", ".map")
out = Path("data")
dl = (out / "downloads").resolve()
dl.mkdir(parents=True, exist_ok=True)

opts = Options()
opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
opts.add_experimental_option("prefs", {
    "download.default_directory": str(dl),
    "download.prompt_for_download": False,
    "safebrowsing.enabled": True,
})
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)


def requests_seen(label):
    print(f"\n== Data requests ({label})")
    seen = set()
    for entry in driver.get_log("performance"):
        msg = json.loads(entry["message"])["message"]
        if msg.get("method") != "Network.requestWillBeSent":
            continue
        req = msg["params"]["request"]
        url = req["url"]
        if url.lower().split("?")[0].endswith(STATIC) or url.startswith("data:"):
            continue
        if msg["params"].get("type") not in ("XHR", "Fetch", "Document", None):
            continue
        key = (req["method"], url, req.get("postData"))
        if key in seen:
            continue
        seen.add(key)
        print(" ", req["method"], url)
        if req.get("postData"):
            print("     body:", req["postData"][:300])


def list_controls():
    for tag, attrs in (("input", ["type", "id", "name", "placeholder", "aria-label", "value"]),
                       ("button", ["id", "aria-label", "type"])):
        els = driver.find_elements(By.TAG_NAME, tag)
        print(f"\n== <{tag}> x{len(els)}")
        for e in els[:40]:
            print("  ", {a: e.get_attribute(a) for a in attrs}, "| text:", (e.text or "")[:30].replace("\n", " "))


def show_downloads():
    files = sorted(p for p in dl.iterdir() if p.is_file())
    print("== Files in data/downloads:", [(p.name, p.stat().st_size) for p in files])
    for p in files:
        if p.suffix.lower() in (".csv", ".txt"):
            print("   first lines of", p.name)
            for line in p.read_text(errors="replace").splitlines()[:6]:
                print("    ", line[:160])


try:
    driver.get(URL)
    time.sleep(10)
    requests_seen("on page load")

    # 1. Export test (Today selected)
    print("\n== Testing Export")
    btns = [b for b in driver.find_elements(By.TAG_NAME, "button") if "export" in (b.text or "").lower()]
    print("   Export buttons found:", len(btns))
    if btns:
        btns[0].click()
        time.sleep(3)
        driver.save_screenshot(str(out / "after_export.png"))
        popups = driver.find_elements(
            By.CSS_SELECTOR,
            "[role='dialog'], [role='alertdialog'], .MuiDialog-root, .MuiModal-root, [role='menu']")
        print("   Popup text after Export:", [d.text[:300] for d in popups if d.text.strip()])
        items = driver.find_elements(By.CSS_SELECTOR, "[role='dialog'] button, [role='menuitem'], [role='menu'] li")
        print("   Buttons/items in the popup:", [c.text for c in items])
        for c in items:
            if any(k in c.text.lower() for k in ("csv", "excel", "xls")):
                c.click()
                break
        time.sleep(8)
        # close any popup so the next step can click the dropdown
        from selenium.webdriver.common.keys import Keys
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        time.sleep(1)
    show_downloads()
    requests_seen("after Export")

    # 2. Select range
    selects = driver.find_elements(By.CSS_SELECTOR, ".MuiSelect-select")
    print("\n== MUI dropdowns:", [s.text for s in selects])
    delivery = selects[1] if len(selects) > 1 else selects[0]
    driver.execute_script("arguments[0].click()", delivery)
    time.sleep(2)
    options = driver.find_elements(By.CSS_SELECTOR, "li[role='option']")
    print("== Delivery Period options:", [o.text for o in options])
    for o in options:
        if "select range" in o.text.strip().lower():
            o.click()
            break
    time.sleep(4)

    # 3. What appeared?
    list_controls()
    requests_seen("after choosing Select Range")
    (out / "page_dump_range.html").write_text(driver.page_source, encoding="utf-8")
    driver.save_screenshot(str(out / "page_range.png"))
    print("\nSaved data/page_dump_range.html and data/page_range.png")
finally:
    driver.quit()
