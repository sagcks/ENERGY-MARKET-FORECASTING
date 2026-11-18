"""Test one date range end to end: pick dates, Update Report, read the table, Export Excel.

Run:  python scripts/try_range.py 01-04-2022 07-04-2022
(dates are DD-MM-YYYY). A Chrome window opens and closes by itself.
It prints what happened, saves data/try_range.png, and puts the Excel file (if the
download works) in data/downloads. Paste the printed text back to Claude.
"""
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://www.iexindia.com/market-data/day-ahead-market/market-snapshot"
start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("01-04-2022", "07-04-2022")

out = Path("data")
dl = (out / "downloads").resolve()
dl.mkdir(parents=True, exist_ok=True)

opts = Options()
opts.add_experimental_option("prefs", {
    "download.default_directory": str(dl),
    "download.prompt_for_download": False,
})
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)


def js_click(el):
    driver.execute_script("arguments[0].click()", el)


def button(text):
    return driver.find_element(By.XPATH, f"//button[normalize-space()='{text}']")


def set_date(el, value):
    """Type DD-MM-YYYY into a date box; try digits only first, then with dashes."""
    for attempt in (value.replace("-", ""), value):
        el.click()
        el.send_keys(Keys.COMMAND, "a")
        el.send_keys(Keys.BACKSPACE)
        el.send_keys(attempt)
        time.sleep(0.5)
        if el.get_attribute("value") == value:
            return True
    return False


try:
    driver.get(URL)
    time.sleep(10)

    selects = driver.find_elements(By.CSS_SELECTOR, ".MuiSelect-select")
    js_click(selects[1])
    time.sleep(2)
    for o in driver.find_elements(By.CSS_SELECTOR, "li[role='option']"):
        if "select range" in o.text.strip().lower():
            js_click(o)
            break
    time.sleep(3)

    boxes = [i for i in driver.find_elements(By.CSS_SELECTOR, "input[placeholder='DD-MM-YYYY']")]
    print("date boxes found:", len(boxes))
    ok1 = set_date(boxes[0], start)
    ok2 = set_date(boxes[1], end)
    print("start box value:", boxes[0].get_attribute("value"), "| ok:", ok1)
    print("end box value:  ", boxes[1].get_attribute("value"), "| ok:", ok2)

    js_click(button("Update Report"))
    time.sleep(8)

    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    print("table rows on screen:", len(rows))
    pag = driver.find_elements(By.CSS_SELECTOR, ".MuiTablePagination-displayedRows")
    print("pagination text:", [p.text for p in pag])
    # which dates appear in the table?
    dates = sorted({d for d in (t.text for t in driver.find_elements(By.CSS_SELECTOR, "table td")) if len(d) == 10 and d[2] == "-" and d[5] == "-"})
    print("dates visible in the table:", dates[:12])
    msgs = [e.text for e in driver.find_elements(By.CSS_SELECTOR, "[role='alert'], .MuiAlert-message, .MuiSnackbar-root") if e.text.strip()]
    print("alerts/messages:", msgs)
    driver.save_screenshot(str(out / "try_range.png"))

    # Export Excel
    before = {p.name for p in dl.iterdir()}
    js_click(button("Export"))
    time.sleep(1.5)
    js_click(button("Export Excel"))
    newfile = None
    for _ in range(30):
        time.sleep(1)
        fresh = [p for p in dl.iterdir() if p.name not in before and not p.name.endswith(".crdownload")]
        if fresh:
            newfile = fresh[0]
            break
    print("export file:", (newfile.name, newfile.stat().st_size) if newfile else "NONE after 30s")
    if newfile:
        try:
            import pandas as pd
            x = pd.read_excel(newfile, header=None)
            print("excel shape:", x.shape)
            print(x.head(8).to_string())
        except Exception as exc:  # openpyxl may be missing; the file itself is what matters
            print("could not preview Excel:", exc)
finally:
    driver.quit()
