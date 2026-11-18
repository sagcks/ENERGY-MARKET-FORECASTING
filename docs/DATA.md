# Data notes

## DAM (day-ahead market) snapshot
- Source: IEX market data > Day Ahead Market > Market Snapshot. 15-minute blocks (96/day) are
  available from delivery day 1 April 2022.
- Export: Delivery Period = '-Select Range-', type two dates (DD-MM-YYYY), Update Report,
  Export > Export Excel. Verified for 1-7 April 2022 (672 blocks, 7 days, no gaps).
- Columns: Purchase Bid, Sell Bid, MCV, Final Scheduled Volume (MW), MCP (Rs/MWh).
  MCV equals Final Scheduled Volume in every block of the test week.
- Unknown: the maximum range the site accepts. Default chunk is 31 days ('Last 31 Days' is a menu option).

## Price cap (verified against the full 2022-2026 dataset)
| Regime | Evidence in the data |
|---|---|
| Rs 20,000/MWh | 1 Apr 2022 is the only day with a maximum of exactly 20,000 |
| Rs 12,000/MWh | last day above 12,000 is 2 Apr 2022; first day at exactly 12,000 is 7 Apr 2022; 228 days peak at exactly 12,000, until 29 Mar 2023 |
| Rs 10,000/MWh | last day above 10,000 is 29 Mar 2023; first day at exactly 10,000 is 8 Apr 2023; nothing exceeds 10,000 afterwards (1,054 days peak exactly at 10,000) |

The exact effective dates (3-7 April 2022 and 30 Mar-8 Apr 2023) cannot be fixed from the data, because no day in
those windows reaches either ceiling. CERC order 04/SM/2023 gives 4 April 2023. **TODO: read the 2022 CERC order for the first change.**

Share of blocks at the cap: 15.1% (2022), 19.9% (2023), 15.7% (2024), 12.4% (2025), 28.3% (2026).
CERC's staff paper says about 21% for 2023, so the dataset agrees (the gap is probably tolerance and definition).
Cap hits follow a daily pattern: evening and night blocks, almost none around midday.
Sep-Oct 2026 is exceptional: about half of all blocks at the cap and a mean price of about Rs 6,994 (Sep-Oct 2025: about Rs 3,117). Cause not yet investigated.

## Data gaps
- 21-27 Nov 2025 (7 days): IEX's site has no data for these days (checked by hand on 11 Nov 2026), so they are missing from the dataset.
  Models must treat them as missing, not as zero, and must not use them as lagged inputs.

## Germany (SMARD)
- Source: [SMARD.de](https://www.smard.de), Bundesnetzagentur, licensed CC BY 4.0
  (attribution: "Bundesnetzagentur | SMARD.de").
- Download centre: *Day-ahead prices* (Germany/Luxembourg), *Actual consumption* and *Actual
  generation* (Germany), resolution *Hour*, CSV. The site limits the range per download, so
  download in blocks (for example 31.12.2018–31.12.2022 and 31.12.2022–present) and put all
  files in `data/external/smard/`; `smard.load_many` joins them and removes overlaps.
- Timestamps are German local time. The date picker uses the browser's time zone, so files
  start and end mid-day; `smard.whole_days` trims to complete German days (23/24/25 hours).
- Before 1 Oct 2025 day-ahead prices were hourly; since then they are 15-minute. SMARD's hourly
  series after the switch equals the mean of the four quarter-hours (checked on Oct 2025).
- Checked coverage: 31 Dec 2018 – 12 Nov 2026, 68,976 hours, no gaps. Nuclear is zero after
  15 Apr 2023 and blank from 30 Jan 2024 (treated as zero). The Polish price is blank until
  19 Nov 2019; the old DE/AT/LU zone is blank throughout.
- Price limits: the upper limit never came close (max 936 €/MWh, 2024). The harmonised floor of
  −500 €/MWh was reached on 2 Jul 2023 and nearly on 1 May 2026 (−499). Share of negative
  hours: 2.4% (2019) to 6.5% (2025).

## Open questions
1. IEX terms of use for bulk collection and redistribution (check before sharing data).
2. Regional/bid-area prices (the snapshot is the single MCP).
3. APPCPL: written confirmation for publishing the legacy code is pending; `legacy/` is excluded from the public repository until then.
4. The site loads a tracking script that may be bot detection. Keep collection slow and modest.
5. Germany: add day-ahead load and renewable forecasts (SMARD "Forecasted consumption/generation").
