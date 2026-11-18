# Cap-aware probabilistic forecasting of day-ahead electricity prices

Research code for forecasting the full day-ahead price distribution in markets with a
regulatory price limit. The main case is the Indian Energy Exchange (IEX) day-ahead market,
where a price cap binds in many 15-minute blocks; the transfer test is Germany–Luxembourg,
where the upper limit never binds but prices often fall below zero.

Author: Sagnik Chakravarty. Working paper in preparation; a two-page research summary is
available on request.

## Main results

Rolling-origin evaluation, models retrained every quarter, test period January 2024 to
November 2026. Pinball loss is averaged over the 10%–90% quantiles (lower is better).

| Market | Empirical quantiles | One-part LightGBM | Two-part (regime) | Two-part vs one-part |
|---|---|---|---|---|
| India, 99,456 blocks, Rs/MWh | 376.1 | 217.7 (−42.1%) | 208.5 (−44.6%) | −4.2%, DM t = −7.1 |
| India without bid data | 376.1 | 221.9 (−41.0%) | 215.0 (−42.8%) | −3.1%, DM t = −4.1 |
| Germany, 25,128 hours, €/MWh | 11.63 | 9.02 (−22.5%) | 9.00 (−22.7%) | −0.25%, DM t = −1.1 |

* The method transfers: in Germany it beats the empirical benchmark clearly
  (Diebold–Mariano t = −9.8) and predicts negative-price hours with a Brier score of 0.036
  against 0.062 for persistence (India, cap hits: 0.036 against 0.061).
* The two-part model, which predicts the extreme regime separately, helps only where a limit
  binds: Indian prices pile up at the cap, German negative prices spread out down to −500 €/MWh.
* Recalibration from past errors brings 80% interval coverage from 72% to 82% (India) and
  from 72% to 79% (Germany).

## How the models work

All features use only information available before the auction closes (prices up to the day
before delivery; demand and generation lagged two days; calendar fields). The **one-part**
model forecasts nine quantiles directly with LightGBM. The **two-part** model combines a
classifier for the extreme regime (a cap hit in India, a negative price in Germany) with
quantile models trained separately on each regime. Quantiles are optionally recalibrated
with the empirical error quantiles of the previous 60 days (ending two days before delivery).

## Layout

| Path | What it does |
|---|---|
| `src/iexfc/xlsx.py`, `store.py` | parse and combine IEX Excel exports, with checks |
| `src/iexfc/cap.py` | IEX price-cap regimes |
| `src/iexfc/features.py` | leak-free India feature table (96 blocks per day) |
| `src/iexfc/smard.py` | load SMARD CSV exports (time zones, daylight saving, the 2025 switch to 15-minute prices) |
| `src/iexfc/features_de.py` | leak-free Germany feature table (24 hours per day) |
| `scripts/collect_ranges.py`, `build_dataset.py` | download IEX data and build `data/dam_all.csv` |
| `scripts/baselines_naive.py` | naive and empirical-quantile benchmarks |
| `scripts/model_gbm.py` | India one-part model, rolling evaluation |
| `scripts/model_capaware.py` | India two-part (cap-aware) model |
| `scripts/recalibrate.py` | recalibration from past errors |
| `scripts/ablation_nobid.py` | India without bid and volume features |
| `scripts/model_de.py` | Germany: all models, recalibration and summary |
| `tests/` | `pytest` tests for the parsers and loaders |
| `docs/DATA.md` | data sources, price-cap notes, open questions |

## Reproduce

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```
On macOS, LightGBM needs OpenMP: `brew install libomp`.

**Data are not included.** IEX data must be downloaded from IEX's website (see
`docs/DATA.md` and `scripts/collect_ranges.py`); its terms for redistribution have not been
checked. German data come from [SMARD](https://www.smard.de) (Bundesnetzagentur | SMARD.de,
CC BY 4.0): download *Day-ahead prices* (Germany/Luxembourg), *Actual consumption* and
*Actual generation* (Germany), resolution *Hour*, CSV, 31.12.2018 to the present, and put the
files in `data/external/smard/`. Grid-India daily data: `data/external/POSOCO_data.csv`
(see `docs/DATA.md`).

```
# India
python scripts/model_gbm.py --prices data/dam_all.csv
python scripts/model_capaware.py --prices data/dam_all.csv
python scripts/recalibrate.py --pred data/results/predictions_capaware.csv --prefix k --out predictions_capaware_cal.csv
python scripts/ablation_nobid.py --prices data/dam_all.csv
# Germany
python scripts/model_de.py --smard data/external/smard
```
Run times on a laptop: about 6–7 minutes per India script and 6–9 minutes for Germany.
Results reproduce across machines to within the last decimal (LightGBM sampling).

## Limitations

Simple benchmarks only so far (no quantile regression averaging, ARX or neural networks);
no weather or holiday features; the German model uses lagged actuals rather than day-ahead
forecasts; Germany is hourly and India 15-minute; design choices were made on the test period.

## Licence

Code: MIT (see `LICENSE`). Data remain under their providers' terms.
