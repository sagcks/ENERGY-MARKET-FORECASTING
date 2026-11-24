# Cap-aware probabilistic forecasting of day-ahead electricity prices

Research code for forecasting the full day-ahead price distribution in markets with a
regulatory price limit. The main case is the Indian Energy Exchange (IEX) day-ahead market,
where a price cap binds in many 15-minute blocks; the transfer test is Germany–Luxembourg,
where the upper limit never binds but prices often fall below zero.

Author: Sagnik Chakravarty. Paper: *Probabilistic Day-Ahead Electricity Price Forecasting Under a
Binding Price Cap: Evidence from India and Germany* (preprint on
[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7726358); revised version in preparation).

## Main results

Rolling-origin evaluation, LightGBM retrained every quarter. Common sample where every model has a
forecast: 2 March 2024 to 31 October 2026. Pinball loss averaged over the 10%–90% quantiles
(lower is better); DM t compares the two-part model with each row (negative = two-part better).

| Model | India (Rs/MWh) | DM t | Germany (€/MWh) | DM t |
|---|---|---|---|---|
| Empirical 28-day quantiles | 382.0 | −11.9 | 11.61 | −10.7 |
| QR-X (pooled) | 256.7 | −11.4 | 10.11 | −8.6 |
| QRA (pooled) | 236.6 | −7.7 | 9.58 | −6.0 |
| QRA (per block/hour) | 246.2 | −9.1 | 9.64 | −5.0 |
| LEAR-QRA | 269.7 | −11.0 | 8.72 | +3.2 |
| QRA on LEAR + LightGBM | 245.3 | −8.2 | **8.54** | +5.3 |
| One-part LightGBM | 223.8 | −6.9 | 9.19 | −1.8 |
| Two-part QRA (cap structure on QRA) | 215.3 | −0.7 | – | – |
| **Two-part LightGBM** | **213.9** | – | 9.16 | – |

* **India** (86,304 blocks): the two-part model beats every benchmark. The gain is made in capped
  blocks, holds in all six half-years and without bid data (−3.1%, DM t = −4.1), and the same cap
  structure on top of QRA does almost as well, so the structure rather than the learner drives it.
* **Germany** (23,376 hours): prices do not pile up at a limit; the two-part structure brings no
  significant gain and QRA on LEAR + LightGBM is best.
* Cap-hit probability (India): Brier 0.036 against 0.061 for persistence; AUC 0.986.
* Recalibration from past errors brings 80% interval coverage from 72% to 82–83% (India).

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
| `scripts/benchmarks_qr.py` | QR-X and pooled QRA benchmarks |
| `scripts/benchmarks_lear.py` | LEAR (Lago et al. 2021), LEAR-QRA and per-block QRA |
| `scripts/twopart_qra.py` | India: the cap structure on top of QRA |
| `scripts/eval_benchmarks2.py` | comparison table with Diebold–Mariano tests |
| `scripts/robustness.py` | recalibration windows and results by half-year |
| `run_all.sh` | runs everything above in order |
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

Optional: SMARD *Forecasted consumption* and *Forecasted generation Day-Ahead* (Germany, Hour,
CSV, same dates) in the same folder, for the run with day-ahead forecasts.

Run everything (from the project folder, with the virtual environment active):
```
bash run_all.sh               # India and Germany, all models and benchmarks
bash run_all.sh germany-fc    # Germany with day-ahead load, wind and solar forecasts
```
Logs and tables go to `data/results/logs/`. LEAR is the slow step (a few hours on a laptop; it
uses all CPU cores but one); the other steps take minutes each.

## Limitations

No distributional neural networks or censored (Tobit) benchmarks yet; no weather or holiday
features; the main German run uses lagged actuals rather than day-ahead forecasts (see
`run_all.sh germany-fc`); Germany is hourly and India 15-minute; some design choices were made
with the test period in view.

## Licence

Code: MIT (see `LICENSE`). Data remain under their providers' terms.
