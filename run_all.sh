#!/usr/bin/env bash
# Reproduce every result in the paper, in order. Run from anywhere:
#     bash run_all.sh            # India + Germany (lagged actuals), all benchmarks
#     bash run_all.sh germany-fc # Germany again with SMARD day-ahead forecasts (needs the 4 forecast files)
# Logs go to data/results/logs/. LEAR is the slow step (hours on a laptop); everything else takes minutes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/results/logs
step() { local name=$1; shift; echo "[$(date +%H:%M)] $name"; "$@" > "data/results/logs/$name.log" 2>&1 || { echo "FAILED: $name (see data/results/logs/$name.log)"; exit 1; }; }

if [ "${1:-all}" = "germany-fc" ]; then
  step de_fc_models      python scripts/model_de.py --forecasts
  step de_fc_qr          python scripts/benchmarks_qr.py --market germany --forecasts
  step de_fc_lear        python scripts/benchmarks_lear.py --market germany --forecasts
  step de_fc_table       python scripts/eval_benchmarks2.py --market germany --forecasts
  cat data/results/logs/de_fc_table.log
  exit 0
fi

# ---------- India ----------
step in_onepart        python scripts/model_gbm.py
step in_twopart        python scripts/model_capaware.py
step in_recal_one      python scripts/recalibrate.py --pred data/results/predictions.csv --prefix q --out predictions_cal.csv
step in_recal_two      python scripts/recalibrate.py --pred data/results/predictions_capaware.csv --prefix k --out predictions_capaware_cal.csv
step in_nobid          python scripts/ablation_nobid.py
step in_qr             python scripts/benchmarks_qr.py --market india
step in_lear           python scripts/benchmarks_lear.py --market india
step in_twopart_qra    python scripts/twopart_qra.py
step in_table          python scripts/eval_benchmarks2.py --market india
# ---------- Germany ----------
step de_models         python scripts/model_de.py
step de_qr             python scripts/benchmarks_qr.py --market germany
step de_lear           python scripts/benchmarks_lear.py --market germany
step de_table          python scripts/eval_benchmarks2.py --market germany
# ---------- robustness ----------
step robustness        python scripts/robustness.py
echo; cat data/results/logs/in_table.log data/results/logs/de_table.log
