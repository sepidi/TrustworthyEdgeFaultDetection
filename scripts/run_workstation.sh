#!/usr/bin/env bash
# Workstation stage: preprocess -> train dense models -> prune + fine-tune -> bearing-level calibration
# models (tcfd.lobo) -> export for Jetson.
#   bash scripts/run_workstation.sh                 # everything
#   bash scripts/run_workstation.sh --folds A       # one fold first (recommended for day 1)
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${CONFIG:-config.yaml}"
mkdir -p logs
ARGS="$*"
python -m tcfd.data_pu -c "$CFG"           2>&1 | tee logs/01_preprocess.log
python -m tcfd.train   -c "$CFG" $ARGS     2>&1 | tee logs/02_train.log
python -m tcfd.prune   -c "$CFG" $ARGS     2>&1 | tee logs/03_prune.log
python -m tcfd.lobo    -c "$CFG" $ARGS     2>&1 | tee logs/03b_lobo.log
python -m tcfd.export  -c "$CFG" $ARGS     2>&1 | tee logs/04_export.log
echo "Workstation stage done. Next: JETSON=user@host bash scripts/sync_to_jetson.sh (see README, Section 5.2)"
