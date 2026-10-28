#!/usr/bin/env bash
# End-to-end self test on SYNTHETIC data (CPU only, ~5-15 min). Verifies the code, not the science.
# Runs every stage of the paper pipeline (preprocess -> train -> prune -> lobo -> export -> Jetson dry run with
# ONNX Runtime -> evaluate -> report -> paper_numbers) in a scratch folder; your data/ and results/ are not touched.
#   bash tools/selftest.sh [/tmp/tcfd_selftest]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-/tmp/tcfd_selftest}"
rm -rf "$WORK"; mkdir -p "$WORK"
cp "$REPO/config.yaml" "$WORK/config.yaml"
export PYTHONPATH="$REPO"
CFG="$WORK/config.yaml"
FAST="train.epochs=10 train.device=cpu prune.finetune_epochs=3 prune.ratios=[0.5,0.85]"
cd "$REPO"

echo "== synthetic PU-format data"; python3 tools/make_synthetic_pu.py --config "$CFG" --out "$WORK/data/raw" --len 70000 >/dev/null
echo "== preprocess";  python3 -m tcfd.data_pu -c "$CFG" --workers 2 | tail -3
echo "== train";       python3 -m tcfd.train -c "$CFG" --set $FAST | grep -E "saved|windows"
echo "== prune";       python3 -m tcfd.prune -c "$CFG" --set $FAST | grep -E "saved"
echo "== lobo";        python3 -m tcfd.lobo -c "$CFG" --archs wdcnn --set $FAST | grep -E "saved|pool"
echo "== export";      python3 -m tcfd.export -c "$CFG" --set $FAST | grep -E "exported|Wrote"
echo "== jetson dry run (ONNX Runtime CPU)"
python3 jetson/smoke_test.py --backend ort --export-dir "$WORK/export" --out-dir "$WORK/jetson_out"
python3 jetson/dump_logits.py --backend ort --precisions fp32 --export-dir "$WORK/export" --out-dir "$WORK/jetson_out" | tail -1
python3 jetson/bench.py --backend ort --precisions fp32 --duration 1 --py-iters 50 --idle-seconds 0.2 \
        --export-dir "$WORK/export" --out-dir "$WORK/jetson_out" | tail -1
python3 tools/fake_precisions.py --out-dir "$WORK/jetson_out"
echo "== evaluate";    python3 -m tcfd.evaluate -c "$CFG" | tail -1
echo "== report";      python3 -m tcfd.report -c "$CFG" | tail -1
echo "== paper numbers"; python3 -m tcfd.paper_numbers -c "$CFG" | tail -1
ls "$WORK/results/figures" "$WORK/results/tables"
echo "SELFTEST OK  ->  $WORK"
