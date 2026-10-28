#!/usr/bin/env bash
# Full Jetson pipeline: smoke test -> build all engines -> dump logits -> benchmark.
# Run from the repo root on the Jetson:   bash jetson/run_all.sh
# Long-running: start it inside tmux/screen so an SSH drop does not kill it.
set -euo pipefail
EXPORT=${EXPORT:-export}
OUT=${OUT:-jetson_out}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"

echo "== Environment"; cat /etc/nv_tegra_release 2>/dev/null || true
python3 -c "import tensorrt as t; print('TensorRT', t.__version__)"
python3 -c "import polygraphy; print('Polygraphy', polygraphy.__version__)"

echo "== Power mode (lock clocks for stable latency numbers)"
sudo nvpmodel -q || true
sudo jetson_clocks || echo "[WARN] jetson_clocks failed - latency numbers will be noisier"

echo "== 1/4 smoke test"
python3 jetson/smoke_test.py --export-dir "$EXPORT" --out-dir "$OUT"

echo "== 2/4 build engines (all folds)"
python3 jetson/build_engines.py --export-dir "$EXPORT" --out-dir "$OUT" 2>&1 | tee "$OUT/build.log"

echo "== 3/4 dump logits (all folds, all eval sets)"
python3 jetson/dump_logits.py --export-dir "$EXPORT" --out-dir "$OUT" 2>&1 | tee "$OUT/dump.log"

echo "== 4/4 benchmark (bench fold only)"
python3 jetson/bench.py --export-dir "$EXPORT" --out-dir "$OUT" 2>&1 | tee "$OUT/bench.log"

echo "ALL DONE. Copy $OUT/logits and $OUT/bench back to the workstation (scripts/sync_from_jetson.sh)."
