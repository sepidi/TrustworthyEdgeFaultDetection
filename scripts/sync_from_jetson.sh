#!/usr/bin/env bash
# Bring logits + benchmark results back (engines stay on the board).
#   JETSON=user@jetson-ip bash scripts/sync_from_jetson.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${JETSON:?set JETSON=user@host}"
REMOTE_DIR="${REMOTE_DIR:-~/tcfd}"
OUT="${OUT:-jetson_out}"
mkdir -p "$OUT"
rsync -avP "$JETSON:$REMOTE_DIR/$OUT/logits" "$JETSON:$REMOTE_DIR/$OUT/bench" "$OUT/"
rsync -avP "$JETSON:$REMOTE_DIR/$OUT/*.log" "$OUT/" 2>/dev/null || true
echo "Next: bash scripts/run_analysis.sh"
