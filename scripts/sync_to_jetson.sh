#!/usr/bin/env bash
# Copy the Jetson scripts + exported models/data to the board.
#   JETSON=user@jetson-ip bash scripts/sync_to_jetson.sh
# (default remote dir ~/tcfd; override with REMOTE_DIR=...)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${JETSON:?set JETSON=user@host}"
REMOTE_DIR="${REMOTE_DIR:-~/tcfd}"
ssh "$JETSON" "mkdir -p $REMOTE_DIR"
rsync -avP --delete jetson requirements-jetson.txt "$JETSON:$REMOTE_DIR/"
rsync -avP "${EXPORT:-export}" "$JETSON:$REMOTE_DIR/"
echo "On the Jetson:  cd $REMOTE_DIR && tmux new -s tcfd  then  bash jetson/run_all.sh"
