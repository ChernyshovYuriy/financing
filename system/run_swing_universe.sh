#!/usr/bin/env bash
# Runs the swing universe builder and pushes the resulting
# data/can_tickers_swing_universe to git if it changed.
#
# Invoked by financing-swing-universe.service (see system/info for install steps).

set -euo pipefail

REPO_DIR="/home/yurii/dev/financing"
PY="$REPO_DIR/.venv/bin/python3"
OUT_FILE="data/can_tickers_swing_universe"

cd "$REPO_DIR/py"
"$PY" swing_universe.py

cd "$REPO_DIR"

if [ -n "$(git status --porcelain -- "$OUT_FILE")" ]; then
    git add "$OUT_FILE"
    git commit -m "Update swing universe ($(date -u +%Y-%m-%d))"
    git push origin main
    echo "Pushed updated $OUT_FILE"
else
    echo "No change in $OUT_FILE; nothing to commit."
fi
