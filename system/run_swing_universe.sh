#!/usr/bin/env bash
# Runs the swing universe builder and pushes the resulting
# data/can_tickers_swing_universe to git if it changed.
#
# Invoked by financing-swing-universe.service (see system/info for install steps).

set -euo pipefail

REPO_DIR="/home/pi/dev/financing"
# No venv on this box, same as stockscanner's units: rely on the system
# python3 having yfinance/pandas/numpy installed globally.
PY="python3"
OUT_FILE="data/can_tickers_swing_universe"

cd "$REPO_DIR/py"
"$PY" swing_universe.py

cd "$REPO_DIR"

if [ -n "$(git status --porcelain -- "$OUT_FILE")" ]; then
    git add "$OUT_FILE"
    git -c user.name="Financing Bot" -c user.email="chernyshov.yuriy@gmail.com" \
        commit -m "Update swing universe ($(date -u +%Y-%m-%d))"
    git push origin main
    echo "Pushed updated $OUT_FILE"
else
    echo "No change in $OUT_FILE; nothing to commit."
fi
