#!/usr/bin/env bash
# Pull labbox cbf_rl_mvp/go2/ run artifacts back to local.
#
# Usage:   ./scripts/sync_pull.sh
# Override the source path with LABBOX_PATH=... ./scripts/sync_pull.sh
#
# Brings down ONLY run artifacts (CSVs, logs, plots, checkpoints) -- never
# overwrites source files, so you can't clobber local edits.
set -euo pipefail

cd "$(dirname "$0")/.."   # cbf_rl_mvp/go2/
LOCAL_DIR="$(pwd)/"
REMOTE_HOST="${LABBOX_HOST:-labbox}"
REMOTE_PATH="${LABBOX_PATH:-~/Desktop/cbf_rl_mvp/go2/}"

echo "[sync_pull] $REMOTE_HOST:$REMOTE_PATH -> $LOCAL_DIR"
rsync -av \
    --include='*/' \
    --include='*.csv' \
    --include='*.log' \
    --include='*.png' \
    --include='*.pt' \
    --include='*.onnx' \
    --include='*.json' \
    --include='*.txt' \
    --include='*.mp4' \
    --include='logs/***' \
    --include='outputs/***' \
    --exclude='*' \
    "$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_DIR"
