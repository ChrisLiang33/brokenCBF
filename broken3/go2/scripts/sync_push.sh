#!/usr/bin/env bash
# Push local cbf_rl_mvp/go2/ to labbox.
#
# Usage:   ./scripts/sync_push.sh
# Override the destination path with LABBOX_PATH=... ./scripts/sync_push.sh
#
# Uses --update (newer-only, no delete) so remote-only artifacts (logs,
# CSVs from training runs) survive. Run sync_pull.sh to retrieve those.
set -euo pipefail

cd "$(dirname "$0")/.."   # cbf_rl_mvp/go2/
LOCAL_DIR="$(pwd)/"
REMOTE_HOST="${LABBOX_HOST:-labbox}"
REMOTE_PATH="${LABBOX_PATH:-~/Desktop/cbf_rl_mvp/go2/}"

echo "[sync_push] $LOCAL_DIR -> $REMOTE_HOST:$REMOTE_PATH"
# macOS ships an older rsync that doesn't support --mkpath, so create the
# parent dir up front. The path expansion happens remotely so ~ resolves.
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PATH"
rsync -av --update \
    --exclude='__pycache__/' \
    --exclude='*.py[cod]' \
    --exclude='.DS_Store' \
    --exclude='*.csv' \
    --exclude='*.log' \
    --exclude='*.png' \
    --exclude='*.pt' \
    --exclude='*.onnx' \
    --exclude='logs/' \
    --exclude='outputs/' \
    "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_PATH"
