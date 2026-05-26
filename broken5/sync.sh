#!/usr/bin/env bash
# Push the project to the GPU lab machine. One-way: laptop -> lab.
# Configure once: edit LAB_HOST / LAB_PATH below, or set them in your shell.
#
# Usage:
#   ./sync.sh              # push changes
#   ./sync.sh --dry-run    # show what would change without copying
#   ./sync.sh --pull       # pull changes from lab -> laptop (e.g., trained weights)

set -euo pipefail

LAB_HOST="${LAB_HOST:-chrisliang@130.64.84.163}"
LAB_PATH="${LAB_PATH:-~/Desktop/mvp}"

HERE="$(cd "$(dirname "$0")" && pwd)"

RSYNC_FLAGS=(-avz --delete --human-readable --progress --stats)
EXCLUDES=(
  --exclude='.git/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.DS_Store'
  --exclude='.venv/'
  --exclude='runs/'        # training logs/checkpoints stay on the lab machine
  --exclude='wandb/'
)

mode="push"
extra=()
for arg in "$@"; do
  case "$arg" in
    --pull) mode="pull" ;;
    --dry-run|-n) extra+=(--dry-run) ;;
    *) extra+=("$arg") ;;
  esac
done

if [[ "$mode" == "push" ]]; then
  echo ">> push  $HERE/  ->  $LAB_HOST:$LAB_PATH/"
  rsync "${RSYNC_FLAGS[@]}" "${EXCLUDES[@]}" ${extra[@]+"${extra[@]}"} \
    "$HERE/" "$LAB_HOST:$LAB_PATH/"
else
  echo ">> pull  $LAB_HOST:$LAB_PATH/  ->  $HERE/"
  rsync "${RSYNC_FLAGS[@]}" "${EXCLUDES[@]}" ${extra[@]+"${extra[@]}"} \
    "$LAB_HOST:$LAB_PATH/" "$HERE/"
fi
