#!/usr/bin/env bash
# Sync Mac-side changes to the lab workstation via rsync over SSH.
#
# Usage (from Mac):
#     ./scripts/sync_to_lab.sh          # push everything, no deletes
#     ./scripts/sync_to_lab.sh --dry    # preview what would change
#     ./scripts/sync_to_lab.sh --prune  # also delete remote files missing locally
#
# Assumes SSH access to chrisliang@130.64.84.163 is already set up
# (same username, no key prompt — add SSH key to lab box if prompting).
#
# What gets synced:
#   - IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
#     → the main dev surface, all the CBF teacher code
#   - scripts/
#     → smoke-test + locomotion inference helpers
#
# What does NOT get synced:
#   - logs/ (lab-generated; pulling them is a separate concern)
#   - __pycache__/ (Python bytecode, regenerated on import)
#   - .git/, .venv/, .DS_Store (obvious)
#   - PROGRESS.md, LOG.md, TODO_training.md (kept on Mac only as source of truth)

set -euo pipefail

LAB_USER="chrisliang"
LAB_HOST="130.64.84.163"
LAB_ROOT="${LAB_USER}@${LAB_HOST}:~/Desktop/safety-go2"
LOCAL_ROOT="${HOME}/Desktop/safety-go2"

# Rsync flags:
#   -a   archive (preserve perms, times, symlinks, recursive)
#   -v   verbose (prints each file transferred)
#   -h   human-readable sizes
#   -z   compress during transfer
#   --progress  per-file progress (older flag; macOS rsync 2.6 lacks info=progress2)
RSYNC_FLAGS=(-avhz --progress)

EXCLUDES=(
  --exclude='__pycache__/'
  --exclude='.DS_Store'
  --exclude='*.pyc'
  --exclude='logs/'
  --exclude='.git/'
  --exclude='.venv/'
)

# Dry-run vs. prune flags
case "${1:-}" in
  --dry)
    RSYNC_FLAGS+=(--dry-run)
    echo "=== DRY RUN: no files will change ==="
    ;;
  --prune)
    RSYNC_FLAGS+=(--delete)
    echo "=== PRUNE MODE: remote files missing locally will be DELETED ==="
    read -r -p "Continue? [y/N] " ans
    [[ "${ans}" =~ ^[yY]$ ]] || { echo "Aborted."; exit 1; }
    ;;
  "")
    ;;
  *)
    echo "Unknown flag: $1"
    echo "Usage: $0 [--dry|--prune]"
    exit 1
    ;;
esac

# The two paths that actually need syncing. Trailing slashes matter:
# they tell rsync to merge contents INTO the target dir, not nest it.
PATHS_TO_SYNC=(
  "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/"
  "scripts/"
)

for rel_path in "${PATHS_TO_SYNC[@]}"; do
  echo ""
  echo "→ Syncing ${rel_path}"
  rsync "${RSYNC_FLAGS[@]}" "${EXCLUDES[@]}" \
    "${LOCAL_ROOT}/${rel_path}" \
    "${LAB_ROOT}/${rel_path}"
done

echo ""
echo "✓ Sync complete. On the lab box, clear pycache to force reimport:"
echo "  find IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety \\"
echo "       -name __pycache__ -type d -exec rm -rf {} +"
