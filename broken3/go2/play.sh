#!/usr/bin/env bash
# Quick play helper. Usage:
#   bash play.sh <policy_checkpoint> [task] [num_envs] [steps]
# Example:
#   bash play.sh phase6_slalom_intervention0_teacher_outputs/rsl_rl/model_final.pt
#   bash play.sh phase6_decorr_intervention0_teacher_outputs/rsl_rl/model_final.pt Isaac-CBF-Adaptive-Go2-Decorr-v0
#
# Defaults: task=Slalom, num_envs=1, steps=1500.
# Requires GUI session (DISPLAY set). Won't work from a pure-SSH terminal.

set -euo pipefail

LOCO_CKPT="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt"
POLICY_CKPT="${1:?Usage: bash play.sh <policy_checkpoint> [task] [num_envs] [steps]}"
TASK="${2:-Isaac-CBF-Adaptive-Go2-Slalom-v0}"
NUM_ENVS="${3:-1}"
STEPS="${4:-1500}"

if [[ -z "${DISPLAY:-}" ]]; then
    echo "ERROR: \$DISPLAY is not set. You're in a text-only TTY or SSH session."
    echo "       Run this from a terminal opened ON THE LAB BOX DESKTOP."
    exit 1
fi

cd ~/Desktop/cbf_rl_mvp/go2
~/IsaacLab/isaaclab.sh -p phase6_play.py \
    --checkpoint "${LOCO_CKPT}" \
    --policy_checkpoint "${POLICY_CKPT}" \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --steps "${STEPS}"
