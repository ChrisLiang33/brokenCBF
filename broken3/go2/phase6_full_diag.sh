#!/usr/bin/env bash
# Run all 5 diagnostics on a trained teacher in sequence, in one terminal.
#
# Usage:
#   bash phase6_full_diag.sh <policy_checkpoint> [task] [decorr_task]
#
# Defaults:
#   task         = Isaac-CBF-Adaptive-Go2-Unified-v0
#   decorr_task  = Isaac-CBF-Adaptive-Go2-Decorr-v0
#
# Example:
#   bash phase6_full_diag.sh phase6_unified_teacher_outputs/rsl_rl/model_final.pt
#
# Total wall time: ~5-7 min (each script restarts Isaac Sim ~20s).
# Outputs:
#   - per-script JSONs in their respective out_dirs
#   - combined console log at phase6_full_diag_<name>.log
#
set -euo pipefail

LOCO_CKPT="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt"
POLICY_CKPT="${1:?Usage: bash phase6_full_diag.sh <policy_checkpoint> [task] [decorr_task]}"
TASK="${2:-Isaac-CBF-Adaptive-Go2-Unified-v0}"
DECORR_TASK="${3:-Isaac-CBF-Adaptive-Go2-Decorr-v0}"
NUM_ENVS=256

# derive a name from the checkpoint dir for the log file
RUN_NAME=$(basename "$(dirname "$(dirname "${POLICY_CKPT}")")")
LOG="phase6_full_diag_${RUN_NAME}.log"
: > "${LOG}"

# pretty section banner
banner () {
    echo "" | tee -a "${LOG}"
    echo "================================================================================" | tee -a "${LOG}"
    echo "  $1" | tee -a "${LOG}"
    echo "  started: $(date)" | tee -a "${LOG}"
    echo "================================================================================" | tee -a "${LOG}"
}

cd ~/Desktop/cbf_rl_mvp/go2

banner "DIAG 1/5  --  ENCODER HEALTH  (architecture + activation health)"
~/IsaacLab/isaaclab.sh -p phase6_encoder_health.py \
    --checkpoint "${LOCO_CKPT}" \
    --policy_checkpoint "${POLICY_CKPT}" \
    --task "${TASK}" --num_envs "${NUM_ENVS}" --headless 2>&1 | tee -a "${LOG}"

banner "DIAG 2/5  --  PRIV ATTENTION  (per-channel usage)"
~/IsaacLab/isaaclab.sh -p phase6_priv_attention.py \
    --checkpoint "${LOCO_CKPT}" \
    --policy_checkpoint "${POLICY_CKPT}" \
    --task "${TASK}" --num_envs "${NUM_ENVS}" --headless 2>&1 | tee -a "${LOG}"

banner "DIAG 3/5  --  LIDAR ATTENTION  (Part B span)"
~/IsaacLab/isaaclab.sh -p phase6_lidar_attention.py \
    --checkpoint "${LOCO_CKPT}" \
    --policy_checkpoint "${POLICY_CKPT}" \
    --task "${TASK}" --num_envs "${NUM_ENVS}" --headless 2>&1 | tee -a "${LOG}"

banner "DIAG 4/5  --  DECORRELATION  (lidar vs goal-proxy on Decorr env)"
~/IsaacLab/isaaclab.sh -p phase6_decorrelation_test.py \
    --checkpoint "${LOCO_CKPT}" \
    --policy_checkpoint "${POLICY_CKPT}" \
    --task "${DECORR_TASK}" --num_envs "${NUM_ENVS}" \
    --rollout_steps 600 --headless 2>&1 | tee -a "${LOG}"

banner "DIAG 5/5  --  PHASE5 DISTURBANCE SWEEP  (does policy adapt to d?)"
# phase5_train_teacher.py with max_iterations=0 just loads + evals
# the existing checkpoint. The eval output is the disturbance sweep.
~/IsaacLab/isaaclab.sh -p phase5_train_teacher.py \
    --checkpoint "${LOCO_CKPT}" \
    --task "${TASK}" \
    --policy_checkpoint "${POLICY_CKPT}" \
    --num_envs "${NUM_ENVS}" --max_iterations 0 \
    --out_dir "$(dirname "$(dirname "${POLICY_CKPT}")")" \
    --headless 2>&1 | tee -a "${LOG}"

# --------------------------------------------------------------------------
# FINAL SUMMARY  --  cherry-pick verdict lines from each diagnostic
# --------------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "================================================================================" | tee -a "${LOG}"
echo "  FULL DIAG COMPLETE  --  $(date)" | tee -a "${LOG}"
echo "================================================================================" | tee -a "${LOG}"
echo "  checkpoint: ${POLICY_CKPT}" | tee -a "${LOG}"
echo "  task:       ${TASK}" | tee -a "${LOG}"
echo "  log:        ${LOG}" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"
echo "  --- verdict lines ---" | tee -a "${LOG}"
grep -E "verdict|VERDICT|PASS|FAIL|FLAT|MIXED|SIGNAL|fraction" "${LOG}" \
     | grep -v "^[+0-9]" | tail -30 | tee -a "${LOG}"
echo "================================================================================" | tee -a "${LOG}"
