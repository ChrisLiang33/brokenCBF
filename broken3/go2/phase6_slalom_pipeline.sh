#!/usr/bin/env bash
# Phase 6 slalom + intervention=0 pipeline:
#   1. retrain teacher on slalom
#   2. encoder_health on the final checkpoint
#   3. lidar_attention on the final checkpoint
#
# Run on labbox:
#     cd ~/Desktop/cbf_rl_mvp/go2
#     bash phase6_slalom_pipeline.sh 2>&1 | tee phase6_slalom_pipeline.log
#
# Or detached (survives ssh disconnect):
#     nohup bash phase6_slalom_pipeline.sh > phase6_slalom_pipeline.log 2>&1 &
#     disown
#
# Expected wall time: 1-2 hr (mostly the retrain).

set -euo pipefail

LOCO_CKPT="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt"
TASK="Isaac-CBF-Adaptive-Go2-Slalom-v0"
OUT_DIR="phase6_slalom_intervention0_teacher_outputs"
NUM_ENVS=256
MAX_ITER=1500

MASTER_LOG="${OUT_DIR}_master.log"
mkdir -p "${OUT_DIR}"
: > "${MASTER_LOG}"

echo "=========================================================================" | tee -a "${MASTER_LOG}"
echo "  PHASE 6 SLALOM PIPELINE" | tee -a "${MASTER_LOG}"
echo "  task:       ${TASK}" | tee -a "${MASTER_LOG}"
echo "  out_dir:    ${OUT_DIR}" | tee -a "${MASTER_LOG}"
echo "  num_envs:   ${NUM_ENVS}" | tee -a "${MASTER_LOG}"
echo "  max_iter:   ${MAX_ITER}" | tee -a "${MASTER_LOG}"
echo "  started:    $(date)" | tee -a "${MASTER_LOG}"
echo "=========================================================================" | tee -a "${MASTER_LOG}"

# -------------------------------------------------------------------------
# 1. RETRAIN
# -------------------------------------------------------------------------
echo "" | tee -a "${MASTER_LOG}"
echo "-------------------------------------------------------------------------" | tee -a "${MASTER_LOG}"
echo "  STAGE 1: TEACHER RETRAIN" | tee -a "${MASTER_LOG}"
echo "  started: $(date)" | tee -a "${MASTER_LOG}"
echo "-------------------------------------------------------------------------" | tee -a "${MASTER_LOG}"

~/IsaacLab/isaaclab.sh -p phase5_train_teacher.py \
    --checkpoint "${LOCO_CKPT}" \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITER}" \
    --out_dir "${OUT_DIR}" \
    --headless 2>&1 | tee -a "${MASTER_LOG}"

echo "  retrain finished: $(date)" | tee -a "${MASTER_LOG}"

# find the latest model_*.pt the training wrote (rsl_rl saves to
# <out_dir>/rsl_rl/model_*.pt by default -- use find to be robust)
LATEST_CKPT=$(find "${OUT_DIR}" -name "model_final.pt" | head -1)
if [[ -z "${LATEST_CKPT}" ]]; then
    LATEST_CKPT=$(find "${OUT_DIR}" -name "model_*.pt" -printf "%T@ %p\n" \
                  | sort -nr | head -1 | awk '{print $2}')
fi
if [[ -z "${LATEST_CKPT}" ]]; then
    echo "  ERROR: no model_*.pt found in ${OUT_DIR}/. Aborting." | tee -a "${MASTER_LOG}"
    exit 1
fi
echo "  latest checkpoint: ${LATEST_CKPT}" | tee -a "${MASTER_LOG}"

# -------------------------------------------------------------------------
# 2. ENCODER HEALTH
# -------------------------------------------------------------------------
echo "" | tee -a "${MASTER_LOG}"
echo "-------------------------------------------------------------------------" | tee -a "${MASTER_LOG}"
echo "  STAGE 2: ENCODER HEALTH" | tee -a "${MASTER_LOG}"
echo "  started: $(date)" | tee -a "${MASTER_LOG}"
echo "-------------------------------------------------------------------------" | tee -a "${MASTER_LOG}"

~/IsaacLab/isaaclab.sh -p phase6_encoder_health.py \
    --checkpoint "${LATEST_CKPT}" \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --headless 2>&1 | tee -a "${MASTER_LOG}"

echo "  encoder_health finished: $(date)" | tee -a "${MASTER_LOG}"

# -------------------------------------------------------------------------
# 3. LIDAR ATTENTION
# -------------------------------------------------------------------------
echo "" | tee -a "${MASTER_LOG}"
echo "-------------------------------------------------------------------------" | tee -a "${MASTER_LOG}"
echo "  STAGE 3: LIDAR ATTENTION  (THE HEADLINE)" | tee -a "${MASTER_LOG}"
echo "  started: $(date)" | tee -a "${MASTER_LOG}"
echo "-------------------------------------------------------------------------" | tee -a "${MASTER_LOG}"

~/IsaacLab/isaaclab.sh -p phase6_lidar_attention.py \
    --checkpoint "${LOCO_CKPT}" \
    --policy_checkpoint "${LATEST_CKPT}" \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --headless 2>&1 | tee -a "${MASTER_LOG}"

echo "  lidar_attention finished: $(date)" | tee -a "${MASTER_LOG}"

# -------------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------------
echo "" | tee -a "${MASTER_LOG}"
echo "=========================================================================" | tee -a "${MASTER_LOG}"
echo "  PIPELINE COMPLETE" | tee -a "${MASTER_LOG}"
echo "  finished: $(date)" | tee -a "${MASTER_LOG}"
echo "=========================================================================" | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"
echo "  checkpoint:        ${LATEST_CKPT}" | tee -a "${MASTER_LOG}"
echo "  encoder_health:    phase6_encoder_health_outputs/" | tee -a "${MASTER_LOG}"
echo "  lidar_attention:   phase6_lidar_attention_outputs/" | tee -a "${MASTER_LOG}"
echo "" | tee -a "${MASTER_LOG}"

# pull the headline verdict from the lidar_attention json
if [[ -f phase6_lidar_attention_outputs/phase6_lidar_attention.json ]]; then
    echo "  LIDAR ATTENTION VERDICT:" | tee -a "${MASTER_LOG}"
    python3 -c "
import json
with open('phase6_lidar_attention_outputs/phase6_lidar_attention.json') as f:
    d = json.load(f)
print('   ', d['verdict'])
print('    phi_span (Part B):  ', d['interventional']['phi_span'])
print('    alpha_span (Part B):', d['interventional']['alpha_span'])
" | tee -a "${MASTER_LOG}"
fi
echo "=========================================================================" | tee -a "${MASTER_LOG}"
