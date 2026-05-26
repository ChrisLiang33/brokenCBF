#!/bin/bash
# Re-eval V3's already-trained checkpoint with patched eval_baseline.py.
# Original V3 eval had all-zero B0/B1/B2 because of a 0/0 NaN bug when
# c_param_range = (-0.05, -0.05) (V3 in-dist and FROZEN_AC both hit it).
# Patched _encode_dim to return zeros for degenerate ranges; this script
# just re-runs the eval portion against the existing ckpt.
#
# Usage on lab box:
#   tmux new -s wk3reeval
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/reeval_v3_baselines.sh \
#     2>&1 | tee logs/reeval_wk3tight3.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

# Use the V3 ckpt from the training run that just finished.
CKPT="logs/rsl_rl/cbf_go2_teacher_rma/2026-05-18_18-41-33/model_1499.pt"
[ -f "$CKPT" ] || { echo "ERROR: V3 ckpt missing: $CKPT"; exit 1; }

echo "================================================================"
echo "V3 re-eval with patched eval_baseline.py (degenerate-range NaN fix)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "================================================================"

export CBF_AUX_COEF=0.0

echo ""
echo "[1/2] in-dist eval at $(date '+%H:%M:%S')"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V3-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight3_indist_v2" --headless

echo ""
echo "[2/2] FROZEN_AC eval at $(date '+%H:%M:%S')"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight3_frozenac_v2" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  In-dist v2: logs/baseline_eval_wk3tight3_indist_v2/baseline.csv"
echo "  FROZEN_AC v2: logs/baseline_eval_wk3tight3_frozenac_v2/baseline.csv"
echo "================================================================"
