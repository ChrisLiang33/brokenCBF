#!/bin/bash
# Verification eval: PHIWIN_TIGHTCOR ckpt on DEPLOY_REALISTIC_FROZEN_AC.
# Tests the deploy-collapse hypothesis — freezing a + c at eval time
# (to match training) should restore deploy generalization.
#
# Comparison:
#   PHIWIN_TIGHTCOR on DEPLOY_REALISTIC (a/c released):
#     j_act 0.189, fall 0.683, a≈1.3, c≈+0.07  (collapsed)
#   PHIWIN_TIGHTCOR on DEPLOY_REALISTIC_FROZEN_AC (a/c clamped to training):
#     ??? — if hypothesis is correct, fall should drop significantly

set -e
cd ~/Desktop/safety-go2/IsaacLab

CKPT="logs/rsl_rl/cbf_go2_teacher_rma/2026-05-18_13-16-52/model_1499.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: PHIWIN_TIGHTCOR ckpt not found: $CKPT"
    exit 1
fi

echo "================================================================"
echo "Verification eval: PHIWIN_TIGHTCOR on DEPLOY_REALISTIC_FROZEN_AC"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "================================================================"

export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_tightcor_frozenac" --headless

echo ""
echo "Done. CSV at logs/baseline_eval_tightcor_frozenac/baseline.csv"
