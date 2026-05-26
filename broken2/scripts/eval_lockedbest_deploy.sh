#!/bin/bash
# Eval locked-best (LAYER3_PUSH_A_C) ckpt on DEPLOY_REALISTIC.
# Reference point for the deploy-collapse investigation — locked-best
# is the only iteration we believe deploys cleanly (low fall rate).
#
# Compare against PHIWIN_v1, PHIWIN_CURR, PHIWIN_TIGHTCOR deploy CSVs to
# identify what training changes caused deploy-time fall_rate to jump
# from ~0.05 to ~0.65.

set -e
cd ~/Desktop/safety-go2/IsaacLab

CKPT="logs/rsl_rl/cbf_go2_teacher_rma/2026-05-16_12-45-39/model_1499.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: locked-best ckpt not found: $CKPT"
    exit 1
fi

echo "================================================================"
echo "Eval locked-best on DEPLOY_REALISTIC"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "================================================================"

export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,4.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_lockedbest_deploy" --headless

echo ""
echo "Done. CSV at logs/baseline_eval_lockedbest_deploy/baseline.csv"
