#!/bin/bash
# Resume the LAYER3_PUSH pipeline at the diagnostic+eval phase.
# Training finished 2026-05-16 03:33; diagnostics crashed at the
# Unsupported priv_dim=31 layout in diagnose_phi_corr.py. Patched.
# Usage:
#   ~/Desktop/safety-go2/scripts/eval_only_layer3_push.sh
#       2>&1 | tee ~/Desktop/safety-go2/IsaacLab/logs/eval_wk3push.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

CKPT="${CKPT:-logs/rsl_rl/cbf_go2_teacher_rma/2026-05-16_02-07-29/model_1499.pt}"
test -f "$CKPT" || { echo "Checkpoint not found: $CKPT"; exit 1; }
echo "Using checkpoint: $CKPT"

export CBF_AUX_COEF=0.0

echo ""
echo "==== φ-CORR ===="
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --use_locked \
  --output diagnose_phi_corr_wk3push.json \
  --headless

echo ""
echo "==== α-CORR ===="
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_alpha_corr_wk3push.json \
  --headless

echo ""
echo "==== PROBE Z ===="
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3push.json \
  --headless

echo ""
echo "==== HEADLINE EVAL (B0 + B1 + B2 + BR) ===="
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,4.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir logs/baseline_eval_wk3push_indist \
  --headless

unset CBF_AUX_COEF
echo ""
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Eval CSV: logs/baseline_eval_wk3push_indist/baseline.csv"
