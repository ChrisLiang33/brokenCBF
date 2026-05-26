#!/bin/bash
# V11 post-training: diagnostics + 4 evals against an existing ckpt.
# Used when resume_v11.sh's training phase finished but the diagnostic/eval
# phases didn't run (chain script jumped to V12 prematurely).
#
# Usage:
#   tmux new -s wk3v11_postdiag
#   cd ~/Desktop/safety-go2/IsaacLab
#   CKPT=~/Desktop/safety-go2/IsaacLab/logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_11-43-29/model_2999.pt \
#     ~/Desktop/safety-go2/scripts/v11_postdiag.sh \
#     2>&1 | tee logs/v11_postdiag.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

CKPT=${CKPT:?must set CKPT=path/to/model_XXXX.pt}
[ -f "$CKPT" ] || { echo "ERROR: ckpt missing: $CKPT"; exit 1; }

echo "================================================================"
echo "V11 POSTDIAG: $CKPT"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/2] DIAGNOSTICS at $(date '+%H:%M:%S')"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 12 \
  --output diagnose_phi_corr_wk3v11.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 12 \
  --output diagnose_alpha_corr_wk3v11.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 12 \
  --output probe_z_linear_wk3v11.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_grad_sensitivity.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 50 --priv_dim 12 \
  --alpha_min 0.5 --alpha_max 3.0 \
  --output diagnose_grad_sensitivity_wk3v11.json --headless

echo ""
echo "[2/2] 4-EVAL at $(date '+%H:%M:%S')"

# 1. in-dist
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v11_indist" --headless

# 2. trainmatch
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V11-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v11_trainmatch" --headless

# 3. OOD
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V11-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v11_ood" --headless

# 4. STRESSOR
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V11-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v11_stressor" --headless

echo ""
echo "================================================================"
echo "V11 POSTDIAG DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo "================================================================"
