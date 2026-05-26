#!/bin/bash
# PHIWIN_TIGHTCOR_V2 (2026-05-18) — push corridor tightness further.
# PHIWIN_TIGHTCOR landed Pearson(φ, σ_act) = +0.15 (positive but below
# 0.3 gate). y_center=0.46 (was 0.48) gives ~0.5cm clearance per side
# at worst-radius obstacles. scene_corridor_prob=0.70 (was 0.50) for
# more exposure. Same isolation as PHIWIN_TIGHTCOR otherwise.
#
# Usage on lab box (in tmux):
#   tmux new -s wk3tight2
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_phiwin_tightcor_v2.sh \
#     2>&1 | tee logs/train_and_eval_wk3tight2.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-1500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "PHIWIN_TIGHTCOR_V2: y_center=0.46, prob=0.70"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

grep -q "CbfGo2EnvCfg_LAYER3_PHIWIN_TIGHTCOR_V2" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ TIGHTCOR_V2 config present" \
  || { echo "  ✗ TIGHTCOR_V2 config missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V2-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ TIGHTCOR_V2 task registered" \
  || { echo "  ✗ TIGHTCOR_V2 task not registered"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V2-v0 \
  --num_envs "${ENVS}" --max_iterations "${ITERATIONS}" \
  --headless

unset CBF_AUX_COEF
LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1 "${LATEST_DIR}"/model_*.pt 2>/dev/null \
    | awk -F'model_|\\.pt' '{print $2, $0}' \
    | sort -n | awk '{print $2}' | tail -1)
[ -f "$CKPT" ] || { echo "ERROR: no ckpt"; exit 1; }
echo "Using checkpoint: $CKPT"

echo ""
echo "[2/3] DIAGNOSTICS at $(date '+%H:%M:%S')"
export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V2-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3tight2.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V2-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3tight2.json --headless

echo ""
echo "[3/3] EVAL at $(date '+%H:%M:%S')"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V2-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight2_indist" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight2_deploy" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo "Gate: Pearson(φ, σ_act) > 0.3, within-env std > 0.5"
echo "================================================================"
