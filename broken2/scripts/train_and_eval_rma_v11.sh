#!/bin/bash
# RMA_V11 (2026-05-20) — V9 priv MINUS base_height. 12-D truly-hidden priv.
#
# V8 gradient diagnostic: base_height was the dominant signal for both
# heads (φ sens 0.83, α sens 0.34). It's partially observable on hardware
# (IMU + foot kinematics), so reading it doesn't require RMA.
# V9 (no tracking_err, but kept base_height) showed heads still leaned on
# base_height as the residual "things bad" signal.
#
# V11 removes base_height too. Forces the policy to either:
#   (a) Find adaptation signal in truly-hidden priv (friction, σ_act, COM,
#       mass, force/torque) — would validate RMA architecture.
#   (b) Collapse further than V9 (V9 was already globally cautious) — would
#       confirm reward landscape is the binding constraint regardless of
#       priv design.
#
# Single variable change from V9:
#   priv: 13 dims → 12 dims (drop base_height)
#
# Usage on lab box (in tmux):
#   tmux new -s wk3v11
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   CBF_ITERATIONS=2500 ~/Desktop/safety-go2/scripts/train_and_eval_rma_v11.sh \
#     2>&1 | tee logs/train_and_eval_wk3v11.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-2500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "RMA_V11: V9 priv − base_height. 12-D truly-hidden priv only."
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')  iters=${ITERATIONS}"
echo "================================================================"

grep -q "CbfGo2EnvCfg_LAYER3_RMA_V11" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V11 train cfg present" \
  || { echo "  ✗ V11 train cfg missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ V11 train task registered" \
  || { echo "  ✗ V11 train task not registered"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
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
echo "[3/3] EVAL at $(date '+%H:%M:%S')"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v11_indist" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V11-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v11_trainmatch" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V11-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v11_ood" --headless

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
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "Critical comparisons:"
echo "  gradient sens σ_act:   V8 0.006, V9 0.016. V11 should reveal if"
echo "                          removing the last partially-observable signal"
echo "                          finally activates σ_act use."
echo "  gradient sens friction: V8 0.07, V9 0.07. Same test."
echo "  total head sensitivity: V8 ~2.40, V9 ~1.10. V11 lower → reward"
echo "                          landscape is binding constraint."
echo "================================================================"
