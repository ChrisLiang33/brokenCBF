#!/bin/bash
# PHIWIN_TIGHTCOR_V7 (2026-05-19) — structural perception-bias fix on V5
# base. Single variable change: c clamp widened from (-0.05, -0.05) to
# (-0.10, -0.10) to fully compensate SHIELD_R_SAFETY_MARGIN.
#
# Audit context: the LAYER3 chain uses synthetic LiDAR clustering
# (shield_v0c) which structurally inflates every perceived obstacle
# radius by +0.10 m. CBF theory says c* = −0.10 should fully cancel
# this. PHIWIN_TIGHTCOR (V1) clamped c at −0.05, only undoing half the
# bias. Net effect: in the y_center=0.48 corridor, the policy perceived
# clearance as −0.02 m (after partial compensation) instead of the
# theoretical +0.03 m true clearance. The policy was training to
# navigate corridors it perceived as IMPASSABLE.
#
# V7 sets c clamp = (-0.10, -0.10) so perceived clearance matches true
# clearance.
#
# V7 does NOT include V6's additional changes (wider friction, dynamic
# obstacles, bigger σ_act amplitude). V6 regressed; we revert to V5 base
# and isolate the perception-bias fix. V6's distribution amplifications
# can stack on top of V7 as V8 if V7 succeeds.
#
# Usage on lab box (in tmux):
#   tmux new -s wk3tight7
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   CBF_ITERATIONS=3000 ~/Desktop/safety-go2/scripts/train_and_eval_phiwin_tightcor_v7.sh \
#     2>&1 | tee logs/train_and_eval_wk3tight7.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-3000}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "PHIWIN_TIGHTCOR_V7: V5 + c clamp (-0.10, -0.10) — SHIELD fix"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')  iters=${ITERATIONS}"
echo "================================================================"

grep -q "CbfGo2EnvCfg_LAYER3_PHIWIN_TIGHTCOR_V7" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V7 train cfg present" \
  || { echo "  ✗ V7 train cfg missing"; exit 1; }
grep -q "CbfGo2EnvCfg_DEPLOY_REALISTIC_FROZEN_AC_TRAINMATCH_V7" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V7 deploy cfg present" \
  || { echo "  ✗ V7 deploy cfg missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V7-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ V7 train task registered" \
  || { echo "  ✗ V7 train task not registered"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V7-v0 \
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
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V7-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3tight7.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V7-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_alpha_corr_wk3tight7.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V7-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3tight7.json --headless

echo ""
echo "[3/3] EVAL at $(date '+%H:%M:%S')"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V7-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight7_indist" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V7-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight7_trainmatch" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "What to look for:"
echo "  φ_mean   : expected lower than V5 (2.39) — less hedging vs perceived walls"
echo "  φ within-ep Pearson(φ_t, h_t) : ideally < -0.20 (TISSf-shape adaptation)"
echo "  α within-ep Pearson(α_t, |tracking_err|_t) : should be < -0.20"
echo "  In-dist BR vs best fixed     : >5pp improvement = SHIELD bias was bottleneck"
echo "  Deploy BR vs best fixed       : same gate"
echo ""
echo "If V7 closes the gap: perception bias was the binding constraint."
echo "If V7 doesn't close it: the bottleneck is reward/architecture/training-length."
echo "================================================================"
