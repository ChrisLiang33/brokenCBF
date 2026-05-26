#!/bin/bash
# RMA_CANONICAL_V9 (2026-05-19) — canonical RMA priv obs only.
#
# Same V8 dynamics (SHIELD c-comp at −0.10, per-step φ, within-episode
# σ_act regime at 5s windows, lock removed). Only the priv obs layout
# changes: from 33-D (V8) to 13-D (canonical RMA-style env factors).
#
# V8 evals showed BR matches oracle best fixed on trainmatch but loses
# on both in-dist (rank 4) and OOD (rank 8/16) by ~10pp safety_score.
# Shared-signal regression found heads gate primarily off tracking_err
# + base_height — partially observable signals, not true privileged info.
#
# V9 hypothesis: forcing heads to read TRUE privileged signals (friction,
# mass, σ_act, COM, applied force/torque) should improve OOD generalization,
# since these are the actual environmental factors the encoder needs to
# distill for student deploy.
#
# Train length: 2500 iters (~3.3h on lab box RTX 5090). Same as V8.
#
# Usage on lab box (in tmux):
#   tmux new -s wk3rma9
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   CBF_ITERATIONS=2500 ~/Desktop/safety-go2/scripts/train_and_eval_rma_canonical_v9.sh \
#     2>&1 | tee logs/train_and_eval_wk3rma9.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-2500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "RMA_CANONICAL_V9: V8 minus non-canonical priv channels."
echo "13-D priv = friction + mass + base_height + applied_force(3)"
echo "             + applied_torque(3) + com_offset(3) + σ_act"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')  iters=${ITERATIONS}"
echo "================================================================"

grep -q "CbfGo2EnvCfg_LAYER3_RMA_CANONICAL_V9" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V9 train cfg present" \
  || { echo "  ✗ V9 train cfg missing"; exit 1; }
grep -q "CbfGo2EnvCfg_DEPLOY_REALISTIC_FROZEN_AC_TRAINMATCH_V9" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V9 deploy cfg present" \
  || { echo "  ✗ V9 deploy cfg missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-RMACanon-V9-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ V9 train task registered" \
  || { echo "  ✗ V9 train task not registered"; exit 1; }
grep -q "def set_priv_dim" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py \
  && echo "  ✓ set_priv_dim() runtime override present" \
  || { echo "  ✗ set_priv_dim() missing"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMACanon-V9-v0 \
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

# Note: priv_dim=13 for V9 (was 33 for V8). Pass it through.
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMACanon-V9-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 13 \
  --output diagnose_phi_corr_wk3rma9.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMACanon-V9-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 13 \
  --output diagnose_alpha_corr_wk3rma9.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMACanon-V9-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3rma9.json --headless

echo ""
echo "[3/3] EVAL at $(date '+%H:%M:%S')"

# In-dist eval (same as train task)
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMACanon-V9-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3rma9_indist" --headless

# Trainmatch eval (deploy-realistic, DR matches training)
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V9-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3rma9_trainmatch" --headless

# OOD eval (the key question)
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3rma9_ood" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "What V9 tests:"
echo "  Hypothesis: stripping tracking_err + base_ang_vel + δ_R from priv"
echo "  forces heads to use true privileged signals (friction, σ_act, COM,"
echo "  mass, force/torque). Should improve OOD generalization if V8's"
echo "  OOD loss was caused by gating off pseudo-priv signals."
echo ""
echo "Headline checks:"
echo "  Pearson(φ, σ_act) — was −0.20 in V8; if larger negative in V9,"
echo "                      φ is now leaning on σ_act more"
echo "  Pearson(α, friction) — V8 had ≈ 0 (β=−0.04 partial); V9 should"
echo "                          activate this if friction is needed"
echo "  OOD BR vs B0 α=2 — V8 lost by 9.6pp (0.626 vs 0.722)."
echo "                      V9 should at minimum match B0, ideally beat."
echo "  z_priv R²(friction) — V8 had 0.07; V9 forces encoder to extract"
echo "                         friction more cleanly (no tracking_err shortcut)."
echo "================================================================"
