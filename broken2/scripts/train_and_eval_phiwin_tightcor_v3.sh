#!/bin/bash
# PHIWIN_TIGHTCOR_V3 (2026-05-18) — Step 2 of the φ-α-adaptation roadmap.
# Single-variable change from V1 PHIWIN_TIGHTCOR: release α. Everything
# else identical to V1, including:
#   - corridor y_center=0.48, scene_corridor_prob=0.50 (V1 sweet spot)
#   - per-episode φ lock kept (windowed lock is Step 3 V4 territory)
#   - σ_act curriculum + post-QP adversarial noise unchanged
#   - a + c stay frozen as hyperparameters
#
# Clean ablation: if V3 shows α + φ both adapt, Step 2 is done and we
# move to Step 3 (within-episode DR, V4). If only α moves and φ regresses,
# we learn that α release alone breaks φ. If neither moves, we learn the
# config is fundamentally not eliciting adaptation.
#
# Usage on lab box (in tmux):
#   tmux new -s wk3tight3
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_phiwin_tightcor_v3.sh \
#     2>&1 | tee logs/train_and_eval_wk3tight3.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-1500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "PHIWIN_TIGHTCOR_V3: Step 2 — release α on V1 TIGHTCOR base"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Sanity-check the V3 cfg + task are present before we burn 2h of GPU.
grep -q "CbfGo2EnvCfg_LAYER3_PHIWIN_TIGHTCOR_V3" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ TIGHTCOR_V3 config present" \
  || { echo "  ✗ TIGHTCOR_V3 config missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V3-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ TIGHTCOR_V3 task registered" \
  || { echo "  ✗ TIGHTCOR_V3 task not registered"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V3-v0 \
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

# φ diagnostic — per-episode lock still on, so within-env std will be
# near zero (only varies if an env resets within the 100-step rollout).
# The headline number is Pearson(φ, σ_act) — should match or beat V1's
# +0.151 since corridor geometry is unchanged.
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V3-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3tight3.json --headless

# α diagnostic — α released for the first time on this corridor base.
# Looking for Pearson(α, friction) or Pearson(α, |tracking_err|) > +0.3.
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V3-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_alpha_corr_wk3tight3.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V3-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3tight3.json --headless

echo ""
echo "[3/3] EVAL at $(date '+%H:%M:%S')"

# In-distribution sanity (training env).
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V3-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight3_indist" --headless

# FROZEN_AC deploy (apples-to-apples — a and c frozen at training values).
# This is the headline eval: BR vs B0/B1/B2 head-to-head.
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight3_frozenac" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "Step 2 success gates:"
echo "  φ:  Pearson(φ, σ_act) ≥ +0.15 (preserved from V1, ideally > +0.20)"
echo "  α:  Pearson(α, friction) or Pearson(α, |tracking_err|) > +0.30"
echo "  Deploy: BR joint_actual on FROZEN_AC > B0 AND > B1 AND > B2"
echo ""
echo "Diagnostic outputs:"
echo "  φ corr: diagnose_phi_corr_wk3tight3.json"
echo "  α corr: diagnose_alpha_corr_wk3tight3.json"
echo "  z probe: probe_z_linear_wk3tight3.json"
echo ""
echo "Eval outputs:"
echo "  In-dist: logs/baseline_eval_wk3tight3_indist/baseline.csv"
echo "  Deploy:  logs/baseline_eval_wk3tight3_frozenac/baseline.csv"
echo "================================================================"
