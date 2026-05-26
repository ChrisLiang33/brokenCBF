#!/bin/bash
# Wk3 4-param + one-sided φ over-inflation tax (2026-05-16).
#
# LAYER3_PUSH_A_C clean eval showed BR joint 0.255 vs best baseline 0.356
# (gap -10pp). Diagnosis: PPO traded c for φ in the constraint null space.
# c shifted only halfway (-0.06 vs needed -0.10), φ re-inflated 1.80→3.50
# to cover the residual, α stayed pegged at 4.25.
#
# This iteration: one-sided hinge max(φ - 2.0, 0) at weight -0.01.
# Closes the φ-inflation loophole; Kolathaya ISSf floor (φ ≈ 1-3 for
# σ_max=0.10) stays unpenalized. Forces PPO to push c the rest of the way
# negative to absorb the phantom padding.
#
# Single-variable change from LAYER3_PUSH_A_C:
#   - rewards.cbf_phi_above_target.weight = -0.01 (was 0.0)
# All else (push event, a release + L1 tax, c release [-0.20, +0.20],
# priv obs 31-D, perception, planner mix) held constant.
#
# Decision gates:
#   - joint_success > 0.356 (best fixed baseline from clean eval)
#   - mean(φ_used) drops from 3.50 toward 2.0
#   - mean(c) drops further negative, toward -0.10
#   - α mean drops from 4.25
#   - α-tracking coupling magnitude < 0.30
#   - a-tracking coupling stays positive
#
# Hard fallback: if this also fails, lock LAYER3_PUSH_A as the final
# teacher (joint 0.312) and move to student distillation. Treat the
# c-release branch as deferred until Wk4.
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3phitax
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_layer3_push_a_c_phitax.sh \
#     2>&1 | tee logs/train_and_eval_wk3phitax.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

AUX_COEF=0.0

echo "================================================================"
echo "Wk3 4-param + φ over-inflation tax: LAYER3_PUSH_A_C_PHITAX"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
grep -q "CbfGo2EnvCfg_LAYER3_PUSH_A_C_PHITAX" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ CbfGo2EnvCfg_LAYER3_PUSH_A_C_PHITAX config present" \
  || { echo "  ✗ config missing — sync env_cfg.py"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-Push-A-C-PhiTax-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ Isaac-CBF-Go2-RMA-Layer3-Push-A-C-PhiTax-v0 task registered" \
  || { echo "  ✗ task not registered — sync __init__.py"; exit 1; }
grep -q "cbf_phi_above_target_penalty" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_rewards.py \
  && echo "  ✓ cbf_phi_above_target_penalty present" \
  || { echo "  ✗ reward function missing — sync cbf_go2_rewards.py"; exit 1; }
grep -q "self.cbf_phi_param = phi" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py \
  && echo "  ✓ cbf_phi_param cached" \
  || { echo "  ✗ cbf_phi_param not cached — sync cbf_go2_env.py"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] PPO TRAINING: ${CBF_ITERATIONS:-1500} iters, 4096 envs"
echo "      4 params released, a-L1 + φ-above-target taxes active"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=$AUX_COEF
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-PhiTax-v0 \
  --num_envs 4096 --max_iterations ${CBF_ITERATIONS:-1500} \
  --headless

unset CBF_AUX_COEF

echo ""
echo "[1/3] PPO TRAINING done at $(date '+%H:%M:%S')"

LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1t "${LATEST_DIR}"/model_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in ${LATEST_DIR}"
    exit 1
fi
echo "Using checkpoint: $CKPT"

# ---------- PHASE 2: DIAGNOSTICS ----------
echo ""
echo "================================================================"
echo "[2/3] DIAGNOSTICS: α, φ, a, c correlations + linear probe"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-PhiTax-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --use_locked \
  --output diagnose_phi_corr_wk3phitax.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-PhiTax-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_alpha_corr_wk3phitax.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_a_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-PhiTax-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_a_corr_wk3phitax.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_c_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-PhiTax-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --c_lo -0.20 --c_hi 0.20 \
  --output diagnose_c_corr_wk3phitax.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-PhiTax-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3phitax.json \
  --headless

# ---------- PHASE 3: HEADLINE EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] HEADLINE EVAL: B0 + B1 + B2 + BR vs in-dist"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT="logs/baseline_eval_wk3phitax_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-PhiTax-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,4.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT" \
  --headless

echo ""
echo "[3/3] HEADLINE EVAL done at $(date '+%H:%M:%S')"
unset CBF_AUX_COEF

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint:           $CKPT"
echo "Eval CSV:             logs/baseline_eval_wk3phitax_indist/baseline.csv"
echo "================================================================"
