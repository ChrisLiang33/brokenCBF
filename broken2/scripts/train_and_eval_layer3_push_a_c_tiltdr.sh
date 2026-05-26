#!/bin/bash
# Wk3 reward-misalignment fix + symmetric perception DR (2026-05-16).
#
# Three previous iterations on the 4-param branch all regressed joint
# success: 0.312 (3-param) → 0.255 (4-param) → 0.208 (4-param + φ tax).
# Knobs moved in math-predicted directions every time but joint metric
# kept dropping because reward stack is misaligned with joint success:
# dense collision penalty + sparse fall penalty → PPO trades falls for
# collision-rate.
#
# Two orthogonal fixes:
#   1. tilt_penalty -2.0 → -10.0. Dense per-step fall pressure so PPO
#      can no longer cheaply trade falls for collision-rate reduction.
#   2. obstacle_radius_perception_error_range = (-0.15, +0.15). Was
#      one-sided (0, 0.10) → policy memorized a static c-offset.
#      Symmetric DR flips bias sign per episode so `c` has to deduce
#      geometry from z_priv.
#
# DROPPED: cbf_phi_above_target tax (inherits from PUSH_A_C, not PHITAX).
#
# Held constant:
#   - Push event ±1.0 m/s every 5-10 s
#   - Symptom-based priv obs (_PRIV_DIM = 31)
#   - a released ∈ [0, 3] with L1 tax -0.01
#   - c released ∈ [-0.20, +0.20]
#   - Perception (shield_v0c, n_rays=128, persist=50)
#   - Planner mix, corridor scenes
#
# Decision gates:
#   - joint_success > 0.356 (best fixed baseline)
#   - fall_rate < 0.045
#   - mean(c) shows real adaptation: within-env std > 0.05 AND
#     Pearson(c, perception_bias) > 0.2
#   - mean(φ) stays in [1.5, 3.0] (Kolathaya floor preserved)
#   - a-tracking coupling stays positive
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3tiltdr
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_layer3_push_a_c_tiltdr.sh \
#     2>&1 | tee logs/train_and_eval_wk3tiltdr.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

AUX_COEF=0.0

echo "================================================================"
echo "Wk3 reward fix + symmetric perception DR: LAYER3_PUSH_A_C_TILTDR"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
grep -q "CbfGo2EnvCfg_LAYER3_PUSH_A_C_TILTDR" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ CbfGo2EnvCfg_LAYER3_PUSH_A_C_TILTDR config present" \
  || { echo "  ✗ config missing — sync env_cfg.py"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0 task registered" \
  || { echo "  ✗ task not registered — sync __init__.py"; exit 1; }
grep -q "obstacle_radius_perception_error_range" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py \
  && echo "  ✓ symmetric perception DR support present in env" \
  || { echo "  ✗ env.py missing range field handling — sync"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] PPO TRAINING: ${CBF_ITERATIONS:-1500} iters, 4096 envs"
echo "      tilt -10, symmetric perception DR, 4 params released, a-L1 tax"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=$AUX_COEF
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0 \
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
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --use_locked \
  --output diagnose_phi_corr_wk3tiltdr.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_alpha_corr_wk3tiltdr.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_a_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_a_corr_wk3tiltdr.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_c_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --c_lo -0.20 --c_hi 0.20 \
  --output diagnose_c_corr_wk3tiltdr.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3tiltdr.json \
  --headless

# ---------- PHASE 3: HEADLINE EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] HEADLINE EVAL: B0 + B1 + B2 + BR vs in-dist"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT="logs/baseline_eval_wk3tiltdr_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0 \
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
echo "Eval CSV:             logs/baseline_eval_wk3tiltdr_indist/baseline.csv"
echo "================================================================"
