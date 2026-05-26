#!/bin/bash
# Wk3 α-basin clamp (2026-05-16).
#
# Motivation: Bf-α=0.5 diagnostic on TILTDR ckpt showed clamping BR's α
# to 0.5 recovers +10 pp of joint success (0.150 → 0.254). PPO can't
# reach the low-α basin from the high-α one under the current reward
# stack. Analytical solve confirmed α* spans ~1.0 (grippy) to ~3.0
# (slippery) across the friction DR — the convergence target lives in
# this range, not at 4.25.
#
# Single-variable change from PUSH_A_C:
#   - alpha_param_range = (0.5, 3.0)
# (a + L1 tax, c released, push event, priv obs at 31-D all unchanged.
# TILTDR's tilt-bump + symmetric perception DR NOT inherited.)
#
# Decision gates:
#   - joint_success > 0.336 (best fixed baseline)
#   - mean(α) lives in [0.5, 3.0], NOT pegged at 3.0
#   - Pearson(α, friction) meaningfully negative (|r| > 0.2)
#   - Pearson(a, |tracking_err|) positive (a tax still works)
#   - mean(c) varies with perception bias direction
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3aclamp
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_layer3_push_a_c_aclamp.sh \
#     2>&1 | tee logs/train_and_eval_wk3aclamp.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

AUX_COEF=0.0

echo "================================================================"
echo "Wk3 α-basin clamp: LAYER3_PUSH_A_C_ACLAMP"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
grep -q "CbfGo2EnvCfg_LAYER3_PUSH_A_C_ACLAMP" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ CbfGo2EnvCfg_LAYER3_PUSH_A_C_ACLAMP config present" \
  || { echo "  ✗ config missing — sync env_cfg.py"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-Push-A-C-AClamp-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ Isaac-CBF-Go2-RMA-Layer3-Push-A-C-AClamp-v0 task registered" \
  || { echo "  ✗ task not registered — sync __init__.py"; exit 1; }
grep -q "alpha_param_range" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py \
  && echo "  ✓ alpha_param_range field support present in env" \
  || { echo "  ✗ env.py missing alpha range field handling — sync"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] PPO TRAINING: ${CBF_ITERATIONS:-1500} iters, 4096 envs"
echo "      α clamped to [0.5, 3.0]; 4 params released; a-L1 tax"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=$AUX_COEF
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-AClamp-v0 \
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
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-AClamp-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --use_locked \
  --output diagnose_phi_corr_wk3aclamp.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-AClamp-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_alpha_corr_wk3aclamp.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_a_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-AClamp-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_a_corr_wk3aclamp.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_c_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-AClamp-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --c_lo -0.20 --c_hi 0.20 \
  --output diagnose_c_corr_wk3aclamp.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-AClamp-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3aclamp.json \
  --headless

# ---------- PHASE 3: HEADLINE EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] HEADLINE EVAL: B0 + B1 + B2 + BR vs in-dist"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT="logs/baseline_eval_wk3aclamp_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-AClamp-v0 \
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
echo "Eval CSV:             logs/baseline_eval_wk3aclamp_indist/baseline.csv"
echo "================================================================"
