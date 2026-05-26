#!/bin/bash
# PHIWIN_CURR (2026-05-18) — curriculum on σ_act to escape PPO's hedge basin.
#
# Sandbox proved analytical φ*(σ) is monotonic-adaptive regardless of
# collision cost. PPO just doesn't find this from random init — it
# converges to a hedge value. Curriculum: start σ_act narrow ([0, 0.03])
# and ramp to [0, 0.20] over the first 18000 policy steps (~50% of
# training).
#
# α frozen at 2.0 (isolate φ adaptation). a + c hyperparameter values.
# Post-QP adversarial noise from PHIWIN_v1.
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3curr
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_phiwin_curr.sh \
#     2>&1 | tee logs/train_and_eval_wk3curr.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-1500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "PHIWIN_CURR: σ_act curriculum + α frozen"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
grep -q "CbfGo2EnvCfg_LAYER3_PHIWIN_CURR" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ PHIWIN_CURR config present" \
  || { echo "  ✗ PHIWIN_CURR config missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-PhiWin-Curr-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ PHIWIN_CURR task registered" \
  || { echo "  ✗ PHIWIN_CURR task not registered"; exit 1; }
grep -q "actuation_noise_curriculum" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py \
  && echo "  ✓ env curriculum path wired" \
  || { echo "  ✗ env curriculum path missing"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] PPO TRAINING: ${ITERATIONS} iters, ${ENVS} envs"
echo "      σ_act curriculum 0.03 → 0.20 over first 18000 steps"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-Curr-v0 \
  --num_envs "${ENVS}" --max_iterations "${ITERATIONS}" \
  --headless

unset CBF_AUX_COEF

echo ""
echo "[1/3] PPO TRAINING done at $(date '+%H:%M:%S')"

LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1 "${LATEST_DIR}"/model_*.pt 2>/dev/null \
    | awk -F'model_|\\.pt' '{print $2, $0}' \
    | sort -n | awk '{print $2}' | tail -1)
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in ${LATEST_DIR}"
    exit 1
fi
echo "Using checkpoint: $CKPT"

# ---------- PHASE 2: DIAGNOSTICS ----------
echo ""
echo "================================================================"
echo "[2/3] DIAGNOSTICS at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-Curr-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3curr.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-Curr-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3curr.json --headless

# ---------- PHASE 3: EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] EVAL: training-dist + DEPLOY_REALISTIC"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT_INDIST="logs/baseline_eval_wk3curr_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-Curr-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT_INDIST" --headless

EVAL_OUT_DEPLOY="logs/baseline_eval_wk3curr_deploy"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT_DEPLOY" --headless

echo ""
echo "[3/3] EVAL done at $(date '+%H:%M:%S')"

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo ""
echo "Training-dist eval:    ${EVAL_OUT_INDIST}/baseline.csv"
echo "DEPLOY_REALISTIC eval: ${EVAL_OUT_DEPLOY}/baseline.csv"
echo ""
echo "Decision gates:"
echo "  Pearson(φ_output, σ_act) > 0.3 (the φ-adapts demonstration)"
echo "  φ population std > 0.3 (not pinned to a single value)"
echo "  training-dist joint_actual stays in same band as locked-best (no major regression)"
echo "================================================================"
