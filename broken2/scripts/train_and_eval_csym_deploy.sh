#!/bin/bash
# Symmetric c-DR fix + DEPLOY_REALISTIC headline eval (2026-05-17).
#
# Single-variable change from locked-best LAYER3_PUSH_A_C: flip the radius
# perception DR symmetric so c has bidirectional adaptation pressure.
#
# Then eval the new teacher on DEPLOY_REALISTIC vs the three fixed-CBF
# baselines (α-only, α+φ, α+φ+ε₀+λ). The headline paper claim is that
# joint_actual_BR > all three.
#
# Wall-time: ~1h25m train + ~30 min diagnostics + ~10 min eval ≈ 2h.
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3csym
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_csym_deploy.sh \
#     2>&1 | tee logs/train_and_eval_wk3csym.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-1500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "Symmetric c-DR fix + DEPLOY_REALISTIC headline eval"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
grep -q "CbfGo2EnvCfg_LAYER3_PUSH_A_C_CSYM" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ LAYER3_PUSH_A_C_CSYM config present" \
  || { echo "  ✗ CSYM config missing"; exit 1; }
grep -q "CbfGo2EnvCfg_DEPLOY_REALISTIC" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ DEPLOY_REALISTIC config present" \
  || { echo "  ✗ DEPLOY_REALISTIC config missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-Push-A-C-CSym-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ CSYM task registered" \
  || { echo "  ✗ CSYM task not registered"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Deploy-Realistic-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ DEPLOY_REALISTIC task registered" \
  || { echo "  ✗ DEPLOY_REALISTIC task not registered"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] PPO TRAINING: ${ITERATIONS} iters, ${ENVS} envs"
echo "      Symmetric c-DR (−0.10, +0.10) on LAYER3_PUSH_A_C base"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-CSym-v0 \
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
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-CSym-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 --use_locked \
  --output diagnose_phi_corr_wk3csym.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-CSym-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_alpha_corr_wk3csym.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_a_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-CSym-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_a_corr_wk3csym.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_c_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-CSym-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 --c_lo -0.20 --c_hi 0.20 \
  --output diagnose_c_corr_wk3csym.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-CSym-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3csym.json --headless

# ---------- PHASE 3: HEADLINE EVAL ON DEPLOY_REALISTIC ----------
echo ""
echo "================================================================"
echo "[3/3] HEADLINE EVAL: DEPLOY_REALISTIC vs B0/B1/B2/BR"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT="logs/baseline_eval_wk3csym_deploy"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,4.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT" --headless

# Bonus: also eval on the training-distribution (matches archive eval setup).
EVAL_OUT_INDIST="logs/baseline_eval_wk3csym_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-CSym-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,4.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT_INDIST" --headless

echo ""
echo "[3/3] HEADLINE EVAL done at $(date '+%H:%M:%S')"

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint:           $CKPT"
echo ""
echo "DEPLOY_REALISTIC eval:    ${EVAL_OUT}/baseline.csv"
echo "Training-dist eval:       ${EVAL_OUT_INDIST}/baseline.csv"
echo ""
echo "Decision gates:"
echo "  Headline (DEPLOY_REALISTIC): joint_actual_BR > best of {B0, B1, B2}"
echo "  c-fix worked:                mean(c) ≈ 0 with bidirectional Pearson coupling"
echo "  Adaptive locked-best regression: training-dist joint_actual ≥ 0.724 (don't lose)"
echo "================================================================"
