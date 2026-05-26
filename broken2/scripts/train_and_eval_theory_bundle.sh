#!/bin/bash
# Theory-bundle training + eval (2026-05-17).
#
# Three coordinated changes from locked-best LAYER3_PUSH_A_C, all targeting
# the training/theory misalignment uncovered by the per-param sandbox:
#   1. Drop L1 tax on `a` (was suppressing a parameter that doesn't need it)
#   2. Drop adversarial planner from training mix (φ should stop hedging high)
#   3. Add δ_R to priv obs (mean_signed + max_abs) + symmetric perception DR
#      (gives c the analytical signal AND bidirectional training pressure)
#
# Eval on:
#   - Training distribution (regression check vs locked-best 0.724)
#   - DEPLOY_REALISTIC (paper-style headline)
#
# Known distillation gap: δ_R in priv obs doesn't transfer to deployed
# student. This iteration is a theory-validation, not a deployable.
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3theory
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_theory_bundle.sh \
#     2>&1 | tee logs/train_and_eval_wk3theory.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-1500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "Theory-bundle training + eval"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
grep -q "CbfGo2EnvCfg_LAYER3_THEORY_BUNDLE" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ THEORY_BUNDLE config present" \
  || { echo "  ✗ THEORY_BUNDLE config missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ THEORY_BUNDLE task registered" \
  || { echo "  ✗ THEORY_BUNDLE task not registered"; exit 1; }
grep -q "_PRIV_DIM = 33" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py \
  && echo "  ✓ _PRIV_DIM bumped to 33" \
  || { echo "  ✗ _PRIV_DIM not bumped to 33"; exit 1; }
grep -q "def mean_signed_delta_R" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_observations.py \
  && echo "  ✓ mean_signed_delta_R priv obs present" \
  || { echo "  ✗ mean_signed_delta_R priv obs missing"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] PPO TRAINING: ${ITERATIONS} iters, ${ENVS} envs"
echo "      Theory bundle on LAYER3_PUSH_A_C base"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
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
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3theory.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_alpha_corr_wk3theory.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_a_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_a_corr_wk3theory.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_c_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --c_lo -0.20 --c_hi 0.20 \
  --output diagnose_c_corr_wk3theory.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3theory.json --headless

# ---------- PHASE 3: HEADLINE EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] EVAL: training-dist + DEPLOY_REALISTIC"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT_INDIST="logs/baseline_eval_wk3theory_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,4.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT_INDIST" --headless

EVAL_OUT_DEPLOY="logs/baseline_eval_wk3theory_deploy"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,4.0" \
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
echo "Checkpoint:        $CKPT"
echo ""
echo "Training-dist eval:    ${EVAL_OUT_INDIST}/baseline.csv"
echo "DEPLOY_REALISTIC eval: ${EVAL_OUT_DEPLOY}/baseline.csv"
echo ""
echo "Decision gates:"
echo "  Training-dist joint_actual >= 0.724 (locked-best ceiling)"
echo "  mean(c)–mean_signed_delta_R Pearson > 0.5 (c actually adapts)"
echo "  φ population mean drops from ~3.6 toward ~1 (adversarial gone)"
echo "  mean(a) stays near 0 (sandbox prediction)"
echo "  Encoder R²(base_height, σ_act, mean_signed_delta_R) all > 0.5"
echo "================================================================"
