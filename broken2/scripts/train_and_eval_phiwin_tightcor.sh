#!/bin/bash
# PHIWIN_TIGHTCOR (2026-05-18) — tight corridors force φ adaptation.
#
# After PHIWIN_CURR (σ_act curriculum) confirmed Pearson(φ, σ_act) ≈ 0,
# Ryan's reframe is the operating thesis: PPO won't learn φ adaptation
# because in our current sim, low φ is never strictly better than high φ.
# Tight corridors create the missing scenario where high φ literally
# fails the goal (CBF's φ-bubble exceeds the physical corridor width).
#
# All PHIWIN_CURR isolation preserved: α frozen at 2.0, a + c
# hyperparameters, post-QP adversarial noise, σ_act curriculum.
#
# Tight-corridor changes:
#   scene_corridor_prob: 0.30 → 0.50
#   corridor_y_center:   0.55 → 0.48
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3tight
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_phiwin_tightcor.sh \
#     2>&1 | tee logs/train_and_eval_wk3tight.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-1500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "PHIWIN_TIGHTCOR: σ_act curriculum + α frozen + TIGHT CORRIDORS"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
grep -q "CbfGo2EnvCfg_LAYER3_PHIWIN_TIGHTCOR" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ PHIWIN_TIGHTCOR config present" \
  || { echo "  ✗ PHIWIN_TIGHTCOR config missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ PHIWIN_TIGHTCOR task registered" \
  || { echo "  ✗ PHIWIN_TIGHTCOR task not registered"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] PPO TRAINING: ${ITERATIONS} iters, ${ENVS} envs"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-v0 \
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
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3tight.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3tight.json --headless

# ---------- PHASE 3: EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] EVAL: training-dist + DEPLOY_REALISTIC"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT_INDIST="logs/baseline_eval_wk3tight_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT_INDIST" --headless

EVAL_OUT_DEPLOY="logs/baseline_eval_wk3tight_deploy"
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
echo "  Pearson(φ_output, σ_act) > 0.3   — primary metric"
echo "  φ within-env std > 0.5            — varies meaningfully across states"
echo "  Corridor scene completion > 0.5   — policy can navigate tight corridors"
echo "================================================================"
