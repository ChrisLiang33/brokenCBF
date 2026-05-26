#!/bin/bash
# PHIWIN_v1 — post-QP noise with adversarial component (2026-05-17).
#
# AP_ADAPT failed because the legacy noise model (u_des noise BEFORE QP)
# gives φ no real safety job — the QP's safety guarantee holds on the
# noisy command. Sandbox proved this; PHIWIN_v1 fixes it structurally:
#
#   1. Apply noise AFTER the QP, directly to u_safe (bypasses safety guarantee)
#   2. Adversarial component aligned with −∇h (-cached gradient from QP)
#   3. σ_act_max rolled back to 0.10 (was 0.30, broke locomotion)
#
# All AP_ADAPT freezes preserved: freeze_a=0.05, c_clamped=-0.05,
# α range [0.5, 3.0], cbf_a_l1 weight 0.
#
# Success criterion: BR joint_actual > ALL three fixed baselines on
# DEPLOY_REALISTIC + φ–σ_act correlation > 0.2.
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3phiwin
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_phiwin_v1.sh \
#     2>&1 | tee logs/train_and_eval_wk3phiwin.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-1500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "PHIWIN_v1: post-QP noise with adversarial component"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
grep -q "CbfGo2EnvCfg_LAYER3_PHIWIN_V1" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ PHIWIN_V1 config present" \
  || { echo "  ✗ PHIWIN_V1 config missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-PhiWin-v1" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ PHIWIN_v1 task registered" \
  || { echo "  ✗ PHIWIN_v1 task not registered"; exit 1; }
grep -q "noise_after_qp" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py \
  && echo "  ✓ env post-QP noise path wired" \
  || { echo "  ✗ env post-QP noise path missing"; exit 1; }

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
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-v1 \
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
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-v1 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3phiwin.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-v1 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_alpha_corr_wk3phiwin.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-v1 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3phiwin.json --headless

# ---------- PHASE 3: EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] EVAL: training-dist + DEPLOY_REALISTIC"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT_INDIST="logs/baseline_eval_wk3phiwin_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-v1 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT_INDIST" --headless

EVAL_OUT_DEPLOY="logs/baseline_eval_wk3phiwin_deploy"
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
echo "  1. BR joint_actual > all three fixed baselines on DEPLOY_REALISTIC"
echo "  2. φ–σ_act correlation > 0.2 (the central test for φ adaptation)"
echo "  3. α–base_height correlation magnitude > 0.4 (preserved)"
echo "  4. Training-dist joint_actual >= 0.70 (no locomotion-break regression)"
echo "================================================================"
