#!/bin/bash
# Two-adaptive-param teacher (Wk3, 2026-05-17).
#
# α and φ adaptive, a and c fixed as hyperparameters. After sandbox theory
# checks revealed only α and φ have meaningful adaptive cases in our sim.
#
# Five changes from locked-best LAYER3_PUSH_A_C:
#   1. freeze_a_value = 0.05               (small safety cushion)
#   2. c_param_range  = (-0.05, -0.05)     (analytical mean optimum, DR active)
#   3. alpha_param_range = (0.5, 3.0)      (clamp from 5.0, sandbox showed > 3 is over-aggressive)
#   4. actuation_noise_sigma_max = 0.30    (3×, gives φ bidirectional pressure)
#   5. rewards.cbf_a_l1.weight = 0.0       (a frozen, no point)
#
# Success criterion: BR joint_actual > all three fixed baselines on
# DEPLOY_REALISTIC. Deployable-teacher gate.
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3ap
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_ap_adapt.sh \
#     2>&1 | tee logs/train_and_eval_wk3ap.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-1500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "Two-adaptive-param teacher (α + φ adaptive, a + c fixed)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
grep -q "CbfGo2EnvCfg_LAYER3_AP_ADAPT" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ AP_ADAPT config present" \
  || { echo "  ✗ AP_ADAPT config missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-AP-Adapt-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ AP_ADAPT task registered" \
  || { echo "  ✗ AP_ADAPT task not registered"; exit 1; }

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
  --task Isaac-CBF-Go2-RMA-Layer3-AP-Adapt-v0 \
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
  --task Isaac-CBF-Go2-RMA-Layer3-AP-Adapt-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3ap.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-AP-Adapt-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_alpha_corr_wk3ap.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-AP-Adapt-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3ap.json --headless

# ---------- PHASE 3: EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] EVAL: training-dist + DEPLOY_REALISTIC"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT_INDIST="logs/baseline_eval_wk3ap_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-AP-Adapt-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT_INDIST" --headless

EVAL_OUT_DEPLOY="logs/baseline_eval_wk3ap_deploy"
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
echo "Checkpoint:        $CKPT"
echo ""
echo "Training-dist eval:    ${EVAL_OUT_INDIST}/baseline.csv"
echo "DEPLOY_REALISTIC eval: ${EVAL_OUT_DEPLOY}/baseline.csv"
echo ""
echo "Decision gates:"
echo "  1. BR joint_actual > ALL three fixed baselines on DEPLOY_REALISTIC (deployable gate)"
echo "  2. α–base_height correlation magnitude > 0.4 (α still adapts)"
echo "  3. φ–σ_act correlation > 0.2 (φ NOW adapts — new demonstration)"
echo "  4. Training-dist joint_actual >= 0.70 (no major regression)"
echo "================================================================"
