#!/bin/bash
# Omniscient-perception teacher retrain (2026-05-17).
#
# Archive re-eval (joint_actual collision split) locked LAYER3_PUSH_A_C
# as the best so far: joint_actual = 0.724, beats best fixed baseline by
# +1.6 pp. The 4-param adaptive teacher actually wins in actual safety
# terms. The PHITAX/TILTDR/ACLAMP iterations regressed it.
#
# This iteration measures the policy ceiling by removing the perception
# bottleneck entirely. Truth obstacle positions for both QP h(x) and
# policy observation grid. With perceived ≡ true, no over-protection
# confound, the CBF math operates as analytical bounds prescribe.
#
# Single-variable change from LAYER3_PUSH_A_C:
#   - perception_mode = "priv"
#   - observations = CbfObservationsCfg() (truth grid)
#   - obstacle_tracker_enabled = False
# All other settings (push, a + L1 tax, c released, priv obs 31-D,
# planner mix, corridor scenes, reward stack) inherited.
#
# Decision gates:
#   - joint_actual > 0.724 (LAYER3_PUSH_A_C ceiling)
#   - Ideally > 0.85 (significant headroom showing perception is the
#     dominant residual bottleneck)
#   - All four CBF parameters still genuinely adaptive
#   - Encoder R²(COM, σ_actuation) preserved
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3omni
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_layer3_push_a_c_omni.sh \
#     2>&1 | tee logs/train_and_eval_wk3omni.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

AUX_COEF=0.0

echo "================================================================"
echo "Omniscient-perception teacher: LAYER3_PUSH_A_C_OMNI"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
grep -q "CbfGo2EnvCfg_LAYER3_PUSH_A_C_OMNI" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ CbfGo2EnvCfg_LAYER3_PUSH_A_C_OMNI config present" \
  || { echo "  ✗ config missing — sync env_cfg.py"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-Push-A-C-Omni-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ task registered" \
  || { echo "  ✗ task not registered — sync __init__.py"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] PPO TRAINING: ${CBF_ITERATIONS:-1500} iters, 4096 envs"
echo "      4 params released; truth-obstacle QP + truth grid"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=$AUX_COEF
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-Omni-v0 \
  --num_envs 4096 --max_iterations ${CBF_ITERATIONS:-1500} \
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
echo "[2/3] DIAGNOSTICS"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-Omni-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 --use_locked \
  --output diagnose_phi_corr_wk3omni.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-Omni-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_alpha_corr_wk3omni.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_a_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-Omni-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_a_corr_wk3omni.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_c_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-Omni-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 --c_lo -0.20 --c_hi 0.20 \
  --output diagnose_c_corr_wk3omni.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-Omni-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3omni.json --headless

# ---------- PHASE 3: HEADLINE EVAL (with collision split) ----------
echo ""
echo "================================================================"
echo "[3/3] HEADLINE EVAL: B0 + B1 + B2 + BR (collision split)"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT="logs/baseline_eval_wk3omni_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-Omni-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,4.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT" --headless

unset CBF_AUX_COEF
echo ""
echo "[3/3] HEADLINE EVAL done at $(date '+%H:%M:%S')"

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint:           $CKPT"
echo "Eval CSV:             logs/baseline_eval_wk3omni_indist/baseline.csv"
echo ""
echo "Compare:"
echo "  Locked best (PUSH_A_C):  joint_actual = 0.724  (+1.6 pp vs best fixed)"
echo "  Target (gate):           joint_actual > 0.724"
echo "  Stretch:                 joint_actual > 0.85"
echo "================================================================"
