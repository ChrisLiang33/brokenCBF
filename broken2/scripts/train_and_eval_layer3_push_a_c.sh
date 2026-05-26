#!/bin/bash
# Wk3 within-episode push + `a` + `c` release (2026-05-16).
#
# Motivation: LAYER3_PUSH_A halved the joint-success gap (0.28 → 0.31,
# -12pp → -5pp vs best fixed baseline). The a-correlation diagnostic
# confirmed Pearson(a, |tracking_err|) within-episode = +0.259 (positive
# sign, above the +0.15 gate). `a` is doing exactly what we hypothesized.
#
# Remaining gap: α still pegged at 4.27 with strong wrong-direction
# coupling under push (Pearson(α, |tracking_err|) = -0.571). shield_v0c's
# per-cluster radius fit adds SHIELD_R_SAFETY_MARGIN = 0.10m to every
# cylinder, so every perceived h is biased low by ≥ 0.10m. With c frozen
# at 0, the policy can't undo this static perception bias; it has to keep
# α aggressive during calm walking to compensate.
#
# This iteration:
#   - Release c ∈ [-0.20, +0.20] (symmetric, so policy can shift the
#     perceived h-boundary EITHER way; expected: mean(c) < 0)
#
# Held constant from LAYER3_PUSH_A:
#   - Push event ±1.0 m/s every 5-10 s
#   - Symptom-based + God-mode priv obs (_PRIV_DIM = 31)
#   - Perception (shield_v0c, n_rays=128, persist=50)
#   - Reward stack including cbf_a_l1 = -0.01
#   - a ∈ [0, 3] released
#
# Decision gates:
#   - joint_success > 0.36 (current best fixed baseline)
#   - mean(c) < 0 (compensating the radius over-estimate)
#   - α population mean drops from 4.27 toward 2-3 range
#   - α-tracking coupling magnitude drops below 0.55
#   - a-tracking coupling stays positive
#   - Encoder R²(COM, σ) preserved
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3pushac
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_layer3_push_a_c.sh \
#     2>&1 | tee logs/train_and_eval_wk3pushac.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

AUX_COEF=0.0

echo "================================================================"
echo "Wk3 within-episode push + a + c release: LAYER3_PUSH_A_C"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
grep -q "CbfGo2EnvCfg_LAYER3_PUSH_A_C" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ CbfGo2EnvCfg_LAYER3_PUSH_A_C config present" \
  || { echo "  ✗ LAYER3_PUSH_A_C config missing — sync env_cfg.py"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0 task registered" \
  || { echo "  ✗ task not registered — sync __init__.py"; exit 1; }
grep -q "^_PRIV_DIM = 31" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py \
  && echo "  ✓ _PRIV_DIM still 31" \
  || { echo "  ✗ _PRIV_DIM mismatch — sync cbf_go2_teacher_rma.py"; exit 1; }
grep -q "self.cbf_c_param = c_param" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py \
  && echo "  ✓ cbf_c_param cached for diagnostic" \
  || { echo "  ✗ cbf_c_param not cached — sync cbf_go2_env.py"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] PPO TRAINING: ${CBF_ITERATIONS:-1500} iters, 4096 envs"
echo "      a + c released, L1 tax on a, push event, symptom priv"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=$AUX_COEF
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0 \
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
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --use_locked \
  --output diagnose_phi_corr_wk3pushac.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_alpha_corr_wk3pushac.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_a_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_a_corr_wk3pushac.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_c_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --c_lo -0.20 --c_hi 0.20 \
  --output diagnose_c_corr_wk3pushac.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3pushac.json \
  --headless

# ---------- PHASE 3: HEADLINE EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] HEADLINE EVAL: B0 + B1 + B2 + BR vs in-dist"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT="logs/baseline_eval_wk3pushac_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0 \
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
echo "Diagnostics:"
echo "  φ-corr:             diagnose_phi_corr_wk3pushac.json"
echo "  α-corr:             diagnose_alpha_corr_wk3pushac.json"
echo "  a-corr:             diagnose_a_corr_wk3pushac.json"
echo "  c-corr:             diagnose_c_corr_wk3pushac.json"
echo "  linear probe:       probe_z_linear_wk3pushac.json"
echo "Eval CSV:             logs/baseline_eval_wk3pushac_indist/baseline.csv"
echo "================================================================"
