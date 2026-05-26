#!/bin/bash
# Wk3 within-episode push + `a` release (2026-05-16).
#
# Motivation: LAYER3_PUSH (the prior iteration) showed wrong-knob
# adaptation. Pearson(α, |tracking_err|) = -0.545 (strong but wrong sign):
# under push, policy dialed α down → harder QP projection → u_safe far
# from u_des → locomotion can't track → fall. `a` (additive slack) was
# frozen at 0, so the policy had no shock-absorber knob.
#
# This iteration:
#   1. Release `a ∈ [0, 3]` (freeze_a_value = None)
#   2. Add L1 regularizer -0.01 · |a| on the reward stack so PPO doesn't
#      peg `a` at the upper cap as a free constraint-relaxation.
#
# Held constant from LAYER3_PUSH:
#   - Push event ±1.0 m/s every 5-10 s
#   - Symptom-based + God-mode priv obs (_PRIV_DIM = 31)
#   - Perception (shield_v0c, n_rays=128, persist=50)
#   - Reward stack, planner mix, corridor scenes
#   - `c` still frozen at 0
#
# Decision gates:
#   - joint_success = (no col) ∧ (no fall) ∧ (goal) on Layer3-Push-A eval
#     beats best fixed-(α, φ) baseline
#   - mean(`a`) bounded, NOT pegged at 0 or 3
#   - mean(`a`) during calm walking (no push in last 2 s) < 0.5
#   - Pearson(`a`, |tracking_err|) > +0.15 (POSITIVE sign — more push →
#     more slack)
#   - α-tracking coupling magnitude drops below 0.55
#   - Encoder R²(COM, σ) preserved
#
# If `a` still pegs near 3.0: bump L1 weight from -0.01 → -0.05 next.
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3pusha
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_layer3_push_a.sh \
#     2>&1 | tee logs/train_and_eval_wk3pusha.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

AUX_COEF=0.0

echo "================================================================"
echo "Wk3 within-episode push + `a` release: LAYER3_PUSH_A"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "Pre-flight checks"
test -f source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py \
  && echo "  ✓ cbf_go2_teacher_rma.py present" \
  || { echo "  ✗ cbf_go2_teacher_rma.py missing — sync first"; exit 1; }
grep -q "CbfGo2EnvCfg_LAYER3_PUSH_A" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ CbfGo2EnvCfg_LAYER3_PUSH_A config present" \
  || { echo "  ✗ LAYER3_PUSH_A config missing — sync env_cfg.py"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-Push-A-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ Isaac-CBF-Go2-RMA-Layer3-Push-A-v0 task registered" \
  || { echo "  ✗ task not registered — sync __init__.py"; exit 1; }
grep -q "^_PRIV_DIM = 31" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py \
  && echo "  ✓ _PRIV_DIM still 31" \
  || { echo "  ✗ _PRIV_DIM mismatch — sync cbf_go2_teacher_rma.py"; exit 1; }
grep -q "cbf_a_l1_penalty" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_rewards.py \
  && echo "  ✓ cbf_a_l1_penalty reward function present" \
  || { echo "  ✗ cbf_a_l1_penalty missing — sync cbf_go2_rewards.py"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] PPO TRAINING: ${CBF_ITERATIONS:-1500} iters, 4096 envs"
echo "      RMA + push event + symptom priv (31-D) + a released + L1 tax"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=$AUX_COEF
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-v0 \
  --num_envs 4096 --max_iterations ${CBF_ITERATIONS:-1500} \
  --headless

unset CBF_AUX_COEF

echo ""
echo "[1/3] PPO TRAINING done at $(date '+%H:%M:%S')"

# ---------- LOCATE CHECKPOINT ----------
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
echo "[2/3] φ-CORR + α-CORR + PROBE DIAGNOSTICS"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --use_locked \
  --output diagnose_phi_corr_wk3pusha.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_alpha_corr_wk3pusha.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3pusha.json \
  --headless

# ---------- PHASE 3: HEADLINE EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] HEADLINE EVAL: B0 + B1 + B2 + BR vs LAYER3_PUSH_A in-dist"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT="logs/baseline_eval_wk3pusha_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-A-v0 \
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
echo "α-corr JSON:          diagnose_alpha_corr_wk3pusha.json"
echo "φ-corr JSON:          diagnose_phi_corr_wk3pusha.json"
echo "Linear probe JSON:    probe_z_linear_wk3pusha.json"
echo "Eval CSV:             logs/baseline_eval_wk3pusha_indist/baseline.csv"
echo "================================================================"
