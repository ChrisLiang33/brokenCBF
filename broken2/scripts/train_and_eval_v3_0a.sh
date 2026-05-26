#!/bin/bash
# v3.0 Layer 1 — α-only adaptation test.
#
# Why this exists:
#   v2.15 + diagnostics revealed the trained policy varies its outputs
#   across envs but with the SAME magnitude regardless of regime
#   (cbf_alpha_std ≈ 1.78 on calm in-dist AND on perturbed OOD). The
#   variation is calibrated for "training-time average noise," not
#   responsive to current state. On in-dist: harmful; on OOD: accidentally
#   useful. Net: loses to fixed baselines.
#
#   New approach: strip down to one parameter (α) and prove it can
#   adapt usefully before adding any complexity. Build up layer by
#   layer. Each layer adds one CBF math term + its target DR axis
#   only after the prior layer is verified.
#
# Layer 1 (this script):
#   CBF constraint: L_g h · u ≥ -α · h
#     - φ-term gone (FREEZE_PHI_VALUE = 0.0)
#     - a-term gone (FREEZE_A_VALUE = 0.0)
#     - c-shift gone (FREEZE_C_VALUE = 0.0; h - c = h)
#   α adaptive over [0.1, 5.0] (ALPHA_MIN relaxed from 1.0).
#   Slot-specific DR axes auto-disabled by FREEZE coupling.
#
# Kept (locomotion-relevant DR — α-relevant):
#   - friction (0.30, 1.20) static / (0.20, 1.00) dynamic
#   - COM offset ±5cm xy / ±3cm z
#   - base force/torque ±10N / ±2Nm
#   - obstacle motion max_speed_range (0, 0.4)
#
# Reward (v2.15 REWARD-3 unchanged):
#   collision -100, base_contact_penalty -500, stuck -1.0,
#   proximity -0.5, u_safe_deviation -0.1, action_rate -0.005,
#   infeasibility -10
#
# Comparison modes (after training):
#   - BR (adaptive α — v3.0a teacher)
#   - B0 sweep across 7 α values: {0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0}
#     — this replaces the redundant Bf-α ablation, since with φ/a/c
#     all forced to 0, Bf-α at target=X is identical to B0 at α=X.
#
# Tasks:
#   - in-dist (Isaac-CBF-Go2-v0)
#   - HeavyCOM (closest existing locomotion-stress edge — α-relevant)
#
# Slot-irrelevant OOD tasks SKIPPED (NoisyPerception / RadiusError /
# HighActuationNoise / etc. — those exercise φ/a/c slots that don't
# exist in this layer).
#
# Decision criterion (locked):
#   PASS: BR combined beats best-of-B0-sweep by ≥3pp on in-dist OR HeavyCOM
#   FAIL: BR ≤ best-of-B0-sweep on both tasks
#
# If PASS → proceed to Layer 2 (add φ + actuation-noise DR).
# If FAIL → α-only adaptation isn't useful even isolated; investigate
#   before any more training (obs encoder, PPO hyperparams, network
#   capacity, policy architecture).
#
# Time budget:
#   Training: ~3h (3K iters, 4096 envs — fewer iters than v2.15 since
#                  only 1 param to learn)
#   Eval: ~15-25 min sequential (8 configs × 2 tasks)
#   Total: ~3.5h
#
# Usage on lab box:
#   ~/Desktop/safety-go2/scripts/train_and_eval_v3_0a.sh 2>&1 | tee logs/train_and_eval_v3_0a.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v3.0 Layer 1 (α-only adaptation test) pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ---------- TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] TRAINING: 3000 iters, 4096 envs"
echo "      α adaptive in [0.1, 5.0]; φ=a=c=0; slot DRs auto-disabled"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-v0 \
  --num_envs 4096 --max_iterations 3000 \
  --headless

echo ""
echo "[1/3] TRAINING done at $(date '+%H:%M:%S')"

# ---------- LOCATE CHECKPOINT ----------
echo ""
echo "================================================================"
echo "Locating most recent checkpoint..."
echo "================================================================"

LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1t "${LATEST_DIR}"/model_*.pt 2>/dev/null | head -1)

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in ${LATEST_DIR}"
    exit 1
fi

echo "Using checkpoint: $CKPT"

# ---------- HEADLINE 2-EVAL (sequential) ----------
echo ""
echo "================================================================"
echo "[2/2] HEADLINE EVAL: 2 tasks (in-dist + HeavyCOM)"
echo "      Modes: B0 (sweep 7 α values) + BR"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_TASKS=("v0" "HeavyCOM-v0")

task_tag() {
  if [ "$1" = "v0" ]; then echo "indist"; else echo "${1%-v0}"; fi
}

for TASK in "${EVAL_TASKS[@]}"; do
  TAG=$(task_tag "$TASK")
  OUT="logs/baseline_eval_v3_0a_${TAG}"
  echo ""
  echo "  >>> [$TASK] -> $OUT  ($(date '+%H:%M:%S'))"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "Isaac-CBF-Go2-${TASK}" \
    --num_envs 64 --steps_per_config 1500 \
    --modes B0,BR \
    --alpha_grid "0.1,0.5,1.0,2.0,3.0,4.0,5.0" \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --headless > "${OUT}.stdout.log" 2>&1
  echo "  >>> [$TASK] done at $(date '+%H:%M:%S')"
done

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "Eval CSVs in: logs/baseline_eval_v3_0a_{indist,HeavyCOM}/"
echo ""
echo "Decision criterion:"
echo "  PASS: BR combined beats best-of-B0-sweep by ≥3pp on ≥1 task"
echo "  FAIL: BR ≤ best-of-B0-sweep on both tasks"
echo "================================================================"
