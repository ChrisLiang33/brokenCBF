#!/bin/bash
# v2.9 = REWARD-2 (reward-side fixes on top of v2.8 structural config).
#
# B-α' (revert PLANNER-2a) was rejected: mid-episode switching during
# training would fix stuck *artificially* via external disturbance
# kicking the policy out of the zero-velocity attractor; the policy
# never learns intrinsic recovery, so any improvement disappears at
# deployment with one stable nav stack. The legitimate fix is reward
# shaping that creates direct gradient pressure on the failure modes.
#
# v2.9 keeps v2.8's structural config (locked planner / PLANNER-2a,
# walk + adversarial dropped / PLANNER-2b, mild DR widening) and
# changes only the reward stack:
#
#   1. NEW base_contact_penalty (-50 terminal): closes a structural
#      gap. The collision -100 reward only fires on obstacle_contact
#      (0% in v2.8 evals). Falls (base_contact, 26.5% in v2.8) had no
#      terminal penalty — falling was value-positive vs continuing.
#      -50 (half of collision) closes the gap without dominating the
#      local gradient and locking policy into pure caution → stuck.
#   2. NEW stuck (-2.0 per step when ‖v_xy‖ < 0.15 m/s): direct
#      gradient pressure to escape the zero-velocity attractor.
#      Targets v2.8's 22.8% stuck rate.
#   3. CHANGE proximity weight (-1.0 → -0.5): episode-mean -0.195
#      was the dominant per-step term, creating over-cautious "stop
#      near obstacle" pressure. Halve dominance, keep gradient.
#
# A u_safe_rate term was considered and rejected: action_rate already
# handles controllable u_safe jerk (smooth CBF params); penalizing
# the remaining (geometric) jerk would conflict with the proximity
# reduction. Code is in place but unregistered; revisit as REWARD-3.
#
# Decision criteria:
#   - In-dist combined ≤ 0.31 (v2.6 paper level) → REWARD-2 closes the
#     gap; v2.9 = new working ckpt; move to Wk3.
#   - 0.31 < combined < 0.40 → partial recovery; iterate on REWARD-2
#     weights (likely tune stuck weight or proximity).
#   - combined ≥ 0.40 → reward shaping isn't enough; deeper issue;
#     reconsider full v2.8 retrospective.
#
# Usage on lab box:
#   ./scripts/train_and_eval_v29.sh 2>&1 | tee logs/train_and_eval_v29.log

set -e  # abort on any error

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v2.9 (REWARD-2) TRAIN + EVAL pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ---------- TRAINING ----------
echo ""
echo "================================================================"
echo "[1/2] TRAINING: 5000 iters, 4096 envs"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-v0 \
  --num_envs 4096 --max_iterations 5000 \
  --headless

echo ""
echo "[1/2] TRAINING done at $(date '+%H:%M:%S')"

# ---------- LOCATE CHECKPOINT ----------
echo ""
echo "================================================================"
echo "Locating most recent checkpoint..."
echo "================================================================"

# Most recently modified run dir
LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}  # strip trailing slash

# Most recently modified .pt file in that dir (likely model_4999.pt)
CKPT=$(ls -1t "${LATEST_DIR}"/model_*.pt 2>/dev/null | head -1)

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in ${LATEST_DIR}"
    exit 1
fi

echo "Using checkpoint: $CKPT"

# ---------- EVALS ----------
echo ""
echo "================================================================"
echo "[2/2] EVALUATION: 7-task sweep"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_TASKS=(
  "v0"
  "DensePack-v0"
  "Slippery-v0"
  "HighDisturbance-v0"
  "HeavyCOM-v0"
  "FastObstacles-v0"
  "RealisticCompound-v0"
)

for TASK in "${EVAL_TASKS[@]}"; do
  TAG="${TASK%-v0}"        # strip -v0 suffix
  [ "$TAG" = "v" ] && TAG="indist"  # rename "v0" -> "indist"
  OUT="logs/baseline_eval_v29_${TAG}"

  echo ""
  echo "  >>> Eval [$TASK] -> $OUT  (start $(date '+%H:%M:%S'))"

  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "Isaac-CBF-Go2-$TASK" \
    --num_envs 64 --steps_per_config 2000 \
    --modes B0,B1,B2,BR \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --headless

  echo "  >>> Eval [$TASK] done at $(date '+%H:%M:%S')"
done

# ---------- SUMMARY ----------
echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "CSVs:"
ls -la logs/baseline_eval_v29_*/baseline.csv 2>/dev/null
echo "================================================================"
