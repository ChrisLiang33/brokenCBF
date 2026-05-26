#!/bin/bash
# v2.9b = REWARD-2 retune. ONE knob change from v2.9:
#   base_contact_penalty: -50 → -100
#
# v2.9 evidence (in-dist BR):
#   - fall:    26.5% (v2.8) → 40.8% (v2.9)  — got WORSE
#   - stuck:   22.8% (v2.8) →  7.7% (v2.9)  — fixed (back to v2.6 level)
#   - combined: 49.3% → 48.6% (basically tied with v2.8)
#
# REWARD-2's stuck term (-2.0/step) worked — robot now moves. But
# base_contact_penalty -50 wasn't strong enough to keep the moving
# policy safe. Net trade: stuck for fall.
#
# `--no_obstacles` BR diagnostic on v2.9 ckpt: fall 8.3% = loco-
# internal floor under v2.8/v2.9 DR. So 80% of v2.9's 40.8% in-dist
# fall is CBF-attributable (the controllable lever). Reward shaping
# has substantial room.
#
# Initial worry about -100 (caution lock-in → stuck) is empirically
# refuted: v2.9 stuck is 7.7% with -50 plus stuck-term, way under
# v2.8's 22.8%. We have headroom on the stuck axis to add fall
# pressure. -100 matches collision -100 symmetrically (both are
# "robot-broken-terminal" events).
#
# v2.9b reward stack (only base_contact_penalty changes vs v2.9):
#   - base_contact_penalty -100  (was -50)
#   - stuck                -2.0  (unchanged)
#   - proximity            -0.5  (unchanged)
#   - collision           -100   (unchanged)
#   - infeasibility        -10   (unchanged)
#   - u_safe_deviation     -0.1  (unchanged)
#   - action_rate         -0.005 (unchanged)
#
# Structural config from v2.8/v2.9 unchanged: locked planner
# (PLANNER-2a), walk + adversarial dropped (PLANNER-2b), mild DR
# widening. 5K iters, 4096 envs.
#
# Predicted: fall ~15-20%, stuck ~7-10%, combined ~25-30%.
# If combined ≤ 0.30 → v2.9b BEATS v2.6's 30.6% in-dist baseline.
#
# Decision criteria:
#   - In-dist combined ≤ 0.31 → v2.9b = new working ckpt; move to Wk3.
#   - 0.31 < combined < 0.40 → partial recovery; iterate (likely
#     tune stuck or re-add u_safe_rate as REWARD-3).
#   - combined ≥ 0.40 → reward shaping exhausted; deeper issue.
#
# Usage on lab box:
#   ./scripts/train_and_eval_v29b.sh 2>&1 | tee logs/train_and_eval_v29b.log

set -e  # abort on any error

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v2.9b (REWARD-2 retune; base_contact -50 → -100) TRAIN + EVAL pipeline"
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
  OUT="logs/baseline_eval_v29b_${TAG}"

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
ls -la logs/baseline_eval_v29b_*/baseline.csv 2>/dev/null
echo "================================================================"
