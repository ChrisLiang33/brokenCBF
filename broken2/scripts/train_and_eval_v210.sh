#!/bin/bash
# v2.10 = v2.6 DR + v2.8 planner regime + v2.9b reward stack.
#
# Why this exists:
#   v2.9b finished with in-dist combined 0.479 (vs v2.6's 0.306).
#   The REWARD-2 retune (-100 base_contact + -2.0 stuck + -0.5 proximity)
#   worked partially — fall dropped 40.8% → 35.0% — but stuck rose
#   12.9% and combined barely moved. Pattern across all 7 evals: BR
#   absolute combined values uniformly higher than v2.6's, even when
#   margins (vs best B) are decent. Diagnosis: WIDER TRAINING DR
#   (v2.8 inheritance) is making the env harder for everyone, but the
#   policy can't converge as cleanly at 5K iters as v2.6 did on its
#   narrow DR.
#
# v2.10 reverts ONLY the training DR (and matching OOD eval ranges)
# back to v2.6 levels. Keeps everything else from v2.9b:
#
#   Training DR (REVERTED to v2.6):
#     - obstacle max_speed:  0.5 → 0.2 m/s
#     - friction static:    (0.20, 1.30) → (0.30, 1.20)
#     - friction dynamic:   (0.15, 1.10) → (0.20, 1.00)
#     - force:              ±15N → ±10N
#     - torque:             ±3Nm → ±2Nm
#     - COM:                ±5cm/±3cm (unchanged — already v2.6)
#
#   Planner (KEPT from v2.8/v2.9):
#     - PLANNER-2a: locked planner per episode (resampling 100,100)
#     - PLANNER-2b: walk + adversarial dropped (smooth_goal 0.45 /
#       waypoint 0.30 / mpc 0.20 / legacy_goal 0.05)
#
#   Reward (KEPT from v2.9b):
#     - base_contact_penalty -100 (terminal on fall)
#     - stuck -2.0/step when ‖v_xy‖<0.15 m/s
#     - proximity -0.5 (halved from v2.6's -1.0)
#     - collision -100, infeasibility -10, u_safe_dev -0.1,
#       action_rate -0.005 (all unchanged)
#
#   OOD eval ranges (REVERTED to v2.6 calibration):
#     - Slippery: friction (0.15, 1.50)/(0.10, 1.30)
#     - HighDisturbance: ±18N / ±3.5Nm
#     - FastObstacles: motion 0.4 m/s
#     - HeavyCOM: COM ±8cm/±5cm (unchanged across all versions)
#     - DensePack: separation_buffer 0.2m (unchanged)
#     - RealisticCompound: all 5 reverted
#     This makes v2.10 OOD numbers directly comparable to v2.6's
#     paper baseline numbers (apples-to-apples).
#
# Hypothesis: v2.6's narrow DR was the dominant regression cause for
# v2.7/v2.8/v2.9/v2.9b. With training DR back to v2.6 levels, the
# policy can converge cleanly while still benefiting from REWARD-2's
# fix-the-structural-gaps reward stack.
#
# Predicted: training base_contact ~6-8% (v2.6 was 5.8%), eval in-dist
# combined ~25-30% (v2.6 was 30.6%). Wins on every single-axis OOD;
# possibly also wins compound (v2.6 tied at -0.3pp; REWARD-2 may close
# that gap).
#
# Decision criteria:
#   - In-dist combined ≤ 0.31 → v2.10 ≥ v2.6 paper baseline → ship.
#   - 0.31 < combined < 0.40 → partial recovery; v2.6 stays canonical;
#     consider further reward tuning or just ship v2.6.
#   - combined ≥ 0.40 → DR revert wasn't the bottleneck; deeper issue.
#     Most likely we ship v2.6 + locked-eval as the paper claim.
#
# Usage on lab box:
#   ./scripts/train_and_eval_v210.sh 2>&1 | tee logs/train_and_eval_v210.log

set -e  # abort on any error

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v2.10 (v2.6 DR + v2.8 planner + v2.9b reward) TRAIN + EVAL pipeline"
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
  OUT="logs/baseline_eval_v210_${TAG}"

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
ls -la logs/baseline_eval_v210_*/baseline.csv 2>/dev/null
echo "================================================================"
