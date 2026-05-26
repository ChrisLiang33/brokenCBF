#!/bin/bash
# v2.8 train-and-eval pipeline. One-shot script:
#   1. Trains v2.8 (~5h, 5000 PPO iters)
#   2. Locates the resulting checkpoint
#   3. Runs the full 7-eval suite sequentially (~1h 45min)
#
# Usage on lab box:
#   ./scripts/train_and_eval_v28.sh 2>&1 | tee logs/train_and_eval_v28.log

set -e  # abort on any error

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v2.8 TRAIN + EVAL pipeline"
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
  OUT="logs/baseline_eval_v28_${TAG}"

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
ls -la logs/baseline_eval_v28_*/baseline.csv 2>/dev/null
echo "================================================================"
