#!/bin/bash
# Multi-seed eval sweep — gives publishable error bars on the paper headline.
#
# Runs the full B0,B1,B2,BR sweep on the 3 deploy-relevant distributions
# (trainmatch, OOD, stressor) for each model, across 3 seeds.
#
# Total: 2 models × 3 dists × 3 seeds = 18 evaluations × ~5-7 min = ~2h.
#
# Usage on lab box (after V13.2 finishes training):
#   tmux new -s multiseed
#   cd ~/Desktop/safety-go2/IsaacLab
#   TEACHER_V13_1=<path> TEACHER_V13_2=<path> \
#     bash ~/Desktop/safety-go2/scripts/multi_seed_sweep.sh \
#     2>&1 | tee logs/multi_seed_sweep.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

# Allow override via env. If unset, default to known paths.
TEACHER_V13_1=${TEACHER_V13_1:-/home/chrisliang/Desktop/safety-go2/IsaacLab/logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_18-25-01/model_2499.pt}
TEACHER_V13_2=${TEACHER_V13_2:-}

if [ -z "$TEACHER_V13_2" ]; then
  # Auto-find: latest run dir from after V13.1, take its final ckpt.
  echo "[multiseed] TEACHER_V13_2 not set, auto-detecting latest run dir..."
  # Skip V13.1's known dir.
  LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ \
    | grep -v "2026-05-20_18-25-01" | head -1)
  LATEST_DIR=${LATEST_DIR%/}
  TEACHER_V13_2=$(ls -1 "${LATEST_DIR}"/model_*.pt 2>/dev/null \
    | awk -F'model_|\\.pt' '{print $2, $0}' \
    | sort -n | awk '{print $2}' | tail -1)
  echo "[multiseed] auto-detected: $TEACHER_V13_2"
fi

if [ -n "$TEACHER_V13_1" ] && [ ! -f "$TEACHER_V13_1" ]; then
  echo "WARNING: TEACHER_V13_1 path given but file missing: $TEACHER_V13_1 — skipping V13.1 leg."
  TEACHER_V13_1=""
fi
# 2026-05-21: SKIP_V13_2=1 short-circuits the V13.2 leg. Cosner directed:
# focus on V13.1 for the paper, V14/V13.2 are extra. Set this env var to
# skip the V13.2 multi-seed and only run V13.1 × 3 seeds × 3 dists.
SKIP_V13_2=${SKIP_V13_2:-0}
if [ "$SKIP_V13_2" = "1" ]; then
  echo "[multiseed] SKIP_V13_2=1 — only running V13.1 leg."
elif [ -z "$TEACHER_V13_2" ] || [ "$TEACHER_V13_2" = "$TEACHER_V13_1" ] || [ ! -f "$TEACHER_V13_2" ]; then
  echo "[multiseed] TEACHER_V13_2 unset / missing / same as V13.1 — skipping V13.2 leg."
  SKIP_V13_2=1
fi

echo "================================================================"
echo "MULTI-SEED SWEEP"
echo "  V13.1: $TEACHER_V13_1"
echo "  V13.2: $TEACHER_V13_2"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

SEEDS=(42 123 7)

# Distribution name → (task_id_V13_1, task_id_V13_2).
declare -A TASK_V13_1
TASK_V13_1[trainmatch]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V13-1-v0"
TASK_V13_1[ood]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V13-1-v0"
TASK_V13_1[stressor]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V13-1-v0"

declare -A TASK_V13_2
TASK_V13_2[trainmatch]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V13-2-v0"
TASK_V13_2[ood]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V13-2-v0"
TASK_V13_2[stressor]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V13-2-v0"

run_one() {
  local model=$1; local dist=$2; local seed=$3
  local task=$4; local teacher=$5
  local out="logs/multiseed_${model}_${dist}_seed${seed}"
  echo ""
  echo "─── ${model} / ${dist} / seed=${seed} ───"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "$task" \
    --num_envs 64 --steps_per_config 1000 \
    --modes B0,B1,B2,BR \
    --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
    --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
    --checkpoint "$teacher" \
    --seed "$seed" \
    --output_dir "$out" --headless
}

if [ -n "$TEACHER_V13_1" ]; then
  echo ""
  echo "─── V13.1 leg ───"
  for dist in trainmatch ood stressor; do
    for seed in "${SEEDS[@]}"; do
      run_one "v13_1" "$dist" "$seed" "${TASK_V13_1[$dist]}" "$TEACHER_V13_1"
    done
  done
else
  echo ""
  echo "─── V13.1 leg SKIPPED (no teacher path) ───"
fi

if [ "$SKIP_V13_2" != "1" ]; then
  echo ""
  echo "─── V13.2 leg ───"
  for dist in trainmatch ood stressor; do
    for seed in "${SEEDS[@]}"; do
      run_one "v13_2" "$dist" "$seed" "${TASK_V13_2[$dist]}" "$TEACHER_V13_2"
    done
  done
else
  echo ""
  echo "─── V13.2 leg SKIPPED ───"
fi

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Pull all logs/multiseed_*/baseline.csv to laptop for aggregation."
echo "================================================================"
