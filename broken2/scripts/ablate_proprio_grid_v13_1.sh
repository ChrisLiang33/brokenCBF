#!/bin/bash
# Slim follow-up to ablate_z_env_v13_1.sh — runs only proprio + z_grid ablations.
# Use this if z_env ablation has already finished and you don't want to redo it.
#
# ETA: ~4 dists × 2 ablations × ~1.5 min = ~13 min.

set -e
cd ~/Desktop/safety-go2/IsaacLab

CKPT=${CKPT:-/home/chrisliang/Desktop/safety-go2/IsaacLab/logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_18-25-01/model_2499.pt}
[ -f "$CKPT" ] || { echo "ckpt not found: $CKPT"; exit 1; }

declare -A TASKS
TASKS[indist]="Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-1-v0"
TASKS[trainmatch]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V13-1-v0"
TASKS[ood]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V13-1-v0"
TASKS[stressor]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V13-1-v0"

run_one() {
  local tag=$1; local var=$2; local dist=$3
  echo ""
  echo "─── ablate=${tag} ${dist} ─── $(date '+%H:%M:%S')"
  unset CBF_ABLATE_Z_ENV CBF_ABLATE_PROPRIO CBF_ABLATE_Z_GRID
  export "$var=mean"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "${TASKS[$dist]}" \
    --num_envs 64 --steps_per_config 1000 \
    --modes BR \
    --alpha_grid "2.0" --phi_grid "0.5" \
    --epsilon0_grid "0.5" --lambda_grid "1.0" \
    --checkpoint "$CKPT" \
    --output_dir "logs/ablate_${tag}_v13_1_${dist}" --headless
}

for dist in indist trainmatch ood stressor; do
  run_one proprio  CBF_ABLATE_PROPRIO  "$dist"
  run_one z_grid   CBF_ABLATE_Z_GRID   "$dist"
done

unset CBF_ABLATE_Z_ENV CBF_ABLATE_PROPRIO CBF_ABLATE_Z_GRID

echo ""
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
