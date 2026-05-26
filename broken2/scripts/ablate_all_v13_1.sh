#!/bin/bash
# Three-pathway ablation eval for V13.1 (2026-05-21).
#
# Single-axis ablations to decompose what α/φ actually adapt to:
#   1. z_env    — hidden env-class latent (friction/σ_act/COM/force/δR)
#   2. proprio  — observable passthrough (tracking_err history + base_height + ang_vel)
#   3. z_grid   — LiDAR-like occupancy grid latent (obstacle proximity)
#
# Each ablation replaces its pathway with batch-mean (kills per-env signal,
# preserves population stats). BR mode only — fixed-α baselines aren't affected.
#
# Pair this with the existing baseline.csv files in
# data_from_lab/baseline_eval_wk3v13_1_<dist>/ to read off contributions:
#
#   Δ_pathway = ablate_combined − baseline_combined
#
# Large Δ → pathway is load-bearing. Δ ≈ 0 → policy ignores it.
#
# ETA: ~4 dists × 3 ablations × ~1.5 min = ~20 min.
#
# Usage on lab box:
#   tmux new -s ablate3
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/ablate_all_v13_1.sh 2>&1 | tee logs/ablate_all_v13_1.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

CKPT=${CKPT:-/home/chrisliang/Desktop/safety-go2/IsaacLab/logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_18-25-01/model_2499.pt}
[ -f "$CKPT" ] || { echo "ckpt not found: $CKPT"; exit 1; }

echo "================================================================"
echo "3-pathway ablation eval — V13.1"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt:    $CKPT"
echo "================================================================"

declare -A TASKS
TASKS[indist]="Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-1-v0"
TASKS[trainmatch]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V13-1-v0"
TASKS[ood]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V13-1-v0"
TASKS[stressor]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V13-1-v0"

run_one() {
  # $1 = ablation tag (z_env / proprio / z_grid)
  # $2 = env var name
  # $3 = dist key
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
  run_one z_env    CBF_ABLATE_Z_ENV    "$dist"
  run_one proprio  CBF_ABLATE_PROPRIO  "$dist"
  run_one z_grid   CBF_ABLATE_Z_GRID   "$dist"
done

unset CBF_ABLATE_Z_ENV CBF_ABLATE_PROPRIO CBF_ABLATE_Z_GRID

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results in:"
echo "  logs/ablate_z_env_v13_1_<dist>/baseline.csv"
echo "  logs/ablate_proprio_v13_1_<dist>/baseline.csv"
echo "  logs/ablate_z_grid_v13_1_<dist>/baseline.csv"
echo ""
echo "Baselines for comparison in data_from_lab/baseline_eval_wk3v13_1_<dist>/"
echo "================================================================"
