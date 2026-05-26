#!/bin/bash
# z_env ablation eval for V13.1 (2026-05-21).
#
# Tests whether π_teacher actually uses the hidden-env latent z_env. Two modes:
#   mean → z_env := batch-mean (kills per-env env-class signal, preserves population stats)
#   zero → z_env := zeros (more aggressive — also kills population stats)
#
# Skips B0/B1/B2 baselines (the ablation doesn't affect fixed-α policies).
# Runs BR only across all 4 V13.1 deploy distributions.
#
# Reads:
#   - Patched cbf_go2_teacher_rma.py with CBF_ABLATE_Z_ENV env-var hook
#   - V13.1 ckpt at logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_18-25-01/model_2499.pt
#
# Usage on lab box:
#   tmux new -s zenvablate
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/ablate_z_env_v13_1.sh 2>&1 | tee logs/ablate_z_env_v13_1.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

CKPT=${CKPT:-/home/chrisliang/Desktop/safety-go2/IsaacLab/logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_18-25-01/model_2499.pt}
[ -f "$CKPT" ] || { echo "ckpt not found: $CKPT"; exit 1; }

echo "================================================================"
echo "z_env ablation eval — V13.1"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt:    $CKPT"
echo "================================================================"

declare -A TASKS
TASKS[indist]="Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-1-v0"
TASKS[trainmatch]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V13-1-v0"
TASKS[ood]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V13-1-v0"
TASKS[stressor]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V13-1-v0"

for ABLATE in mean zero; do
  echo ""
  echo "################################################################"
  echo "# CBF_ABLATE_Z_ENV=${ABLATE}"
  echo "################################################################"
  export CBF_ABLATE_Z_ENV="${ABLATE}"

  for dist in indist trainmatch ood stressor; do
    echo ""
    echo "─── ${dist} ─── $(date '+%H:%M:%S')"
    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
      --task "${TASKS[$dist]}" \
      --num_envs 64 --steps_per_config 1000 \
      --modes BR \
      --alpha_grid "2.0" --phi_grid "0.5" \
      --epsilon0_grid "0.5" --lambda_grid "1.0" \
      --checkpoint "$CKPT" \
      --output_dir "logs/ablate_z_env_${ABLATE}_v13_1_${dist}" --headless
  done
done

unset CBF_ABLATE_Z_ENV

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Compare BR fall+stuck in:"
echo "  logs/ablate_z_env_mean_v13_1_<dist>/baseline.csv"
echo "  logs/ablate_z_env_zero_v13_1_<dist>/baseline.csv"
echo "vs existing baselines:"
echo "  data_from_lab/baseline_eval_wk3v13_1_<dist>/baseline.csv"
echo ""
echo "Interpretation:"
echo "  ablate=mean stays at ~baseline → policy ignores z_env (architecture dead weight)"
echo "  ablate=mean craters significantly → policy depends on z_env after all"
echo "================================================================"
