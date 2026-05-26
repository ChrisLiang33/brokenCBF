#!/bin/bash
# Re-eval V8 and V9 ckpts on the STRESSOR distribution for cross-model
# comparison. ~30 min total. V8 uses 33-D priv task; V9 uses 13-D priv task.
#
# Usage on lab box (separate tmux from V10 training; can run in parallel
# if GPU has headroom, or after V10 finishes):
#   tmux new -s stressor_eval
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/eval_stressor_v8_v9.sh \
#     2>&1 | tee logs/stressor_eval_v8_v9.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

V8_CKPT=${V8_CKPT:-logs/rsl_rl/cbf_go2_teacher_rma/2026-05-19_14-18-46/model_2499.pt}
V9_CKPT=${V9_CKPT:-logs/rsl_rl/cbf_go2_teacher_rma/2026-05-19_18-29-46/model_2499.pt}

[ -f "$V8_CKPT" ] || { echo "ERROR: V8 ckpt missing: $V8_CKPT"; exit 1; }
[ -f "$V9_CKPT" ] || { echo "ERROR: V9 ckpt missing: $V9_CKPT"; exit 1; }

echo "================================================================"
echo "STRESSOR re-eval for V8 + V9 ckpts"
echo "  V8 ckpt: $V8_CKPT"
echo "  V9 ckpt: $V9_CKPT"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo ""
echo "[1/2] V8 ckpt on STRESSOR at $(date '+%H:%M:%S')"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$V8_CKPT" \
  --output_dir "logs/baseline_eval_v8_on_stressor" --headless

echo ""
echo "[2/2] V9 ckpt on STRESSOR_V9 at $(date '+%H:%M:%S')"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V9-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$V9_CKPT" \
  --output_dir "logs/baseline_eval_v9_on_stressor" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Output:"
echo "  logs/baseline_eval_v8_on_stressor/baseline.csv"
echo "  logs/baseline_eval_v9_on_stressor/baseline.csv"
echo "================================================================"
