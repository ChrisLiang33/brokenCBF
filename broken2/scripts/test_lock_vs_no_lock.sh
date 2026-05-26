#!/bin/bash
# Test 1 (2026-05-19) — eval an existing trained-with-lock checkpoint
# BOTH with and without the φ lock to see how much the lock matters at
# inference time.
#
# The trained policy was optimized under windowed lock. Question: does
# its behavior degrade when run at deploy without the lock? Or is the
# trained network smooth enough that per-step inference works fine?
#
# We use V5's ckpt as the baseline (most thoroughly trained, 4500 iters).
# If V7 has any saved iterations, prefer that.
#
# Two eval passes, same ckpt:
#   (a) FROZEN_AC_TRAINMATCH_V7  — windowed lock @ 5s (training match)
#   (b) FROZEN_AC_TRAINMATCH_V8  — per-step φ, no lock
#
# Compare BR's numbers side-by-side. ~20 minutes total.
#
# Usage on lab box:
#   tmux new -s wk3lock_test
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   CKPT=logs/rsl_rl/cbf_go2_teacher_rma/<run_dir>/model_<iter>.pt \
#     ~/Desktop/safety-go2/scripts/test_lock_vs_no_lock.sh \
#     2>&1 | tee logs/lock_vs_no_lock_test.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

# Find a checkpoint. Prefer explicit override, else V5 (most-trained
# baseline), else latest available.
if [ -z "$CKPT" ]; then
    # Try V5 known timestamp first.
    V5_DEFAULT="logs/rsl_rl/cbf_go2_teacher_rma/2026-05-18_22-50-50/model_1499.pt"
    if [ -f "$V5_DEFAULT" ]; then
        CKPT="$V5_DEFAULT"
        echo "Using V5 default ckpt: $CKPT"
    else
        echo "V5 default not found; using latest..."
        LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
        LATEST_DIR=${LATEST_DIR%/}
        CKPT=$(ls -1 "${LATEST_DIR}"/model_*.pt 2>/dev/null \
            | awk -F'model_|\\.pt' '{print $2, $0}' \
            | sort -n | awk '{print $2}' | tail -1)
        echo "Using latest ckpt: $CKPT"
    fi
fi
[ -f "$CKPT" ] || { echo "ERROR: ckpt missing: $CKPT"; exit 1; }

echo "================================================================"
echo "Test 1: lock vs no-lock eval on existing ckpt"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "================================================================"

# Sanity check.
grep -q "CbfGo2EnvCfg_DEPLOY_REALISTIC_FROZEN_AC_TRAINMATCH_V7" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ TRAINMATCH_V7 (windowed lock) cfg present" \
  || { echo "  ✗ V7 deploy cfg missing"; exit 1; }
grep -q "CbfGo2EnvCfg_DEPLOY_REALISTIC_FROZEN_AC_TRAINMATCH_V8" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ TRAINMATCH_V8 (per-step) cfg present" \
  || { echo "  ✗ V8 deploy cfg missing"; exit 1; }

export CBF_AUX_COEF=0.0

echo ""
echo "[1/2] WITH lock (TRAINMATCH_V7, windowed @ 5s) at $(date '+%H:%M:%S')"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V7-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/lock_test_with_lock" --headless

echo ""
echo "[2/2] NO lock (TRAINMATCH_V8, per-step φ) at $(date '+%H:%M:%S')"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V8-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/lock_test_no_lock" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "What to compare in the two CSVs (BR row in each):"
echo "  - φ_within_env_std    : should jump up when no lock (φ varies every step)"
echo "  - φ_mean              : should be similar (same network output mean)"
echo "  - goal_reach_rate     : if no-lock comparable to locked → lock not critical"
echo "  - fall_rate           : if no-lock spikes → policy depends on the lock"
echo "  - safety_score        : the headline question"
echo ""
echo "Outputs:"
echo "  logs/lock_test_with_lock/baseline.csv"
echo "  logs/lock_test_no_lock/baseline.csv"
echo "================================================================"
