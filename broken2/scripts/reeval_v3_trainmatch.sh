#!/bin/bash
# Re-eval V3's already-trained checkpoint on FROZEN_AC_TRAINMATCH —
# the train-distribution-matched eval env (corridors + full σ_act range
# + post-QP adversarial restored on top of FROZEN_AC's a/c freezes and
# deploy-realistic planner mix).
#
# Why: V3's first FROZEN_AC eval showed BR losing to fixed baselines by
# ~11pp, but the audit revealed FROZEN_AC eval distribution was missing
# the corridor scenes and most of the σ_act range that training had.
# This eval exposes the policy to the scenario regime it actually trained
# on, while keeping FROZEN_AC's deploy-side narrowing (planner, push).
#
# Usage on lab box:
#   tmux new -s wk3tm
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/reeval_v3_trainmatch.sh \
#     2>&1 | tee logs/reeval_wk3tight3_trainmatch.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

CKPT="logs/rsl_rl/cbf_go2_teacher_rma/2026-05-18_18-41-33/model_1499.pt"
[ -f "$CKPT" ] || { echo "ERROR: V3 ckpt missing: $CKPT"; exit 1; }

echo "================================================================"
echo "V3 re-eval on FROZEN_AC_TRAINMATCH (corridors + σ_act 0-0.20 + adversarial)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "================================================================"

# Sanity-check cfg + task present after sync.
grep -q "CbfGo2EnvCfg_DEPLOY_REALISTIC_FROZEN_AC_TRAINMATCH" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ TRAINMATCH cfg present" \
  || { echo "  ✗ TRAINMATCH cfg missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ TRAINMATCH task registered" \
  || { echo "  ✗ TRAINMATCH task not registered"; exit 1; }

export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight3_trainmatch" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: logs/baseline_eval_wk3tight3_trainmatch/baseline.csv"
echo "================================================================"
