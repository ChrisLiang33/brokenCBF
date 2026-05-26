#!/bin/bash
# PHIWIN_TIGHTCOR_V6 (2026-05-19) — distribution designed to expose B2's
# structural shortcomings. V5 showed BR matches B2 on j_act but already
# wins on safety metrics (lower collision_rate_actual, higher mean_h,
# less time near boundary). V6 amplifies the conditions where adaptive
# should win.
#
# Three coordinated changes from V5:
#   1. σ_act regime amplitude up (0.20 → 0.30) + faster windows (5s → 2s)
#      — B2 is σ-blind; bigger amplitude + more transitions = bigger gap.
#   2. Friction wider (modest): (0.30, 1.20) → (0.20, 1.30) static.
#      — B2 is friction-blind. Avoided ultra-slippery 0.10 to not collapse locomotion.
#   3. Dynamic obstacles: 30% move at 0-0.3 m/s per-episode.
#      — B2 has no obstacle-velocity awareness. BR sees grid temporal changes.
#
# Default ITERATIONS=3000 (~4h) since the distribution is harder and
# user noted shorter training. Override CBF_ITERATIONS=4500 for overnight.
#
# Usage on lab box:
#   tmux new -s wk3tight6
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   CBF_ITERATIONS=3000 ~/Desktop/safety-go2/scripts/train_and_eval_phiwin_tightcor_v6.sh \
#     2>&1 | tee logs/train_and_eval_wk3tight6.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-3000}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "PHIWIN_TIGHTCOR_V6: V5 + bigger σ + wider friction + dynamic obs"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')  iters=${ITERATIONS}"
echo "================================================================"

grep -q "CbfGo2EnvCfg_LAYER3_PHIWIN_TIGHTCOR_V6" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V6 train cfg present" \
  || { echo "  ✗ V6 train cfg missing"; exit 1; }
grep -q "CbfGo2EnvCfg_DEPLOY_REALISTIC_FROZEN_AC_TRAINMATCH_V6" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V6 deploy cfg present" \
  || { echo "  ✗ V6 deploy cfg missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V6-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ V6 train task registered" \
  || { echo "  ✗ V6 train task not registered"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V6-v0 \
  --num_envs "${ENVS}" --max_iterations "${ITERATIONS}" \
  --headless

unset CBF_AUX_COEF
LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1 "${LATEST_DIR}"/model_*.pt 2>/dev/null \
    | awk -F'model_|\\.pt' '{print $2, $0}' \
    | sort -n | awk '{print $2}' | tail -1)
[ -f "$CKPT" ] || { echo "ERROR: no ckpt"; exit 1; }
echo "Using checkpoint: $CKPT"

echo ""
echo "[2/3] DIAGNOSTICS at $(date '+%H:%M:%S')"
export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V6-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3tight6.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V6-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_alpha_corr_wk3tight6.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V6-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3tight6.json --headless

echo ""
echo "[3/3] EVAL at $(date '+%H:%M:%S')"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V6-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight6_indist" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V6-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight6_trainmatch" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "Headline metrics to report (per memo on multi-metric scoring):"
echo "  COMPLETION:   goal_reach_rate"
echo "  SAFETY:       collision_rate_actual, fall_rate, mean_min_h"
echo "  EFFICIENCY:   mean_v_xy, mean_time_to_goal"
echo "  SMOOTHNESS:   avg_qp_active_rate, avg_deflection_mean"
echo ""
echo "Composite scores worth computing (parse CSV):"
echo "  j_act          = (1-fall) × (1-stuck) × goal               [legacy]"
echo "  safety_score   = goal × (1-fall) × (1-stuck) × (1-coll_actual)  [BR's likely win]"
echo "  efficiency     = goal × mean_v_xy                          [BR's likely win]"
echo "================================================================"
