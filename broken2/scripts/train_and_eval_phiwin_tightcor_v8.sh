#!/bin/bash
# PHIWIN_TIGHTCOR_V8 (2026-05-19) — per-step φ (no lock) test. Same setup
# as V7 (SHIELD c-compensation, within-ep σ_act regime, all V5 + V4
# stuff) but with the φ output lock removed. Tests whether per-step φ
# trains cleanly in our current setup, given α already does.
#
# σ_act regime is still resampled every 5s within episode — that's a
# DR axis, NOT a lock. The φ OUTPUT lock is what's removed: policy emits
# φ every control step, CBF uses it every step. No artificial 5s hold.
#
# Default ITERATIONS=2500 (~3.3h) since V7 was killed mid-run and time
# is tight. Override CBF_ITERATIONS for longer.
#
# Usage on lab box (in tmux):
#   tmux new -s wk3tight8
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   CBF_ITERATIONS=2500 ~/Desktop/safety-go2/scripts/train_and_eval_phiwin_tightcor_v8.sh \
#     2>&1 | tee logs/train_and_eval_wk3tight8.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-2500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "PHIWIN_TIGHTCOR_V8: V7 - lock. Per-step φ, SHIELD-fix kept."
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')  iters=${ITERATIONS}"
echo "================================================================"

grep -q "CbfGo2EnvCfg_LAYER3_PHIWIN_TIGHTCOR_V8" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V8 train cfg present" \
  || { echo "  ✗ V8 train cfg missing"; exit 1; }
grep -q "CbfGo2EnvCfg_DEPLOY_REALISTIC_FROZEN_AC_TRAINMATCH_V8" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V8 deploy cfg present" \
  || { echo "  ✗ V8 deploy cfg missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V8-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ V8 train task registered" \
  || { echo "  ✗ V8 train task not registered"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V8-v0 \
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

# φ diagnostic — note: with no lock, --use_locked is NOT applicable.
# Per-step φ is captured directly from policy output.
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V8-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_phi_corr_wk3tight8.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V8-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_alpha_corr_wk3tight8.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V8-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3tight8.json --headless

echo ""
echo "[3/3] EVAL at $(date '+%H:%M:%S')"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V8-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight8_indist" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V8-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight8_trainmatch" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "What V8 tests:"
echo "  Does per-step φ output train cleanly under current setup?"
echo "  α already does; this checks if φ can too."
echo ""
echo "Headline metrics:"
echo "  φ within-env std       : was 0.25 under windowed lock → expect higher"
echo "                           under per-step (φ varies freely now)"
echo "  Pearson(φ_t, h_t) within-ep: V5 had -0.10 under lock"
echo "                                under per-step, should be cleaner if"
echo "                                LLN argument was wrong. Hope < -0.20."
echo "  Pearson(φ, σ_act) between-env: V5 had ~0. If V8 hits +0.20+, the"
echo "                                  lock was actively hurting σ tracking."
echo ""
echo "If V8 trains clean AND BR competitive: lock was unnecessary baggage."
echo "If V8 fails or BR regresses: per-step φ genuinely doesn't work here."
echo "================================================================"
