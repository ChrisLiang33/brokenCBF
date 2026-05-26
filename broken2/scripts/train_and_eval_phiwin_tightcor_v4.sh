#!/bin/bash
# PHIWIN_TIGHTCOR_V4 (2026-05-18) — Step 3 of the φ-α-adaptation roadmap.
# Single variable change from V3: switch φ lock from per_episode to
# windowed @ 5s (250 control steps = 5s at 50 Hz, giving 4 φ samples
# per 20s episode).
#
# Why: V3 showed α adaptation works clean (Pearson(α,|tracking_err|) =
# -0.476) but φ adaptation regressed (Pearson(φ, σ_act) = +0.077 vs V1's
# +0.151) AND BR lost to B2 (state-conditional φ(h)) on deploy. Audit
# revealed the per-episode φ lock is structurally preventing φ from
# using the CNN-grid features that z_grid (64-D) already feeds into
# π_teacher every step. Unlocking via windowed should let φ track
# obstacle proximity within episode — same adaptation pattern that
# makes B2 win.
#
# Eval is on FROZEN_AC_TRAINMATCH (corridors + full σ_act range +
# adversarial restored on top of FROZEN_AC's a/c freezes). Same eval
# we used for V3 head-to-head — apples-to-apples comparison.
#
# Usage on lab box (in tmux):
#   tmux new -s wk3tight4
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_phiwin_tightcor_v4.sh \
#     2>&1 | tee logs/train_and_eval_wk3tight4.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-1500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "PHIWIN_TIGHTCOR_V4: V3 + windowed φ lock @ 5s (250 steps)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Sanity-check cfg + task + env plumbing all present.
grep -q "CbfGo2EnvCfg_LAYER3_PHIWIN_TIGHTCOR_V4" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V4 config present" \
  || { echo "  ✗ V4 config missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V4-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ V4 task registered" \
  || { echo "  ✗ V4 task not registered"; exit 1; }
grep -q 'self\._phi_lock_mode == "windowed"' \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py \
  && echo "  ✓ windowed φ lock branch in env" \
  || { echo "  ✗ env missing windowed branch"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V4-v0 \
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

# φ diagnostic. With windowed lock, within-env std should now be
# meaningfully > 0 (φ varies within episode across the 4 windows).
# The headline metric shifts: we want Pearson(φ_t, h_t) within-episode
# to be strongly negative (φ ramps up as obstacle proximity increases).
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V4-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3tight4.json --headless

# α diagnostic. Goal: confirm α adaptation preserved (V3 baseline was
# Pearson(α, |tracking_err|) = -0.476).
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V4-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_alpha_corr_wk3tight4.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V4-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3tight4.json --headless

echo ""
echo "[3/3] EVAL at $(date '+%H:%M:%S')"

# In-dist sanity (training distribution).
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V4-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight4_indist" --headless

# FROZEN_AC_TRAINMATCH — the deploy eval that exposes the policy to
# the training distribution's scene + disturbance regime. The headline
# Step 2/3 gate.
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight4_trainmatch" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "Success gates:"
echo "  φ: within-env std > 0.3 AND Pearson(φ_t, h_t) within-episode < -0.3"
echo "  α: Pearson(α, |tracking_err|) magnitude > 0.30 (preserved)"
echo "  Deploy: BR joint_actual > best B0/B1/B2 on FROZEN_AC_TRAINMATCH"
echo ""
echo "Diagnostic outputs:"
echo "  φ corr: diagnose_phi_corr_wk3tight4.json"
echo "  α corr: diagnose_alpha_corr_wk3tight4.json"
echo "  z probe: probe_z_linear_wk3tight4.json"
echo ""
echo "Eval outputs:"
echo "  In-dist: logs/baseline_eval_wk3tight4_indist/baseline.csv"
echo "  Deploy:  logs/baseline_eval_wk3tight4_trainmatch/baseline.csv"
echo "================================================================"
