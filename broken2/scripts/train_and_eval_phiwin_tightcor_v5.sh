#!/bin/bash
# PHIWIN_TIGHTCOR_V5 (2026-05-19) — the "where adaptive beats fixed" run.
# Single variable change from V4: enable within-episode σ_act regime
# resampling. Every 5s within an episode, σ_act is re-sampled from
# Uniform(0, σ_max_curriculum). Now no single fixed φ can be optimal
# across the regime changes — fixed φ tuned for the average σ_act is
# wasteful in calm windows and unsafe in noisy windows.
#
# V4 showed BR is implementation-correct (gap to fixed was within eval
# variance under static within-episode DR). V5 tests whether adaptive
# wins decisively when there's a real within-episode signal to adapt to.
#
# Theoretical claim: in V5's distribution, fixed structurally cannot win,
# because no single config is optimal across regime changes. BR's
# windowed φ should track per-window σ_act and outperform fixed by a
# margin above the ~5pp eval variance.
#
# Eval uses FROZEN_AC_TRAINMATCH_V5 — same regime-change structure on
# the deploy-realistic planner side. Apples-to-apples.
#
# Usage on lab box (in tmux):
#   tmux new -s wk3tight5
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_phiwin_tightcor_v5.sh \
#     2>&1 | tee logs/train_and_eval_wk3tight5.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-1500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "PHIWIN_TIGHTCOR_V5: V4 + within-episode σ_act regime (5s window)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Sanity checks.
grep -q "CbfGo2EnvCfg_LAYER3_PHIWIN_TIGHTCOR_V5" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V5 train cfg present" \
  || { echo "  ✗ V5 train cfg missing"; exit 1; }
grep -q "CbfGo2EnvCfg_DEPLOY_REALISTIC_FROZEN_AC_TRAINMATCH_V5" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V5 deploy cfg present" \
  || { echo "  ✗ V5 deploy cfg missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V5-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ V5 train task registered" \
  || { echo "  ✗ V5 train task not registered"; exit 1; }
grep -q "dr_window_sigma_act" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py \
  && echo "  ✓ env within-ep DR plumbing present" \
  || { echo "  ✗ env missing dr_window_sigma_act"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V5-v0 \
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
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V5-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3tight5.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V5-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_alpha_corr_wk3tight5.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V5-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3tight5.json --headless

echo ""
echo "[3/3] EVAL at $(date '+%H:%M:%S')"

# In-dist: V5 task. Fixed baselines see the same within-episode DR they
# can't track. This is where adaptive should win.
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V5-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight5_indist" --headless

# Deploy: FROZEN_AC_TRAINMATCH_V5 — same within-ep DR on deploy planner.
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V5-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3tight5_trainmatch" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "Success gates:"
echo "  φ:  within-episode Pearson(φ_t, σ_act_t) > +0.30 (rises with regime)"
echo "  α:  Pearson(α, |tracking_err|) magnitude > 0.20 (preserved)"
echo "  In-dist: BR j_act > best fixed B0/B1/B2 by >5pp (above eval variance)"
echo "  Deploy:  same on FROZEN_AC_TRAINMATCH_V5"
echo ""
echo "Diagnostic outputs:"
echo "  φ corr: diagnose_phi_corr_wk3tight5.json"
echo "  α corr: diagnose_alpha_corr_wk3tight5.json"
echo "  z probe: probe_z_linear_wk3tight5.json"
echo ""
echo "Eval outputs:"
echo "  In-dist: logs/baseline_eval_wk3tight5_indist/baseline.csv"
echo "  Deploy:  logs/baseline_eval_wk3tight5_trainmatch/baseline.csv"
echo "================================================================"
