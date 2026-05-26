#!/bin/bash
# V13.2 (2026-05-20) — V13.1 architecture + wider DR.
# Layered change: V13.1 + σ_act_max 0.20→0.40 + friction (0.10, 1.30).
# Targets V13.1's stressor regression (was −13pp vs best fixed).
#
# Usage on lab box:
#   tmux new -s wk3v13_2
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   CBF_ITERATIONS=2500 ~/Desktop/safety-go2/scripts/train_and_eval_twostream_v13_2.sh \
#     2>&1 | tee logs/train_and_eval_wk3v13_2.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-2500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "V13.2: V13.1 + wider DR (σ_act 0.40, friction 0.10-1.30)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')  iters=${ITERATIONS}"
echo "================================================================"

grep -q "CbfGo2EnvCfg_LAYER3_TWOSTREAM_V13_2" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V13.2 train cfg present" \
  || { echo "  ✗ V13.2 train cfg missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-2-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ V13.2 train task registered" \
  || { echo "  ✗ V13.2 train task not registered"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-2-v0 \
  --num_envs "${ENVS}" --max_iterations "${ITERATIONS}" \
  --headless

unset CBF_AUX_COEF
LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1 "${LATEST_DIR}"/model_*.pt 2>/dev/null \
    | awk -F'model_|\\.pt' '{print $2, $0}' \
    | sort -n | awk '{print $2}' | tail -1)
[ -f "$CKPT" ] || { echo "ERROR: no ckpt"; exit 1; }
echo "Final ckpt: $CKPT"

echo ""
echo "[2/3] DIAGNOSTICS at $(date '+%H:%M:%S')"
export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-2-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --priv_layout v13 \
  --output diagnose_phi_corr_wk3v13_2.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-2-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --priv_layout v13 \
  --output diagnose_alpha_corr_wk3v13_2.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-2-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3v13_2.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_grad_sensitivity.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-2-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 50 --priv_dim 33 \
  --priv_layout v13 \
  --alpha_min 0.5 --alpha_max 3.0 \
  --output diagnose_grad_sensitivity_wk3v13_2.json --headless

echo ""
echo "[3/3] 4-EVAL at $(date '+%H:%M:%S')"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-2-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v13_2_indist" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V13-2-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v13_2_trainmatch" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V13-2-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v13_2_ood" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V13-2-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v13_2_stressor" --headless

echo ""
echo "================================================================"
echo "V13.2 TRAIN+EVAL DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────
# [4/4] CHAINED MULTI-SEED SWEEP
# Runs V13.1 + V13.2 × {trainmatch, OOD, stressor} × seeds {42, 123, 7}
# for publishable error bars. ~2h.
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] CHAINED MULTI-SEED SWEEP at $(date '+%H:%M:%S')"

TEACHER_V13_1=${TEACHER_V13_1:-/home/chrisliang/Desktop/safety-go2/IsaacLab/logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_18-25-01/model_2499.pt}
if [ ! -f "$TEACHER_V13_1" ]; then
  echo "WARNING: V13.1 teacher not found at $TEACHER_V13_1, skipping V13.1 leg"
  TEACHER_V13_1=""
fi
TEACHER_V13_2="$CKPT"

if [ -f ~/Desktop/safety-go2/scripts/multi_seed_sweep.sh ]; then
  TEACHER_V13_1="$TEACHER_V13_1" TEACHER_V13_2="$TEACHER_V13_2" \
    bash ~/Desktop/safety-go2/scripts/multi_seed_sweep.sh
else
  echo "WARNING: multi_seed_sweep.sh not found. Sync from laptop and run manually."
fi

echo ""
echo "================================================================"
echo "V13.2 + MULTI-SEED FULLY DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "Targets vs V13.1:"
echo "  V13.1 indist  BR=0.618  (lost  by 8pp)"
echo "  V13.1 trainmatch BR=0.861 (WON +7pp on 1 seed; revisit w/ multi-seed)"
echo "  V13.1 OOD     BR=0.797  (WON +4pp on 1 seed; revisit w/ multi-seed)"
echo "  V13.1 stress  BR=0.664  (lost  by 13pp)  ← V13.2 should fix"
echo ""
echo "Next: pull logs/multiseed_* to laptop and run aggregate_multiseed.py."
echo "================================================================"
