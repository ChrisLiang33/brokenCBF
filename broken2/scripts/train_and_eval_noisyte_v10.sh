#!/bin/bash
# NOISYTE_V10 (2026-05-19) — V8 priv (33-D, keeps tracking_err) + per-step
# Gaussian noise σ=0.05 on tracking_err. Matches Go2 IMU + leg-odometry
# noise realism. Tests whether deploy-realistic noise preserves the
# within-episode adaptation signal (which V9 lost by stripping tracking_err
# entirely) while reducing OOD brittleness.
#
# Single variable change from V8:
#   tracking_err_noise_sigma: 0.0 → 0.05
#
# Usage on lab box (in tmux):
#   tmux new -s wk3v10
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   CBF_ITERATIONS=2500 ~/Desktop/safety-go2/scripts/train_and_eval_noisyte_v10.sh \
#     2>&1 | tee logs/train_and_eval_wk3v10.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-4500}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "NOISYTE_V10: V8 + Gaussian noise σ=0.05 on tracking_err"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')  iters=${ITERATIONS}"
echo "================================================================"

grep -q "CbfGo2EnvCfg_LAYER3_NOISYTE_V10" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V10 train cfg present" \
  || { echo "  ✗ V10 train cfg missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-NoisyTE-V10-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ V10 train task registered" \
  || { echo "  ✗ V10 train task not registered"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-NoisyTE-V10-v0 \
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
  --task Isaac-CBF-Go2-RMA-Layer3-NoisyTE-V10-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_phi_corr_wk3v10.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-NoisyTE-V10-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_alpha_corr_wk3v10.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-NoisyTE-V10-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 16 \
  --output probe_z_linear_wk3v10.json --headless

echo ""
echo "[3/3] EVAL at $(date '+%H:%M:%S')"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-NoisyTE-V10-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v10_indist" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V10-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v10_trainmatch" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V10-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v10_ood" --headless

# STRESSOR eval — bumped σ_act/friction to expose adaptive vs fixed.
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v10_stressor" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "What V10 tests:"
echo "  Hypothesis: noisy tracking_err preserves adaptation signal while"
echo "  forcing the policy to be robust to deploy-realistic noise."
echo ""
echo "Headline checks:"
echo "  BR trainmatch safety_score — V8 was 0.768 (tied best fixed)."
echo "  BR in-dist safety_score — V8 was 0.614 (rank 4)."
echo "  BR OOD safety_score — V8 was 0.626 (rank 8/16). V9 was 0.615."
echo "                          V10 should beat V8 here if hypothesis correct."
echo "  φ within-env std — should stay near V8's 1.15 (vs V9's 1.30)."
echo "  Pearson(φ, σ_act) between-env — V8 had −0.20, V9 had −0.06."
echo "                                   V10 should be near V8 (signal preserved)."
echo "================================================================"
