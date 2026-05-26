#!/bin/bash
# Resume V11 training from ckpt_500 and complete to 2500 iters,
# then run diagnostics + 4 evals.
#
# V11 = strip both tracking_err AND base_height from priv obs.
# Tests "what happens if heads can't gate off ANY partially-observable
# channels" — the cleanest architectural test.
#
# V11's first attempt was crashed at iter 500 by play.py launch on
# 2026-05-20. Ckpt_500 survives. We resume from there.
#
# Usage on lab box (after reboot):
#   tmux new -s wk3v11_resume
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/resume_v11.sh \
#     2>&1 | tee logs/train_and_eval_wk3v11_resume.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

# Resume from this run dir
RESUME_RUN=${RESUME_RUN:-2026-05-20_09-28-18}
RESUME_CKPT=${RESUME_CKPT:-model_500.pt}
TARGET_ITERS=${TARGET_ITERS:-2500}
ENVS=${CBF_NUM_ENVS:-4096}

CKPT_PATH=logs/rsl_rl/cbf_go2_teacher_rma/${RESUME_RUN}/${RESUME_CKPT}
[ -f "$CKPT_PATH" ] || { echo "ERROR: ckpt missing: $CKPT_PATH"; exit 1; }

echo "================================================================"
echo "V11 RESUME: from ${RESUME_RUN}/${RESUME_CKPT} → iter ${TARGET_ITERS}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/3] RESUME PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --num_envs "${ENVS}" --max_iterations "${TARGET_ITERS}" \
  --resume --load_run "${RESUME_RUN}" --checkpoint "${RESUME_CKPT}" \
  --headless

unset CBF_AUX_COEF
LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1 "${LATEST_DIR}"/model_*.pt 2>/dev/null \
    | awk -F'model_|\\.pt' '{print $2, $0}' \
    | sort -n | awk '{print $2}' | tail -1)
[ -f "$CKPT" ] || { echo "ERROR: no ckpt after resume"; exit 1; }
echo "Final ckpt: $CKPT"

echo ""
echo "[2/3] DIAGNOSTICS at $(date '+%H:%M:%S')"
export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 12 \
  --output diagnose_phi_corr_wk3v11.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 12 \
  --output diagnose_alpha_corr_wk3v11.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 12 \
  --output probe_z_linear_wk3v11.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_grad_sensitivity.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 50 --priv_dim 12 \
  --alpha_min 0.5 --alpha_max 3.0 \
  --output diagnose_grad_sensitivity_wk3v11.json --headless

echo ""
echo "[3/3] 4-EVAL at $(date '+%H:%M:%S')"

# 1. in-dist
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-RMAv11-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v11_indist" --headless

# 2. trainmatch
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V11-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v11_trainmatch" --headless

# 3. OOD
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V11-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v11_ood" --headless

# 4. STRESSOR
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V11-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v11_stressor" --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "What V11 tests:"
echo "  Both tracking_err and base_height stripped from priv obs."
echo "  Heads have only truly-hidden priv (friction, mass, σ_act, COM,"
echo "  force, torque) + the grid encoder to work with."
echo "  V11 iter 500 preview: total head sens dropped, grid ratio jumped."
echo "  V11 iter 2500 will tell us if more training:"
echo "    (a) activates σ_act / friction / mass adaptation"
echo "    (b) collapses to globally cautious like V9"
echo "    (c) shifts entirely to grid-based behavior"
echo ""
echo "Performance bar: V8 BR trainmatch=0.768, OOD=0.626, STRESSOR=0.743"
echo "================================================================"
