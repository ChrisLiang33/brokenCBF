#!/bin/bash
# Resume theory-bundle pipeline at phase 2 (diagnostics + eval).
#
# The training already finished at logs/rsl_rl/cbf_go2_teacher_rma/2026-05-17_19-18-08/
# (model_1499.pt). The diagnostic scripts failed because they didn't know
# priv_dim=33. After patching, this script picks up from where the pipeline
# crashed without re-running PPO.
#
# Usage on lab box (same tmux session is fine):
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/resume_eval_theory_bundle.sh \
#     2>&1 | tee -a logs/train_and_eval_wk3theory.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

CKPT="logs/rsl_rl/cbf_go2_teacher_rma/2026-05-17_19-18-08/model_1499.pt"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

echo "================================================================"
echo "Resume theory-bundle pipeline at PHASE 2 (diagnostics + eval)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "================================================================"

export CBF_AUX_COEF=0.0

# ---------- PHASE 2: DIAGNOSTICS ----------
echo ""
echo "[2/3] DIAGNOSTICS at $(date '+%H:%M:%S')"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3theory.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_alpha_corr_wk3theory.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_a_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 \
  --output diagnose_a_corr_wk3theory.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_c_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --c_lo -0.20 --c_hi 0.20 \
  --output diagnose_c_corr_wk3theory.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3theory.json --headless

# ---------- PHASE 3: EVAL ----------
echo ""
echo "[3/3] EVAL at $(date '+%H:%M:%S')"

EVAL_OUT_INDIST="logs/baseline_eval_wk3theory_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Theory-Bundle-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,4.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT_INDIST" --headless

EVAL_OUT_DEPLOY="logs/baseline_eval_wk3theory_deploy"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,4.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT_DEPLOY" --headless

echo ""
echo "================================================================"
echo "RESUME PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  diagnostics: diagnose_*_wk3theory.json"
echo "  eval indist: ${EVAL_OUT_INDIST}/baseline.csv"
echo "  eval deploy: ${EVAL_OUT_DEPLOY}/baseline.csv"
echo "================================================================"
