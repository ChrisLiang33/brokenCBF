#!/bin/bash
# V4 variance check + within-episode φ diagnostic re-run.
#
# Goals:
#   1. Confirm V4 in-dist BR-over-B2 (+0.3pp) isn't RNG noise. Re-run with
#      different seed; if BR still ≥ B2, the win is real. If BR drops below,
#      original result was variance.
#   2. Run the updated diagnose_phi_corr.py (now captures within-episode
#      Pearson(φ_t, h_t) and Pearson(φ_t, ‖L_g h‖²_t)) on the V4 ckpt to
#      see if φ's within-episode variance is tracking obstacle proximity
#      (TISSf-shape) or just random.
#
# Usage on lab box:
#   tmux new -s wk3v4check
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/v4_variance_check.sh \
#     2>&1 | tee logs/v4_variance_check.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

CKPT="logs/rsl_rl/cbf_go2_teacher_rma/2026-05-18_22-50-50/model_1499.pt"
# If the V4 ckpt path is different, find the most recent one with V4 in
# the training task name. Override CKPT inline above if needed.
if [ ! -f "$CKPT" ]; then
    echo "Configured CKPT not found; auto-finding latest V4 ckpt..."
    LATEST=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
    LATEST=${LATEST%/}
    CKPT=$(ls -1 "${LATEST}"/model_*.pt 2>/dev/null \
        | awk -F'model_|\\.pt' '{print $2, $0}' \
        | sort -n | awk '{print $2}' | tail -1)
fi
[ -f "$CKPT" ] || { echo "ERROR: no V4 ckpt found"; exit 1; }

echo "================================================================"
echo "V4 variance check + within-episode φ diagnostic"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "================================================================"

export CBF_AUX_COEF=0.0

echo ""
echo "[1/2] In-dist eval with seed=2026 (different from training seed=42)"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V4-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" --seed 2026 \
  --output_dir "logs/baseline_eval_wk3tight4_indist_seed2026" --headless

echo ""
echo "[2/2] φ diagnostic with within-episode Pearson(φ_t, h_t)"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V4-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --use_locked \
  --output diagnose_phi_corr_wk3tight4_v2.json --headless

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Variance check:"
echo "  CSV: logs/baseline_eval_wk3tight4_indist_seed2026/baseline.csv"
echo "  Compare BR j_act to original V4 in-dist eval (which had BR=0.907)."
echo "  If new BR still beats best fixed → win is real."
echo "  If new BR drops below B2 (orig 0.904) → original was variance."
echo ""
echo "Within-episode adaptation:"
echo "  JSON: diagnose_phi_corr_wk3tight4_v2.json"
echo "  Look for: within_episode_pearson.phi_vs_h.mean"
echo "    < -0.30 → STRONG TISSf-shape adaptation (φ ramps as h shrinks)"
echo "    -0.30 to -0.15 → weak negative — directionally right but soft"
echo "    -0.15 to +0.15 → no within-ep adaptation, φ varying randomly"
echo "    > +0.15 → WRONG SIGN (φ rising when robot is FAR from obstacles)"
echo "================================================================"
