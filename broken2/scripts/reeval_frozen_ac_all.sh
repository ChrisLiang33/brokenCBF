#!/bin/bash
# Re-evaluate AP_ADAPT-onwards teachers on DEPLOY_REALISTIC_FROZEN_AC.
# Hypothesis test confirmed: those iterations trained with a/c frozen,
# but were eval'd on DEPLOY_REALISTIC which releases a and c. Policy's
# untrained a/c heads output garbage → 0.62-0.68 fall_rate.
#
# This re-eval fixes the eval config to match training: freeze a=0.05
# and clamp c=(-0.05, -0.05). Expected fall_rate drops to ~0.10-0.15
# range based on PHIWIN_TIGHTCOR verification (0.683 → 0.113).
#
# Usage on lab box (in tmux, after TIGHTCOR_V2 finishes — heavy GPU load):
#   tmux new -s wk3reeval
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/reeval_frozen_ac_all.sh \
#     2>&1 | tee logs/reeval_frozen_ac.log

set -u
cd ~/Desktop/safety-go2/IsaacLab

# (label, ckpt_path)
runs=(
    "ap|logs/rsl_rl/cbf_go2_teacher_rma/2026-05-17_21-47-00/model_1499.pt"
    "phiwin|logs/rsl_rl/cbf_go2_teacher_rma/2026-05-17_23-50-10/model_1499.pt"
    "curr|logs/rsl_rl/cbf_go2_teacher_rma/2026-05-18_11-27-32/model_1499.pt"
    "tight|logs/rsl_rl/cbf_go2_teacher_rma/2026-05-18_13-16-52/model_1499.pt"
)

echo "================================================================"
echo "Re-eval AP_ADAPT-onwards teachers on DEPLOY_REALISTIC_FROZEN_AC"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0

for r in "${runs[@]}"; do
    IFS='|' read -r label ckpt <<< "$r"
    if [ ! -f "$ckpt" ]; then
        echo "── ${label}: MISSING ckpt: $ckpt"
        continue
    fi
    out_dir="logs/baseline_eval_${label}_frozenac"
    echo ""
    echo "── ${label} ──"
    echo "ckpt: ${ckpt}"
    echo "out:  ${out_dir}"
    echo "started at $(date '+%H:%M:%S')"

    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
        --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-v0 \
        --num_envs 64 --steps_per_config 1000 \
        --modes BR \
        --alpha_grid "0.5,2.0,3.0" \
        --phi_grid "0.5,2.0" \
        --epsilon0_grid "0.5" \
        --lambda_grid "1.0,3.0" \
        --checkpoint "$ckpt" \
        --output_dir "$out_dir" --headless \
        || echo "  ⚠ ${label} eval failed (continuing)"
done

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Per-iteration FROZEN_AC eval CSVs:"
for r in "${runs[@]}"; do
    IFS='|' read -r label _ <<< "$r"
    echo "  ${label}: logs/baseline_eval_${label}_frozenac/baseline.csv"
done
echo "================================================================"
