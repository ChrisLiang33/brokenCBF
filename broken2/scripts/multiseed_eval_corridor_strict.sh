#!/bin/bash
# Multi-seed eval of two policies + fixed-α baselines on eval-corridor-strict
# (Isaac-CBF-Go2-Navigation-V15-Strict-v0).
#
# Naming convention (per 2026-05-22 chat-convention update):
#   policy-twostream-base  = V13.1 ckpt (trained 5/20)
#   policy-wider-dr        = V14.5 ckpt (trained 5/22)
#   B0/B1                  = fixed-α baselines run inside eval_baseline.py
#
# Per (policy, seed):
#   64 envs × 1000 sim steps × modes {B0, B1, BR} × α grid × φ grid
#   → ~5-7 min per run on RTX 5090.
#   2 policies × 3 seeds = 6 runs = ~30-45 min total.
#
# Headline metric: `navigation_success_rate` in each baseline.csv
# (termination-based: robot got within 0.5m of locked goal (6, 0)).
# Distinct from `goal_reach_rate` which uses the 1.5m wandering threshold.
#
# Usage:
#   tmux new -s ms_corr_strict
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/multiseed_eval_corridor_strict.sh \
#     2>&1 | tee logs/multiseed_corridor_strict.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

TASK="Isaac-CBF-Go2-Navigation-V15-Strict-v0"

# policy-twostream-base (V13.1)
CKPT_TWOSTREAM=${CKPT_TWOSTREAM:-logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_18-25-01/model_2499.pt}

# policy-wider-dr (V14.5) — latest 2026-05-2x run
CKPT_WIDER_DR=${CKPT_WIDER_DR:-$(ls -1 $(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/2026-05-2*/ | head -1)/model_*.pt | tail -1)}

[ -f "$CKPT_TWOSTREAM" ] || { echo "twostream ckpt not found: $CKPT_TWOSTREAM"; exit 1; }
[ -f "$CKPT_WIDER_DR" ] || { echo "wider-dr ckpt not found: $CKPT_WIDER_DR"; exit 1; }

echo "================================================================"
echo "Multi-seed eval on eval-corridor-strict"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  task    : $TASK"
echo "  ckpt-1  : $CKPT_TWOSTREAM  (policy-twostream-base)"
echo "  ckpt-2  : $CKPT_WIDER_DR   (policy-wider-dr)"
echo "================================================================"

SEEDS=(42 123 7)

run_one() {
  local policy_tag=$1
  local ckpt=$2
  local seed=$3
  echo ""
  echo "─── ${policy_tag} seed=${seed} ─── $(date '+%H:%M:%S')"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "$TASK" \
    --num_envs 64 --steps_per_config 1000 \
    --modes B0,B1,BR \
    --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
    --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
    --checkpoint "$ckpt" \
    --seed "$seed" \
    --output_dir "logs/multiseed_corridor_strict_${policy_tag}_seed${seed}" --headless
}

for seed in "${SEEDS[@]}"; do
  run_one twostream_base "$CKPT_TWOSTREAM" "$seed"
  run_one wider_dr       "$CKPT_WIDER_DR"  "$seed"
done

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Sync results back to Mac:"
echo "  rsync -avz 'chrisliang@130.64.84.163:/home/chrisliang/Desktop/safety-go2/IsaacLab/logs/multiseed_corridor_strict_*' \\"
echo "    /Users/chrisliang8/Desktop/safety-go2/data_from_lab/"
echo "================================================================"
