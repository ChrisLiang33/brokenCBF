#!/bin/bash
# α-basin reachability diagnostic.
#
# Hypothesis: PPO's collision penalty hill-climbs toward HIGH α, even
# though the best fixed baseline lives at α=0.5. Test by running BR with
# its α dimension forced to a fixed low value while letting the policy
# choose (φ, a, c). If Bf-α=0.5 beats vanilla BR, the policy genuinely
# learned the wrong α basin and α-control is load-bearing for the gap.
#
# Sweep: α_target ∈ {0.5, 1.0, 2.0}. Compare against vanilla BR (joint
# 0.150 from the TILTDR eval).
#
# Usage on lab box:
#   bash ~/Desktop/safety-go2/scripts/diag_alpha_basin.sh \
#       /home/chrisliang/Desktop/safety-go2/IsaacLab/logs/rsl_rl/cbf_go2_teacher_rma/2026-05-16_17-52-25/model_1499.pt \
#       2>&1 | tee ~/Desktop/safety-go2/IsaacLab/logs/diag_alpha_basin.log

set -e

CKPT="${1:?Usage: $0 <checkpoint_path>}"
TASK="${TASK:-Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0}"
NUM_ENVS="${NUM_ENVS:-64}"
STEPS="${STEPS:-1000}"

cd ~/Desktop/safety-go2/IsaacLab

if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found: $CKPT"
    exit 1
fi

echo "================================================================"
echo "α-basin diagnostic"
echo "Task:       $TASK"
echo "Checkpoint: $CKPT"
echo "Num envs:   $NUM_ENVS, steps per config: $STEPS"
echo "Started:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Vanilla BR + B1(α=0.5, φ=2.0) included once for reference in each run.
for ALPHA_TARGET in 0.5 1.0 2.0; do
    echo ""
    echo "─── Bf-α = $ALPHA_TARGET ───"
    OUT_DIR="logs/diag_alpha_basin_${ALPHA_TARGET}"
    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
      --task "$TASK" \
      --num_envs "$NUM_ENVS" --steps_per_config "$STEPS" \
      --modes Bf-alpha \
      --bf_alpha_target "$ALPHA_TARGET" \
      --checkpoint "$CKPT" \
      --output_dir "$OUT_DIR" \
      --headless
    echo "  → $OUT_DIR/baseline.csv"
done

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo "Read collision/fall/goal from each CSV and compute joint success:"
echo "  joint = (1 - collision) * (1 - fall) * goal_reach"
echo ""
echo "Reference points:"
echo "  BR (TILTDR vanilla): collision=0.824, fall=0.054, goal=0.905 → joint=0.150"
echo "  B1 α=0.5 φ=2.0:      collision=0.591, fall=0.015, goal=0.833 → joint=0.336"
echo ""
echo "Interpretation:"
echo "  - If Bf-α=0.5 joint >> 0.150  → α basin reachability IS the bottleneck."
echo "  - If Bf-α=0.5 joint ≈ 0.150  → policy's learned φ/a/c are also bad."
echo "  - If Bf-α=0.5 joint ≈ 0.336  → α is the WHOLE story; just clamp it."
