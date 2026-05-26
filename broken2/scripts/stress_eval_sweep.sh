#!/bin/bash
# Per-axis stress eval sweep (Wk3, 2026-05-17).
#
# Runs eval_baseline.py on all 6 stress configs for a single teacher
# checkpoint. Each config narrows every DR axis except one. The
# resulting per-axis joint_actual gap (BR − best fixed) tells us which
# axis is generating the adaptive signal in training.
#
# The φ grid is intentionally widened on the SigmaAct sweep so that the
# per-axis-tuned fixed baseline can find a φ near the Kolathaya ISSf
# optimum (φ* ∝ 1/(2σ²)) — the default eval grid {0.5, 2.0} doesn't
# cover that span.
#
# Usage on lab box (in tmux, after deflection sweep finishes):
#   tmux new -s wk3stress
#   conda activate isaaclab
#   cd ~/Desktop/safety-go2/IsaacLab
#   ~/Desktop/safety-go2/scripts/stress_eval_sweep.sh <ckpt_path> <out_label> \
#     2>&1 | tee logs/stress_eval_<out_label>.log
#
# Args:
#   $1 = absolute or repo-relative path to model_*.pt checkpoint
#   $2 = label for output dirs (e.g. "defl", "omnidefl", "lockedbest")

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <checkpoint_path> <out_label>"
    echo "  checkpoint_path: path to model_*.pt"
    echo "  out_label:       short tag for output dirs"
    exit 1
fi

CKPT="$1"
LABEL="$2"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "Per-axis stress eval sweep"
echo "Checkpoint: $CKPT"
echo "Label:      $LABEL"
echo "Started:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# (label, task_id, alpha_grid, phi_grid)
# Default grids match eval_baseline.py defaults except SigmaAct widens φ.
run_axis () {
    local axis="$1"
    local task="$2"
    local phi_grid="${3:-0.5,2.0}"
    local alpha_grid="${4:-0.5,2.0,4.0}"

    local out_dir="logs/stress_eval_${LABEL}/${axis}"
    echo ""
    echo "── ${axis} ──"
    echo "task: ${task}"
    echo "out:  ${out_dir}"
    echo "alpha_grid=${alpha_grid}  phi_grid=${phi_grid}"
    echo "started at $(date '+%H:%M:%S')"

    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
        --task "${task}" \
        --num_envs 64 --steps_per_config 1000 \
        --modes B0,B1,B2,BR \
        --alpha_grid "${alpha_grid}" \
        --phi_grid "${phi_grid}" \
        --epsilon0_grid "0.5" \
        --lambda_grid "1.0,3.0" \
        --checkpoint "$CKPT" \
        --output_dir "$out_dir" --headless \
        || echo "  ⚠ ${axis} eval failed (continuing)"
}

# Reference: all DR collapsed. Adaptive should ≈ fixed.
run_axis "narrow"       "Isaac-CBF-Go2-RMA-Stress-Narrow-v0"

# Per-axis wide variants.
run_axis "friction"     "Isaac-CBF-Go2-RMA-Stress-Friction-v0"
run_axis "com"          "Isaac-CBF-Go2-RMA-Stress-COM-v0"
# Widen φ grid on the SigmaAct axis to cover the Kolathaya floor for
# σ_act ∈ [0, 0.20] (φ* ≈ 1/(2σ²) → ~12 at σ=0.20, but the action range
# caps φ at ~5, so {0.5, 1.0, 2.0, 3.0, 5.0} suffices).
run_axis "sigma_act"    "Isaac-CBF-Go2-RMA-Stress-SigmaAct-v0" "0.5,1.0,2.0,3.0,5.0"
run_axis "radius_error" "Isaac-CBF-Go2-RMA-Stress-RadiusError-v0"
run_axis "push"         "Isaac-CBF-Go2-RMA-Stress-Push-v0"

echo ""
echo "================================================================"
echo "SWEEP DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Output dirs:"
for axis in narrow friction com sigma_act radius_error push; do
    echo "  logs/stress_eval_${LABEL}/${axis}/baseline.csv"
done
echo ""
echo "Wide-everything anchor: logs/baseline_eval_${LABEL}_indist/baseline.csv"
echo ""
echo "Analyse with:"
echo "  python3 ~/Desktop/safety-go2/scripts/parse_stress_eval.py --label ${LABEL}"
echo "================================================================"
