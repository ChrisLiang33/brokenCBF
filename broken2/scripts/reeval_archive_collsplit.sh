#!/bin/bash
# Re-evaluate every iteration's checkpoint on its corresponding env using
# the patched eval_baseline.py (collision split into actual vs perceived-
# only). The training reward was always grounded in true contact; only the
# eval metric was conflated. Past "regression" iterations might look very
# different in joint_actual.
#
# Iterations (PUSH branch):
#   push      Isaac-CBF-Go2-RMA-Layer3-Push-v0           2026-05-16_02-07
#   push_a    Isaac-CBF-Go2-RMA-Layer3-Push-A-v0         2026-05-16_10-04
#   push_a_c  Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0       2026-05-16_12-45
#   phitax    Isaac-CBF-Go2-RMA-Layer3-Push-A-C-PhiTax-v0  2026-05-16_14-56
#   tiltdr    Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0  2026-05-16_17-52
#   (aclamp already done — output at logs/baseline_eval_aclamp_collsplit)
#
# Usage:
#   conda activate isaaclab
#   bash ~/Desktop/safety-go2/scripts/reeval_archive_collsplit.sh \
#       2>&1 | tee ~/Desktop/safety-go2/IsaacLab/logs/reeval_archive_collsplit.log

set -u

cd ~/Desktop/safety-go2/IsaacLab

# (name, gym_task, ckpt_dir_timestamp)
runs=(
    "push|Isaac-CBF-Go2-RMA-Layer3-Push-v0|2026-05-16_02-07-29"
    "push_a|Isaac-CBF-Go2-RMA-Layer3-Push-A-v0|2026-05-16_10-04-27"
    "push_a_c|Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0|2026-05-16_12-45-39"
    "phitax|Isaac-CBF-Go2-RMA-Layer3-Push-A-C-PhiTax-v0|2026-05-16_14-56-27"
    "tiltdr|Isaac-CBF-Go2-RMA-Layer3-Push-A-C-TiltDR-v0|2026-05-16_17-52-25"
)

echo "================================================================"
echo "Archive re-eval with collision split (actual vs perceived-only)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

for r in "${runs[@]}"; do
    IFS='|' read -r name task ts <<< "$r"
    ckpt_dir="logs/rsl_rl/cbf_go2_teacher_rma/${ts}"
    # Pick the highest-numbered checkpoint in the directory.
    ckpt=$(ls -1 "${ckpt_dir}"/model_*.pt 2>/dev/null \
        | awk -F'model_|\\.pt' '{print $2, $0}' \
        | sort -n \
        | awk '{print $2}' \
        | tail -1)
    if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
        echo "── ${name}: MISSING ckpt under ${ckpt_dir}"
        continue
    fi

    out_dir="logs/baseline_eval_${name}_collsplit"
    echo ""
    echo "── ${name} ──"
    echo "task: ${task}"
    echo "ckpt: ${ckpt}"
    echo "out:  ${out_dir}"

    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
        --task "$task" \
        --num_envs 64 --steps_per_config 1000 \
        --modes B0,B1,B2,BR \
        --alpha_grid "0.5,2.0,4.0" \
        --phi_grid "0.5,2.0" \
        --epsilon0_grid "0.5" \
        --lambda_grid "1.0,3.0" \
        --checkpoint "$ckpt" \
        --output_dir "$out_dir" \
        --headless \
        || echo "  ⚠ eval failed for ${name} (continuing)"
done

echo ""
echo "================================================================"
echo "DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""
echo "joint_actual = (1 - collision_rate_actual) * (1 - fall_rate) * goal_reach_rate"
echo ""
echo "Per-iteration CSVs:"
for r in "${runs[@]}"; do
    IFS='|' read -r name task ts <<< "$r"
    echo "  ${name}: logs/baseline_eval_${name}_collsplit/baseline.csv"
done
echo "  aclamp (already done): logs/baseline_eval_aclamp_collsplit/baseline.csv"
