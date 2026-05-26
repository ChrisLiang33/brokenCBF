#!/usr/bin/env bash
# Phase 5 full pipeline -- fire and forget.
# Chains: teacher train -> per-channel sweep -> jitter diag ->
#         student train -> deployment substitution eval -> summary.
#
# Total wall-clock: ~40 min on RTX 5090.
# Each stage's stdout/stderr goes to a separate log under
# `phase5_pipeline_logs/`. set -e bails the chain on any failure.
#
# Run on labbox:
#     cd ~/Desktop/cbf_rl_mvp/go2
#     bash phase5_full_pipeline.sh
#
# Or in background with nohup so you can disconnect:
#     nohup bash phase5_full_pipeline.sh > pipeline.out 2>&1 &
#     # then `tail -f pipeline.out` to monitor.

set -euo pipefail
cd "$(dirname "$0")"

LOCO="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt"
ISAACLAB="$HOME/IsaacLab/isaaclab.sh"
LOGS="phase5_pipeline_logs"
mkdir -p "$LOGS"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
banner() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "  start: $(stamp)"
    echo "================================================================"
}

START_TIME=$(date +%s)

banner "[1/5] Teacher PPO training  (~15 min,  256 envs,  1500 iters)"
"$ISAACLAB" -p phase5_train_teacher.py \
    --checkpoint "$LOCO" \
    --num_envs 256 --max_iterations 1500 --headless \
    > "$LOGS/1_teacher.log" 2>&1
echo "  done: $(stamp)"

banner "[2/5] Per-channel response sweep  (~7 min)"
"$ISAACLAB" -p phase5_per_channel_sweep.py \
    --checkpoint "$LOCO" \
    --policy_checkpoint phase5_teacher_outputs/rsl_rl/model_final.pt \
    --num_envs 256 --headless \
    > "$LOGS/2_per_channel.log" 2>&1
echo "  done: $(stamp)"

banner "[3/5] Action smoothness diagnostic  (~3 min)"
"$ISAACLAB" -p phase5_action_jitter_diag.py \
    --checkpoint "$LOCO" \
    --policy_checkpoint phase5_teacher_outputs/rsl_rl/model_final.pt \
    --num_envs 256 --headless \
    > "$LOGS/3_jitter.log" 2>&1
echo "  done: $(stamp)"

banner "[4/5] Student training  (~6 min,  collect 2000 steps + train MLP)"
"$ISAACLAB" -p phase5_train_student.py \
    --checkpoint "$LOCO" \
    --policy_checkpoint phase5_teacher_outputs/rsl_rl/model_final.pt \
    --num_envs 256 --collect_steps 2000 --headless \
    > "$LOGS/4_student.log" 2>&1
echo "  done: $(stamp)"

banner "[5/5] Deployment substitution eval  (~5 min)"
"$ISAACLAB" -p phase5_deploy_eval.py \
    --checkpoint "$LOCO" \
    --policy_checkpoint phase5_teacher_outputs/rsl_rl/model_final.pt \
    --student_checkpoint phase5_student_outputs/student.pt \
    --num_envs 256 --headless \
    > "$LOGS/5_deploy.log" 2>&1
echo "  done: $(stamp)"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "================================================================"
echo "  PIPELINE COMPLETE  --  elapsed $(($ELAPSED / 60))m $(($ELAPSED % 60))s"
echo "  $(stamp)"
echo "================================================================"
echo ""

# Final consolidated summary -- BR vs B2 vs all the diagnostics
python3 phase5_final_summary.py || echo "  (summary script failed, see logs)"

echo ""
echo "Per-stage stdout in $LOGS/*.log"
echo "Pull artifacts with: ./scripts/sync_pull.sh"
