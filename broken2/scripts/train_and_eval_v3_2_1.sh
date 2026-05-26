#!/bin/bash
# v3.2.1 (2026-05-11): RMA-style architecture (from v3.1) +
# adversarial-heavy planner mix + AUX_COEF=0.
#
# The bet: v3.0/v3.1 suffered from sparse CBF-active states in training
# data — most cooperative planners avoided obstacles, so the CBF rarely
# fired. v3.0f's aux loss tried to extract per-step gradient signal from
# this sparse-event data, but instead pre-saturated the policy.
#
# v3.2.1 fixes the data, not the loss:
#   - 45% adversarial planner episodes (was 5%) — robot is commanded
#     toward obstacles ~half the time, CBF stress-tests dense.
#   - AUX_COEF=0 — no monkey-patched PPO. The existing reward stack
#     (collision -100, base_contact -500, stuck -1, proximity -0.5,
#     cbf_lhs_margin -0.1) gets dense natural signal because the data
#     itself exercises the CBF in nearly every episode.
#   - Inherits v3.1 split encoders + cbf_state-excluded policy input.
#
# Sync before launch (from local Mac):
#   rsync -av ~/Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/{cbf_go2_env_cfg.py,__init__.py} \
#       chrisliang@130.64.84.163:Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
#   rsync -av ~/Desktop/safety-go2/scripts/train_and_eval_v3_2_1.sh \
#       chrisliang@130.64.84.163:Desktop/safety-go2/scripts/
#
# Usage on lab box (in tmux):
#   tmux new -s v32
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_v3_2_1.sh 2>&1 | tee logs/train_and_eval_v3_2_1.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

# v3.2.1: aux loss OFF. Adversarial planner provides the dense signal.
AUX_COEF=0.0

echo "================================================================"
echo "v3.2.1 (RMA split-encoder + 45% adversarial planner) pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Pre-flight: confirm v3.2.1 changes are in place
echo ""
echo "Pre-flight checks"
test -f source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py \
  && echo "  ✓ cbf_go2_teacher_rma.py present" \
  || { echo "  ✗ cbf_go2_teacher_rma.py missing — sync first"; exit 1; }
test -f source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/agents/rsl_rl_ppo_rma_cfg.py \
  && echo "  ✓ rsl_rl_ppo_rma_cfg.py present" \
  || { echo "  ✗ rsl_rl_ppo_rma_cfg.py missing — sync first"; exit 1; }
grep -q "CbfGo2EnvCfg_HARD_PLANNER" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ CbfGo2EnvCfg_HARD_PLANNER config present" \
  || { echo "  ✗ HARD_PLANNER config missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-V32-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ Isaac-CBF-Go2-RMA-V32-v0 task registered" \
  || { echo "  ✗ V32 task not registered in __init__.py"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/2] PPO TRAINING: 3000 iters, 4096 envs"
echo "      RMA split encoders + 45% adversarial planner mix + AUX_COEF=0"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

# Aux loss OFF by setting CBF_AUX_COEF=0 — the monkey-patch in __init__.py
# only activates when CBF_AUX_COEF>0, so the patched PPO.update is NOT
# loaded. Vanilla rsl_rl PPO is used.
export CBF_AUX_COEF=$AUX_COEF

# Defensive: don't accidentally inherit v3.0e/f's pretrain-load env vars.
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-V32-v0 \
  --num_envs 4096 --max_iterations 3000 \
  --headless

unset CBF_AUX_COEF

echo ""
echo "[1/2] PPO TRAINING done at $(date '+%H:%M:%S')"

# ---------- LOCATE CHECKPOINT ----------
LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1t "${LATEST_DIR}"/model_*.pt 2>/dev/null | head -1)

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in ${LATEST_DIR}"
    exit 1
fi
echo "Using checkpoint: $CKPT"

# ---------- PHASE 2: EVAL + DIAGNOSTICS ----------
echo ""
echo "================================================================"
echo "[2/2] HEADLINE EVAL + linear probe + alpha-corr + Z diag"
echo "      Eval on REGULAR planner mix (5% adversarial — same as v3.0/v3.1)"
echo "      so headline numbers compare directly to prior runs."
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

# Aux coef irrelevant for eval but set to 0 to skip the patch path.
export CBF_AUX_COEF=0.0

# Eval on standard task (5% adversarial) — same eval distribution as v3.0/v3.1
# so combined-metric comparison to prior runs is fair. Adversarial mix is
# training-only.
EVAL_TASKS=("RMA-v0" "RMA-HeavyCOM-v0")
task_tag() {
  if [ "$1" = "RMA-v0" ]; then echo "indist"; else echo "${1#RMA-}"; tag=${tag%-v0}; fi
}

for TASK in "${EVAL_TASKS[@]}"; do
  TAG=$(echo "$TASK" | sed -E 's/^RMA-//; s/-v0$//; s/^v0$/indist/')
  OUT="logs/baseline_eval_v3_2_1_${TAG}"
  echo ""
  echo "  >>> [Isaac-CBF-Go2-${TASK}] -> $OUT  ($(date '+%H:%M:%S'))"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "Isaac-CBF-Go2-${TASK}" \
    --num_envs 64 --steps_per_config 1500 \
    --modes B0,BR \
    --alpha_grid "0.1,0.5,1.0,2.0,3.0,4.0,5.0" \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --headless > "${OUT}.stdout.log" 2>&1
  echo "  >>> [Isaac-CBF-Go2-${TASK}] done at $(date '+%H:%M:%S')"
done

echo ""
echo "Running alpha-correlation diagnostic..."
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output diagnose_alpha_corr_v3_2_1.json \
  --headless

echo ""
echo "Running linear probe Z_priv → priv features..."
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_v3_2_1.json \
  --headless

echo ""
echo "Running Z diagnostic..."
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_z.py \
  --task Isaac-CBF-Go2-RMA-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output diagnose_z_v3_2_1.json \
  --headless

unset CBF_AUX_COEF

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint:        $CKPT"
echo "Eval CSVs:         logs/baseline_eval_v3_2_1_{indist,HeavyCOM}/"
echo "α-corr JSON:       diagnose_alpha_corr_v3_2_1.json"
echo "Linear probe JSON: probe_z_linear_v3_2_1.json"
echo "Z diagnostic JSON: diagnose_z_v3_2_1.json"
echo ""
echo "Decision criterion (same as v3.0/v3.1):"
echo "  PASS:  Pearson(α, friction) OR Pearson(α, |com_offset|) > 0.20"
echo "         AND BR combined beats best-of-B0 by ≥3pp on ≥1 task"
echo "  AMBIG: corrs improve but combined metric doesn't beat fixed-α"
echo "  FAIL:  all DR corrs <0.10, OR α still saturated"
echo "================================================================"
