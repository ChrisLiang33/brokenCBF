#!/bin/bash
# v3.1 (2026-05-11): RMA-style split-encoder teacher.
#
# Architectural pivot from v3.0a-f. Instead of one monolithic encoder
# trying to compress 8211-D obs into Z (and losing the priv signal to
# the grid CNN's gradient dominance), we use two independent encoders:
#
#   priv (15)     ──► priv encoder   ──► z_priv (8)   ┐
#   grid (2x64x64) ─► grid CNN       ──► z_grid (64)  ├─► π_teacher → action
#   cbf_state (4) ──────────────────────────────────── ┘
#
# Key consequences:
#   - No pretrain phase. Priv encoder has clean input (the 15 priv numbers
#     directly), so joint RL training works — no need to compete with the
#     grid CNN for shared bottleneck capacity. This is RMA Phase 1.
#   - No encoder freeze. Both encoders train end-to-end with PPO.
#   - aux_loss + velocity_along_cmd reward stay enabled (orthogonal to
#     architecture — they address sparse-gradient and saturation).
#
# Decision criterion (locked):
#   PASS:  Pearson(α, friction) OR Pearson(α, |com_offset|) > 0.20
#          AND BR combined beats best-of-B0 by ≥3pp on ≥1 task
#   AMBIG: alpha-corrs improve but combined metric ties
#   FAIL:  α saturated OR DR corrs <0.10 → re-think (see TODO)
#
# Time: ~3h PPO + ~25 min eval = ~3.5h. Faster than v3.0e/f (no pretrain).
#
# Sync before launch (from local Mac):
#   rsync -av ~/Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/{cbf_go2_teacher_rma.py,__init__.py} \
#       chrisliang@130.64.84.163:Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
#   rsync -av ~/Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/agents/rsl_rl_ppo_rma_cfg.py \
#       chrisliang@130.64.84.163:Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/agents/
#   rsync -av ~/Desktop/safety-go2/scripts/train_and_eval_v3_1.sh \
#       chrisliang@130.64.84.163:Desktop/safety-go2/scripts/
#
# Usage on lab box (in tmux):
#   tmux new -s v31
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_v3_1.sh 2>&1 | tee logs/train_and_eval_v3_1.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

# v3.1 (2026-05-11) tuning: AUX_COEF reduced 0.02 → 0.005 based on v3.0f
# diagnostic. At 0.02, aux loss was strong enough to homogenize the policy
# toward α=5 (slack always positive → aux loss silent → no per-state signal).
# At 0.005, the dense gradient on α persists but doesn't overwhelm the
# safety penalty + state-conditioning emerging from z_priv variation.
# velocity_along_cmd reward also dropped to weight=0.0 in env_cfg (was 0.2)
# — reinforced rather than counteracted α-saturation.
AUX_COEF=0.005

echo "================================================================"
echo "v3.1 (RMA split-encoder, joint RL, no pretrain) pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Pre-flight: confirm v3.1 files are in place
echo ""
echo "Pre-flight checks"
test -f source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py \
  && echo "  ✓ cbf_go2_teacher_rma.py present" \
  || { echo "  ✗ cbf_go2_teacher_rma.py missing — sync first"; exit 1; }
test -f source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/agents/rsl_rl_ppo_rma_cfg.py \
  && echo "  ✓ rsl_rl_ppo_rma_cfg.py present" \
  || { echo "  ✗ rsl_rl_ppo_rma_cfg.py missing — sync first"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ Isaac-CBF-Go2-RMA-v0 task registered" \
  || { echo "  ✗ RMA task not registered in __init__.py"; exit 1; }
grep -q "def velocity_along_cmd" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_rewards.py \
  && echo "  ✓ velocity_along_cmd reward (inherited from v3.0f)" \
  || { echo "  ✗ velocity_along_cmd reward missing"; exit 1; }

# ---------- PHASE 1: PPO TRAINING (no pretrain) ----------
echo ""
echo "================================================================"
echo "[1/2] PPO TRAINING: 3000 iters, 4096 envs, JOINT RL"
echo "      Split encoders + CBF_AUX_COEF=$AUX_COEF (v_along_cmd weight=0.0)"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=$AUX_COEF
# Make sure we don't accidentally inherit v3.0e's pretrain-load env vars.
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-v0 \
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
echo "[2/2] HEADLINE EVAL + linear probe + alpha-corr"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

# Eval needs aux coef set to 0 to bypass the monkey-patch path (no aux
# loss at eval time).
export CBF_AUX_COEF=0.0

EVAL_TASKS=("v0" "HeavyCOM-v0")
task_tag() {
  if [ "$1" = "v0" ]; then echo "indist"; else echo "${1%-v0}"; fi
}

for TASK in "${EVAL_TASKS[@]}"; do
  TAG=$(task_tag "$TASK")
  OUT="logs/baseline_eval_v3_1_${TAG}"
  echo ""
  echo "  >>> [Isaac-CBF-Go2-RMA-${TASK}] -> $OUT  ($(date '+%H:%M:%S'))"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "Isaac-CBF-Go2-RMA-${TASK}" \
    --num_envs 64 --steps_per_config 1500 \
    --modes B0,BR \
    --alpha_grid "0.1,0.5,1.0,2.0,3.0,4.0,5.0" \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --headless > "${OUT}.stdout.log" 2>&1
  echo "  >>> [Isaac-CBF-Go2-RMA-${TASK}] done at $(date '+%H:%M:%S')"
done

echo ""
echo "Running alpha-correlation diagnostic..."
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output diagnose_alpha_corr_v3_1.json \
  --headless

echo ""
echo "Running linear probe Z_priv → priv features..."
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_v3_1.json \
  --headless

echo ""
echo "Running Z diagnostic..."
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_z.py \
  --task Isaac-CBF-Go2-RMA-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output diagnose_z_v3_1.json \
  --headless

unset CBF_AUX_COEF

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint:        $CKPT"
echo "Eval CSVs:         logs/baseline_eval_v3_1_{indist,HeavyCOM}/"
echo "α-corr JSON:       diagnose_alpha_corr_v3_1.json"
echo "Linear probe JSON: probe_z_linear_v3_1.json"
echo "Z diagnostic JSON: diagnose_z_v3_1.json"
echo ""
echo "Decision criterion:"
echo "  PASS:  Pearson(α, friction) OR Pearson(α, |com_offset|) > 0.20"
echo "         AND BR combined beats best-of-B0 by ≥3pp on ≥1 task"
echo "  AMBIG: corrs improve but combined metric doesn't beat fixed-α"
echo "  FAIL:  α saturated OR all DR corrs <0.10"
echo "================================================================"
