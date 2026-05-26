#!/bin/bash
# v3.0d — architectural skip connection: raw dynamics features bypass the
# encoder and reach π_teacher directly.
#
# Why this exists (the diagnostic story):
#   v3.0a (Layer 1 baseline): α distribution identical across tasks.
#                             Gradient bottleneck hypothesized.
#   v3.0c (obs + LHS reward): same fingerprint. State-conditioning still
#                             absent. Suspect: encoder.
#   diagnose_z (Z latent):    encoder is "alive" (12/12 dims active,
#                             healthy spread) BUT produces near-identical
#                             Z for indist and HeavyCOM:
#                               ||μ_indist − μ_HeavyCOM|| = 0.16
#                               per-dim std            ≈ 2.0
#                             Cross-task mean shift is < 10% of within-task
#                             variance. Encoder learned to encode obstacle
#                             layout (8192-D grid path, 530K params) and
#                             ignore the dynamics path (19-D, 1.3K params).
#                             Including the cbf_state we so carefully added
#                             in v3.0b.
#
# What v3.0d does:
#   _SplitCNNMLP.forward() now concatenates the raw first 19 dims of obs
#   (friction, mass, height, force, torque, tracking_err, com_offset,
#   cbf_state) to Z BEFORE π_teacher consumes it. The policy now always
#   gets a side-channel view of the env-class features, regardless of
#   what the encoder learned. π_teacher input dim goes from z_dim=12 to
#   z_dim + 19 = 31.
#
#   Tests the hypothesis directly: if state-conditional α emerges with
#   this architecture, the encoder bottleneck WAS the problem and the
#   fix is essentially zero-cost. If state-conditioning still doesn't
#   emerge, the issue is deeper (reward, optimizer, capacity).
#
# Code change for v3.0d (vs v3.0c):
#   cbf_go2_teacher_cnn.py: _SplitCNNMLP forward + π_teacher input dim.
#   Everything else unchanged from v3.0b/c (obs structure, LHS reward,
#   env caching, ALPHA_MIN=0.1, FREEZE_*=0 for φ/a/c).
#
# Eval: same 2-task sweep as v3.0a/b/c for direct comparison.
#
# Decision criterion (locked):
#   PASS:  v3.0d BR has alpha_mean OR alpha_std that differs ≥0.3
#          between indist and HeavyCOM.  (signal-of-life test)
#          AND v3.0d BR combined beats best-of-B0-sweep by ≥3pp on ≥1 task.
#   AMBIG: state-conditional signal visible in α stats but combined
#          metric not better than fixed-α. (Adaptation emerged but
#          isn't useful — tune reward / curriculum.)
#   FAIL:  alpha distribution still identical across tasks. The skip
#          connection didn't matter — bottleneck is downstream of encoder.
#          Reconsider π_teacher capacity or reward landscape.
#
# Time: ~3h training + ~25 min eval = ~3.5h wall.
#
# Sync command before launch:
#   rsync -av ~/Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_cnn.py \
#       chrisliang@130.64.84.163:Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
#
# Usage on lab box:
#   ~/Desktop/safety-go2/scripts/train_and_eval_v3_0d.sh 2>&1 | tee logs/train_and_eval_v3_0d.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v3.0d (α-only + obs + LHS reward + dynamics skip-connection)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Pre-flight: confirm v3.0b + v3.0c + v3.0d markers present
echo ""
echo "Pre-flight: confirm v3.0b+c+d changes are in place"
grep -q "_DYN_DIM = 19" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_cnn.py \
  && echo "  ✓ _DYN_DIM = 19 (v3.0b)" \
  || { echo "  ✗ _DYN_DIM still 15 — sync v3.0b changes first!"; exit 1; }
grep -q "dyn_dim_skip" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_cnn.py \
  && echo "  ✓ _SplitCNNMLP has dyn_dim_skip (v3.0d)" \
  || { echo "  ✗ dyn_dim_skip not present — sync v3.0d changes first!"; exit 1; }
grep -q "def cbf_state_b" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_observations.py \
  && echo "  ✓ cbf_state_b() (v3.0b)" \
  || { echo "  ✗ cbf_state_b not found — sync v3.0b changes first!"; exit 1; }
grep -q "def cbf_lhs_margin_penalty" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_rewards.py \
  && echo "  ✓ cbf_lhs_margin_penalty() (v3.0c)" \
  || { echo "  ✗ cbf_lhs_margin_penalty not found — sync v3.0c changes first!"; exit 1; }

# ---------- TRAINING ----------
echo ""
echo "================================================================"
echo "[1/2] TRAINING: 3000 iters, 4096 envs"
echo "      α adaptive in [0.1, 5.0]; φ=a=c=0; cbf_state in obs;"
echo "      LHS-margin reward weight = -0.1;"
echo "      raw 19-D dynamics features skip-connected into π_teacher."
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-v0 \
  --num_envs 4096 --max_iterations 3000 \
  --headless

echo ""
echo "[1/2] TRAINING done at $(date '+%H:%M:%S')"

# ---------- LOCATE CHECKPOINT ----------
echo ""
echo "================================================================"
echo "Locating most recent checkpoint..."
echo "================================================================"

LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1t "${LATEST_DIR}"/model_*.pt 2>/dev/null | head -1)

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in ${LATEST_DIR}"
    exit 1
fi

echo "Using checkpoint: $CKPT"

# ---------- HEADLINE 2-EVAL (sequential) ----------
echo ""
echo "================================================================"
echo "[2/2] HEADLINE EVAL: 2 tasks (in-dist + HeavyCOM)"
echo "      Modes: B0 (sweep 7 α values) + BR"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_TASKS=("v0" "HeavyCOM-v0")

task_tag() {
  if [ "$1" = "v0" ]; then echo "indist"; else echo "${1%-v0}"; fi
}

for TASK in "${EVAL_TASKS[@]}"; do
  TAG=$(task_tag "$TASK")
  OUT="logs/baseline_eval_v3_0d_${TAG}"
  echo ""
  echo "  >>> [$TASK] -> $OUT  ($(date '+%H:%M:%S'))"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "Isaac-CBF-Go2-${TASK}" \
    --num_envs 64 --steps_per_config 1500 \
    --modes B0,BR \
    --alpha_grid "0.1,0.5,1.0,2.0,3.0,4.0,5.0" \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --headless > "${OUT}.stdout.log" 2>&1
  echo "  >>> [$TASK] done at $(date '+%H:%M:%S')"
done

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "Eval CSVs in: logs/baseline_eval_v3_0d_{indist,HeavyCOM}/"
echo ""
echo "Decision criterion:"
echo "  PASS:  alpha_mean OR alpha_std differs by ≥0.3 between tasks"
echo "         AND BR combined beats best-of-B0 by ≥3pp on ≥1 task"
echo "  AMBIG: state-conditional in α stats but no combined-metric win"
echo "  FAIL:  alpha distribution still identical → encoder skip didn't help"
echo "================================================================"
