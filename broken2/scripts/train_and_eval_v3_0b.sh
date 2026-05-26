#!/bin/bash
# v3.0b — Layer 1 + CBF state in observation (LHS-as-input experiment).
#
# Why this exists:
#   v3.0a Layer 1 isolated α as the only adaptive parameter to test
#   whether single-param state-conditional adaptation is learnable. The
#   CBF math review (cbf_math_refresher.html §6-9) showed why even
#   single-α might fail: ∂u_safe/∂α = 0 whenever the QP is idle
#   (slack ≥ 0), which is most steps on calm in-dist. The policy gets
#   no gradient on α from those steps and converges to the on-average
#   best α — a single fixed magnitude regardless of state.
#
#   v3.0b tests one fix: expose the CBF constraint geometry directly
#   to the policy as observations. With h, L_g h·u_des, ‖L_g h‖², and
#   lagged slack in the obs, the policy can compute slack at any
#   candidate α directly — adapting α(state) without needing the QP
#   gradient to flow.
#
# Code change for v3.0b (vs v3.0a):
#   cbf_go2_env.py        : caches h, L_g h, slack per env in _cbf_filter
#   cbf_go2_observations  : new cbf_state_b() returns (N, 4) tensor
#   cbf_go2_env_cfg.py    : new ObsTerm `cbf_state` in TeacherPrivCfg
#   cbf_go2_teacher_cnn.py: _DYN_DIM 15 → 19 (Linear(19→64) instead of 15→64)
#   ALPHA / FREEZE constants UNCHANGED from v3.0a:
#     ALPHA_MIN = 0.1
#     FREEZE_ALPHA_VALUE = None   (adaptive, range [0.1, 5.0])
#     FREEZE_PHI_VALUE   = 0.0    (term off, axis off)
#     FREEZE_A_VALUE     = 0.0    (term off, axis off)
#     FREEZE_C_VALUE     = 0.0    (term off, axis off)
#
# The obs-space change means this CANNOT use a v3.0a checkpoint — must
# train fresh. Trained policy will not load on the v3.0a env (different
# obs dimension) and vice versa.
#
# Eval: same 2-task sweep as v3.0a so we can directly compare:
#   - in-dist (Isaac-CBF-Go2-v0)
#   - HeavyCOM
#   - Modes: B0 sweep {0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0} + BR
#
# Decision criterion (locked):
#   PASS: BR combined beats best-of-B0-sweep by ≥3pp on ≥1 task
#         AND v3.0b BR beats v3.0a BR by ≥3pp on the same task
#         (both conditions: not just better than fixed, but adaptation
#          gained more from richer obs than the gradient bottleneck cost)
#   FAIL otherwise — observation enrichment isn't enough; the
#         bottleneck is somewhere else (reward, architecture, data
#         distribution).
#
# Time: ~3h training + ~25 min eval = ~3.5h wall.
#
# PREREQUISITE: Run AFTER v3.0a's training + eval is complete.
# The lab needs to receive the obs-space changes (env, observations,
# env_cfg, teacher_cnn) BEFORE this launches but AFTER v3.0a eval has
# finished its baseline.csv generation.
#
# Sync command before launch:
#   rsync -av ~/Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/{cbf_go2_env.py,cbf_go2_observations.py,cbf_go2_env_cfg.py,cbf_go2_teacher_cnn.py} \
#       chrisliang@130.64.84.163:Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
#
# Usage on lab box:
#   ~/Desktop/safety-go2/scripts/train_and_eval_v3_0b.sh 2>&1 | tee logs/train_and_eval_v3_0b.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v3.0b (α-only + CBF state in obs) pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Sanity check: confirm v3.0b obs structure is on this machine
echo ""
echo "Pre-flight: confirm v3.0b obs structure is in place"
grep -q "_DYN_DIM = 19" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_cnn.py \
  && echo "  ✓ _DYN_DIM = 19 (teacher_cnn)" \
  || { echo "  ✗ _DYN_DIM still 15 — sync v3.0b changes first!"; exit 1; }
grep -q "def cbf_state_b" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_observations.py \
  && echo "  ✓ cbf_state_b() registered (observations)" \
  || { echo "  ✗ cbf_state_b not found — sync v3.0b changes first!"; exit 1; }
grep -q "cbf_state = ObsTerm" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ cbf_state ObsTerm (env_cfg)" \
  || { echo "  ✗ cbf_state ObsTerm not registered — sync v3.0b changes first!"; exit 1; }
grep -q "last_h_for_obs" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py \
  && echo "  ✓ env caches CBF state per env (env)" \
  || { echo "  ✗ env caching not in place — sync v3.0b changes first!"; exit 1; }

# ---------- TRAINING ----------
echo ""
echo "================================================================"
echo "[1/2] TRAINING: 3000 iters, 4096 envs"
echo "      α adaptive in [0.1, 5.0]; φ=a=c=0; CBF state in obs"
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

# ---------- HEADLINE 2-EVAL (sequential, mirrors v3.0a for direct comparison) ----------
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
  OUT="logs/baseline_eval_v3_0b_${TAG}"
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
echo "Eval CSVs in: logs/baseline_eval_v3_0b_{indist,HeavyCOM}/"
echo ""
echo "Decision criterion:"
echo "  PASS: BR combined beats best-of-B0-sweep by ≥3pp on ≥1 task"
echo "        AND v3.0b BR beats v3.0a BR by ≥3pp on the same task"
echo "  FAIL otherwise — observation enrichment isn't enough."
echo "================================================================"
