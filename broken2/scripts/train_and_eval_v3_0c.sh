#!/bin/bash
# v3.0c — Layer 1 + obs enrichment + dense LHS-margin reward.
#
# Stack-up of fixes to the QP-idle gradient bottleneck identified in
# the v3.0a CBF math review (cbf_math_refresher.html §1-10):
#
#   v3.0a: Layer 1 baseline (α-only, φ=a=c=0). Reproduces sparse-gradient
#          symptom in clean form.
#   v3.0b: + h, L_g h·u_des, ‖L_g h‖², slack into the policy obs.
#          Lets the policy SEE the constraint margin so it can output
#          α(state) without having to learn h-from-grid first.
#          Doesn't add a new gradient pathway — fixes function class only.
#   v3.0c: + dense LHS-margin reward (this script). Keeps v3.0b's obs
#          AND adds reward = -λ · softplus(-slack). Now every step has
#          nonzero advantage signal on α via PPO's policy-gradient,
#          including idle steps. This is the "always-on α gradient"
#          the math review pointed at.
#
# Code change for v3.0c (vs v3.0b):
#   cbf_go2_rewards.py    : new cbf_lhs_margin_penalty(env) function
#   cbf_go2_env_cfg.py    : new RewTerm `cbf_lhs_margin` weight = -0.1
#   ALL OTHER FILES UNCHANGED from v3.0b (env caching, obs func,
#   _DYN_DIM=19 — same Layer 1 α-only setup).
#
# Reward weight tuning note:
#   Starting at -0.1 (small). softplus(-slack) saturates at |slack|;
#   typical |slack| in active zone is O(1), so per-step penalty is O(0.1).
#   Compared to other reward magnitudes (-500 fall, -1.0 stuck, -0.5
#   proximity, -0.1 u_safe_dev), this is in the "shaping nudge" range
#   rather than dominant. Crank to -0.5 if effect is invisible; back off
#   to -0.05 if it overrides task reward and stops the robot from moving.
#
# The math one more time:
#   slack       = L_g h · u_des - rhs                    (LHS at u=u_des)
#   r_aux       = -λ · softplus(-slack)
#   ∂r_aux/∂α   = λ · sigmoid(-slack) · (h - c)         (always nonzero)
#                                  (>0 in idle, ≈1 in active, etc.)
#
#   In Layer 1, c=0, so ∂r_aux/∂α = λ · sigmoid(-slack) · h. Whenever
#   the robot has positive clearance (h > 0), bigger α improves r_aux.
#   Task reward (forward motion) provides the opposing pressure. The
#   tension produces state-conditional α — IF PPO can find it.
#
# Eval: same 2-task sweep as v3.0a/b for direct comparison.
#
# Decision criterion (locked):
#   PASS  : v3.0c BR beats best-of-B0-sweep by ≥3pp on ≥1 task
#           AND v3.0c BR beats v3.0b BR by ≥3pp on the same task
#           (richer reward gained more than richer obs alone).
#   AMBIG : v3.0c ≈ v3.0b   → reward shaping didn't add over obs.
#           Either obs was the binding fix, or LHS reward weight
#           was too small / too large.
#   FAIL  : v3.0c worse than v3.0a/b   → reward shaping is harmful;
#           either weight is wrong or it's pushing α the wrong way.
#
# Time: ~3h training + ~25 min eval = ~3.5h wall.
#
# PREREQUISITE: Run AFTER v3.0a's eval is complete AND v3.0b has either
# launched or is being skipped. If you're running v3.0c instead of v3.0b,
# you still need the v3.0b file changes synced (obs structure with
# _DYN_DIM=19, etc.) — v3.0c builds on v3.0b's env/obs/cnn changes.
#
# Sync command before launch (everything from both v3.0b and v3.0c):
#   rsync -av ~/Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/{cbf_go2_env.py,cbf_go2_observations.py,cbf_go2_env_cfg.py,cbf_go2_teacher_cnn.py,cbf_go2_rewards.py} \
#       chrisliang@130.64.84.163:Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
#
# Usage on lab box:
#   ~/Desktop/safety-go2/scripts/train_and_eval_v3_0c.sh 2>&1 | tee logs/train_and_eval_v3_0c.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v3.0c (α-only + CBF state obs + LHS-margin reward) pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Pre-flight: confirm v3.0b + v3.0c changes are on this machine
echo ""
echo "Pre-flight: confirm v3.0b + v3.0c changes are in place"
grep -q "_DYN_DIM = 19" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_cnn.py \
  && echo "  ✓ _DYN_DIM = 19 (v3.0b, teacher_cnn)" \
  || { echo "  ✗ _DYN_DIM still 15 — sync v3.0b+c changes first!"; exit 1; }
grep -q "def cbf_state_b" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_observations.py \
  && echo "  ✓ cbf_state_b() (v3.0b, observations)" \
  || { echo "  ✗ cbf_state_b not found — sync v3.0b+c changes first!"; exit 1; }
grep -q "cbf_state = ObsTerm" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ cbf_state ObsTerm (v3.0b, env_cfg)" \
  || { echo "  ✗ cbf_state ObsTerm not registered — sync v3.0b+c changes first!"; exit 1; }
grep -q "def cbf_lhs_margin_penalty" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_rewards.py \
  && echo "  ✓ cbf_lhs_margin_penalty() (v3.0c, rewards)" \
  || { echo "  ✗ cbf_lhs_margin_penalty not found — sync v3.0c changes first!"; exit 1; }
grep -q "cbf_lhs_margin = RewTerm" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ cbf_lhs_margin RewTerm (v3.0c, env_cfg)" \
  || { echo "  ✗ cbf_lhs_margin RewTerm not registered — sync v3.0c changes first!"; exit 1; }
grep -q "last_h_for_obs" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py \
  && echo "  ✓ env caches CBF state per env (v3.0b, env)" \
  || { echo "  ✗ env caching not in place — sync v3.0b+c changes first!"; exit 1; }

# ---------- TRAINING ----------
echo ""
echo "================================================================"
echo "[1/2] TRAINING: 3000 iters, 4096 envs"
echo "      α adaptive in [0.1, 5.0]; φ=a=c=0; CBF state in obs;"
echo "      LHS-margin reward weight = -0.1 (dense penalty on low slack)."
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

# ---------- HEADLINE 2-EVAL (sequential, mirrors v3.0a/b for comparison) ----------
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
  OUT="logs/baseline_eval_v3_0c_${TAG}"
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
echo "Eval CSVs in: logs/baseline_eval_v3_0c_{indist,HeavyCOM}/"
echo ""
echo "Decision criterion:"
echo "  PASS:  BR combined beats best-of-B0 by ≥3pp on ≥1 task"
echo "         AND v3.0c BR beats v3.0b BR by ≥3pp on the same task"
echo "  AMBIG: v3.0c ≈ v3.0b — reward shaping not adding over obs"
echo "         (try weight=-0.5; or accept obs-only fix)"
echo "  FAIL:  v3.0c worse than v3.0a/b — reward weight bad"
echo "================================================================"
