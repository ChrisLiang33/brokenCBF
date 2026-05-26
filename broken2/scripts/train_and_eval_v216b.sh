#!/bin/bash
# v2.16b — pair-isolation B: a + c adaptive, α + φ FROZEN at v2.15 BR means.
#
# Why this exists:
#   v2.15 Bf-X showed a / c slots looked load-bearing (Bf-a +35.9pp on
#   NoisyPerception; Bf-c +14.8pp on RadiusError) — bigger margins than
#   α / φ. v2.16a tested whether α + φ become useful when isolated; the
#   answer was no (BR ≈ Bf-α=3.52 and BR ≈ Bf-φ=2.26 in 2-param mode,
#   plus φ overcompensated and HighActuationNoise regressed). v2.16b is
#   the symmetric check: with α + φ frozen at v2.15 BR's typical values,
#   can a + c learn meaningful state-conditional behavior in 2-param
#   adaptive mode?
#
# Code change for v2.16b (only):
#   cbf_go2_env.py module-level:
#     FREEZE_ALPHA_VALUE = 3.52   (frozen at v2.15 BR cbf_alpha_mean)
#     FREEZE_PHI_VALUE   = 2.26   (frozen at v2.15 BR cbf_phi_mean)
#     FREEZE_A_VALUE     = None   (adaptive)
#     FREEZE_C_VALUE     = None   (adaptive)
#   Auto-coupled DR disable (in env.__init__): with φ frozen, actuation-
#   noise DR is auto-disabled (φ can't adapt to it). a/c slots' DR axes
#   (perception bias, radius error) STAY ACTIVE — those are the slots
#   under test. Everything else from v2.15: REWARD-3, ALPHA_MIN=1.0,
#   C_MIN=0.10, per-episode φ lock (no-op since φ frozen), 6K iters,
#   4096 envs, cylinder-only pool, 6-planner mix.
#
# Reduced eval scope (no 10-eval; pair-isolation needs less coverage):
#   Headline 3 tasks (sequential):
#     v0 (in-dist), NoisyPerception (a axis), RadiusError (c axis)
#   Bf-X 2 ablations on in-dist (sequential):
#     Bf-a target=0.015 (v2.15 BR a-mean) — does adaptive a beat fixed-at-mean?
#     Bf-c eval-target=0.02 → env c=0.118 (compensated for env range).
#       Tests does adaptive c beat fixed-at-v2.15-BR-mean?
#
# Bf-c target compensation:
#   eval_baseline encodes c-target over PARAM_RANGES['c']=(0.0, 1.0).
#   env applies c = C_MIN + (squashed+1)/2 * (1-C_MIN) over [C_MIN=0.10, 1.0].
#   To get env c=0.118, eval needs target X such that env-tanh-scale produces
#   0.118: solve → X = 0.02. Used below.
#
# Total: ~5h training + ~70 min eval = ~6h wall.
#
# Decision criteria post-eval:
#   - If BR beats Bf-a AND Bf-c on in-dist by ≥ 3pp → a+c adaptation IS
#     doing useful work in 2-param isolation. Combined with v2.16a's null
#     result, the architectural takeaway is: adaptation lives in a/c, not
#     in α/φ.
#   - If BR ≈ Bf-a and BR ≈ Bf-c → a+c adaptation also doesn't add value
#     in isolation. The architecture genuinely isn't learning per-step
#     state-conditional CBF params under PPO.
#   - If BR LOSES to fixed-at-mean → a+c "load-bearing" claim from v2.15
#     was an artifact (Bf-X was vs sub-optimal targets, not vs BR's actual
#     adaptation). Time to rethink.
#
# Usage on lab box:
#   ~/Desktop/safety-go2/scripts/train_and_eval_v216b.sh 2>&1 | tee logs/train_and_eval_v216b.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v2.16b (a+c adaptive, α+φ FROZEN at v2.15 BR means) pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ---------- TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] TRAINING: 6000 iters, 4096 envs"
echo "      Frozen: α=3.52, φ=2.26 (v2.15 BR means)"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-v0 \
  --num_envs 4096 --max_iterations 6000 \
  --headless

echo ""
echo "[1/3] TRAINING done at $(date '+%H:%M:%S')"

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

# ---------- HEADLINE 3-EVAL (sequential) ----------
echo ""
echo "================================================================"
echo "[2/3] HEADLINE EVAL: 3 tasks (in-dist + a axis + c axis)"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_TASKS=("v0" "NoisyPerception-v0" "RadiusError-v0")

task_tag() {
  if [ "$1" = "v0" ]; then echo "indist"; else echo "${1%-v0}"; fi
}

for TASK in "${EVAL_TASKS[@]}"; do
  TAG=$(task_tag "$TASK")
  OUT="logs/baseline_eval_v216b_${TAG}"
  echo ""
  echo "  >>> [$TASK] -> $OUT  ($(date '+%H:%M:%S'))"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "Isaac-CBF-Go2-${TASK}" \
    --num_envs 64 --steps_per_config 1500 \
    --modes B0,B1,B2,BR \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --headless > "${OUT}.stdout.log" 2>&1
  echo "  >>> [$TASK] done at $(date '+%H:%M:%S')"
done

# ---------- BF-X ABLATIONS (Bf-a and Bf-c at v2.15 BR means, on in-dist) ----------
echo ""
echo "================================================================"
echo "[3/3] BF-X ABLATIONS: Bf-a (target=0.015) + Bf-c (target=0.02 → env=0.118) on in-dist"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

# Bf-a at v2.15 BR's cbf_a_mean = 0.015. PARAM_RANGES['a']=(0,3) and env
# range [0,3] match → no compensation needed.
echo ""
echo "  >>> Bf-a target=0.015 on in-dist  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-v0" \
  --num_envs 64 --steps_per_config 1500 \
  --modes "Bf-a,BR" \
  --bf_a_target 0.015 \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v216b_bfa_indist" \
  --headless > "logs/baseline_eval_v216b_bfa_indist.stdout.log" 2>&1
echo "  >>> Bf-a done at $(date '+%H:%M:%S')"

# Bf-c at v2.15 BR's cbf_c_mean = 0.118 in env terms. PARAM_RANGES['c']=(0,1)
# but env applies C_MIN floor → range [0.10, 1.0]. Need compensation:
# eval_target=0.02 → env c=0.118.
echo ""
echo "  >>> Bf-c eval_target=0.02 (env c=0.118) on in-dist  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-v0" \
  --num_envs 64 --steps_per_config 1500 \
  --modes "Bf-c,BR" \
  --bf_c_target 0.02 \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v216b_bfc_indist" \
  --headless > "logs/baseline_eval_v216b_bfc_indist.stdout.log" 2>&1
echo "  >>> Bf-c done at $(date '+%H:%M:%S')"

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "Headline 3-eval CSVs in: logs/baseline_eval_v216b_{indist,NoisyPerception,RadiusError}/"
echo "Bf-X CSVs in: logs/baseline_eval_v216b_{bfa,bfc}_indist/"
echo "================================================================"
