#!/bin/bash
# v2.16a — pair-isolation A: α + φ adaptive, a + c FROZEN at v2.15 BR means.
#
# Why this exists:
#   The full v2.15 (4-param adaptive) lost the headline 10-eval 0W/0T/10L,
#   and Bf-α sweeps showed BR's adaptive α was at the worst spot of a U-curve
#   (α≈3.5) while a/c were doing useful work per Bf-X. Hypothesis: inter-slot
#   gradient interference is preventing α and φ from learning. Isolating them
#   into a 2-param adaptive setup (with a/c frozen at v2.15 BR's typical
#   values) tests whether α + φ can learn meaningful state-conditional
#   behavior alone.
#
# Code change for v2.16a (only):
#   cbf_go2_env.py module-level:
#     FREEZE_ALPHA_VALUE = None    (adaptive)
#     FREEZE_PHI_VALUE   = None    (adaptive)
#     FREEZE_A_VALUE     = 0.015   (frozen at v2.15 BR cbf_a_mean)
#     FREEZE_C_VALUE     = 0.118   (frozen at v2.15 BR cbf_c_mean)
#   Everything else from v2.15 unchanged: REWARD-3 (-500/-1.0), ALPHA_MIN=1.0,
#   C_MIN=0.10, per-episode φ lock, actuation-noise DR, radius-error DR,
#   6K iters, 4096 envs, cylinder-only pool, 6-planner mix.
#
# Reduced eval scope (no 10-eval; pair-isolation needs less coverage):
#   Headline 3 tasks (sequential):
#     v0 (in-dist), HighActuationNoise (φ axis), HeavyCOM (locomotion / α axis)
#   Bf-X 2 ablations on in-dist (sequential):
#     Bf-α target=3.52 (v2.15 BR α-mean) — does adaptive α beat fixed-at-BR-mean?
#     Bf-φ target=2.26 (v2.15 BR φ-mean) — does adaptive φ beat fixed-at-BR-mean?
#
# Total: ~5h training + ~70 min eval = ~6h wall.
#
# Decision criteria post-eval:
#   - If BR beats Bf-α AND Bf-φ on in-dist by ≥ 3pp → 2-param α+φ adaptation
#     is doing useful work, proceed to v2.16b (swap)
#   - If BR ≈ Bf-α and BR ≈ Bf-φ → 2-param α+φ is no better than fixed at
#     BR-mean values; the architectural slot might genuinely not learn
#   - If BR LOSES to fixed-at-mean → α+φ adaptation is anti-helpful even
#     in isolation; rethink reward landscape
#
# Usage on lab box:
#   ./scripts/train_and_eval_v216a.sh 2>&1 | tee logs/train_and_eval_v216a.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v2.16a (α+φ adaptive, a+c FROZEN at v2.15 BR means) pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ---------- TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] TRAINING: 6000 iters, 4096 envs"
echo "      Frozen: a=0.015, c=0.118 (v2.15 BR means)"
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
echo "[2/3] HEADLINE EVAL: 3 tasks (in-dist + φ axis + α axis)"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_TASKS=("v0" "HighActuationNoise-v0" "HeavyCOM-v0")

task_tag() {
  if [ "$1" = "v0" ]; then echo "indist"; else echo "${1%-v0}"; fi
}

for TASK in "${EVAL_TASKS[@]}"; do
  TAG=$(task_tag "$TASK")
  OUT="logs/baseline_eval_v216a_${TAG}"
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

# ---------- BF-X ABLATIONS (Bf-α and Bf-φ at v2.15 BR means, on in-dist) ----------
echo ""
echo "================================================================"
echo "[3/3] BF-X ABLATIONS: Bf-α (target=3.52) + Bf-φ (target=2.26) on in-dist"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

# Bf-α at v2.15 BR's cbf_alpha_mean = 3.52 in env terms.
# eval_baseline.py uses PARAM_RANGES['alpha'] = (0.1, 5.0) for encoding,
# but env applies tanh+scale over [ALPHA_MIN=1.0, 5.0]. Compensate target
# so actual env α matches 3.52: target_eval = 3.187 → env α = 3.52.
echo ""
echo "  >>> Bf-α actual_target=3.52 (eval_target=3.187) on in-dist  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-v0" \
  --num_envs 64 --steps_per_config 1500 \
  --modes "Bf-alpha,BR" \
  --bf_alpha_target 3.187 \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v216a_bfalpha_indist" \
  --headless > "logs/baseline_eval_v216a_bfalpha_indist.stdout.log" 2>&1
echo "  >>> Bf-α done at $(date '+%H:%M:%S')"

# Bf-φ at v2.15 BR's cbf_phi_mean
echo ""
echo "  >>> Bf-φ target=2.26 on in-dist  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-v0" \
  --num_envs 64 --steps_per_config 1500 \
  --modes "Bf-phi,BR" \
  --bf_phi_target 2.26 \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v216a_bfphi_indist" \
  --headless > "logs/baseline_eval_v216a_bfphi_indist.stdout.log" 2>&1
echo "  >>> Bf-φ done at $(date '+%H:%M:%S')"

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "Headline 3-eval CSVs in: logs/baseline_eval_v216a_{indist,HighActuationNoise,HeavyCOM}/"
echo "Bf-X CSVs in: logs/baseline_eval_v216a_{bfalpha,bfphi}_indist/"
echo "================================================================"
