#!/bin/bash
# v2.11 = v2.10 + variable resample DR + variable obstacle motion DR + L_f h
# obstacle drift + widened a/c ranges + B-fixed-X eval ablations + parallel eval.
#
# Why this exists:
#   v2.10 landed in the partial-recovery zone (in-dist combined 0.343 vs v2.6's
#   0.306). 4 wins / 2 ties / 1 LOSS (HeavyCOM -7.8pp). HeavyCOM mid-switch
#   diagnostic showed -8.8pp combined recovery just from changing eval-time
#   planner regime — SMOKING GUN that PLANNER-2a (locked-planner training)
#   was the dominant HeavyCOM regression cause.
#
# v2.11 changes (additive on v2.10's base):
#
#   Code changes (rollback-safe via flags / new args):
#     1. Variable resample DR (BIMODAL): each episode rolls
#        P(0.5) → uniform [5, 15]s mid-switch; P(0.5) → 100s locked.
#        Restores v2.6's stuck-recovery regularizer that PLANNER-2a stripped
#        WHILE retaining heterogeneous training distribution. Eval always
#        forces locked planner (deploy-realistic). (cbf_go2_commands.py +
#        cbf_go2_env_cfg.py)
#     2. Variable obstacle-motion DR: per-episode v_obs sampled in [0, 0.4] m/s
#        (sometimes static, sometimes fast). Exercises `c` for kinematic
#        margin scaling. (cbf_go2_events.py + cbf_go2_env_cfg.py)
#     3. L_f h obstacle-drift term: QP constraint augmented with
#        ∂h/∂p_obs · v_obs. Mathematically equivalent to constraining the
#        relative velocity (u - v_obs). Correctness fix; pairs with #2.
#        Behind USE_LFH_OBSTACLE_DRIFT flag. (cbf_go2_env.py)
#     4. Wider `a` ([0,1]→[0,3]) and `c` ([0,0.5]→[0,1]) ranges behind
#        WIDE_PARAM_RANGES flag. Headroom for Dean 2019 measurement
#        uncertainty + boundary correction. (cbf_go2_env.py)
#     5. B-fixed-{α,φ,a,c} eval modes: paper Table 2 ablations. Run BR
#        policy but clamp one CBF slot to a fixed physical value; compare
#        BR vs Bf-X. (eval_baseline.py)
#     6. Training health logging: 12 new CBF stats (mean/std for α/φ/a/c,
#        h_min/h_mean, qp_active_rate, u_safe_clamp_rate) surfaced into
#        rsl_rl per-iter log. extract_training_summary.py extended to
#        capture them. (cbf_go2_env.py + extract_training_summary.py)
#     7. Watch script: scripts/watch_training_health.sh polls the in-progress
#        log every 5 min, prints colored trip-wire warnings.
#
# Reward stack: UNCHANGED from v2.10 (REWARD-2 retune kept — base_contact -100,
# stuck -2.0, proximity -0.5, etc.). REWARD-2 contributed compound flip + DensePack
# improvement; not the cause of HeavyCOM regression.
#
# Eval matrix (parallel, ~50% time savings):
#   Headline (mandatory): 7 tasks under LOCKED planner — deploy-realistic.
#     v0, DensePack, Slippery, HighDisturbance, HeavyCOM, FastObstacles, Compound
#   Diagnostic regime sweep: v0 + HeavyCOM under MID-SWITCH (resample 10s) —
#     apples-to-apples vs v2.6 paper baseline; verifies HeavyCOM regression fixed.
#   Table-2 ablations: B-fixed-α (DensePack + v0), B-fixed-φ (HighDist),
#     B-fixed-c (HeavyCOM + FastObs). Skipped Bf-a (NoisyPerception is v2.12).
#
# Time budget: ~5h training + ~4.5h parallel eval = ~9.5h total.
#
# Predicted (vs v2.10 baseline): in-dist combined 0.28-0.32, HeavyCOM margin
# recovers from -7.8pp to ≥ +3pp, compound + DensePack wins retained.
#
# Decision criteria (post-eval):
#   - In-dist combined ≤ 0.31 AND HeavyCOM ≥ +3pp → ship as paper baseline.
#   - 0.31 < combined < 0.40 → partial recovery; consider further iteration.
#   - combined ≥ 0.40 → need to investigate (DR widening over-shot? ranges?).
#
# Usage on lab box:
#   ./scripts/train_and_eval_v211.sh 2>&1 | tee logs/train_and_eval_v211.log
#
# Optional: in a tmux pane B during training, run the health watcher:
#   bash ~/Desktop/safety-go2/scripts/watch_training_health.sh \
#     ~/Desktop/safety-go2/IsaacLab/logs/train_and_eval_v211.log

set -e  # abort on any error

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v2.11 (variable resample + obstacle motion DR + L_f h + Table-2) pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ---------- TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] TRAINING: 5000 iters, 4096 envs"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-v0 \
  --num_envs 4096 --max_iterations 5000 \
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

# ---------- HEADLINE 7-EVAL (LOCKED PLANNER, 2-UP PARALLEL) ----------
echo ""
echo "================================================================"
echo "[2/3] HEADLINE EVAL: 7 tasks under LOCKED planner (deploy-realistic)"
echo "      Running 2-up parallel to halve wall-clock time"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_TASKS=(
  "v0"
  "DensePack-v0"
  "Slippery-v0"
  "HighDisturbance-v0"
  "HeavyCOM-v0"
  "FastObstacles-v0"
  "RealisticCompound-v0"
)

# Helper: map task name (v0 / DensePack-v0 / etc) → dir tag (indist / DensePack / etc).
# v2.11 BUGFIX (2026-05-07): the old `${TASK%-v0}` rename only stripped the
# trailing `-v0`, which doesn't apply to the bare "v0" task → in-dist dir
# came out as `_v0` instead of `_indist`. Special-case it explicitly so all
# headline + dual-regime + Bf-X dirs use consistent `_indist` naming.
task_tag() {
  if [ "$1" = "v0" ]; then
    echo "indist"
  else
    echo "${1%-v0}"
  fi
}

# Stagger 2-up: launch task[i] + task[i+1], wait for both, repeat.
# Stagger by 30s within a pair to avoid Isaac Sim init contention.
for ((i=0; i<${#EVAL_TASKS[@]}; i+=2)); do
  TASK1="${EVAL_TASKS[$i]}"
  TASK2="${EVAL_TASKS[$((i+1))]:-}"

  TAG1=$(task_tag "$TASK1")
  OUT1="logs/baseline_eval_v211_${TAG1}"

  echo ""
  echo "  >>> [pair $((i/2+1))] launch [$TASK1] -> $OUT1  ($(date '+%H:%M:%S'))"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "Isaac-CBF-Go2-${TASK1}" \
    --num_envs 64 --steps_per_config 2000 \
    --modes B0,B1,B2,BR \
    --checkpoint "$CKPT" \
    --output_dir "$OUT1" \
    --headless > "${OUT1}.stdout.log" 2>&1 &
  PID1=$!

  if [ -n "$TASK2" ]; then
    sleep 30   # Stagger Isaac Sim asset-loading contention
    TAG2=$(task_tag "$TASK2")
    OUT2="logs/baseline_eval_v211_${TAG2}"

    echo "  >>> [pair $((i/2+1))] launch [$TASK2] -> $OUT2  ($(date '+%H:%M:%S'))"
    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
      --task "Isaac-CBF-Go2-${TASK2}" \
      --num_envs 64 --steps_per_config 2000 \
      --modes B0,B1,B2,BR \
      --checkpoint "$CKPT" \
      --output_dir "$OUT2" \
      --headless > "${OUT2}.stdout.log" 2>&1 &
    PID2=$!
  else
    PID2=""
  fi

  echo "  >>> [pair $((i/2+1))] waiting for PID1=$PID1 ${PID2:+PID2=$PID2}"
  wait $PID1
  echo "  >>> [pair $((i/2+1))] [$TASK1] done at $(date '+%H:%M:%S')"
  if [ -n "$PID2" ]; then
    wait $PID2
    echo "  >>> [pair $((i/2+1))] [$TASK2] done at $(date '+%H:%M:%S')"
  fi
done

# ---------- DUAL-REGIME DIAGNOSTIC: v0 + HeavyCOM under MID-SWITCH ----------
echo ""
echo "================================================================"
echo "[3a/3] DUAL-REGIME DIAGNOSTIC: v0 + HeavyCOM under mid-switch (resample 10s)"
echo "      Apples-to-apples vs v2.6 paper baseline regime"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

# Run as a single 2-up pair (both tasks, both mid-switch)
echo ""
echo "  >>> launch [v0 mid-switch] + [HeavyCOM mid-switch]  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes B0,B1,B2,BR \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v211_indist_midswitch" \
  --planner_resample_s 10 \
  --headless > "logs/baseline_eval_v211_indist_midswitch.stdout.log" 2>&1 &
PID1=$!

sleep 30
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-HeavyCOM-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes B0,B1,B2,BR \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v211_HeavyCOM_midswitch" \
  --planner_resample_s 10 \
  --headless > "logs/baseline_eval_v211_HeavyCOM_midswitch.stdout.log" 2>&1 &
PID2=$!

wait $PID1
echo "  >>> [v0 mid-switch] done at $(date '+%H:%M:%S')"
wait $PID2
echo "  >>> [HeavyCOM mid-switch] done at $(date '+%H:%M:%S')"

# ---------- TABLE-2 ABLATIONS: B-fixed-X on relevant axes ----------
echo ""
echo "================================================================"
echo "[3b/3] TABLE-2 ABLATIONS: B-fixed-{α,φ,c} on matched OOD axes"
echo "      (skip Bf-a — NoisyPerception env is v2.12)"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

# Bf-α paired with DensePack (tight-space adapt), Bf-φ with HighDisturbance,
# Bf-c with HeavyCOM. Run as 2-up pairs to keep wall-clock low.

# Pair 1: Bf-α on v0 + DensePack
echo ""
echo "  >>> launch [Bf-α on v0] + [Bf-α on DensePack]  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes BR,Bf-alpha \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v211_bfalpha_indist" \
  --headless > "logs/baseline_eval_v211_bfalpha_indist.stdout.log" 2>&1 &
PID1=$!

sleep 30
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-DensePack-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes BR,Bf-alpha \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v211_bfalpha_DensePack" \
  --headless > "logs/baseline_eval_v211_bfalpha_DensePack.stdout.log" 2>&1 &
PID2=$!

wait $PID1; echo "  >>> [Bf-α v0] done at $(date '+%H:%M:%S')"
wait $PID2; echo "  >>> [Bf-α DensePack] done at $(date '+%H:%M:%S')"

# Pair 2: Bf-φ on HighDist, Bf-c on HeavyCOM
echo ""
echo "  >>> launch [Bf-φ HighDist] + [Bf-c HeavyCOM]  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-HighDisturbance-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes BR,Bf-phi \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v211_bfphi_HighDist" \
  --headless > "logs/baseline_eval_v211_bfphi_HighDist.stdout.log" 2>&1 &
PID1=$!

sleep 30
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-HeavyCOM-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes BR,Bf-c \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v211_bfc_HeavyCOM" \
  --headless > "logs/baseline_eval_v211_bfc_HeavyCOM.stdout.log" 2>&1 &
PID2=$!

wait $PID1; echo "  >>> [Bf-φ HighDist] done at $(date '+%H:%M:%S')"
wait $PID2; echo "  >>> [Bf-c HeavyCOM] done at $(date '+%H:%M:%S')"

# Pair 3: Bf-c on FastObstacles (solo)
echo ""
echo "  >>> launch [Bf-c FastObs]  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-FastObstacles-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes BR,Bf-c \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v211_bfc_FastObs" \
  --headless > "logs/baseline_eval_v211_bfc_FastObs.stdout.log" 2>&1
echo "  >>> [Bf-c FastObs] done at $(date '+%H:%M:%S')"

# ---------- SUMMARY ----------
echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "Headline CSVs:"
ls -la logs/baseline_eval_v211_indist/baseline.csv \
       logs/baseline_eval_v211_DensePack/baseline.csv \
       logs/baseline_eval_v211_Slippery/baseline.csv \
       logs/baseline_eval_v211_HighDisturbance/baseline.csv \
       logs/baseline_eval_v211_HeavyCOM/baseline.csv \
       logs/baseline_eval_v211_FastObstacles/baseline.csv \
       logs/baseline_eval_v211_RealisticCompound/baseline.csv 2>/dev/null
echo ""
echo "Mid-switch dual-regime CSVs:"
ls -la logs/baseline_eval_v211_indist_midswitch/baseline.csv \
       logs/baseline_eval_v211_HeavyCOM_midswitch/baseline.csv 2>/dev/null
echo ""
echo "Table-2 ablation CSVs:"
ls -la logs/baseline_eval_v211_bf*/baseline.csv 2>/dev/null
echo "================================================================"
