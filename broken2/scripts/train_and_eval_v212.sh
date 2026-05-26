#!/bin/bash
# v2.12 = v2.11 stack + cylinder-only obstacle pool + QP-side obstacle-position
# noise injection + restored 6-planner mix (PLANNER-2b reverted) + new
# NoisyPerception OOD env. SHIELD-style commitment to analytical SDF.
#
# Why this exists:
#   v2.11 failed on every front: in-dist combined REGRESSED to 0.472
#   (-1.9pp LOSS vs best B), HeavyCOM still lost (-4.8pp), 4 of 7 evals lost.
#   Diagnostic gold: 12 CBF training health stats showed `a` slot collapsed
#   to mean 0.06 (range [0, 3.0]) and `c` collapsed to mean 0.09 (range
#   [0, 1.0]) — the WIDE_PARAM_RANGES had no effect because those slots had
#   no gradient signal. Root cause: zero measurement noise in training +
#   analytical h(x) with no boundary error. α and φ were doing all the work.
#
#   v2.12 hypothesis: with QP-side obstacle-position noise, `a`/`c` come
#   alive, load redistributes off α/φ, training converges cleanly even at
#   5K iters with the v2.11 disturbance budget. Cylinder-only pool keeps
#   the math simple and matches what a SHIELD-style cluster-fit-cylinder
#   LiDAR pipeline would produce at deploy.
#
# v2.12 changes (delta from v2.11):
#
#   Code changes:
#     1. Cylinder-only OBSTACLE_SHAPES (drop boxes, walls, rect boxes).
#        20 cylinders, radii 0.10–0.50m. Loses arbitrary-shape generality
#        vs SHIELD; gains analytical-SDF simplicity + train/deploy parity.
#        (cbf_go2_env_cfg.py)
#     2. PLANNER-2b reverted: restore walk + adversarial planners (6-planner
#        mix matching v2.6). PLANNER-2b was bundled into v2.8/v2.10/v2.11
#        but never tested in isolation; restoring v2.6's known-good mix.
#        (cbf_go2_env_cfg.py)
#     3. QP-side obstacle-position noise (Option A): per-episode σ ~
#        Uniform(0, σ_max), per-step ε ~ N(0, σ²·I) added to obstacle
#        positions used by _compute_h() ONLY. Priv obs grid stays clean —
#        policy "sees truth", QP "sees noisy", `a` slot must absorb the
#        gap. σ_max = 0.05m for training, 0.10m for NoisyPerception OOD.
#        (cbf_go2_env_cfg.py + cbf_go2_env.py)
#     4. New Isaac-CBF-Go2-NoisyPerception-v0 OOD env. Single-axis push on
#        measurement-uncertainty axis. Headline number for paper Table 2
#        Bf-a ablation row. (cbf_go2/__init__.py + cbf_go2_env_cfg.py)
#     5. Two new CBF health stats: cbf_obs_noise_sigma_mean / std. Lets
#        the watch script confirm noise injection is active.
#
#   Kept from v2.11 (no change):
#     - Bimodal resample DR (P=0.5 mid-switch [5,15]s / P=0.5 locked 100s)
#     - Variable obstacle motion DR (max_speed_range=(0, 0.4))
#     - L_f h obstacle-drift term (USE_LFH_OBSTACLE_DRIFT)
#     - WIDE_PARAM_RANGES (a [0, 3.0], c [0, 1.0])
#     - REWARD-2 stack (-100 fall, stuck -2.0, proximity -0.5)
#     - 5K iters, 4096 envs
#     - 12 CBF training health stats (now 14 with the noise σ pair)
#     - B-fixed-X eval modes
#     - Parallel 2-up eval, 30s stagger
#
# Eval matrix (parallel; ~8 task headline + dual-regime + Bf-X):
#   Headline (LOCKED planner, deploy-realistic): 8 tasks —
#     v0, DensePack, Slippery, HighDisturbance, HeavyCOM, FastObstacles,
#     RealisticCompound, NoisyPerception (NEW).
#   Diagnostic regime sweep: v0 + HeavyCOM under MID-SWITCH (resample 10s).
#   Table-2 ablations: Bf-α (v0 + DensePack), Bf-φ (HighDist), Bf-a
#     (NoisyPerception — NEW; meaningful now that `a` has signal),
#     Bf-c (HeavyCOM + FastObs).
#
# Time budget: ~5h training + ~5h parallel eval = ~10h total.
#
# Predicted (vs v2.11 baseline + v2.6 paper baseline):
#   - Training term_base_contact: ≤ 7% (v2.11 was 11.9%; bet on cleaner
#     convergence now that `a`/`c` aren't dead)
#   - cbf_a_std at iter 5000: ≥ 0.4 (v2.11 was 0.23 with collapsed mean)
#   - cbf_c_std at iter 5000: ≥ 0.3 (v2.11 was 0.24 with collapsed mean)
#   - In-dist combined: ≤ 0.31 (v2.11 was 0.472, v2.6 was 0.306)
#   - HeavyCOM margin: ≥ +3pp WIN (v2.11 was -4.8pp LOSS)
#   - All 4 B-fixed-X show BR > Bf-X by ≥ 3pp (paper Table 2 lands cleanly)
#
# Decision criteria (post-eval):
#   - In-dist ≤ 0.31 AND HeavyCOM ≥ +3pp AND compound holds → ship as paper baseline.
#   - 0.31 < combined < 0.40 → partial recovery; consider further iteration.
#   - combined ≥ 0.40 OR HeavyCOM still loses → fall back to v2.6 + locked-eval headline.
#
# Usage on lab box:
#   ./scripts/train_and_eval_v212.sh 2>&1 | tee logs/train_and_eval_v212.log
#
# IMPORTANT: in tmux pane B during training, run the health watcher:
#   bash ~/Desktop/safety-go2/scripts/watch_training_health.sh \
#     ~/Desktop/safety-go2/IsaacLab/logs/train_and_eval_v212.log
#   Health log appended to: train_and_eval_v212.health.log
#   Catches `a`/`c` collapse early (iter ≥ 1000 — if cbf_a_std < 0.2 we know
#   the noise injection isn't doing its job and abort).

set -e  # abort on any error

cd ~/Desktop/safety-go2/IsaacLab

echo "================================================================"
echo "v2.12 (cylinder pool + QP-side noise injection + 6-planner mix) pipeline"
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

# ---------- HEADLINE 8-EVAL (LOCKED PLANNER, 2-UP PARALLEL) ----------
echo ""
echo "================================================================"
echo "[2/3] HEADLINE EVAL: 8 tasks under LOCKED planner (deploy-realistic)"
echo "      Running 2-up parallel"
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
  "NoisyPerception-v0"
)

# Helper: map task name (v0 / DensePack-v0 / etc) → dir tag (indist / DensePack / etc).
task_tag() {
  if [ "$1" = "v0" ]; then
    echo "indist"
  else
    echo "${1%-v0}"
  fi
}

# Stagger 2-up: launch task[i] + task[i+1], wait for both, repeat.
for ((i=0; i<${#EVAL_TASKS[@]}; i+=2)); do
  TASK1="${EVAL_TASKS[$i]}"
  TASK2="${EVAL_TASKS[$((i+1))]:-}"

  TAG1=$(task_tag "$TASK1")
  OUT1="logs/baseline_eval_v212_${TAG1}"

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
    OUT2="logs/baseline_eval_v212_${TAG2}"

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

echo ""
echo "  >>> launch [v0 mid-switch] + [HeavyCOM mid-switch]  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes B0,B1,B2,BR \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v212_indist_midswitch" \
  --planner_resample_s 10 \
  --headless > "logs/baseline_eval_v212_indist_midswitch.stdout.log" 2>&1 &
PID1=$!

sleep 30
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-HeavyCOM-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes B0,B1,B2,BR \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v212_HeavyCOM_midswitch" \
  --planner_resample_s 10 \
  --headless > "logs/baseline_eval_v212_HeavyCOM_midswitch.stdout.log" 2>&1 &
PID2=$!

wait $PID1
echo "  >>> [v0 mid-switch] done at $(date '+%H:%M:%S')"
wait $PID2
echo "  >>> [HeavyCOM mid-switch] done at $(date '+%H:%M:%S')"

# ---------- TABLE-2 ABLATIONS: B-fixed-X on relevant axes ----------
echo ""
echo "================================================================"
echo "[3b/3] TABLE-2 ABLATIONS: B-fixed-{α,φ,a,c} on matched OOD axes"
echo "      Bf-a now meaningful: NoisyPerception env exercises `a`"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

# Pair 1: Bf-α on v0 + DensePack
echo ""
echo "  >>> launch [Bf-α on v0] + [Bf-α on DensePack]  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes BR,Bf-alpha \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v212_bfalpha_indist" \
  --headless > "logs/baseline_eval_v212_bfalpha_indist.stdout.log" 2>&1 &
PID1=$!

sleep 30
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-DensePack-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes BR,Bf-alpha \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v212_bfalpha_DensePack" \
  --headless > "logs/baseline_eval_v212_bfalpha_DensePack.stdout.log" 2>&1 &
PID2=$!

wait $PID1; echo "  >>> [Bf-α v0] done at $(date '+%H:%M:%S')"
wait $PID2; echo "  >>> [Bf-α DensePack] done at $(date '+%H:%M:%S')"

# Pair 2: Bf-φ on HighDist, Bf-a on NoisyPerception (NEW for v2.12)
echo ""
echo "  >>> launch [Bf-φ HighDist] + [Bf-a NoisyPerception]  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-HighDisturbance-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes BR,Bf-phi \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v212_bfphi_HighDist" \
  --headless > "logs/baseline_eval_v212_bfphi_HighDist.stdout.log" 2>&1 &
PID1=$!

sleep 30
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-NoisyPerception-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes BR,Bf-a \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v212_bfa_NoisyPerception" \
  --headless > "logs/baseline_eval_v212_bfa_NoisyPerception.stdout.log" 2>&1 &
PID2=$!

wait $PID1; echo "  >>> [Bf-φ HighDist] done at $(date '+%H:%M:%S')"
wait $PID2; echo "  >>> [Bf-a NoisyPerception] done at $(date '+%H:%M:%S')"

# Pair 3: Bf-c on HeavyCOM + FastObstacles
echo ""
echo "  >>> launch [Bf-c HeavyCOM] + [Bf-c FastObs]  ($(date '+%H:%M:%S'))"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-HeavyCOM-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes BR,Bf-c \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v212_bfc_HeavyCOM" \
  --headless > "logs/baseline_eval_v212_bfc_HeavyCOM.stdout.log" 2>&1 &
PID1=$!

sleep 30
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task "Isaac-CBF-Go2-FastObstacles-v0" \
  --num_envs 64 --steps_per_config 2000 \
  --modes BR,Bf-c \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_v212_bfc_FastObs" \
  --headless > "logs/baseline_eval_v212_bfc_FastObs.stdout.log" 2>&1 &
PID2=$!

wait $PID1; echo "  >>> [Bf-c HeavyCOM] done at $(date '+%H:%M:%S')"
wait $PID2; echo "  >>> [Bf-c FastObs] done at $(date '+%H:%M:%S')"

# ---------- SUMMARY ----------
echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: $CKPT"
echo "Headline CSVs:"
ls -la logs/baseline_eval_v212_indist/baseline.csv \
       logs/baseline_eval_v212_DensePack/baseline.csv \
       logs/baseline_eval_v212_Slippery/baseline.csv \
       logs/baseline_eval_v212_HighDisturbance/baseline.csv \
       logs/baseline_eval_v212_HeavyCOM/baseline.csv \
       logs/baseline_eval_v212_FastObstacles/baseline.csv \
       logs/baseline_eval_v212_RealisticCompound/baseline.csv \
       logs/baseline_eval_v212_NoisyPerception/baseline.csv 2>/dev/null
echo ""
echo "Mid-switch dual-regime CSVs:"
ls -la logs/baseline_eval_v212_indist_midswitch/baseline.csv \
       logs/baseline_eval_v212_HeavyCOM_midswitch/baseline.csv 2>/dev/null
echo ""
echo "Table-2 ablation CSVs:"
ls -la logs/baseline_eval_v212_bf*/baseline.csv 2>/dev/null
echo "================================================================"
