#!/bin/bash
# Phase 10 / V2 overnight pipeline -- SHARD-BASED SEQUENTIAL runner.
#
# All jobs are enumerated in a stable order, then partitioned by shard
# (A = even indices, B = odd indices). Run TWO terminals in parallel,
# one per shard, so you can watch real per-job ETA in each terminal:
#
#   Terminal A:  bash phase10_overnight.sh <loco.pt> phase10_outputs A
#   Terminal B:  bash phase10_overnight.sh <loco.pt> phase10_outputs B
#
# Or run a single shard sequentially:
#   bash phase10_overnight.sh <loco.pt> phase10_outputs ALL
#
# Job order is stable across invocations and shards skip already-done
# jobs (existing manifest or JSON), so you can rerun if a terminal dies.
#
# Job inventory:
#   Stage 1 (train):  5 archs × 2 intervention_costs = 10 jobs
#   Stage 2 (eval):   10 policies × 4 scenes         = 40 jobs
#   Stage 3 (basel):  3 baselines × 4 scenes         = 12 jobs
#   Stage 4 (aggreg): 1 final pass                   = 1 job (only on shard A / ALL)
#   Total: 63 jobs (31 per shard for A/B split)
#
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOCO="${1:-}"
OUT="${2:-phase10_outputs}"
SHARD="${3:-ALL}"   # A, B, or ALL
if [ -z "$LOCO" ]; then
  echo "usage: $0 <locomotion_checkpoint.pt> [out_dir] [shard:A|B|ALL]" >&2
  exit 2
fi
mkdir -p "$OUT"
EVAL_OUT="$OUT/eval_results"
mkdir -p "$EVAL_OUT"

ISAACLAB="${ISAACLAB:-$HOME/IsaacLab/isaaclab.sh}"
if [ ! -x "$ISAACLAB" ]; then
  echo "[orchestrator] ISAACLAB not found at $ISAACLAB" >&2
  exit 2
fi

NUM_ENVS_TRAIN="${NUM_ENVS_TRAIN:-2048}"
MAX_ITERS="${MAX_ITERS:-3000}"
DIAG_INTERVAL="${DIAG_INTERVAL:-200}"
NUM_ENVS_EVAL="${NUM_ENVS_EVAL:-512}"
EVAL_STEPS="${EVAL_STEPS:-1250}"
EPS_PER_CELL="${EPS_PER_CELL:-512}"
# Video capture (off by default; recording adds ~10-30% runtime). Set
# VIDEO_TRAIN=1 / VIDEO_EVAL=1 in the environment to enable.
VIDEO_TRAIN="${VIDEO_TRAIN:-0}"
VIDEO_EVAL="${VIDEO_EVAL:-0}"
VIDEO_TRAIN_INTERVAL="${VIDEO_TRAIN_INTERVAL:-15000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1250}"

ARCHS=("V2Full" "V2NoPriv" "V2NoProprio" "V2RMAClassic" "V2History")
COSTS=("0.0" "-0.05")
SCENES=("E1Gap" "E2Slalom" "E3Wall" "E4Field")
# (phi, alpha) baselines: trivial, alpha-max-only, both-max
BASELINES=("0.0,2.5" "0.0,4.0" "1.0,4.0")

_fmt_hms() {
  local s=$1
  local h=$((s / 3600)) m=$(((s % 3600) / 60)) sec=$((s % 60))
  printf "%d:%02d:%02d" $h $m $sec
}

_cost_tag() {
  # 0.0 -> 0_0, -0.05 -> n0_05  (shell-safe)
  echo "$1" | sed 's/^-/n/; s/\./_/g'
}

# ---- job dispatch helpers ----
_run_train() {
  local arch=$1 cost=$2
  local tag; tag=$(_cost_tag "$cost")
  local outdir="$OUT/${arch}_int${tag}"
  if [ -f "$outdir/manifest.txt" ]; then
    echo "  SKIP (manifest exists)"
    return 0
  fi
  mkdir -p "$outdir"
  local video_args=()
  if [ "$VIDEO_TRAIN" = "1" ]; then
    video_args=(--video --video_interval "$VIDEO_TRAIN_INTERVAL" --video_length "$VIDEO_LENGTH")
  fi
  PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase10_train_unified.py \
      --task "Isaac-CBF-Adaptive-Go2-${arch}-v0" \
      --intervention_cost "$cost" \
      --checkpoint "$LOCO" \
      --num_envs "$NUM_ENVS_TRAIN" --max_iterations "$MAX_ITERS" \
      --diag_interval "$DIAG_INTERVAL" \
      --out_dir "$outdir" --headless \
      "${video_args[@]}"
}

_run_eval() {
  local arch=$1 cost=$2 scene=$3
  local tag; tag=$(_cost_tag "$cost")
  local policy_dir="$OUT/${arch}_int${tag}"
  local result="$EVAL_OUT/eval_${arch}_int${tag}_${scene}.json"
  if [ ! -f "$policy_dir/manifest.txt" ]; then
    echo "  SKIP (policy not trained yet)"
    return 0
  fi
  # The eval script writes eval_<policy_label>_<scene>.json where the
  # policy_label is basename(policy_dir). Check for that file too.
  local policy_label; policy_label=$(basename "$policy_dir")
  local result_eval="$EVAL_OUT/eval_${policy_label}_${scene}.json"
  if [ -f "$result_eval" ]; then
    echo "  SKIP (eval result exists)"
    return 0
  fi
  local video_args=()
  if [ "$VIDEO_EVAL" = "1" ]; then
    video_args=(--video --video_length "$VIDEO_LENGTH")
  fi
  PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase10_eval_unified.py \
      --policy_dir "$policy_dir" \
      --scene "$scene" \
      --checkpoint "$LOCO" \
      --num_envs "$NUM_ENVS_EVAL" --eval_steps "$EVAL_STEPS" \
      --eps_per_cell "$EPS_PER_CELL" \
      --out_dir "$EVAL_OUT" --headless \
      "${video_args[@]}"
}

_run_baseline() {
  local b=$1 scene=$2
  local phi; phi=$(echo "$b" | cut -d',' -f1)
  local alpha; alpha=$(echo "$b" | cut -d',' -f2)
  local result_eval="$EVAL_OUT/eval_baseline_phi${phi}_alpha${alpha}_${scene}.json"
  if [ -f "$result_eval" ]; then
    echo "  SKIP (baseline result exists)"
    return 0
  fi
  local video_args=()
  if [ "$VIDEO_EVAL" = "1" ]; then
    video_args=(--video --video_length "$VIDEO_LENGTH")
  fi
  PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase10_eval_unified.py \
      --baseline_phi "$phi" --baseline_alpha "$alpha" \
      --scene "$scene" \
      --checkpoint "$LOCO" \
      --num_envs "$NUM_ENVS_EVAL" --eval_steps "$EVAL_STEPS" \
      --eps_per_cell "$EPS_PER_CELL" \
      --out_dir "$EVAL_OUT" --headless \
      "${video_args[@]}"
}

_run_aggregate() {
  python3 phase10_aggregate.py --eval_dir "$EVAL_OUT" --out "$OUT/phase10_summary.csv"
}

# ---- build full job list (stable order) ----
# Each job is "KIND|arg1|arg2|arg3" so a split on '|' recovers args.
JOBS=()
for ARCH in "${ARCHS[@]}"; do
  for COST in "${COSTS[@]}"; do
    JOBS+=("train|$ARCH|$COST|")
  done
done
for ARCH in "${ARCHS[@]}"; do
  for COST in "${COSTS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
      JOBS+=("eval|$ARCH|$COST|$SCENE")
    done
  done
done
for B in "${BASELINES[@]}"; do
  for SCENE in "${SCENES[@]}"; do
    JOBS+=("baseline|$B|$SCENE|")
  done
done
# aggregate runs only on shard A or ALL (it reads all per-cell JSONs;
# shard B finishing last is racy if it runs concurrently, so let A own it)
JOBS+=("aggregate|||")

TOTAL=${#JOBS[@]}

# ---- filter by shard ----
MY_JOBS=()
MY_INDICES=()
for ((i=0; i<TOTAL; i++)); do
  case "$SHARD" in
    A)   if [ $((i % 2)) -eq 0 ]; then MY_JOBS+=("${JOBS[$i]}"); MY_INDICES+=("$i"); fi ;;
    B)   if [ $((i % 2)) -eq 1 ]; then MY_JOBS+=("${JOBS[$i]}"); MY_INDICES+=("$i"); fi ;;
    ALL) MY_JOBS+=("${JOBS[$i]}"); MY_INDICES+=("$i") ;;
    *) echo "[orchestrator] bad shard: $SHARD" >&2; exit 2 ;;
  esac
done
N=${#MY_JOBS[@]}

echo "[shard $SHARD] $N of $TOTAL jobs assigned to this shard"
echo "[shard $SHARD] cwd=$SCRIPT_DIR  out=$OUT  isaaclab=$ISAACLAB"
echo "[shard $SHARD] train=${NUM_ENVS_TRAIN}envs×${MAX_ITERS}it  eval=${NUM_ENVS_EVAL}envs×${EVAL_STEPS}s×${EPS_PER_CELL}ep"
echo "[shard $SHARD] video: train=${VIDEO_TRAIN}  eval=${VIDEO_EVAL}  (interval=${VIDEO_TRAIN_INTERVAL} steps, length=${VIDEO_LENGTH} steps)"
echo

COMPLETED=0
ELAPSED_TOTAL=0
SHARD_T0=$SECONDS
for ((j=0; j<N; j++)); do
  spec="${MY_JOBS[$j]}"
  abs_idx="${MY_INDICES[$j]}"
  IFS='|' read -r KIND A1 A2 A3 <<< "$spec"
  REMAINING=$((N - COMPLETED))
  if [ "$COMPLETED" -gt 0 ]; then
    MEAN_T=$((ELAPSED_TOTAL / COMPLETED))
    ETA=$((REMAINING * MEAN_T))
    ETA_STR="ETA $(_fmt_hms $ETA) (~$(_fmt_hms $MEAN_T)/job)"
  else
    ETA_STR="ETA ?"
  fi
  case "$KIND" in
    train)     DESC="train $A1 int=$A2" ;;
    eval)      DESC="eval $A1 int=$A2 on $A3" ;;
    baseline)  DESC="baseline phi,alpha=$A1 on $A2" ;;
    aggregate) DESC="aggregate JSONs -> CSV" ;;
  esac
  echo "[shard $SHARD] === job $((j+1))/$N (abs #$abs_idx): $DESC === $ETA_STR"
  JOB_T0=$SECONDS
  case "$KIND" in
    train)     _run_train "$A1" "$A2" ;;
    eval)      _run_eval "$A1" "$A2" "$A3" ;;
    baseline)  _run_baseline "$A1" "$A2" ;;
    aggregate) _run_aggregate ;;
  esac
  RC=$?
  JOB_DT=$((SECONDS - JOB_T0))
  ELAPSED_TOTAL=$((ELAPSED_TOTAL + JOB_DT))
  COMPLETED=$((COMPLETED + 1))
  if [ "$RC" -eq 0 ]; then
    echo "[shard $SHARD]   done in $(_fmt_hms $JOB_DT)  (shard total elapsed $(_fmt_hms $((SECONDS - SHARD_T0))))"
  else
    echo "[shard $SHARD]   FAILED (rc=$RC) in $(_fmt_hms $JOB_DT)  -- continuing"
  fi
  echo
done

echo "[shard $SHARD] all $N jobs done. shard wall-clock = $(_fmt_hms $((SECONDS - SHARD_T0)))"
