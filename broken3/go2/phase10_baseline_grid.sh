#!/bin/bash
# Phase 10 / V2 baseline-grid sweep.
#
# Runs the B0 / B1 / B2 grids on every V2 eval scene. Same shard-based
# sequential pattern as phase10_overnight.sh (use A and B in two terminals).
# Resulting JSONs land alongside the V2 policy eval JSONs under
# phase10_outputs/eval_results/ -- phase10_aggregate.py picks them up.
#
# B0  (Exponential CBF, Ames 2017):       phi=0, alpha const.        3 configs.
# B1  (ECBF + ISSf):                       (phi, alpha) const.        6 configs.
# B2  (TISSf-CBF, Cohen 2024/Molnar 2023): alpha const, phi(h)=
#         (1/eps0)*exp(-lam*h).                                       6 configs.
# Total: 15 configs * 4 scenes = 60 invocations. ~3 h per shard.
#
# Usage:
#   Terminal A:  bash phase10_baseline_grid.sh <loco.pt> phase10_outputs A
#   Terminal B:  bash phase10_baseline_grid.sh <loco.pt> phase10_outputs B
#
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOCO="${1:-}"
OUT="${2:-phase10_outputs}"
SHARD="${3:-ALL}"
if [ -z "$LOCO" ]; then
  echo "usage: $0 <locomotion_checkpoint.pt> [out_dir] [shard:A|B|ALL]" >&2
  exit 2
fi
EVAL_OUT="$OUT/eval_results"
mkdir -p "$EVAL_OUT"

ISAACLAB="${ISAACLAB:-$HOME/IsaacLab/isaaclab.sh}"
NUM_ENVS_EVAL="${NUM_ENVS_EVAL:-512}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
EPS_PER_CELL="${EPS_PER_CELL:-512}"

SCENES=("E1Gap" "E2Slalom" "E3Wall" "E4Field")
# Slimmed grids: B0/B1 are sanity baselines (single config each), B2 is
# the bar V2 needs to beat so we keep the full 6-config sweep.
# Total: 1 + 1 + 6 = 8 configs * 4 scenes = 32 invocations (~48 min/shard).
B0_ALPHAS=("2.5")                       # ECBF sanity
B1_PHIS=("1.0")
B1_ALPHAS=("2.5")                       # ECBF+ISSf sanity
B2_ALPHAS=("1.0" "2.5")
B2_PARAMS=("1.0,0.5" "1.0,1.0" "2.0,1.0")

_fmt_hms() {
  local s=$1; local h=$((s/3600)) m=$(((s%3600)/60)) sec=$((s%60))
  printf "%d:%02d:%02d" $h $m $sec
}

_run_b0() {
  local alpha=$1 scene=$2
  local label="baseline_B0_phi0.0_alpha${alpha}"
  local result="$EVAL_OUT/eval_${label}_${scene}.json"
  if [ -f "$result" ]; then echo "  SKIP (exists)"; return 0; fi
  PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase10_eval_unified.py \
      --baseline_type B0 --baseline_alpha "$alpha" \
      --scene "$scene" --checkpoint "$LOCO" \
      --num_envs "$NUM_ENVS_EVAL" --eval_steps "$EVAL_STEPS" \
      --eps_per_cell "$EPS_PER_CELL" \
      --out_dir "$EVAL_OUT" --headless
}

_run_b1() {
  local phi=$1 alpha=$2 scene=$3
  local label="baseline_B1_phi${phi}_alpha${alpha}"
  local result="$EVAL_OUT/eval_${label}_${scene}.json"
  if [ -f "$result" ]; then echo "  SKIP (exists)"; return 0; fi
  PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase10_eval_unified.py \
      --baseline_type B1 --baseline_phi "$phi" --baseline_alpha "$alpha" \
      --scene "$scene" --checkpoint "$LOCO" \
      --num_envs "$NUM_ENVS_EVAL" --eval_steps "$EVAL_STEPS" \
      --eps_per_cell "$EPS_PER_CELL" \
      --out_dir "$EVAL_OUT" --headless
}

_run_b2() {
  local alpha=$1 eps0=$2 lam=$3 scene=$4
  local label="baseline_B2_alpha${alpha}_eps${eps0}_lam${lam}"
  local result="$EVAL_OUT/eval_${label}_${scene}.json"
  if [ -f "$result" ]; then echo "  SKIP (exists)"; return 0; fi
  PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase10_eval_unified.py \
      --baseline_type B2 --baseline_alpha "$alpha" \
      --baseline_eps0 "$eps0" --baseline_lam "$lam" \
      --scene "$scene" --checkpoint "$LOCO" \
      --num_envs "$NUM_ENVS_EVAL" --eval_steps "$EVAL_STEPS" \
      --eps_per_cell "$EPS_PER_CELL" \
      --out_dir "$EVAL_OUT" --headless
}

# ---- job list (stable order) ----
JOBS=()
for SCENE in "${SCENES[@]}"; do
  for A in "${B0_ALPHAS[@]}"; do
    JOBS+=("B0|$A||$SCENE")
  done
  for P in "${B1_PHIS[@]}"; do
    for A in "${B1_ALPHAS[@]}"; do
      JOBS+=("B1|$P|$A|$SCENE")
    done
  done
  for A in "${B2_ALPHAS[@]}"; do
    for EL in "${B2_PARAMS[@]}"; do
      EPS=$(echo "$EL" | cut -d',' -f1)
      LAM=$(echo "$EL" | cut -d',' -f2)
      JOBS+=("B2|$A|$EPS,$LAM|$SCENE")
    done
  done
done
TOTAL=${#JOBS[@]}

# ---- shard filter ----
MY_JOBS=()
for ((i=0; i<TOTAL; i++)); do
  case "$SHARD" in
    A)   if [ $((i % 2)) -eq 0 ]; then MY_JOBS+=("${JOBS[$i]}"); fi ;;
    B)   if [ $((i % 2)) -eq 1 ]; then MY_JOBS+=("${JOBS[$i]}"); fi ;;
    ALL) MY_JOBS+=("${JOBS[$i]}") ;;
    *) echo "bad shard: $SHARD" >&2; exit 2 ;;
  esac
done
N=${#MY_JOBS[@]}
echo "[bgrid $SHARD] $N of $TOTAL baseline cells assigned"
echo

COMPLETED=0
ELAPSED_TOTAL=0
SHARD_T0=$SECONDS
for ((j=0; j<N; j++)); do
  spec="${MY_JOBS[$j]}"
  IFS='|' read -r KIND A1 A2 SCENE <<< "$spec"
  REMAINING=$((N - COMPLETED))
  if [ "$COMPLETED" -gt 0 ]; then
    MEAN_T=$((ELAPSED_TOTAL / COMPLETED))
    ETA=$((REMAINING * MEAN_T))
    ETA_STR="ETA $(_fmt_hms $ETA) (~$(_fmt_hms $MEAN_T)/cell)"
  else
    ETA_STR="ETA ?"
  fi
  case "$KIND" in
    B0) DESC="B0 alpha=$A1  scene=$SCENE" ;;
    B1) DESC="B1 phi=$A1 alpha=$A2  scene=$SCENE" ;;
    B2) DESC="B2 alpha=$A1 (eps,lam)=$A2  scene=$SCENE" ;;
  esac
  echo "[bgrid $SHARD] === cell $((j+1))/$N: $DESC === $ETA_STR"
  JOB_T0=$SECONDS
  case "$KIND" in
    B0) _run_b0 "$A1" "$SCENE" ;;
    B1) _run_b1 "$A1" "$A2" "$SCENE" ;;
    B2) IFS=',' read -r EPS LAM <<< "$A2"; _run_b2 "$A1" "$EPS" "$LAM" "$SCENE" ;;
  esac
  RC=$?
  JOB_DT=$((SECONDS - JOB_T0))
  ELAPSED_TOTAL=$((ELAPSED_TOTAL + JOB_DT))
  COMPLETED=$((COMPLETED + 1))
  if [ "$RC" -eq 0 ]; then
    echo "[bgrid $SHARD]   done in $(_fmt_hms $JOB_DT)  (shard elapsed $(_fmt_hms $((SECONDS - SHARD_T0))))"
  else
    echo "[bgrid $SHARD]   FAILED (rc=$RC) -- continuing"
  fi
  echo
done

echo "[bgrid $SHARD] all $N cells done. wall-clock = $(_fmt_hms $((SECONDS - SHARD_T0)))"
