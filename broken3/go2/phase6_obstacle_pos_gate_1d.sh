#!/usr/bin/env bash
# Cleaner 1D obstacle-position gate: vary obstacle x while keeping
# y fixed at 0.5 (avoids the on-path "dead zone" at y=0). This is
# the version the professor will see -- monotonic, easy to read.
#
# 4 positions × 7 alphas + 4 Isaac startups = ~10 min.

set -euo pipefail
cd "$(dirname "$0")"

LOCO="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt"
ISAACLAB="$HOME/IsaacLab/isaaclab.sh"
OUT="phase6_obs_pos_gate_outputs"
LOGS="phase6_obs_pos_gate_logs"
mkdir -p "$OUT" "$LOGS"

# 1D sweep: obstacle x at fixed y=0.5 (off-path, all feasible).
# 1.5 = close/early, 4.5 = late/almost-passed.
POSITIONS=(
    "1.5 0.5"
    "2.5 0.5"
    "3.5 0.5"
    "4.5 0.5"
)

for POS in "${POSITIONS[@]}"; do
    read -r OX OY <<< "$POS"
    TAG="x${OX}_y${OY}"
    echo ""
    echo "================================================================"
    echo "  Obs-pos gate 1D: ($OX, $OY)  (started $(date '+%H:%M:%S'))"
    echo "================================================================"
    "$ISAACLAB" -p phase6_obstacle_pos_gate.py \
        --checkpoint "$LOCO" --num_envs 256 \
        --obstacle_x "$OX" --obstacle_y "$OY" --headless \
        > "$LOGS/pos1d_${TAG}.log" 2>&1
done

echo ""
echo "================================================================"
echo "  STRONG ALPHA GATE (1D OBSTACLE-X SWEEP @ y=0.5) -- consolidated"
echo "================================================================"
python3 <<'EOF'
import json, glob
OUT = "phase6_obs_pos_gate_outputs"
# pick up just the 1D-sweep cells (y=0.5, x in the swept positions).
# Filename has signed format `x+1.5_y+0.5_best.json` (note the +), so we
# load everything and filter in Python rather than fight glob escapes.
TARGET_XS = {1.5, 2.5, 3.5, 4.5}
rows = []
for f in sorted(glob.glob(f"{OUT}/obs_pos_*_best.json")):
    d = json.load(open(f))
    if abs(d["obs_y"] - 0.5) < 0.01 and d["obs_x"] in TARGET_XS:
        rows.append(d)
rows.sort(key=lambda r: r["obs_x"])
print(f"  {'obs_x':>6}  {'obs_y':>6}  {'best_alpha':>11}  {'coll':>5}  "
      f"{'reach':>5}  {'int':>6}  {'track':>7}  tag")
for r in rows:
    print(f"  {r['obs_x']:>+6.2f}  {r['obs_y']:>+6.2f}  {r['alpha']:>11.2f}  "
          f"{r['collision_rate']:>5.2f}  {r['reach_rate']:>5.2f}  "
          f"{r['intervention_mean']:>6.0f}  {r['tracking_err_mean']:>7.3f}  "
          f"{r['tag']}")
alpha_span = max(r['alpha'] for r in rows) - min(r['alpha'] for r in rows)
int_span = max(r['intervention_mean'] for r in rows) - min(r['intervention_mean'] for r in rows)
pct = 100 * alpha_span / 3.8
print()
print(f"  best-alpha span:    {alpha_span:.2f}  ({pct:.0f}% of bound width [0.2, 4.0])")
print(f"  intervention span:  {int_span:.0f}")
print()
verdict = ("PASS" if alpha_span >= 1.0
           else "WEAK" if alpha_span >= 0.5
           else "FAIL")
print(f"  verdict: {verdict}")
with open(f"{OUT}/obs_pos_1d_consolidated.json", "w") as f:
    json.dump({"per_position": rows, "alpha_span": alpha_span,
               "int_span": int_span, "pct_bound": pct,
               "verdict": verdict}, f, indent=2)
print(f"  saved -> {OUT}/obs_pos_1d_consolidated.json")
EOF
