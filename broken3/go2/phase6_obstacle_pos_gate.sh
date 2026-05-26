#!/usr/bin/env bash
# Sweep obstacle position; for each position find the best fixed alpha.
# Looks for whether optimal alpha shifts across positions -- a real
# state-conditional alpha signal that terrain failed to provide.
#
# 5 positions × 7 alpha values × ~15s/cell + 5 Isaac startups
# = ~12-15 min total.

set -euo pipefail
cd "$(dirname "$0")"

LOCO="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt"
ISAACLAB="$HOME/IsaacLab/isaaclab.sh"
OUT="phase6_obs_pos_gate_outputs"
LOGS="phase6_obs_pos_gate_logs"
mkdir -p "$OUT" "$LOGS"

# (x, y) obstacle positions to test. Robot starts at (0, 0), goal at (6, 0).
# Mix of "directly on path" vs "off to the side" at near/mid distances.
POSITIONS=(
    "2.5  0.0"
    "3.0  0.5"
    "3.0 -0.5"
    "3.5  1.0"
    "4.0  1.5"
)

for POS in "${POSITIONS[@]}"; do
    read -r OX OY <<< "$POS"
    TAG="x${OX}_y${OY}"
    echo ""
    echo "================================================================"
    echo "  Obstacle position gate: ($OX, $OY)  (started $(date '+%H:%M:%S'))"
    echo "================================================================"
    "$ISAACLAB" -p phase6_obstacle_pos_gate.py \
        --checkpoint "$LOCO" --num_envs 256 \
        --obstacle_x "$OX" --obstacle_y "$OY" --headless \
        > "$LOGS/pos_${TAG}.log" 2>&1
done

echo ""
echo "================================================================"
echo "  STRONG ALPHA GATE (OBSTACLE POSITION) -- consolidated"
echo "================================================================"
python3 <<'EOF'
import json, glob, os
OUT = "phase6_obs_pos_gate_outputs"
files = sorted(glob.glob(f"{OUT}/obs_pos_*_best.json"))
rows = [json.load(open(f)) for f in files]
print(f"  {'obs_x':>6}  {'obs_y':>6}  {'best_alpha':>11}  {'coll':>5}  "
      f"{'reach':>5}  {'int':>6}  {'track':>7}  tag")
for r in rows:
    print(f"  {r['obs_x']:>+6.2f}  {r['obs_y']:>+6.2f}  {r['alpha']:>11.2f}  "
          f"{r['collision_rate']:>5.2f}  {r['reach_rate']:>5.2f}  "
          f"{r['intervention_mean']:>6.0f}  {r['tracking_err_mean']:>7.3f}  "
          f"{r['tag']}")
alpha_span = max(r['alpha'] for r in rows) - min(r['alpha'] for r in rows)
int_span = max(r['intervention_mean'] for r in rows) - min(r['intervention_mean'] for r in rows)
print()
print(f"  best-alpha span across positions:   {alpha_span:.2f}")
print(f"  intervention span across positions: {int_span:.0f}")
if alpha_span >= 1.0:
    verdict = "PASS -- optimal alpha shifts meaningfully with obstacle position"
elif alpha_span >= 0.5:
    verdict = "WEAK -- some shift, may not be strong enough"
else:
    verdict = "FAIL -- optimal alpha is roughly flat across positions"
print(f"  verdict: {verdict}")
with open(f"{OUT}/obs_pos_consolidated.json", "w") as f:
    json.dump({"per_position": rows, "alpha_span": alpha_span,
               "int_span": int_span, "verdict": verdict}, f, indent=2)
print(f"  saved -> {OUT}/obs_pos_consolidated.json")
EOF
