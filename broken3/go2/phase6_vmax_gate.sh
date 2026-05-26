#!/usr/bin/env bash
# Strong α gate via commanded velocity (v_max).
# Theory: stopping distance ~ v² / (2α). Doubling v should ~halve
# optimal α if α is genuinely tracking kinematic urgency.
#
# 4 v_max values × 7 alphas + 4 Isaac startups = ~10 min.

set -euo pipefail
cd "$(dirname "$0")"

LOCO="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt"
ISAACLAB="$HOME/IsaacLab/isaaclab.sh"
OUT="phase6_vmax_gate_outputs"
LOGS="phase6_vmax_gate_logs"
mkdir -p "$OUT" "$LOGS"

# obstacle pinned at off-path feasible position so v_max is the only var
OBS_X=3.0
OBS_Y=0.5

V_MAXES=(0.5 1.0 1.5 2.0)

for V in "${V_MAXES[@]}"; do
    echo ""
    echo "================================================================"
    echo "  v_max gate: v=$V  obs=($OBS_X,$OBS_Y)  (started $(date '+%H:%M:%S'))"
    echo "================================================================"
    "$ISAACLAB" -p phase6_vmax_gate.py \
        --checkpoint "$LOCO" --num_envs 256 \
        --v_max "$V" --obstacle_x "$OBS_X" --obstacle_y "$OBS_Y" --headless \
        > "$LOGS/v_${V}.log" 2>&1
done

echo ""
echo "================================================================"
echo "  STRONG ALPHA GATE (v_max SWEEP) -- consolidated"
echo "================================================================"
python3 <<'EOF'
import json, glob
OUT = "phase6_vmax_gate_outputs"
files = sorted(glob.glob(f"{OUT}/vmax_*_best.json"))
rows = [json.load(open(f)) for f in files]
rows.sort(key=lambda r: r["v_max"])
print(f"  {'v_max':>6}  {'best_alpha':>11}  {'coll':>5}  {'reach':>5}  "
      f"{'int':>6}  {'track':>7}  tag")
for r in rows:
    print(f"  {r['v_max']:>6.2f}  {r['alpha']:>11.2f}  "
          f"{r['collision_rate']:>5.2f}  {r['reach_rate']:>5.2f}  "
          f"{r['intervention_mean']:>6.0f}  {r['tracking_err_mean']:>7.3f}  "
          f"{r['tag']}")
alpha_span = max(r['alpha'] for r in rows) - min(r['alpha'] for r in rows)
int_span = max(r['intervention_mean'] for r in rows) - min(r['intervention_mean'] for r in rows)
pct = 100 * alpha_span / 3.8
print()
print(f"  best-alpha span:    {alpha_span:.2f}  ({pct:.0f}% of bound width)")
print(f"  intervention span:  {int_span:.0f}")
verdict = ("PASS" if alpha_span >= 1.0
           else "WEAK" if alpha_span >= 0.5
           else "FAIL")
print(f"  verdict: {verdict}")
with open(f"{OUT}/vmax_consolidated.json", "w") as f:
    json.dump({"per_v": rows, "alpha_span": alpha_span,
               "int_span": int_span, "pct_bound": pct,
               "verdict": verdict}, f, indent=2)
print(f"  saved -> {OUT}/vmax_consolidated.json")
EOF
