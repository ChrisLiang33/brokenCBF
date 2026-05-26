#!/usr/bin/env bash
# Loop the alpha gate across terrain levels. Each level is a fresh
# Python invocation (Isaac Sim can't cleanly rebuild a scene within
# one process). Per-level outputs are merged into one summary table
# at the end.
#
# Run on labbox:
#     bash phase6_alpha_gate.sh

set -euo pipefail
cd "$(dirname "$0")"

LOCO="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt"
ISAACLAB="$HOME/IsaacLab/isaaclab.sh"
OUT="phase6_alpha_gate_outputs"
LOGS="phase6_alpha_gate_logs"
mkdir -p "$OUT" "$LOGS"

for LEVEL in 4 5 6; do    # levels 0..3 already done; harder levels here
    echo ""
    echo "================================================================"
    echo "  Alpha gate, terrain level $LEVEL  (started $(date '+%H:%M:%S'))"
    echo "================================================================"
    "$ISAACLAB" -p phase6_alpha_gate.py \
        --checkpoint "$LOCO" \
        --num_envs 256 --level "$LEVEL" --headless \
        > "$LOGS/level_${LEVEL}.log" 2>&1
    # echo just the best line + verdict text from the per-level summary
    if [ -f "$OUT/alpha_gate_level${LEVEL}_best.json" ]; then
        BEST_ALPHA=$(python3 -c "import json; d=json.load(open('$OUT/alpha_gate_level${LEVEL}_best.json')); print(d['alpha'])")
        BEST_COLL=$(python3 -c "import json; d=json.load(open('$OUT/alpha_gate_level${LEVEL}_best.json')); print(f\"{d['collision_rate']:.2f}\")")
        BEST_REACH=$(python3 -c "import json; d=json.load(open('$OUT/alpha_gate_level${LEVEL}_best.json')); print(f\"{d['reach_rate']:.2f}\")")
        BEST_TRACK=$(python3 -c "import json; d=json.load(open('$OUT/alpha_gate_level${LEVEL}_best.json')); print(f\"{d['tracking_err_mean']:.3f}\")")
        echo "  best alpha @ level $LEVEL: $BEST_ALPHA  (coll=$BEST_COLL, reach=$BEST_REACH, track=$BEST_TRACK)"
    fi
done

echo ""
echo "================================================================"
echo "  STRONG ALPHA GATE  --  consolidated"
echo "================================================================"
python3 <<'EOF'
import json, glob, os
OUT = "phase6_alpha_gate_outputs"
files = sorted(glob.glob(f"{OUT}/alpha_gate_level*_best.json"))
rows = [json.load(open(f)) for f in files]
rows.sort(key=lambda r: r["level"])
print(f"  {'level':>6}  {'best_alpha':>11}  {'coll':>5}  {'reach':>5}  {'int':>6}  {'track_err':>10}  tag")
for r in rows:
    print(f"  {r['level']:>6}  {r['alpha']:>11.2f}  "
          f"{r['collision_rate']:>5.2f}  {r['reach_rate']:>5.2f}  "
          f"{r['intervention_mean']:>6.0f}  {r['tracking_err_mean']:>10.3f}  "
          f"{r['tag']}")
alpha_span = max(r['alpha'] for r in rows) - min(r['alpha'] for r in rows)
track_span = max(r['tracking_err_mean'] for r in rows) - min(r['tracking_err_mean'] for r in rows)
print()
print(f"  best-alpha span across levels:    {alpha_span:.2f}")
print(f"  tracking_err span across levels:  {track_span:.3f}")
if alpha_span >= 1.0:
    verdict = "PASS -- optimal alpha shifts meaningfully with roughness"
elif alpha_span >= 0.5:
    verdict = "WEAK -- some shift, may not be strong enough"
else:
    verdict = "FAIL -- optimal alpha is roughly flat across roughness"
print(f"  verdict: {verdict}")
with open(f"{OUT}/alpha_gate_consolidated.json", "w") as f:
    json.dump({"per_level": rows, "alpha_span": alpha_span,
               "track_span": track_span, "verdict": verdict}, f, indent=2)
print(f"  saved -> {OUT}/alpha_gate_consolidated.json")
EOF
