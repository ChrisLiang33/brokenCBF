#!/bin/bash
# Run the per-axis stress sweep on all 4 candidate teachers sequentially
# (2026-05-17). Each teacher takes ~30 min (6 axes × ~5 min), so total
# wall time is ~2h.
#
# Teachers:
#   lockedbest  — archive PUSH_A_C, joint_actual 0.724 (locked best)
#   defl        — locked-best + L2 deflection penalty, joint_actual 0.611 (regressed)
#   omni        — archive omniscient teacher, joint_actual 0.503 (regressed)
#   omnidefl    — omni + L2 deflection penalty, joint_actual 0.377 (regressed)
#
# Even the regressed teachers are informative — the per-axis table tells
# us whether the workaround killed adaptation uniformly or selectively.
#
# Usage on lab box (in tmux):
#   tmux new -s wk3stress4
#   conda activate isaaclab
#   cd ~/Desktop/safety-go2/IsaacLab
#   ~/Desktop/safety-go2/scripts/stress_eval_all4.sh \
#     2>&1 | tee logs/stress_eval_all4.log

set -u

cd ~/Desktop/safety-go2/IsaacLab

# Pick latest checkpoint from each training run directory.
pick_ckpt () {
    local ts="$1"
    local d="logs/rsl_rl/cbf_go2_teacher_rma/${ts}"
    ls -1 "${d}"/model_*.pt 2>/dev/null \
        | awk -F'model_|\\.pt' '{print $2, $0}' \
        | sort -n | awk '{print $2}' | tail -1
}

LOCKEDBEST_CKPT=$(pick_ckpt "2026-05-16_12-45-39")
DEFL_CKPT=$(pick_ckpt "2026-05-17_08-57-53")
OMNI_CKPT=$(pick_ckpt "2026-05-17_00-49-10")
OMNIDEFL_CKPT=$(pick_ckpt "2026-05-17_10-31-16")

echo "================================================================"
echo "Stress eval — all 4 teachers, sequential"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Teachers:"
echo "  lockedbest:  $LOCKEDBEST_CKPT"
echo "  defl:        $DEFL_CKPT"
echo "  omni:        $OMNI_CKPT"
echo "  omnidefl:    $OMNIDEFL_CKPT"
echo "================================================================"

for t in lockedbest defl omni omnidefl; do
    case "$t" in
        lockedbest) ckpt="$LOCKEDBEST_CKPT" ;;
        defl)       ckpt="$DEFL_CKPT" ;;
        omni)       ckpt="$OMNI_CKPT" ;;
        omnidefl)   ckpt="$OMNIDEFL_CKPT" ;;
    esac

    if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
        echo ""
        echo "⚠ Skipping ${t}: checkpoint not found ($ckpt)"
        continue
    fi

    echo ""
    echo "################################################################"
    echo "# Teacher: ${t}"
    echo "# Started: $(date '+%H:%M:%S')"
    echo "################################################################"

    ~/Desktop/safety-go2/scripts/stress_eval_sweep.sh "$ckpt" "$t" \
        || echo "⚠ sweep for ${t} failed (continuing)"
done

echo ""
echo "================================================================"
echo "ALL 4 DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Per-teacher attribution tables:"
for t in lockedbest defl omni omnidefl; do
    echo "  python3 ~/Desktop/safety-go2/scripts/parse_stress_eval.py --label ${t}"
done
echo "================================================================"
