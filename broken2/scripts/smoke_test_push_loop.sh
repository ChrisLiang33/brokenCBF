#!/bin/bash
# Push-magnitude smoke test wrapper.
#
# Isaac Lab doesn't cleanly support creating multiple envs in one app
# launch, so we launch the test once per magnitude. Per-magnitude
# stdout/stderr is written to ~/Desktop/safety-go2/logs/smoke_push_mag<m>.log
# so any failure surfaces — earlier "capture with $()" version silently
# ate error output.
#
# Usage (lab box, from anywhere):
#   bash ~/Desktop/safety-go2/scripts/smoke_test_push_loop.sh \
#        2>&1 | tee ~/Desktop/safety-go2/logs/smoke_push_loop.log

set -u

cd ~/Desktop/safety-go2/IsaacLab

MAGNITUDES=${MAGNITUDES:-"0.5 0.75 1.0 1.5"}
TOTAL_STEPS=${TOTAL_STEPS:-1500}
NUM_ENVS=${NUM_ENVS:-64}
INTERVAL_MIN=${INTERVAL_MIN:-5.0}
INTERVAL_MAX=${INTERVAL_MAX:-10.0}
LOG_DIR=${LOG_DIR:-~/Desktop/safety-go2/logs}

mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Push-magnitude smoke test loop"
echo "Magnitudes:    ${MAGNITUDES}"
echo "Total steps:   ${TOTAL_STEPS}"
echo "Num envs:      ${NUM_ENVS}"
echo "Interval (s):  (${INTERVAL_MIN}, ${INTERVAL_MAX})"
echo "Per-mag logs:  ${LOG_DIR}/smoke_push_mag<m>.log"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

RESULTS=()

for mag in $MAGNITUDES; do
    echo ""
    echo "─── magnitude ${mag} m/s ───"
    PER_MAG_LOG="${LOG_DIR}/smoke_push_mag${mag}.log"
    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/smoke_test_push.py \
        --headless \
        --magnitude "$mag" \
        --num_envs "$NUM_ENVS" \
        --total_steps "$TOTAL_STEPS" \
        --interval_s "$INTERVAL_MIN" "$INTERVAL_MAX" \
        > "$PER_MAG_LOG" 2>&1 \
        || echo "  ⚠ non-zero exit for magnitude ${mag} (see ${PER_MAG_LOG})"

    LINE=$(grep "\[smoke_test_push\] RESULT" "$PER_MAG_LOG" | tail -1)
    if [ -n "$LINE" ]; then
        echo "  ${LINE}"
        RESULTS+=("$LINE")
    else
        echo "  ✗ no RESULT line for magnitude ${mag}; tail of log:"
        tail -10 "$PER_MAG_LOG" | sed 's/^/      /'
    fi
done

echo ""
echo "================================================================"
echo "SUMMARY"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
