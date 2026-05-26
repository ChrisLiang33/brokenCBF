#!/bin/bash
# Background training health watcher.
#
# Polls the in-progress training log every 5 minutes, runs
# extract_training_summary.py on it, and writes a clean text log
# of results + trip-wire warnings.
#
# Output is dual-written: terminal AND a derived log file at
# `<training_log>.health.log`. You can cat / tail / copy-paste from
# the file at any time without watching the script live.
#
# Designed to run in a separate tmux pane while training runs in pane A.
# Doesn't block, doesn't interfere with the main run.
#
# Usage:
#   bash ~/Desktop/safety-go2/scripts/watch_training_health.sh \
#     ~/Desktop/safety-go2/IsaacLab/logs/train_and_eval_v211.log
#   # appends to: ~/Desktop/safety-go2/IsaacLab/logs/train_and_eval_v211.health.log
#
# Exits cleanly on Ctrl-C.

set -u

if [ $# -lt 1 ]; then
    echo "Usage: $0 <training_log_path> [poll_interval_seconds]" >&2
    echo "  default poll interval: 300 (5 min)" >&2
    echo "  health output is appended to <training_log>.health.log" >&2
    exit 2
fi

LOG="$1"
INTERVAL="${2:-300}"
EXTRACT="$HOME/Desktop/safety-go2/scripts/extract_training_summary.py"
CSV="${LOG%.log}.training_summary.csv"
HEALTH_LOG="${LOG%.log}.health.log"

# Trip-wire thresholds (mirror the v2.6 paper-baseline gates)
TRIP_R_STUCK="-0.20"        # r_stuck climbing past this means policy is freezing
TRIP_TERM_BC="0.10"         # term_base_contact above this means too many falls

# Header — emit before the redirect so the user sees where the log is
echo "watch_training_health.sh started"
echo "  Training log:  $LOG"
echo "  Health log:    $HEALTH_LOG  (cat / tail / copy-paste from here)"
echo "  Poll interval: ${INTERVAL}s"
echo "  Trip wires:    r_stuck < $TRIP_R_STUCK, term_base_contact > $TRIP_TERM_BC"
echo ""

# Dual-write subsequent output: terminal AND append to the health log.
# Plain text only — no ANSI escape codes, so copy-paste from the file is clean.
exec > >(tee -a "$HEALTH_LOG") 2>&1

trap 'echo ""; echo "Watch stopped at $(date)."; exit 0' INT TERM

# One-time banner inside the redirected stream so the file has provenance
{
    echo "============================================================"
    echo "watch_training_health.sh session started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Training log: $LOG"
    echo "Trip wires:   r_stuck < $TRIP_R_STUCK, term_base_contact > $TRIP_TERM_BC"
    echo "============================================================"
}

while true; do
    echo ""
    echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---"

    if [ ! -f "$LOG" ]; then
        echo "log not found yet (training may not have started writing)"
        sleep "$INTERVAL"
        continue
    fi

    # Re-extract the summary (script handles partial logs gracefully)
    python3 "$EXTRACT" "$LOG" > /tmp/wth_extract_out.txt 2>&1
    EXTRACT_EXIT=$?
    if [ "$EXTRACT_EXIT" -ne 0 ]; then
        echo "extract_training_summary.py returned ${EXTRACT_EXIT}; output:"
        cat /tmp/wth_extract_out.txt
        sleep "$INTERVAL"
        continue
    fi

    # Show the spot-check block (last ~60 lines covers the table comfortably
    # now that we have 27 metrics including the 12 new CBF stats)
    tail -60 /tmp/wth_extract_out.txt

    # Trip-wire check on latest CSV row
    if [ -f "$CSV" ]; then
        # Header line names → column indices (so this stays robust if the
        # extractor adds/removes columns later).
        HEADER=$(head -1 "$CSV")
        IDX_R_STUCK=$(echo "$HEADER" | tr ',' '\n' | grep -nx 'r_stuck' | cut -d: -f1)
        IDX_TERM_BC=$(echo "$HEADER" | tr ',' '\n' | grep -nx 'term_base_contact' | cut -d: -f1)
        IDX_ITER=$(echo "$HEADER" | tr ',' '\n' | grep -nx 'iter' | cut -d: -f1)

        if [ -n "$IDX_R_STUCK" ] && [ -n "$IDX_TERM_BC" ]; then
            LAST=$(tail -1 "$CSV")
            ITER=$(echo "$LAST"     | cut -d, -f"$IDX_ITER")
            R_STUCK=$(echo "$LAST"  | cut -d, -f"$IDX_R_STUCK")
            TERM_BC=$(echo "$LAST"  | cut -d, -f"$IDX_TERM_BC")

            # bc returns 1 if true, 0 if false; -l for floats
            STUCK_CROSSED=$(echo "$R_STUCK < $TRIP_R_STUCK" | bc -l 2>/dev/null || echo 0)
            BC_CROSSED=$(echo "$TERM_BC > $TRIP_TERM_BC"   | bc -l 2>/dev/null || echo 0)

            echo ""
            echo "Latest iter $ITER: r_stuck=$R_STUCK, term_base_contact=$TERM_BC"

            if [ "$STUCK_CROSSED" = "1" ]; then
                echo "*** TRIP WIRE: r_stuck=$R_STUCK past $TRIP_R_STUCK - policy may be freezing ***"
            fi
            if [ "$BC_CROSSED" = "1" ]; then
                echo "*** TRIP WIRE: term_base_contact=$TERM_BC above $TRIP_TERM_BC - too many falls ***"
            fi
            if [ "$STUCK_CROSSED" != "1" ] && [ "$BC_CROSSED" != "1" ]; then
                echo "[OK] trip wires clear"
            fi
        fi
    fi

    echo ""
    sleep "$INTERVAL"
done
