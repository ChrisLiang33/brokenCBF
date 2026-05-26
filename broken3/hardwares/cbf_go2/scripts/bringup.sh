#!/usr/bin/env bash
# Tier 3 bring-up — launch the full cbf_go2 stack in tmux windows.
#
# Order matches data dependencies (each waits ~1 s for its upstream):
#   1. odom_publisher                    (/sportmodestate -> /odom)
#   2. static TFs                        (body, livox_frame, body_link, utlidar_lidar)
#   3. livox_lidar_publisher             (/livox/lidar)
#   4. cloud_merger                      (/poisson_cloud + /occupancy_grid)
#   5. cbf_grid_node                     (/cbf/grid)
#   6. cbf_inference_node                (/cbf/params)
#   7. cbf_filter_node                   (/u_des)
#   8. teleop                            (keyboard -> /u_teleop)
#
# Does NOT start walking_bridge — the robot does not move from this
# script. Start it manually in a separate terminal once perception +
# inference + filter look healthy:
#
#   source ~/cbf_rl_mvp/hardwares/cbf_go2/install/setup.bash
#   ros2 run cbf_go2 walking_bridge
#
# Usage:
#   bash scripts/bringup.sh                 # stub mode (SAFE_DEFAULTS)
#   CHECKPOINT=/path/to/policy.pt bash scripts/bringup.sh
set -euo pipefail

SESSION=cbf_go2
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP="source $WS_ROOT/install/setup.bash"

CHECKPOINT="${CHECKPOINT:-}"
if [ -n "$CHECKPOINT" ] && [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "================================================================"
echo "cbf_go2 bring-up"
echo "  workspace: $WS_ROOT"
[ -n "$CHECKPOINT" ] && echo "  checkpoint: $CHECKPOINT" || echo "  checkpoint: (none — stub mode)"
echo "================================================================"

launch_window() {
    local wname="$1"; local cmd="$2"
    tmux new-window -t "$SESSION" -n "$wname" \
        "bash -c '$SETUP; $cmd; echo; echo \"[$wname exited]\"; exec bash'"
}

# First window starts the session.
tmux new-session -d -s "$SESSION" -n odom \
    "bash -c '$SETUP; ros2 run cbf_go2 odom_publisher; echo; exec bash'"
sleep 1

launch_window "tf_body_livox" \
    "ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 body livox_frame"
launch_window "tf_livox_body_link" \
    "ros2 run tf2_ros static_transform_publisher -0.05 0.0 0.18 0 3.14159 0 livox_frame body_link"
launch_window "tf_body_link_utlidar" \
    "ros2 run tf2_ros static_transform_publisher 0.37 0.0 0.05 0 2.9 0 body_link utlidar_lidar"
sleep 1

# Livox driver — must come before cloud_merger.
launch_window "livox" \
    "ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args -p user_config_path:=$WS_ROOT/install/cbf_go2/share/cbf_go2/config/MID360_config.json -p frame_id:=livox_frame"
sleep 2

launch_window "cloud_merger" "ros2 run cbf_go2 cloud_merger"
sleep 2

launch_window "grid_node" "ros2 run cbf_go2 cbf_grid_node"
sleep 1

INF_CMD="ros2 run cbf_go2 cbf_inference_node"
[ -n "$CHECKPOINT" ] && INF_CMD="$INF_CMD --ros-args -p checkpoint:=$CHECKPOINT"
launch_window "inference_node" "$INF_CMD"
sleep 1

launch_window "filter_node" "ros2 run cbf_go2 cbf_filter_node"
sleep 1

launch_window "teleop" "ros2 run cbf_go2 teleOp"
sleep 2

cat <<EOF

================================================================
All nodes launched in tmux session '$SESSION'.

  tmux attach -t $SESSION       — see all windows
  (Ctrl-B then window# to switch)

Quick health check (in a separate terminal):
  $SETUP
  ros2 node list | grep -E 'cbf|cloud|odom'
  ros2 topic echo /cbf/inference_status    # state=OK or INPUT_STALE
  ros2 topic echo /cbf/filter_status       # state=OK / OK_DEFLECTED / PASSTHROUGH_*

--- ROBOT IS NOT MOVING YET ---
To enable motion (CBF filter active):
  $SETUP
  ros2 run cbf_go2 walking_bridge

Tear down:  tmux kill-session -t $SESSION
================================================================
EOF
