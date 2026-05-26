#!/bin/bash
# Tier 3 bring-up: launch the full V13.1 CBF deploy stack on the Go2.
#
# Starts 8 nodes in a single tmux session (one node per window) IN ORDER:
#   1. odom_publisher
#   2. 3× static_transform_publisher (body_link↔utlidar, livox↔body, body↔livox)
#   3. cloud_merger
#   4. cbf_grid_node
#   5. cbf_inference_node (loads V13.1 teacher + student)
#   6. cbf_filter_node
#   7. teleOp (keyboard)
#
# DOES NOT START walking_bridge. The robot won't move from this script.
# After everything below is healthy, MANUALLY start walking_bridge in a
# fresh terminal when ready:
#
#   source ~/safety-go2/install/setup.bash
#   ros2 run go2_walking_lidar walking_bridge
#
# (no remap this time — walking_bridge consumes /u_des, which is the
# CBF-filtered output of teleop. Filter IS active.)
#
# Usage:
#   bash ~/safety-go2/deploy/scripts/tier3_bringup.sh
#
# Then:
#   tmux attach -t tier3                  # see all panes
#   tmux kill-session -t tier3            # nuke everything when done

set -e

SESSION=tier3
TEACHER=$HOME/safety-go2/checkpoints/model_2499.pt
STUDENT=$HOME/safety-go2/checkpoints/student_v13_1.pt
SETUP="source ~/safety-go2/install/setup.bash"

# Sanity: checkpoints exist
[ -f "$TEACHER" ] || { echo "ERROR: teacher ckpt missing: $TEACHER"; exit 1; }
[ -f "$STUDENT" ] || { echo "ERROR: student ckpt missing: $STUDENT"; exit 1; }

# Kill any previous session
tmux kill-session -t $SESSION 2>/dev/null || true

echo "================================================================"
echo "Tier 3 bring-up — V13.1 full deploy stack"
echo "  teacher: $TEACHER"
echo "  student: $STUDENT"
echo "================================================================"

# Helper: open a new tmux window running a command in a bash shell that
# stays alive after the command exits (so you can see errors).
launch_window() {
  local wname="$1"; local cmd="$2"
  tmux new-window -t $SESSION -n "$wname" \
    "bash -c '$SETUP; $cmd; echo; echo \"[$wname exited]\"; exec bash'"
}

# Start session with the first window
tmux new-session -d -s $SESSION -n odom \
  "bash -c '$SETUP; ros2 run go2_walking_lidar odom_publisher; echo; exec bash'"
sleep 1

# Static TFs
launch_window "tf_body_utlidar" \
  "ros2 run tf2_ros static_transform_publisher 0.37 0.0 0.05 0 2.9 0 body_link utlidar_lidar"
launch_window "tf_livox_body" \
  "ros2 run tf2_ros static_transform_publisher -0.05 0.0 0.18 0 3.14159 0 livox_frame body_link"
launch_window "tf_body_livox" \
  "ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 body livox_frame"

sleep 1

# Cloud merger (depends on TFs)
launch_window "cloud_merger" \
  "ros2 run go2_walking_lidar cloud_merger"
sleep 2

# Grid node (depends on /poisson_cloud from cloud_merger)
launch_window "grid_node" \
  "ros2 run go2_walking_lidar cbf_grid_node"
sleep 1

# Inference node (no LiDAR dependency, but waits for /odom + /u_teleop)
launch_window "inference_node" \
  "ros2 run go2_walking_lidar cbf_inference_node --ros-args -p teacher_ckpt:=$TEACHER -p student_ckpt:=$STUDENT"
sleep 1

# Filter node (consumes /u_teleop + /cbf/params + /poisson_cloud)
launch_window "filter_node" \
  "ros2 run go2_walking_lidar cbf_filter_node"
sleep 1

# Teleop (curses keyboard) — needs interactive terminal, gets one tmux window
launch_window "teleop" \
  "ros2 run go2_walking_lidar teleOp"

sleep 2

echo ""
echo "================================================================"
echo "All 8 nodes launched in tmux session '$SESSION'."
echo ""
echo "  tmux attach -t $SESSION       — see all windows"
echo "  (then Ctrl-B + window# to switch)"
echo ""
echo "Quick verification (in a SEPARATE terminal):"
echo "  $SETUP"
echo "  ros2 node list | grep -E 'cbf|cloud|odom'   # should show 5 nodes"
echo "  ros2 topic echo /cbf/inference_status        # should show state=OK"
echo "  ros2 topic echo /cbf/filter_status           # should show state=OK or OK_DEFLECTED"
echo ""
echo "─── ROBOT IS NOT MOVING YET ───"
echo "When you're ready to enable robot motion (CBF filter active):"
echo ""
echo "  source ~/safety-go2/install/setup.bash"
echo "  ros2 run go2_walking_lidar walking_bridge"
echo ""
echo "Robot will RecoveryStand on startup. Use arrow keys in the teleop"
echo "window to drive. CBF will deflect/intervene on real LiDAR data."
echo ""
echo "To kill everything:  tmux kill-session -t $SESSION"
echo "================================================================"
