#!/bin/bash
# Record a single hardware trial of the V13.1 CBF deploy stack on the Go2.
#
# Captures all the topics we need to plot trajectory, parameter traces,
# h(x), deflection, and ground truth obstacle from the LiDAR cloud.
#
# Usage on Go2:
#   bash deploy/record_trial.sh <condition_name> <trial_idx>
#   bash deploy/record_trial.sh ours_v13_1     1
#   bash deploy/record_trial.sh fixed_b1_a2_p05 1
#   bash deploy/record_trial.sh raw_no_cbf     1
#
# Each trial saves to: ~/safety-go2/trials/<condition_name>/trial_<idx>/
#
# Press Ctrl+C to stop recording. Recommended trial length: ~10-15 seconds.

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <condition_name> <trial_idx>"
  echo "Example: $0 ours_v13_1 1"
  exit 1
fi

CONDITION=$1
TRIAL_IDX=$2
TRIALS_DIR=$HOME/safety-go2/trials/$CONDITION
mkdir -p "$TRIALS_DIR"

# Bag name: trial_<NN>
BAG_NAME=$(printf "trial_%02d" "$TRIAL_IDX")
BAG_PATH=$TRIALS_DIR/$BAG_NAME

echo "================================================================"
echo "RECORDING: $CONDITION / $BAG_NAME"
echo "  Saving to: $BAG_PATH"
echo "  Topics:"
echo "    /odom                  — robot pose + velocity"
echo "    /cbf/params            — α, φ, a, b, c (50 Hz)"
echo "    /cbf/inference_status  — fail-safe state"
echo "    /cbf/filter_h          — h(x), L_g h, slack"
echo "    /cbf/filter_status     — filter state"
echo "    /u_teleop              — raw user command"
echo "    /u_des                 — CBF-filtered command"
echo "    /poisson_cloud         — filtered LiDAR (large; throttled to 5Hz)"
echo "    /sportmodestate        — full Go2 state (IMU, contacts, etc.)"
echo "    /tf, /tf_static        — TF frames"
echo ""
echo "Press Ctrl+C to STOP RECORDING when the trial is done."
echo "================================================================"

# Source ROS env
source ~/safety-go2/install/setup.bash

# Note: /poisson_cloud + /cbf/grid are large (8 MB/s each). We skip /cbf/grid
# (8192 floats / msg @ 50 Hz). poisson_cloud gets logged at native rate;
# user can throttle post-hoc with rosbag2 filter if size is an issue.

ros2 bag record \
  -o "$BAG_PATH" \
  /odom \
  /cbf/params \
  /cbf/inference_status \
  /cbf/filter_h \
  /cbf/filter_status \
  /u_teleop \
  /u_des \
  /poisson_cloud \
  /sportmodestate \
  /tf \
  /tf_static

echo ""
echo "================================================================"
echo "Trial saved: $BAG_PATH"
echo "Inspect with:    ros2 bag info $BAG_PATH"
echo "Replay with:     ros2 bag play $BAG_PATH"
echo "================================================================"
