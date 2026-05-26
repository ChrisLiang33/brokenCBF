#!/usr/bin/env bash
# Source this (don't execute) to put cbf_go2's runtime deps on the path.
#
#   source setup_env.sh
#   ./build.sh
#   ros2 launch cbf_go2 cbf_go2.launch.py
#
# Required external workspaces (cloned/built elsewhere on the Jetson):
#   - ROS 2 (humble preferred, foxy works)
#   - unitree_ros2     (Unitree's SDK + messages, from semantic-safety/submodules/unitree_ros2 or upstream)
#   - livox_ros_driver2 (semantic-safety/submodules/ws_livox)
#
# Adjust the paths below to match the deployment machine. The defaults
# point at the lab Jetson Orin layout used by semantic-safety/.

set -u

# --- ROS 2 base ---
if [ -z "${ROS_DISTRO:-}" ]; then
    if [ -f /opt/ros/humble/setup.bash ]; then
        source /opt/ros/humble/setup.bash
    elif [ -f /opt/ros/foxy/setup.bash ]; then
        source /opt/ros/foxy/setup.bash
    else
        echo "[setup_env] ERROR: ROS 2 not found at /opt/ros/{humble,foxy}." >&2
        return 1 2>/dev/null || exit 1
    fi
fi

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Unitree SDK + messages ---
# Default: alongside this workspace under ../semantic-safety/submodules/unitree_ros2.
UNITREE_WS="${UNITREE_WS:-$WS_ROOT/../semantic-safety/submodules/unitree_ros2}"
if [ -f "$UNITREE_WS/install/setup.bash" ]; then
    source "$UNITREE_WS/install/setup.bash"
else
    echo "[setup_env] WARN: unitree_ros2 overlay not built at $UNITREE_WS/install."
    echo "             Build it before running on the Go2."
fi

# --- Livox Mid360 driver ---
LIVOX_WS="${LIVOX_WS:-$WS_ROOT/../semantic-safety/submodules/ws_livox}"
if [ -f "$LIVOX_WS/install/setup.bash" ]; then
    source "$LIVOX_WS/install/setup.bash"
else
    echo "[setup_env] WARN: livox_ros_driver2 overlay not built at $LIVOX_WS/install."
fi

# --- Our overlay (if built) ---
if [ -f "$WS_ROOT/install/setup.bash" ]; then
    source "$WS_ROOT/install/setup.bash"
fi

echo "[setup_env] ROS_DISTRO=$ROS_DISTRO  WS=$WS_ROOT"
