#!/usr/bin/env bash
# Build cbf_go2 in this workspace.
#
# Assumes setup_env.sh has been sourced first (ROS 2 + Livox + Unitree
# on the path). Builds the single `cbf_go2` package with symlink-install
# so Python node edits don't require a rebuild.
set -euo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WS_ROOT"

colcon build --packages-select cbf_go2 --symlink-install
echo
echo "Build complete. Source the overlay:"
echo "  source $WS_ROOT/install/setup.bash"
