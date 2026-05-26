#!/bin/bash
# Render a video of V8 BR teacher running on a chosen deploy task.
# Uses Isaac Lab's default viewport resolution (typically 1280x720).
# For higher-res, set env_cfg.viewer.resolution in cfg instead of CLI.
#
# Usage:
#   tmux new -s render
#   cd ~/Desktop/safety-go2/IsaacLab
#   ~/Desktop/safety-go2/scripts/render_v8_video.sh
#
# Customize via env vars:
#   TASK=Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-v0 ...

set -e
cd ~/Desktop/safety-go2/IsaacLab

# Robust safety: just check GPU memory free (not process names).
GPU_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$GPU_FREE_MIB" -lt 6000 ]; then
    echo "ERROR: only ${GPU_FREE_MIB} MiB GPU memory free. Need ~6 GiB. Wait for other jobs."
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    exit 1
fi

# Clean up stale Isaac Sim lock from any prior crash.
rm -f /tmp/hub-chrisliang.lock 2>/dev/null
rm -f /tmp/.usd-*.lock 2>/dev/null

CKPT=${CKPT:-logs/rsl_rl/cbf_go2_teacher_rma/2026-05-19_14-18-46/model_2499.pt}
TASK=${TASK:-Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V8-v0}
NUM_ENVS=${NUM_ENVS:-1}
VIDEO_LENGTH=${VIDEO_LENGTH:-800}

[ -f "$CKPT" ] || { echo "ckpt missing: $CKPT"; exit 1; }

echo "================================================================"
echo "Rendering"
echo "  task: $TASK"
echo "  ckpt: $CKPT"
echo "  envs: $NUM_ENVS, video_length: $VIDEO_LENGTH steps"
echo "  GPU free: ${GPU_FREE_MIB} MiB"
echo "================================================================"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task "$TASK" \
  --checkpoint "$CKPT" \
  --num_envs "$NUM_ENVS" \
  --video --video_length "$VIDEO_LENGTH" \
  --headless

echo ""
echo "Video should be at: $(dirname "$CKPT")/videos/play/"
ls -lt "$(dirname "$CKPT")/videos/play/" 2>/dev/null | head -3
echo "================================================================"
