#!/usr/bin/env python3
"""Minimal play script that records video without triggering JIT export.

play.py from rsl_rl's standard scripts ends with `runner.export_policy_to_jit()`
which tries to TorchScript the policy. Our model uses module-level globals
(_PRIV_DIM etc.) that JIT can't resolve, so the export crashes — even though
the actual rollout works fine in eager mode.

This script does everything play.py does (env + policy + video) but skips
the JIT export. Pure eager-mode inference.

Usage on lab box:
  cd ~/Desktop/safety-go2/IsaacLab
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/play_render.py \\
    --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V8-v0 \\
    --checkpoint logs/rsl_rl/cbf_go2_teacher_rma/2026-05-19_14-18-46/model_2499.pt \\
    --num_envs 1 --video --video_length 800 \\
    --headless
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True, type=str)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video", action="store_true",
                    help="Record videos via gym.wrappers.RecordVideo.")
parser.add_argument("--video_length", type=int, default=800)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video_dir", type=str, default=None,
                    help="Where to save videos. Defaults to ckpt_dir/videos/play.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Cameras must be enabled BEFORE AppLauncher to support rgb_array rendering.
if args.video:
    args.enable_cameras = True

print(f"[play_render] starting AppLauncher (enable_cameras={getattr(args, 'enable_cameras', False)})...",
      flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[play_render] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main() -> int:
    torch.manual_seed(args.seed)
    device = "cuda:0"

    print(f"[play_render] task={args.task}, num_envs={args.num_envs}, "
          f"video={args.video}, video_length={args.video_length}",
          flush=True)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)

    # Decide video output dir.
    ckpt_path = Path(args.checkpoint).resolve()
    if args.video_dir:
        video_dir = Path(args.video_dir).resolve()
    else:
        video_dir = ckpt_path.parent / "videos" / "play"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Build env with render mode if video requested.
    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    inner = env.unwrapped

    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,  # record one video starting at step 0
            video_length=args.video_length,
            name_prefix="play",
        )
        print(f"[play_render] video will be saved to: {video_dir}", flush=True)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    print(f"[play_render] runner loaded.", flush=True)

    obs, _ = env.reset()

    # Roll out until video_length steps have elapsed.
    steps_to_run = args.video_length + 5
    print(f"[play_render] rolling out {steps_to_run} steps...", flush=True)
    for step in range(steps_to_run):
        with torch.no_grad():
            action = policy(obs)
        step_out = env.step(action)
        obs = step_out[0]
        if step % 50 == 0 or step == steps_to_run - 1:
            print(f"[play_render] step {step:>4}/{steps_to_run}", flush=True)

    env.close()
    print(f"[play_render] DONE. Video in: {video_dir}", flush=True)

    # List videos produced
    for v in sorted(video_dir.glob("*.mp4")):
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"  {v.name}  {size_mb:.1f} MB", flush=True)

    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception as e:
        import traceback
        print(f"[play_render] FATAL: {e}", flush=True)
        traceback.print_exc()
    os._exit(rc if rc is not None else 0)
