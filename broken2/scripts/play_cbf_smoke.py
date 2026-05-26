"""Step 2f: hand-tuned CBF smoke test.

Drives CbfGo2Env with FIXED 5D CBF params (no RL policy), deterministic
u_des (constant forward), and prints per-step diagnostics. Optionally
records a video.

Goal: confirm the CBF-QP actually intervenes as the robot approaches the
obstacle — u_safe should deviate from u_des, h should stay positive.

Usage (on lab, from IsaacLab dir):
    cd /home/chrisliang/Desktop/safety-go2/IsaacLab
    ./isaaclab.sh -p ../scripts/play_cbf_smoke.py \\
        --num_steps 400 --headless

    # with video:
    ./isaaclab.sh -p ../scripts/play_cbf_smoke.py \\
        --num_steps 400 --headless --video --video_length 400
"""

import argparse

from isaaclab.app import AppLauncher

# --- CLI parsing has to happen BEFORE sim app launches ---
parser = argparse.ArgumentParser(description="CBF smoke test with fixed params.")
parser.add_argument("--num_steps", type=int, default=400, help="Total env steps.")
parser.add_argument("--print_every", type=int, default=20, help="Diagnostic print interval.")
parser.add_argument("--alpha", type=float, default=1.0, help="CBF decay rate.")
parser.add_argument("--phi", type=float, default=0.1, help="Actuation uncertainty.")
parser.add_argument("--a_param", type=float, default=0.01, help="Measurement uncertainty (constant).")
parser.add_argument("--b_param", type=float, default=0.0, help="Input-scaled meas unc (unused — SOCP).")
parser.add_argument("--c_param", type=float, default=0.1, help="Safe-set shrink.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=400, help="Video length in steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Cameras required for video wrapper
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after sim app starts ---

import os

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  -- registers tasks
from isaaclab_tasks.utils import parse_env_cfg


TASK = "Isaac-CBF-Go2-Play-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_folder = os.path.abspath("logs/cbf_smoke_videos")
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[video] Recording to {video_folder}")

    env.reset()
    inner = env.unwrapped

    # Fixed 5D CBF params, broadcast to all envs
    cbf_params = torch.tensor(
        [[args_cli.alpha, args_cli.phi, args_cli.a_param, args_cli.b_param, args_cli.c_param]],
        device=inner.device,
        dtype=torch.float32,
    ).expand(inner.num_envs, -1).contiguous()

    print("\n[CBF SMOKE] Fixed params:")
    print(f"  alpha = {args_cli.alpha}")
    print(f"  phi   = {args_cli.phi}")
    print(f"  a     = {args_cli.a_param}")
    print(f"  b     = {args_cli.b_param}  (unused)")
    print(f"  c     = {args_cli.c_param}")
    print(f"[CBF SMOKE] num_envs = {inner.num_envs}, num_steps = {args_cli.num_steps}")
    print(f"[CBF SMOKE] obstacle at /World/Obstacle world pos {inner.scene['obstacle'].data.root_pos_w[0].tolist()}")
    print()
    print(f"{'step':>5s} {'x':>7s} {'y':>7s} {'dist':>6s} {'h':>7s} "
          f"{'u_des_x':>8s} {'u_safe_x':>9s} {'|du|':>7s}  filtered?")

    n_filtered = 0
    for step in range(args_cli.num_steps):
        env.step(cbf_params)

        if step % args_cli.print_every == 0 or step == args_cli.num_steps - 1:
            robot_pos = inner.scene["robot"].data.root_pos_w[0, :2]
            obs_pos = inner.scene["obstacle"].data.root_pos_w[0, :2]
            diff = robot_pos - obs_pos
            dist = torch.linalg.norm(diff).item()

            # Recompute h with current state (uses live e_i, no caching).
            h_vals, _ = inner._compute_h()
            h = h_vals[0].item()

            u_des = inner.last_u_des[0]
            u_safe = inner.last_u_safe[0]
            du = torch.linalg.norm(u_des - u_safe).item()
            filtered = "YES" if du > 0.01 else ""
            if du > 0.01:
                n_filtered += 1

            print(f"{step:>5d} {robot_pos[0].item():7.3f} {robot_pos[1].item():7.3f} "
                  f"{dist:6.2f} {h:7.3f} "
                  f"{u_des[0].item():8.3f} {u_safe[0].item():9.3f} {du:7.3f}  {filtered}")

        if args_cli.video and step >= args_cli.video_length:
            break

    print(f"\n[CBF SMOKE] {n_filtered} printed diagnostic rows showed CBF filtering (|du| > 0.01)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
