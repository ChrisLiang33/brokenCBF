"""Smoke test for u_des noise injection.

Verifies the v12 noise-injection change doesn't destabilize the locomotion
controller at a given σ_max. Runs Layer 2 env with FIXED CBF params for
N steps, reports fall rate, mean velocity, NaN occurrence, terminations.

Background:
  v5 and earlier — noise on u_safe (between QP and locomotion). Absorbed
                  by locomotion smoothing; R²(σ) ≈ 0.05.
  v6–v11        — noise on joint targets (after locomotion). Bypassed
                  locomotion but injected at the wrong abstraction level;
                  CBF teacher couldn't form a usable relationship.
  v12+          — noise on u_des (before the CBF filter). QP sees a noisy
                  target and must choose CBF params that are robust to it.
                  σ_max units are m/s now (linear vel), applied to vx, vy
                  only — angular rate left clean.

The smoke test below just exercises the env at a fixed σ; the actual
injection point lives in env.py and follows whatever the current branch
implements. PASS criterion is unchanged: locomotion should stay stable
(fall_rate within ~5pp of σ=0 baseline, mean_v_xy ≥ 70% of baseline).

Usage (lab box, headless):
    cd ~/Desktop/safety-go2/IsaacLab
    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/smoke_noise_injection.py \\
        --sigma 0.0 --headless         # baseline (no noise)
    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/smoke_noise_injection.py \\
        --sigma 0.02 --headless        # mild
    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/smoke_noise_injection.py \\
        --sigma 0.05 --headless        # moderate
    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/smoke_noise_injection.py \\
        --sigma 0.10 --headless        # aggressive

Compare each σ run against the σ=0 baseline. Locomotion is stable if
fall_rate stays within ~5pp of baseline and mean_v_xy doesn't drop >30%.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for joint-target noise injection.")
parser.add_argument("--sigma", type=float, default=0.02,
                    help="actuation_noise_sigma_max override (rad of joint-angle noise).")
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  -- registers tasks
from isaaclab_tasks.utils import parse_env_cfg


TASK = "Isaac-CBF-Go2-RMA-Layer2-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.actuation_noise_sigma_max = args_cli.sigma
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    inner = env.unwrapped

    # Fixed moderate CBF params — α=2 keeps boundary respected, φ=1 mild
    # robust margin, a/b/c off. Let u_des pass through most of the time.
    cbf_params = torch.tensor(
        [[2.0, 1.0, 0.0, 0.0, 0.0]],
        device=inner.device, dtype=torch.float32,
    ).expand(inner.num_envs, -1).contiguous()

    fell_envs = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
    v_xy_sum = 0.0
    nan_count = 0
    term_count = 0

    print(f"\n=== Smoke test: σ_max = {args_cli.sigma:.3f} rad ===")
    print(f"Task: {TASK}")
    print(f"  {args_cli.num_envs} envs × {args_cli.num_steps} steps")
    print(f"  Fixed CBF params: α=2.0, φ=1.0, a=b=c=0")
    print()

    for step in range(args_cli.num_steps):
        out = env.step(cbf_params)
        obs, reward, terminated, truncated, extras = out

        if torch.isnan(reward).any():
            nan_count += 1
        term_count += int(terminated.sum().item())

        robot = inner.scene["robot"]
        gravity_b = robot.data.projected_gravity_b  # (N, 3) in body frame
        # robot is tilted/flipped if z-component is no longer ≈ -1.0
        tilted = (gravity_b[:, 2] > -0.5)
        fell_envs |= tilted

        v_xy = robot.data.root_lin_vel_w[:, :2].norm(dim=-1)
        v_xy_sum += v_xy.mean().item()

    mean_v = v_xy_sum / args_cli.num_steps
    fall_rate = fell_envs.float().mean().item()

    print(f"Results:")
    print(f"  mean_v_xy   : {mean_v:.3f} m/s")
    print(f"  fall_rate   : {fall_rate:.4f}  ({fell_envs.sum().item()}/{args_cli.num_envs} envs tilted)")
    print(f"  terminations: {term_count} total step-events")
    print(f"  NaN reward steps: {nan_count}")
    print()
    print("Compare to σ=0 baseline:")
    print("  Healthy: fall_rate within ~5pp of baseline, mean_v_xy ≥ 70% of baseline")
    print("  Too hot: fall_rate >> baseline + 5pp, OR mean_v_xy collapsed, OR NaN > 0")


if __name__ == "__main__":
    import os
    try:
        main()
    finally:
        # Isaac Sim's simulation_app.close() doesn't always cleanly exit the
        # process (known issue, leaves Python hung). Force-exit so a sequence
        # of `./isaaclab.sh -p smoke_*.py --sigma X` calls actually progresses.
        try:
            simulation_app.close()
        except Exception:
            pass
        os._exit(0)
