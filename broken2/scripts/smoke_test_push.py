"""Locomotion-only push-magnitude smoke test (single-magnitude variant).

Runs the exported Go2 flat locomotion policy (no CBF) on the
CbfGo2LocomotionTrain task with a re-enabled `push_robot` event at ONE
velocity magnitude. Prints fall_rate.

Background: parent flat env defines push_robot at ±0.5 m/s every (10, 15) s
but the lab mate's cbf_go2_locomotion_train_cfg disables it ("Go2 is too
small to tolerate the ±0.5 m/s velocity push without tipping"). LAYER3_PUSH
proposes ±1.0 m/s every (5, 10) s. This smoke test verifies what the
locomotion policy can survive.

This script runs ONE magnitude per launch (Isaac Lab doesn't cleanly
support creating multiple envs in the same simulation app — earlier
multi-magnitude version hung at env #2). Wrap with a shell loop for
sweeps; see scripts/smoke_test_push_loop.sh.

Usage (lab box, from IsaacLab dir):
    cd /home/chrisliang/Desktop/safety-go2/IsaacLab
    ./isaaclab.sh -p ../scripts/smoke_test_push.py --headless --magnitude 1.0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Single-magnitude push smoke test.")
parser.add_argument(
    "--magnitude",
    type=float,
    required=True,
    help="Velocity push magnitude (m/s), applied to both x and y.",
)
parser.add_argument(
    "--interval_s",
    type=float,
    nargs=2,
    default=[5.0, 10.0],
    help="Interval range (min, max) in seconds between pushes.",
)
parser.add_argument("--num_envs", type=int, default=64, help="Parallel envs.")
parser.add_argument(
    "--total_steps",
    type=int,
    default=2000,
    help="Total env steps (sims ~40 s real per env @ 50 Hz).",
)
parser.add_argument(
    "--locomotion_ckpt",
    type=str,
    default=(
        "/home/chrisliang/Desktop/safety-go2/IsaacLab/logs/rsl_rl/"
        "unitree_go2_flat/2026-04-15_19-24-45/exported/policy.pt"
    ),
    help="Path to the exported locomotion TorchScript policy.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401 (registers tasks)
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-CBF-Go2-LocomotionTrain-v0"


def main() -> None:
    mag = args_cli.magnitude
    print(f"[smoke_test_push] task={TASK}")
    print(f"[smoke_test_push] locomotion_ckpt={args_cli.locomotion_ckpt}")
    print(
        f"[smoke_test_push] num_envs={args_cli.num_envs}, "
        f"total_steps={args_cli.total_steps}, "
        f"interval_s={tuple(args_cli.interval_s)}, "
        f"magnitude={mag}"
    )

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.events.push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=tuple(args_cli.interval_s),
        params={"velocity_range": {"x": (-mag, mag), "y": (-mag, mag)}},
    )

    env = gym.make(TASK, cfg=env_cfg)
    device = env.unwrapped.device
    policy = torch.jit.load(args_cli.locomotion_ckpt).to(device).eval()

    obs, _ = env.reset()
    fall_events = 0
    timeout_events = 0
    for _ in range(args_cli.total_steps):
        obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
        with torch.no_grad():
            action = policy(obs_tensor)
        obs, _rew, terminated, truncated, _info = env.step(action)
        fall_events += int(terminated.sum().item())
        timeout_events += int(truncated.sum().item())

    env.close()

    completed_episodes = fall_events + timeout_events
    fall_rate = fall_events / max(completed_episodes, 1)
    print(
        f"[smoke_test_push] RESULT mag={mag:.2f} "
        f"falls={fall_events} timeouts={timeout_events} "
        f"episodes={completed_episodes} fall_rate={fall_rate:.3f}"
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
