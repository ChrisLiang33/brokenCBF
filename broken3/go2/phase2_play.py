"""Phase 2 / 3 play -- record videos of the trained policy in action.

Loads a trained Phase 2/3 model, runs it for `n_steps` at a fixed
disturbance level, and writes one mp4 per env via gymnasium's
RecordVideo wrapper.

Usage (on labbox):
    ./isaaclab.sh -p ~/Desktop/cbf_rl_mvp/go2/phase2_play.py \\
        --loco_checkpoint /path/to/loco_model.pt \\
        --policy_checkpoint /path/to/phase3_outputs/rsl_rl/model_final.pt \\
        --disturbance_force 30 \\
        --num_envs 4 \\
        --video_length 500 \\
        --enable_cameras
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--loco_checkpoint", required=True,
                    help="Frozen Go2 locomotion .pt checkpoint.")
parser.add_argument("--policy_checkpoint", required=True,
                    help="Trained outer-policy rsl_rl .pt (Phase 2/3).")
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-Phase2-v0")
parser.add_argument("--num_envs", type=int, default=4,
                    help="Number of envs to render simultaneously.")
parser.add_argument("--disturbance_force", type=float, default=30.0)
parser.add_argument("--n_steps", type=int, default=500,
                    help="Steps to roll out.")
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--out_dir", default="phase2_videos")
parser.add_argument("--goal", type=float, nargs=2, default=[6.0, 0.0])
parser.add_argument("--obstacle", type=float, nargs=2, default=[2.5, 0.3])
parser.add_argument("--obstacle_radius", type=float, default=0.9)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True   # required for video
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    # 1) locomotion actor (needed for the env's action term)
    loco_ckpt = retrieve_file_path(args_cli.loco_checkpoint)
    print(f"[play] locomotion -> {loco_ckpt}")
    locomotion_actor = load_locomotion_actor(loco_ckpt, device)

    # 2) env cfg (fix disturbance at the requested magnitude for the
    # playback so the recording shows behavior at a single condition)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = None
    at = env_cfg.actions.cbf_param
    at.locomotion_policy_obj = locomotion_actor
    at.goal_xy = tuple(args_cli.goal)
    at.obstacle_xy = tuple(args_cli.obstacle)
    at.obstacle_radius = float(args_cli.obstacle_radius)
    # pin disturbance to a single magnitude (deterministic playback condition)
    at.disturbance_force_range = (float(args_cli.disturbance_force),
                                  float(args_cli.disturbance_force))

    # 3) env + video wrapper
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    video_folder = os.path.abspath(args_cli.out_dir)
    print(f"[play] video output -> {video_folder}")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        step_trigger=lambda step: step == 0,         # record from step 0
        video_length=args_cli.video_length,
        disable_logger=True,
        name_prefix=f"phase2_d{int(args_cli.disturbance_force)}",
    )

    # 4) load policy via rsl_rl
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                            log_dir=None, device=device)
    policy_ckpt = retrieve_file_path(args_cli.policy_checkpoint)
    print(f"[play] policy -> {policy_ckpt}")
    runner.load(policy_ckpt)
    policy = runner.get_inference_policy(device=device)

    # 5) roll out, recording video frames as we go
    print(f"[play] rolling out {args_cli.n_steps} steps at d={args_cli.disturbance_force}N "
          f"({args_cli.num_envs} envs) ...")
    with torch.inference_mode():
        obs = env_wrapped.get_observations()
        for step in range(args_cli.n_steps):
            action = policy(obs)
            obs, _, _, _ = env_wrapped.step(action)

    print(f"[play] done. Videos written under {video_folder}/")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
