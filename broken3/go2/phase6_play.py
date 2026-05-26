"""Roll out a trained teacher and render it.

Two modes (controlled by --headless / --video flags):

  RECORD VIDEO (works over SSH):
    ~/IsaacLab/isaaclab.sh -p phase6_play.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --policy_checkpoint phase6_slalom_intervention0_teacher_outputs/model_1499.pt \\
        --task Isaac-CBF-Adaptive-Go2-Slalom-v0 \\
        --num_envs 4 --steps 600 \\
        --video --video_length 600 --headless

    -> writes mp4 to videos/ under cwd; pull with rsync.

  LIVE WINDOW (needs X11 forwarding or physical access):
    ssh -X labbox
    ~/IsaacLab/isaaclab.sh -p phase6_play.py \\
        --checkpoint <loco.pt> --policy_checkpoint <teacher.pt> \\
        --task Isaac-CBF-Adaptive-Go2-Slalom-v0 --num_envs 1 --steps 1500
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True,
                    help="Frozen Go2 locomotion checkpoint.")
parser.add_argument("--policy_checkpoint", required=True,
                    help="Trained teacher checkpoint (rsl_rl model_*.pt).")
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-Slalom-v0")
parser.add_argument("--num_envs", type=int, default=4,
                    help="Small (1-4) is fine for visualization.")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--disturbance", type=float, default=0.0)
# Isaac Lab's AppLauncher does NOT auto-add --video / --video_length;
# we add them ourselves and force --enable_cameras when --video is set
# (the camera subsystem must be enabled for rgb_array rendering).
parser.add_argument("--video", action="store_true", default=False,
                    help="Record video of the rollout.")
parser.add_argument("--video_length", type=int, default=600,
                    help="Number of steps to record.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.headless = True

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
from cbf_task.agents import rma_actor_critic  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if getattr(args_cli, "video", False) else None)

    # wrap with gym RecordVideo if requested
    if getattr(args_cli, "video", False):
        video_dir = os.path.join(os.getcwd(), "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_length = getattr(args_cli, "video_length", args_cli.steps)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )
        print(f"[play] recording video -> {video_dir}/  (length={video_length} steps)")

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    agent_cfg.max_iterations = 0
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                            log_dir=None, device=device)
    runner.load(retrieve_file_path(args_cli.policy_checkpoint))
    cbf = env_wrapped.unwrapped.action_manager._terms["cbf_param"]

    # pin disturbance
    cbf._disturbance_force_lo = float(args_cli.disturbance)
    cbf._disturbance_force_hi = float(args_cli.disturbance)

    env_wrapped.unwrapped.reset()
    obs = env_wrapped.get_observations()
    policy = runner.get_inference_policy(device=device)

    print(f"[play] rolling {args_cli.steps} steps x {args_cli.num_envs} envs ...")
    with torch.inference_mode():
        for step in range(args_cli.steps):
            action = policy(obs)
            obs, _, _, _ = env_wrapped.step(action)
            if step % 100 == 0:
                phi_mean = float(cbf.last_phi.mean())
                alpha_mean = float(cbf.last_alpha.mean())
                h_mean = float(cbf.last_h_realized.mean())
                print(f"  step {step:>4d}  phi={phi_mean:+.3f}  "
                      f"alpha={alpha_mean:+.3f}  h={h_mean:+.3f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
