"""Smoke-test for one eval scene: gym.make, take 5 steps, confirm
obstacle layout matches the cfg. Single-scene per invocation because
Isaac Lab only allows one sim context per process. Run from a bash
loop to test all scenes:

    for s in E1 E2 E3 E4; do
      ~/IsaacLab/isaaclab.sh -p _smoke_eval_scenes.py \\
          --scene $s \\
          --locomotion_ckpt /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt
    done

Disposable -- delete after the eval-scenes wiring stabilizes."""
from __future__ import annotations
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True, choices=["E1", "E2", "E3", "E4"])
parser.add_argument("--locomotion_ckpt", required=True)
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa
from cbf_task.locomotion_loader import load_locomotion_actor

EXPECTED = {
    "E1": (1, (7.0, 0.0)),
    "E2": (3, (7.0, 0.0)),
    "E3": (5, (7.0, 0.0)),
    "E4": (2, (7.0, 0.0)),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
loco = load_locomotion_actor(retrieve_file_path(args_cli.locomotion_ckpt), device)

scene = args_cli.scene
n_obs, goal = EXPECTED[scene]
task = f"Isaac-CBF-Adaptive-Go2-Eval{scene}-v0"

try:
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = device
    cfg.actions.cbf_param.locomotion_policy_obj = loco
    env = gym.make(task, cfg=cfg).unwrapped
    cbf = env.action_manager._terms["cbf_param"]
    actual_n = len(cfg.actions.cbf_param.obstacles)
    actual_goal = tuple(cfg.actions.cbf_param.goal_xy)
    assert actual_n == n_obs, f"obstacle count: {actual_n} vs expected {n_obs}"
    assert actual_goal == goal, f"goal: {actual_goal} vs {goal}"
    env.reset()
    for _ in range(5):
        a = torch.zeros((args_cli.num_envs, 2), device=device)
        env.step(a)
    assert cbf._n_obstacles == n_obs, \
        f"cbf._n_obstacles {cbf._n_obstacles} vs cfg {n_obs}"
    obs = env.observation_manager.compute()["policy"]
    assert obs.shape[-1] == 198, f"obs dim: {obs.shape[-1]}"
    env.close()
    print(f"PASS  {scene}  n_obs={actual_n} goal={actual_goal} "
          f"cbf._n_obstacles={cbf._n_obstacles} obs_dim=198")
    sim_app.close()
    sys.exit(0)
except Exception as e:
    import traceback
    print(f"FAIL  {scene}  {e}")
    traceback.print_exc()
    sim_app.close()
    sys.exit(1)
