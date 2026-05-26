"""Smoke test for the RMA-classic + random-topology env.

Verifies:
  1. Task registers cleanly
  2. Env instantiates without import errors
  3. obs dim is the expected 195 (= 4+45+2+72+72)
  4. priv obs returns 4 channels
  5. Random topology samples DIFFERENT positions across resets

Disposable -- delete after the wiring stabilizes."""
from __future__ import annotations
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
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
from cbf_task.agents import rma_actor_critic  # noqa
from cbf_task.agents import rma_classic_actor_critic  # noqa: registers RMAClassicMLPModel
from cbf_task.locomotion_loader import load_locomotion_actor

device = "cuda" if torch.cuda.is_available() else "cpu"
loco = load_locomotion_actor(retrieve_file_path(args_cli.locomotion_ckpt), device)

task = "Isaac-CBF-Adaptive-Go2-RMAStatic-v0"
print(f"\n--- smoke: {task} ---")

cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
cfg.scene.num_envs = args_cli.num_envs
cfg.sim.device = device
cfg.actions.cbf_param.locomotion_policy_obj = loco
env = gym.make(task, cfg=cfg).unwrapped
cbf = env.action_manager._terms["cbf_param"]

# 1) obs dim
obs = env.observation_manager.compute()["policy"]
ok_dim = obs.shape[-1] == 195
print(f"  obs dim = {obs.shape[-1]}  {'OK' if ok_dim else 'FAIL (expected 195)'}")

# 2) priv dim
from cbf_task.mdp import priv_obs_rma_classic
priv = priv_obs_rma_classic(env)
ok_priv = priv.shape[-1] == 4
print(f"  priv dim = {priv.shape[-1]}  {'OK' if ok_priv else 'FAIL (expected 4)'}")

# 3) random topology produces different positions across resets
env.reset()
pos1 = cbf._obs_centers_w.clone()
env.reset()
pos2 = cbf._obs_centers_w.clone()
moved = (pos1 - pos2).abs().mean().item()
ok_random = moved > 0.5
print(f"  random topology: mean obstacle move across resets = {moved:.3f} m  "
      f"{'OK' if ok_random else 'FAIL (expected > 0.5 m, got nominal)'}")

# 4) start exclusion: no obstacle within 0.8m of (0, 0) in env-local coords
env_origins_xy = env.scene.env_origins[:, :2].to(device)
pos_local = pos2 - env_origins_xy.unsqueeze(1)
dist_start = torch.linalg.norm(pos_local, dim=-1)         # (N, K)
ok_start = (dist_start > 0.8 - 1e-3).all().item()
print(f"  start exclusion: min dist to (0,0) = {dist_start.min().item():.3f} m  "
      f"{'OK' if ok_start else 'FAIL'}")

# 5) take a few steps
for _ in range(5):
    env.step(torch.zeros((args_cli.num_envs, 2), device=device))
print(f"  5 random-action steps: OK")

all_ok = ok_dim and ok_priv and ok_random and ok_start
print(f"\n--- {'ALL PASS' if all_ok else 'FAIL'} ---")

env.close()
sim_app.close()
sys.exit(0 if all_ok else 1)
