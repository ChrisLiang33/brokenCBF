"""Smoke test for the obs masking flags: verify that the masked slice
is actually zero in the policy obs, and the unmasked slices look
normal. Disposable."""
from __future__ import annotations
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True,
                    choices=["NoPriv", "NoProprio"])
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

device = "cuda" if torch.cuda.is_available() else "cpu"
loco = load_locomotion_actor(retrieve_file_path(args_cli.locomotion_ckpt), device)

task = f"Isaac-CBF-Adaptive-Go2-RMAStatic{args_cli.scene}-v0"
print(f"\n--- smoke: {task} ---")
cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
cfg.scene.num_envs = args_cli.num_envs
cfg.sim.device = device
cfg.actions.cbf_param.locomotion_policy_obj = loco
env = gym.make(task, cfg=cfg).unwrapped
cbf = env.action_manager._terms["cbf_param"]

# obs dim should still be 195 (slot unchanged, only values masked)
env.reset()
for _ in range(3):
    env.step(torch.zeros((args_cli.num_envs, 2), device=device))

obs = env.observation_manager.compute()["policy"]
print(f"  obs dim = {obs.shape[-1]}  (expected 195)")

# RMA-classic slices: priv [0:4], proprio [4:49], prev_act [49:51],
# lidar_prev [51:123], lidar [123:195]
priv  = obs[..., 0:4]
proprio = obs[..., 4:49]
lidar = obs[..., 51:195]

print(f"  priv:    min={priv.abs().max().item():.5f}  max={priv.abs().max().item():.5f}  "
      f"all zero? {bool((priv.abs() < 1e-5).all().item())}")
print(f"  proprio: max|val|={proprio.abs().max().item():.5f}  "
      f"all zero? {bool((proprio.abs() < 1e-5).all().item())}")
print(f"  lidar:   max|val|={lidar.abs().max().item():.5f}  "
      f"any nonzero? {bool((lidar.abs() > 1e-5).any().item())}")

# Verify the expected slot IS zero and the other slots are NOT zero.
if args_cli.scene == "NoPriv":
    ok = (priv.abs() < 1e-5).all() and (proprio.abs() > 1e-5).any() and (lidar.abs() > 1e-5).any()
elif args_cli.scene == "NoProprio":
    ok = (proprio.abs() < 1e-5).all() and (priv.abs() > 1e-5).any() and (lidar.abs() > 1e-5).any()

print(f"\n  {'PASS' if ok else 'FAIL'}: masking behaves correctly for {args_cli.scene}")

env.close()
sim_app.close()
sys.exit(0 if ok else 1)
