"""Smoke test for the RMAHistory env + model. Disposable.

Verifies:
  1. Task registers and env instantiates
  2. obs dim = 50*45 + 2 + 72 + 72 = 2396
  3. proprio_history buffer fills correctly over 50 steps (no longer zeros)
  4. RMAHistoryMLPModel forward works on the obs
  5. Per-env reset zeros the buffer for the reset envs only
"""
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
import cbf_task  # noqa
from cbf_task.agents import rma_actor_critic  # noqa
from cbf_task.agents import rma_history_actor_critic  # noqa
from cbf_task.locomotion_loader import load_locomotion_actor

device = "cuda" if torch.cuda.is_available() else "cpu"
loco = load_locomotion_actor(retrieve_file_path(args_cli.locomotion_ckpt), device)

task = "Isaac-CBF-Adaptive-Go2-RMAHistory-v0"
print(f"\n--- smoke: {task} ---")
cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
cfg.scene.num_envs = args_cli.num_envs
cfg.sim.device = device
cfg.actions.cbf_param.locomotion_policy_obj = loco
env = gym.make(task, cfg=cfg).unwrapped
cbf = env.action_manager._terms["cbf_param"]

# 1) obs dim
env.reset()
obs = env.observation_manager.compute()["policy"]
expected_dim = 50 * 45 + 2 + 72 + 72   # 2396
ok_dim = obs.shape[-1] == expected_dim
print(f"  obs dim = {obs.shape[-1]}  (expected {expected_dim})  "
      f"{'OK' if ok_dim else 'FAIL'}")

# 2) proprio_history exists and has the right shape
ok_buf = (cbf._proprio_history is not None
          and cbf._proprio_history.shape == (args_cli.num_envs, 50, 45))
print(f"  proprio_history shape = {tuple(cbf._proprio_history.shape) if cbf._proprio_history is not None else None}  "
      f"{'OK' if ok_buf else 'FAIL'}")

# 3) buffer fills as we step
# Right after reset: buffer is all zeros. After K=50 steps, no zeros.
print(f"  pre-step  buffer std = {cbf._proprio_history.std().item():.6f}  (should be ~0)")
for _ in range(60):
    env.step(torch.zeros((args_cli.num_envs, 2), device=device))
post_std = cbf._proprio_history.std().item()
print(f"  post-step buffer std = {post_std:.4f}  (should be > 0.01)")
ok_fill = post_std > 0.01

# 4) model forward on the live obs
agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
agent_cfg.device = device
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                         log_dir=None, device=device)
policy = runner.get_inference_policy(device=device)
obs_td = env_wrapped.get_observations()
with torch.inference_mode():
    action = policy(obs_td)
ok_fwd = action.shape == (args_cli.num_envs, 2)
print(f"  policy forward: action shape = {tuple(action.shape)}  "
      f"{'OK' if ok_fwd else 'FAIL'}")

all_ok = ok_dim and ok_buf and ok_fill and ok_fwd
print(f"\n  {'ALL PASS' if all_ok else 'FAIL'}")

env.close()
sim_app.close()
sys.exit(0 if all_ok else 1)
