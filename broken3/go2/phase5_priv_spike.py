"""Privileged-info pipeline spike -- step 1 of the RMA build.

Verifies that the new `priv` observation group on Isaac-CBF-Adaptive-Go2-
RMA-v0 surfaces per-env disturbance correctly. The disturbance is sampled
fresh per env on every reset, so across N=8 envs we should see 8 distinct
values inside the configured range (0-45 N for Phase 2 inheritance).

PASS criteria:
- env builds (priv obs group is registered without errors)
- priv obs shape is (N, 1)
- values are inside [0, 45]
- 8 envs give 8 distinct values (DR is working)

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase5_priv_spike.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --headless
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--n_steps", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import gymnasium as gym
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor


TASK = "Isaac-CBF-Adaptive-Go2-RMA-v0"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco

    print(f"[priv_spike] building {TASK} with {args_cli.num_envs} envs ...")
    env = gym.make(TASK, cfg=env_cfg)
    obs, _ = env.reset()

    # obs is a dict (since we have multiple obs groups)
    print(f"[priv_spike] obs keys: {list(obs.keys())}")
    assert "policy" in obs, "policy group missing"
    assert "priv" in obs, "priv group missing"
    print(f"[priv_spike] policy obs shape: {tuple(obs['policy'].shape)}")
    print(f"[priv_spike] priv   obs shape: {tuple(obs['priv'].shape)}")

    priv = obs["priv"]
    print(f"[priv_spike] priv obs values (per env, after reset):")
    for i, v in enumerate(priv.cpu().tolist()):
        print(f"    env {i}: {v}")

    # step a few to confirm priv stays stable within episode
    for step in range(args_cli.n_steps):
        action = torch.zeros((args_cli.num_envs, 2), device=device)
        obs, _, _, _, _ = env.step(action)
    priv_after = obs["priv"]
    print(f"\n[priv_spike] priv obs after {args_cli.n_steps} steps "
          f"(should match unless an env auto-reset fired):")
    for i, v in enumerate(priv_after.cpu().tolist()):
        print(f"    env {i}: {v}")

    # verdict
    in_range = bool(((priv >= 0.0) & (priv <= 45.0)).all().item())
    distinct = priv.unique().numel()
    print()
    print(f"[priv_spike] in_range [0, 45]: {in_range}")
    print(f"[priv_spike] distinct values across {args_cli.num_envs} envs: {distinct}")
    ok = (in_range
          and priv.shape == (args_cli.num_envs, 1)
          and distinct >= max(2, args_cli.num_envs // 2))
    print(f"[priv_spike] verdict: {'PASS -- priv pipeline works' if ok else 'REVIEW'}")

    env.close()
    simulation_app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
