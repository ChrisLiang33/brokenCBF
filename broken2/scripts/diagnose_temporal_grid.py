#!/usr/bin/env python3
"""Temporal grid usage diagnostic — does the CNN use the 2-channel grid?

The teacher's occupancy obs is a (2, 64, 64) tensor: channel 0 = current
frame, channel 1 = previous frame. The CNN can in principle compute optical-
flow-like temporal features from this — but does it? An RL-trained CNN may
just learn to use channel 0 and ignore channel 1 if temporal info doesn't
drive returns.

This script measures correlation between policy outputs (α, φ) and the
*magnitude of per-step grid change* per env. Mechanism:

  grid_change[t] = mean over pixels of |frame_t[ch0] - frame_t[ch1]|

Because obstacles in our env are static, grid_change is dominated by:
  - ego-motion (the world appears to translate in the robot's frame)
  - obstacles entering / leaving the 3.2 m FOV
Both are proxies for "robot is moving / situation is dynamic." If the policy
varies α / φ with grid_change, it's using temporal info in some way.

Interpretation:
  |Pearson(α, grid_change)| > 0.20 → CNN encodes temporal info, policy
                                      conditions on it
  in [0.10, 0.20]                   → weak / noisy
  < 0.10                            → CNN ignores temporal info OR policy
                                      doesn't use it
  (negative)                        → policy *backs off* when situation is
                                      dynamic (cautious mode)

Usage:
  cd ~/Desktop/safety-go2/IsaacLab && \\
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_temporal_grid.py \\
      --task Isaac-CBF-Go2-RMA-Layer2-v0 \\
      --checkpoint logs/rsl_rl/cbf_go2_teacher_rma/<TIMESTAMP>/model_1999.pt \\
      --num_envs 256 --rollout_steps 100 \\
      --output diagnose_temporal_grid_v6.json \\
      --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True, type=str)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rollout_steps", type=int, default=100)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--grid_channels", type=int, default=2)
parser.add_argument("--grid_h", type=int, default=64)
parser.add_argument("--grid_w", type=int, default=64)
parser.add_argument("--priv_dim", type=int, default=16,
                    help="Priv slice size. 16 for Layer 2+ (incl actuation_noise_sigma).")
parser.add_argument("--cbf_state_dim", type=int, default=4,
                    help="cbf_state slice size after priv. 4 unless cbf_state is removed from obs.")
parser.add_argument("--proprio_dim", type=int, default=0,
                    help="proprio slice size (after cbf_state). 0 for pre-v14 checkpoints, "
                         "33 for v14+ (base_lin_vel + base_ang_vel + projected_gravity + "
                         "joint_pos_rel + joint_vel_rel).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401  -- registers tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two 1-D arrays; returns 0 if either has zero variance."""
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)

    grid_offset = args.priv_dim + args.cbf_state_dim + args.proprio_dim
    grid_size = args.grid_channels * args.grid_h * args.grid_w
    print(f"[temporal_grid] grid slice = obs[:, {grid_offset}:{grid_offset+grid_size}]")
    print(f"[temporal_grid] reshapes to ({args.grid_channels}, {args.grid_h}, {args.grid_w})")

    obs, _ = env.reset()

    # Collected per (step, env): grid_change, alpha, phi
    grid_change_history = []
    alpha_history = []
    phi_history = []

    for step in range(args.rollout_steps):
        with torch.no_grad():
            if isinstance(obs, dict) or hasattr(obs, "keys"):
                obs_tensor = obs["policy"]
            else:
                obs_tensor = obs
            if obs_tensor.dim() > 2:
                obs_tensor = obs_tensor.reshape(-1, obs_tensor.shape[-1])

            # Extract grid slice and reshape
            grid_flat = obs_tensor[:, grid_offset:grid_offset + grid_size]   # (N, 8192)
            grid = grid_flat.reshape(
                inner.num_envs, args.grid_channels, args.grid_h, args.grid_w,
            )                                                                 # (N, 2, 64, 64)

            # |current - previous|, mean across pixels per env
            grid_diff = (grid[:, 0] - grid[:, 1]).abs().mean(dim=(1, 2))     # (N,)

            action = policy(obs)
            # Action layout: (α, φ, a, b, c)
            alpha = action[:, 0]
            phi = action[:, 1]

            grid_change_history.append(grid_diff.detach().cpu().numpy())
            alpha_history.append(alpha.detach().cpu().numpy())
            phi_history.append(phi.detach().cpu().numpy())

        step_out = env.step(action)
        obs = step_out[0]

        if step % 20 == 0:
            print(f"[temporal_grid] step {step:>3}/{args.rollout_steps}  "
                  f"mean_grid_change={float(grid_diff.mean()):.4f}  "
                  f"mean_α={float(alpha.mean()):.3f}  "
                  f"mean_φ={float(phi.mean()):.3f}")

    GC = np.stack(grid_change_history)    # (steps, N)
    A = np.stack(alpha_history)           # (steps, N)
    P = np.stack(phi_history)             # (steps, N)

    # Per-env means (one sample per env)
    gc_per_env = GC.mean(axis=0)          # (N,)
    a_per_env = A.mean(axis=0)            # (N,)
    p_per_env = P.mean(axis=0)            # (N,)

    # Pooled (every step, every env — flat samples)
    gc_flat = GC.reshape(-1)
    a_flat = A.reshape(-1)
    p_flat = P.reshape(-1)

    r_alpha_per_env = pearson_corr(gc_per_env, a_per_env)
    r_phi_per_env = pearson_corr(gc_per_env, p_per_env)
    r_alpha_flat = pearson_corr(gc_flat, a_flat)
    r_phi_flat = pearson_corr(gc_flat, p_flat)

    summary = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "n_envs": int(args.num_envs),
        "n_rollout_steps": int(args.rollout_steps),
        "grid_change_stats": {
            "mean":  float(GC.mean()),
            "std":   float(GC.std()),
            "min":   float(GC.min()),
            "max":   float(GC.max()),
        },
        "correlations": {
            "alpha_vs_grid_change_per_env": round(r_alpha_per_env, 4),
            "phi_vs_grid_change_per_env":   round(r_phi_per_env, 4),
            "alpha_vs_grid_change_flat":    round(r_alpha_flat, 4),
            "phi_vs_grid_change_flat":      round(r_phi_flat, 4),
        },
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print(f"Temporal grid diagnostic — {args.task}")
    print("=" * 70)
    print(f"  grid_change mean: {GC.mean():.4f}   std: {GC.std():.4f}")
    print(f"  (low values ≈ static / open space; high ≈ dynamic / ego-motion + FOV transitions)")
    print()
    print(f"  Correlations of policy outputs with grid_change:")
    print(f"    Pearson(α, grid_change)  per-env: {r_alpha_per_env:+.4f}   flat: {r_alpha_flat:+.4f}")
    print(f"    Pearson(φ, grid_change)  per-env: {r_phi_per_env:+.4f}   flat: {r_phi_flat:+.4f}")
    print()
    print(f"  Interpretation:")
    if max(abs(r_alpha_per_env), abs(r_phi_per_env)) > 0.20:
        print(f"    STRONG — CNN appears to use temporal info, policy conditions on it")
    elif max(abs(r_alpha_per_env), abs(r_phi_per_env)) > 0.10:
        print(f"    WEAK — some temporal influence, but not strong")
    else:
        print(f"    ABSENT — CNN ignores temporal info OR policy doesn't condition on it")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    import os
    rc = 1
    try:
        rc = main()
    except Exception as e:
        import traceback
        print(f"[diagnose_temporal_grid] FATAL: {e}", flush=True)
        traceback.print_exc()
    # Skip simulation_app.close() — v17 confirmed it hangs unreliably
    # (this script hung 10+ min after writing its JSON output). OS reclaims
    # handles when the process dies via os._exit, so this is safe.
    os._exit(rc if rc is not None else 0)
