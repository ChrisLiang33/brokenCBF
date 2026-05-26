#!/usr/bin/env python3
"""Linear probe of z_grid → engineered obstacle features.

Tells us what z_grid (the 64-D output of the grid encoder) actually encodes
about obstacle geometry and motion. Complements diagnose_grad_sensitivity.py
which showed the heads USE z_grid; this asks what z_grid CARRIES.

Features probed (computed from the 64×64×2 grid input, body-frame ego-centric):
  - occupancy_total      : sum of occupied cells (channel 0, current frame)
  - nearest_dist         : grid-cell distance from robot center to nearest
                           occupied cell (channel 0). Lower = closer obstacle.
  - obstacle_com_x       : x-coordinate of occupied-cell centroid (current)
  - obstacle_com_y       : y-coordinate of occupied-cell centroid (current)
  - motion_total         : sum of |ch0 - ch1| (motion magnitude across grid)
  - prev_occupancy_total : sum of occupied cells (channel 1, previous frame)

For each feature, fit OLS z_grid → feature, report R²_test. Same train/test
80/20 split as probe_z_linear.py.

Usage on lab box:
  cd ~/Desktop/safety-go2/IsaacLab
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_grid.py \\
    --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V8-v0 \\
    --checkpoint logs/rsl_rl/cbf_go2_teacher_rma/2026-05-19_14-18-46/model_2499.pt \\
    --num_envs 256 --rollout_steps 100 --priv_dim 33 \\
    --output probe_z_grid_wk3tight8.json --headless
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True, type=str)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rollout_steps", type=int, default=100)
parser.add_argument("--priv_dim", type=int, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--train_frac", type=float, default=0.80)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[probe_z_grid] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[probe_z_grid] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


GRID_H = 64
GRID_W = 64
GRID_CENTER_ROW = GRID_H // 2
GRID_CENTER_COL = GRID_W // 2


def linear_probe_r2(Z_train, y_train, Z_test, y_test):
    A_train = np.column_stack([Z_train, np.ones(len(Z_train))])
    A_test = np.column_stack([Z_test, np.ones(len(Z_test))])
    w, *_ = np.linalg.lstsq(A_train, y_train, rcond=None)
    pred_train = A_train @ w
    pred_test = A_test @ w
    ss_res_train = ((y_train - pred_train) ** 2).sum()
    ss_tot_train = ((y_train - y_train.mean()) ** 2).sum() + 1e-12
    r2_train = 1.0 - ss_res_train / ss_tot_train
    ss_res_test = ((y_test - pred_test) ** 2).sum()
    ss_tot_test = ((y_test - y_test.mean()) ** 2).sum() + 1e-12
    r2_test = 1.0 - ss_res_test / ss_tot_test
    mse_test = float(((y_test - pred_test) ** 2).mean())
    return float(r2_train), float(r2_test), mse_test


def compute_grid_features(grid_input: np.ndarray) -> dict:
    """grid_input shape: (N, 2, 64, 64). Returns dict of (N,) feature arrays."""
    n = grid_input.shape[0]
    ch0 = grid_input[:, 0]  # (N, 64, 64) current frame
    ch1 = grid_input[:, 1]  # (N, 64, 64) previous frame

    # Occupancy totals
    occ_total = ch0.reshape(n, -1).sum(axis=1)        # (N,)
    prev_occ_total = ch1.reshape(n, -1).sum(axis=1)   # (N,)

    # Nearest-obstacle distance from robot center (cells).
    # If no occupied cells, set to GRID_H (max distance).
    nearest_dist = np.full(n, float(GRID_H), dtype=np.float32)
    rows, cols = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing="ij")
    dist_from_center = np.sqrt(
        (rows - GRID_CENTER_ROW) ** 2 + (cols - GRID_CENTER_COL) ** 2
    )  # (64, 64)
    for i in range(n):
        mask = ch0[i] > 0
        if mask.any():
            nearest_dist[i] = float(dist_from_center[mask].min())

    # Obstacle centroid in current frame (row, col averages weighted by occupancy)
    # If no occupancy, centroid = grid center.
    com_y = np.full(n, float(GRID_CENTER_ROW), dtype=np.float32)  # row
    com_x = np.full(n, float(GRID_CENTER_COL), dtype=np.float32)  # col
    for i in range(n):
        m = ch0[i]
        s = m.sum()
        if s > 0:
            com_y[i] = float((m * rows).sum() / s)
            com_x[i] = float((m * cols).sum() / s)

    # Motion: sum of absolute frame difference
    motion_total = np.abs(ch0 - ch1).reshape(n, -1).sum(axis=1)

    return {
        "occupancy_total":      occ_total.astype(np.float32),
        "prev_occupancy_total": prev_occ_total.astype(np.float32),
        "nearest_dist":         nearest_dist.astype(np.float32),
        "obstacle_com_x":       com_x.astype(np.float32),
        "obstacle_com_y":       com_y.astype(np.float32),
        "motion_total":         motion_total.astype(np.float32),
    }


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
    print(f"[probe_z_grid] runner loaded.", flush=True)

    # Locate actor and inner _SplitRMAMLP (where the grid encoder lives).
    actor = None
    for path in [("alg", "actor_critic", "actor"),
                 ("alg", "actor_critic"),
                 ("alg", "actor")]:
        m = runner
        try:
            for p in path:
                m = getattr(m, p)
            if hasattr(m, "mlp"):
                actor = m
                break
        except AttributeError:
            continue
    if actor is None or not hasattr(actor, "mlp"):
        raise RuntimeError("could not locate actor.mlp on runner")
    inner_mlp = actor.mlp
    # inner_mlp is _SplitRMAMLP with [priv_encoder, grid_encoder, pi_teacher].
    grid_encoder = inner_mlp[1]
    print(f"[probe_z_grid] grid encoder: {type(grid_encoder).__name__}",
          flush=True)

    obs, _ = env.reset()

    N = args.num_envs
    S = args.rollout_steps
    P = args.priv_dim

    z_grid_history = []
    grid_input_history = []

    for step in range(S):
        obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
        if obs_tensor.dim() > 2:
            obs_tensor = obs_tensor.reshape(-1, obs_tensor.shape[-1])

        with torch.no_grad():
            grid_flat = obs_tensor[:, P:]            # (N, 8192)
            grid = grid_flat.reshape(N, 2, GRID_H, GRID_W)
            z_grid = grid_encoder(grid)              # (N, 64)
            z_grid_history.append(z_grid.cpu().float().numpy())
            grid_input_history.append(grid.cpu().float().numpy())
            action = policy(obs)

        step_out = env.step(action)
        obs = step_out[0]

        if step % 20 == 0 or step == S - 1:
            print(f"[probe_z_grid] step {step:>3}/{S}  "
                  f"z_grid.std={float(z_grid.std()):.3f}", flush=True)

    Z = np.concatenate(z_grid_history, axis=0)       # (S*N, 64)
    G = np.concatenate(grid_input_history, axis=0)   # (S*N, 2, 64, 64)
    print(f"[probe_z_grid] computing features for {G.shape[0]} samples...",
          flush=True)
    features = compute_grid_features(G)

    # Train/test split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(Z.shape[0])
    split = int(Z.shape[0] * args.train_frac)
    train_idx = perm[:split]
    test_idx = perm[split:]
    Z_train, Z_test = Z[train_idx], Z[test_idx]

    results = {}
    for name, y in features.items():
        y_train, y_test = y[train_idx], y[test_idx]
        if y_test.std() < 1e-8:
            results[name] = {
                "r2_train": float("nan"),
                "r2_test": float("nan"),
                "mse_test": float("nan"),
                "feature_std": float(y_test.std()),
                "note": "zero variance",
            }
            continue
        r2_tr, r2_te, mse_te = linear_probe_r2(Z_train, y_train, Z_test, y_test)
        results[name] = {
            "r2_train": r2_tr,
            "r2_test": r2_te,
            "mse_test": mse_te,
            "feature_std": float(y_test.std()),
            "feature_mean": float(y_test.mean()),
        }

    output = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "n_envs": int(N),
        "n_rollout_steps": int(S),
        "priv_dim": int(P),
        "z_grid_dim": int(Z.shape[1]),
        "n_samples_total": int(Z.shape[0]),
        "n_train": int(split),
        "n_test": int(Z.shape[0] - split),
        "linear_probe": results,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    # Console summary
    print("", flush=True)
    print("=" * 70, flush=True)
    print(f"z_grid linear probe — {args.task}", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'feature':<22} {'R²_test':>10} {'std(feat)':>10}  bar", flush=True)
    rows = sorted(results.items(), key=lambda kv: -kv[1].get("r2_test", 0.0))
    for name, r in rows:
        r2 = r["r2_test"]
        bar = "█" * int(min(40, max(0, r2 * 40)))
        std = r["feature_std"]
        print(f"  {name:<22} {r2:>+10.3f} {std:>10.3f}  {bar}", flush=True)
    print("", flush=True)
    print(f"  full output → {out_path}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    import os
    rc = 1
    try:
        rc = main()
    except Exception as e:
        import traceback
        print(f"[probe_z_grid] FATAL: {e}", flush=True)
        traceback.print_exc()
    os._exit(rc if rc is not None else 0)
