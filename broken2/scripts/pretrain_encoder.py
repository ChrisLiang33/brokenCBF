#!/usr/bin/env python3
"""v3.0e: supervised pretraining of the teacher's _GridDynamicsEncoder.

After v3.0d, the diagnose_alpha_corr probe showed that the policy IS
state-conditional — but only on cbf_state features (h, slack, L_g h),
NOT on env-class features (friction, mass, COM offset). The bottleneck
is no longer obs-access (we have raw_dyn skip) — it's that PPO's
gradient on env-class features is too weak/indirect to compete with
the strong cbf_state signal.

This script provides a CLEAN supervised gradient that forces the
encoder to encode env-class features into Z BEFORE PPO starts. We
collect obs samples from the env (random actions), train the encoder
+ an auxiliary head to predict the first 15 dynamics features from Z
via MSE, and save the encoder state_dict.

The PPO training script (train_and_eval_v3_0e.sh) then loads this
pretrained encoder and freezes it, so Z is guaranteed to carry env
class through all of PPO.

Usage on lab box:
  cd ~/Desktop/safety-go2/IsaacLab && \\
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/pretrain_encoder.py \\
    --task Isaac-CBF-Go2-v0 \\
    --num_envs 1024 --rollout_steps 200 \\
    --num_epochs 50 --batch_size 4096 \\
    --learning_rate 5e-4 --z_dim 24 \\
    --output pretrained_encoder_v3_0e.pt \\
    --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-CBF-Go2-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--rollout_steps", type=int, default=200,
                    help="Number of env steps with random actions to collect obs.")
parser.add_argument("--num_epochs", type=int, default=50,
                    help="Training epochs on the collected obs buffer.")
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--z_dim", type=int, default=24,
                    help="Bottleneck dimension. Must match the PPO model's z_dim.")
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[pretrain] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[pretrain] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import isaaclab_tasks  # noqa: F401  — registers tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.manager_based.safety.cbf_go2.cbf_go2_teacher_cnn import (
    _GridDynamicsEncoder,
)

# Number of privileged features the aux head predicts from Z.
# Matches the first 15 dims of TeacherPrivCfg's flat layout:
#   friction(1) + base_mass(1) + base_height(1) + applied_force(3)
#   + applied_torque(3) + tracking_err(3) + com_offset(3) = 15
# (cbf_state at obs[15:19] is NOT a target — it's derived per-step.)
_PRIV_FEATURE_DIM = 15


def build_env():
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    return env


def collect_observations(env, num_steps: int) -> torch.Tensor:
    """Roll out env with random actions, return stacked obs."""
    obs_buffer = []
    obs, _ = env.reset()
    obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
    obs_dim = obs_tensor.shape[-1]
    print(f"[pretrain] collecting {num_steps} steps × {args.num_envs} envs "
          f"= {num_steps * args.num_envs} samples (obs_dim={obs_dim})...",
          flush=True)

    for step in range(num_steps):
        action = torch.zeros(
            (args.num_envs, 5),
            device="cuda:0",
            dtype=torch.float32,
        )
        # mild perturbation around zero so the env actually moves
        action.normal_(0.0, 0.5)
        step_out = env.step(action)
        obs = step_out[0]
        obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
        obs_buffer.append(obs_tensor.detach().cpu())
        if step % 20 == 0 or step == num_steps - 1:
            print(f"[pretrain]   collect step {step+1:>3}/{num_steps}", flush=True)

    return torch.cat(obs_buffer, dim=0)


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    env = build_env()
    obs_buffer = collect_observations(env, args.rollout_steps)
    env.close()
    print(f"[pretrain] collected {obs_buffer.shape[0]} samples, "
          f"obs_dim={obs_buffer.shape[1]}", flush=True)

    # Build encoder + aux head (fresh, not loaded).
    encoder = _GridDynamicsEncoder(
        output_dim=args.z_dim,
        head_hidden_dim=128,
        last_activation=True,
    ).to(device)
    aux_head = nn.Sequential(
        nn.Linear(args.z_dim, 64),
        nn.ELU(),
        nn.Linear(64, _PRIV_FEATURE_DIM),
    ).to(device)

    params = list(encoder.parameters()) + list(aux_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-5)

    # DataLoader over the obs buffer.
    dataset = TensorDataset(obs_buffer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"[pretrain] training: {args.num_epochs} epochs, "
          f"batch_size={args.batch_size}, lr={args.learning_rate}", flush=True)

    encoder.train()
    aux_head.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (obs_batch,) in loader:
            obs_batch = obs_batch.to(device, non_blocking=True)
            z = encoder(obs_batch)
            target = obs_batch[:, :_PRIV_FEATURE_DIM]
            pred = aux_head(z)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == args.num_epochs - 1:
            print(f"[pretrain] epoch {epoch+1:>3}/{args.num_epochs}  "
                  f"mse={mean_loss:.5f}", flush=True)

    # ── SAVE FIRST (before eval) so we don't lose the trained encoder ──
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "aux_head": aux_head.state_dict(),
        "z_dim": args.z_dim,
        "priv_feature_dim": _PRIV_FEATURE_DIM,
        "task": args.task,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "num_epochs": args.num_epochs,
    }, out_path)
    print(f"[pretrain] saved → {out_path}", flush=True)

    # Per-feature MSE in batches (full buffer at once OOMs the 5090).
    encoder.eval()
    aux_head.eval()
    with torch.no_grad():
        sq_sum = torch.zeros(_PRIV_FEATURE_DIM, device=device)
        count = 0
        for (obs_batch,) in DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0):
            obs_batch = obs_batch.to(device, non_blocking=True)
            z = encoder(obs_batch)
            pred = aux_head(z)
            target = obs_batch[:, :_PRIV_FEATURE_DIM]
            sq_sum += ((pred - target) ** 2).sum(dim=0)
            count += obs_batch.shape[0]
        per_feat_mse = (sq_sum / max(count, 1)).cpu()
    feat_names = [
        "friction", "base_mass", "base_height",
        "force_x", "force_y", "force_z",
        "torque_x", "torque_y", "torque_z",
        "tracking_x", "tracking_y", "tracking_z",
        "com_x", "com_y", "com_z",
    ]
    print("", flush=True)
    print(f"[pretrain] per-feature final MSE (lower = encoded better):", flush=True)
    for name, mse in zip(feat_names, per_feat_mse.tolist()):
        print(f"    {name:>14}  {mse:.5f}", flush=True)

    # Update the saved file with the per-feature stats (overwrites).
    ckpt = torch.load(out_path, map_location="cpu")
    ckpt["final_per_feat_mse"] = per_feat_mse.tolist()
    torch.save(ckpt, out_path)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception as e:
        import traceback
        print(f"[pretrain] FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(rc)
