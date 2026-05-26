"""RMA fingerprint gate: for each privileged factor we plan to feed the
teacher, can a supervised model recover it from the proprio history a
deployment-realistic student would see? Same methodology as Phase 1.5,
extended to multi-channel priv.

If R^2 is high for a channel, the signal is there and the student can
learn it. If R^2 is low, that channel either:
- doesn't actually affect the simulation (the DR isn't physics-wired),
- is masked by noise in the proprio history,
- or the DR range is too narrow to leave a fingerprint.

Either way, low R^2 means the policy can't usefully condition on that
channel, so we either widen the range, fix the physics-apply path, or
drop the channel from priv.

Threshold: R^2 >= 0.5 per channel (deliberately lenient; Phase 1.5 hit
0.955 for disturbance alone on a window we know is rich).

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase5_fingerprint_gate.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --num_envs 64 --n_episodes 200 --headless
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--n_episodes", type=int, default=200,
                    help="Total per-env episodes to collect data over. Each "
                         "env contributes one (history, priv) pair per "
                         "episode, taken WINDOW_LEN steps in.")
parser.add_argument("--window_len", type=int, default=20,
                    help="Proprio history window (matches Phase 1.5).")
parser.add_argument("--ep_len", type=int, default=100,
                    help="Steps per episode rollout (env auto-resets handle "
                         "longer/shorter; we just sample one window per "
                         "episode at WINDOW_LEN steps in).")
parser.add_argument("--mlp_epochs", type=int, default=200)
parser.add_argument("--mlp_hidden", type=int, default=64)
parser.add_argument("--r2_threshold", type=float, default=0.5,
                    help="Per-channel R^2 PASS threshold.")
parser.add_argument("--out_csv", default="phase5_fingerprint_gate.csv")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import csv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor
from cbf_task.mdp import deployable_obs, priv_obs


TASK = "Isaac-CBF-Adaptive-Go2-RMA-v0"
PRIV_NAMES = ["disturbance", "friction", "mass_delta", "motor_strength"]
PROPRIO_DIM = 48   # `deployable_obs`: proprio + cmd + last_loco_action

# Safe fixed (phi, alpha) used during rollout collection so the robot
# actually walks coherently toward the goal (matching Phase 1.5's
# methodology). Without this, random CBF params would pin the robot
# in place / make motion chaotic, and no priv channel produces a
# stable proprio response.
# (phi=0.3, alpha=2.5) -> normalize via phi_bounds=(0,1), alpha_bounds=(0.2, 4.0)
SAFE_ACTION = torch.tensor([-0.4, 0.21052632], dtype=torch.float32)


def collect_rollouts(env, num_envs, n_episodes, window_len, ep_len, device):
    """Drive the env with random (phi, alpha) actions; at WINDOW_LEN steps
    into each episode, snapshot (history_window, priv) per env. The
    history window is the last `window_len` proprio frames concatenated.

    Returns (X, Y) numpy arrays:
        X: (n_episodes * num_envs, window_len * PROPRIO_DIM)
        Y: (n_episodes * num_envs, 4)
    """
    X_list, Y_list = [], []
    env.reset()
    history_buf = torch.zeros(
        (num_envs, window_len, PROPRIO_DIM), device=device,
    )
    step_in_ep = torch.zeros(num_envs, dtype=torch.long, device=device)
    eps_done = torch.zeros(num_envs, dtype=torch.long, device=device)
    target_eps_per_env = n_episodes
    pbar_step = 0
    safe_action = SAFE_ACTION.to(device).expand(num_envs, 2)
    while (eps_done < target_eps_per_env).any():
        # SAFE fixed (phi, alpha) so the robot actually walks. The gate
        # measures "is the priv signal recoverable from proprio when the
        # locomotion is doing its normal job" -- not "under chaotic CBF."
        obs, _, term, trunc, _ = env.step(safe_action)

        # update history buffer (shift left, append new frame). Use the
        # 48-dim `deployable_obs` -- the same proprio Phase 1.5 hit
        # R^2=0.955 with. This is also what the eventual student
        # `phi(history)` will consume at deployment.
        unwr = env.unwrapped
        proprio_now = deployable_obs(unwr)               # (N, 48)
        history_buf = torch.roll(history_buf, shifts=-1, dims=1)
        history_buf[:, -1, :] = proprio_now
        step_in_ep += 1

        # at exactly WINDOW_LEN steps into the episode, snapshot
        snap_now = (step_in_ep == window_len) & (eps_done < target_eps_per_env)
        if snap_now.any():
            priv = priv_obs(unwr)
            X = history_buf[snap_now].reshape(snap_now.sum().item(), -1)
            Y = priv[snap_now]
            X_list.append(X.detach().cpu())
            Y_list.append(Y.detach().cpu())

        # detect any env that terminated -> reset its history + counters
        done = term | trunc
        if done.any():
            history_buf[done] = 0.0
            step_in_ep[done] = 0
            eps_done[done] += 1

        pbar_step += 1
        if pbar_step % 100 == 0:
            print(f"[gate] step {pbar_step}  collected={sum(x.shape[0] for x in X_list)}  "
                  f"eps_done min/max={int(eps_done.min())}/{int(eps_done.max())}")

    X = torch.cat(X_list, dim=0).numpy()
    Y = torch.cat(Y_list, dim=0).numpy()
    return X, Y


class FingerprintMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def per_channel_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """R^2 per output column."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-9)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco

    print(f"[gate] building {TASK} ({args_cli.num_envs} envs) ...")
    env = gym.make(TASK, cfg=env_cfg)

    print(f"[gate] collecting (history, priv) pairs ...")
    X, Y = collect_rollouts(env, args_cli.num_envs, args_cli.n_episodes,
                            args_cli.window_len, args_cli.ep_len, device)
    print(f"[gate] X shape: {X.shape}  Y shape: {Y.shape}")
    env.close()

    # train/test split
    n = X.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)
    tr, te = idx[:n_train], idx[n_train:]

    # per-channel target standardization (so MSE balances across channels
    # with very different scales: 30-N vs 1-unitless)
    y_mean = Y[tr].mean(axis=0, keepdims=True)
    y_std = Y[tr].std(axis=0, keepdims=True).clip(min=1e-6)
    Y_n = (Y - y_mean) / y_std

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_n, dtype=torch.float32, device=device)

    model = FingerprintMLP(
        in_dim=X.shape[1], hidden=args_cli.mlp_hidden, out_dim=Y.shape[1],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"[gate] training fingerprint MLP for {args_cli.mlp_epochs} epochs ...")
    for ep in range(args_cli.mlp_epochs):
        # mini-batch on train indices
        perm = torch.randperm(len(tr), device=device)
        batch_size = 4096
        total = 0.0
        for i in range(0, len(tr), batch_size):
            sel = torch.tensor(tr[perm[i:i+batch_size].cpu().numpy()],
                               device=device, dtype=torch.long)
            yhat = model(Xt[sel])
            loss = loss_fn(yhat, Yt[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * sel.shape[0]
        if (ep + 1) % 25 == 0 or ep == 0:
            with torch.no_grad():
                te_sel = torch.tensor(te, device=device, dtype=torch.long)
                yhat_te = model(Xt[te_sel])
                te_loss = loss_fn(yhat_te, Yt[te_sel]).item()
            print(f"    epoch {ep+1:>3d}  train_loss={total/len(tr):.4f}  "
                  f"test_loss={te_loss:.4f}")

    # final per-channel R^2 in ORIGINAL units (de-normalize predictions)
    with torch.no_grad():
        yhat_te = model(Xt[torch.tensor(te, device=device, dtype=torch.long)])
        yhat_te = yhat_te.cpu().numpy() * y_std + y_mean
    r2 = per_channel_r2(Y[te], yhat_te)

    # report
    print()
    print("=" * 78)
    print("  RMA FINGERPRINT GATE  -- can the proprio history predict each priv channel?")
    print("=" * 78)
    print(f"  {'channel':>16}  {'range':>14}  {'R^2 (test)':>10}  {'pass?':>6}")
    rows = []
    all_pass = True
    for i, name in enumerate(PRIV_NAMES):
        ymin, ymax = float(Y[:, i].min()), float(Y[:, i].max())
        ok = bool(r2[i] >= args_cli.r2_threshold)
        all_pass = all_pass and ok
        print(f"  {name:>16}  [{ymin:>+6.2f}, {ymax:>+5.2f}]  {r2[i]:>+10.3f}  "
              f"{'PASS' if ok else 'FAIL':>6}")
        rows.append({"channel": name, "range_min": ymin, "range_max": ymax,
                     "r2_test": float(r2[i]),
                     "pass": ok})
    print("=" * 78)
    print(f"  verdict: {'PASS -- all channels learnable' if all_pass else 'REVIEW -- weak channel(s)'}")

    with open(args_cli.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {args_cli.out_csv}")

    simulation_app.close()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
