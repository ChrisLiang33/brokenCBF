#!/usr/bin/env python3
# Z-bottleneck probe for the trained CBF teacher (Wk3.5b).
#
# Feeds realistic random priv_obs vectors through the env_encoder and
# reports per-dim variance of Z. A healthy bottleneck has all dims
# carrying meaningful variance (std >= 0.1). If some dims collapse
# (near-zero std), the encoder isn't using that dim — bottleneck too
# tight, or the DR distribution didn't force the encoder to spread its
# signal.
#
# Pure-torch. Does NOT import Isaac Lab, Isaac Sim, or rsl_rl — rebuilds
# the env_encoder from the checkpoint's state_dict by auto-detecting
# Linear-layer keys under 'mlp.0.*'. Run on Mac or lab box.
#
# Usage:
#     python scripts/probe_z.py --checkpoint path/to/model_2999.pt
#     python scripts/probe_z.py --checkpoint path/to/model_2999.pt --n_samples 5000

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


def find_encoder_layers(state: dict) -> list[str]:
    """Discover env_encoder Linear-layer keys in the state dict.

    Our teacher stores:
        actor.mlp = _SplitMLP(env_encoder, pi_teacher)   # nn.Sequential
    so env_encoder weights land at keys containing 'mlp.0.' and ending
    in '.weight'. rsl_rl may prefix with 'actor.' — be permissive.
    """
    keys = [k for k in state.keys() if "mlp.0." in k and k.endswith(".weight")]
    keys.sort()   # deterministic order matches Linear-layer order
    return keys


def build_encoder(state: dict, keys: list[str]) -> nn.Sequential:
    """Reconstruct env_encoder from discovered Linear weights.

    Assumes architecture: Linear → ELU → Linear → ELU → ... → Linear → ELU
    (matches the teacher's `last_activation='elu'` choice so Z is
    non-linearly activated before π_teacher consumes it).
    """
    layers: list[nn.Module] = []
    for wk in keys:
        bk = wk[: -len(".weight")] + ".bias"
        W = state[wk]
        out_dim, in_dim = W.shape

        lin = nn.Linear(in_dim, out_dim)
        lin.weight.data = W.clone()
        if bk in state:
            lin.bias.data = state[bk].clone()
        layers.append(lin)
        layers.append(nn.ELU())   # activation after every Linear (last_activation=elu)
    return nn.Sequential(*layers)


def sample_realistic_priv_obs(n: int) -> torch.Tensor:
    """Sample 17D priv_obs vectors with ranges matching training DR.

    Layout from CbfObservationsCfg.TeacherPrivCfg:
      [0]      friction                ~ U(0.4, 1.2)
      [1]      base_mass_offset        ~ U(-2, 2) kg
      [2]      base_height             ~ U(0.25, 0.35) m
      [3:6]    applied_force_b         ~ U(-1, 1)^3  N
      [6:9]    applied_torque_b        ~ U(-0.1, 0.1)^3  Nm
      [9:12]   tracking_err (vx,vy,wz) ~ U(-0.5, 0.5)^3  m/s
      [12:14]  obstacle rel_pos_b      ~ U(-3, 3)^2  m
      [14:16]  obstacle rel_vel_b      ~ U(-0.5, 0.5)^2  m/s
      [16]     obstacle radius         = 0.35  (constant)
    """
    o = torch.zeros(n, 17)
    o[:, 0]     = torch.empty(n).uniform_(0.4, 1.2)
    o[:, 1]     = torch.empty(n).uniform_(-2.0, 2.0)
    o[:, 2]     = torch.empty(n).uniform_(0.25, 0.35)
    o[:, 3:6]   = torch.empty(n, 3).uniform_(-1.0, 1.0)
    o[:, 6:9]   = torch.empty(n, 3).uniform_(-0.1, 0.1)
    o[:, 9:12]  = torch.empty(n, 3).uniform_(-0.5, 0.5)
    o[:, 12:14] = torch.empty(n, 2).uniform_(-3.0, 3.0)
    o[:, 14:16] = torch.empty(n, 2).uniform_(-0.5, 0.5)
    o[:, 16]    = 0.35
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe the teacher's Z-bottleneck.")
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Path to rsl_rl model_*.pt checkpoint")
    ap.add_argument("--n_samples", type=int, default=1000,
                    help="# random priv_obs vectors to feed through encoder")
    ap.add_argument("--dump_keys", action="store_true",
                    help="Print all state_dict keys (for debugging)")
    args = ap.parse_args()

    # ---------- 1. Load checkpoint ----------
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # rsl_rl saves actor and critic state dicts separately (keys:
    # actor_state_dict, critic_state_dict, optimizer_state_dict, iter, infos).
    # Older/other save formats use 'model_state_dict'. Try both, fall back
    # to the raw dict.
    state = (
        ckpt.get("actor_state_dict")
        or ckpt.get("model_state_dict")
        or ckpt
    )

    if args.dump_keys:
        print("All state_dict keys:")
        for k in sorted(state.keys()):
            shape = tuple(state[k].shape) if hasattr(state[k], "shape") else "?"
            print(f"  {k}    shape={shape}")
        return 0

    # ---------- 2. Find + rebuild env_encoder ----------
    keys = find_encoder_layers(state)
    if not keys:
        print("No 'mlp.0.*' keys found. Dumping structure for inspection:\n")
        for k in sorted(state.keys())[:40]:
            print(f"  {k}")
        print("\nRe-run with --dump_keys to see all keys.")
        return 1

    print(f"Discovered env_encoder layers ({len(keys)}):")
    for k in keys:
        print(f"  {k}    shape={tuple(state[k].shape)}")

    encoder = build_encoder(state, keys)
    encoder.eval()

    # ---------- 3. Probe ----------
    x = sample_realistic_priv_obs(args.n_samples)
    with torch.no_grad():
        z = encoder(x)

    mean = z.mean(dim=0)
    std = z.std(dim=0)
    z_dim = z.shape[1]

    # ---------- 4. Report ----------
    print(f"\nZ-bottleneck diagnostics (n={args.n_samples}, z_dim={z_dim}):")
    print(f"  {'Dim':<5}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}{'Status':>15}")
    dead = 0
    for i in range(z_dim):
        s = std[i].item()
        if s < 0.01:
            status, dead = "DEAD", dead + 1
        elif s < 0.1:
            status, dead = "near-dead", dead + 1
        else:
            status = "OK"
        print(
            f"  {i:<5}"
            f"{mean[i].item():>10.4f}"
            f"{s:>10.4f}"
            f"{z[:, i].min().item():>10.4f}"
            f"{z[:, i].max().item():>10.4f}"
            f"{status:>15}"
        )

    print()
    if dead == 0:
        print(f"  ✓ All {z_dim} dims carry variance (std >= 0.1). Bottleneck healthy.")
        print(f"  → Safe to proceed with Wk4 student distillation on this Z.")
    else:
        print(f"  ✗ {dead}/{z_dim} dims near-dead (std < 0.1).")
        print(f"  → Options:")
        print(f"      - Retrain with z_dim={max(1, z_dim - dead)} to match effective rank")
        print(f"      - Widen DR so encoder is forced to spread signal across dims")
        print(f"      - Check if input features are themselves near-constant (probe priv obs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
