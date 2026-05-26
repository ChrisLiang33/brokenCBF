#!/usr/bin/env python3
"""Within-distribution state-conditioning diagnostic for the CBF h-shift
parameter `c` (CBF action dim index 4).

For LAYER3_PUSH_A_C onwards. The headline question is whether the policy
learns negative c to compensate for shield_v0c's per-cluster radius
over-estimate (+SHIELD_R_SAFETY_MARGIN = 0.10 m). Expected: mean(c) < 0
during normal walking. Strong gate: Pearson(c, h_perceived) > 0 — when
the perceived constraint reads safer (high h), the policy shifts c
inward to be more conservative.

Usage on lab box:
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_c_corr.py \\
    --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0 \\
    --checkpoint logs/rsl_rl/cbf_go2_teacher_rma/<run>/model_1499.pt \\
    --num_envs 256 --rollout_steps 100 \\
    --priv_dim 31 \\
    --output diagnose_c_corr_wk3pushac.json \\
    --headless
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True, type=str)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rollout_steps", type=int, default=100)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--priv_dim", type=int, default=31)
parser.add_argument("--c_lo", type=float, default=-0.20,
                    help="Lower bound of c_param_range in env_cfg (default -0.20).")
parser.add_argument("--c_hi", type=float, default=0.20,
                    help="Upper bound of c_param_range in env_cfg (default +0.20).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[diagnose_c_corr] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[diagnose_c_corr] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def get_priv_layout(priv_dim: int) -> dict:
    if priv_dim == 31:
        return {
            "friction":              slice(0, 1),
            "base_mass":             slice(1, 2),
            "base_height":           slice(2, 3),
            "applied_force":         slice(3, 6),
            "applied_torque":        slice(6, 9),
            "tracking_err":          slice(9, 24),
            "base_ang_vel":          slice(24, 27),
            "com_offset":            slice(27, 30),
            "actuation_noise_sigma": slice(30, 31),
        }
    if priv_dim == 33:
        # Wk3 theory bundle (2026-05-17): + mean_signed_delta_R + max_abs_delta_R.
        # This is THE diagnostic for the theory-bundle iteration —
        # c–mean_signed_delta_R correlation > 0.5 is the make-or-break gate.
        return {
            "friction":              slice(0, 1),
            "base_mass":             slice(1, 2),
            "base_height":           slice(2, 3),
            "applied_force":         slice(3, 6),
            "applied_torque":        slice(6, 9),
            "tracking_err":          slice(9, 24),
            "base_ang_vel":          slice(24, 27),
            "com_offset":            slice(27, 30),
            "actuation_noise_sigma": slice(30, 31),
            "mean_signed_delta_R":   slice(31, 32),
            "max_abs_delta_R":       slice(32, 33),
        }
    raise ValueError(f"Unsupported priv_dim={priv_dim}")


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float)
    y = y.astype(float)
    mx, my = x.mean(), y.mean()
    sx, sy = x.std() + 1e-12, y.std() + 1e-12
    return float(((x - mx) * (y - my)).mean() / (sx * sy))


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    layout = get_priv_layout(args.priv_dim)
    c_lo, c_hi = args.c_lo, args.c_hi

    print(f"[diagnose_c_corr] task={args.task}, num_envs={args.num_envs}, "
          f"rollout_steps={args.rollout_steps}, priv_dim={args.priv_dim}, "
          f"c_range=[{c_lo}, {c_hi}]", flush=True)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    print(f"[diagnose_c_corr] runner loaded.", flush=True)

    obs, _ = env.reset()

    N = args.num_envs
    S = args.rollout_steps
    c_history = np.zeros((S, N), dtype=np.float32)
    priv_history = np.zeros((S, N, args.priv_dim), dtype=np.float32)

    for step in range(S):
        with torch.no_grad():
            raw_action = policy(obs)
            # c_param = c_lo + (tanh(raw[:, 4]) + 1) * 0.5 * (c_hi - c_lo)
            c_phys = c_lo + (torch.tanh(raw_action[:, 4]) + 1.0) * 0.5 * (c_hi - c_lo)
            c_history[step] = c_phys.cpu().numpy()

            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            priv_history[step] = obs_tensor[:, :args.priv_dim].cpu().numpy()

        step_out = env.step(raw_action)
        obs = step_out[0]
        if step % 20 == 0 or step == S - 1:
            print(f"[diagnose_c_corr] step {step:>3}/{S}  "
                  f"c mean={c_phys.mean().item():+.3f}  "
                  f"c std={c_phys.std().item():.3f}", flush=True)

    c_per_env = c_history.mean(axis=0)
    priv_per_env = priv_history.mean(axis=0)
    c_within_env_std = c_history.std(axis=0).mean()

    c_flat = c_history.reshape(-1)
    priv_flat = priv_history.reshape(-1, args.priv_dim)

    correlations_between_env = {}
    correlations_within_episode = {}
    detail_scalars = {}

    for name in ["friction", "base_mass", "base_height", "actuation_noise_sigma",
                 "mean_signed_delta_R", "max_abs_delta_R"]:
        if name not in layout:
            continue
        feat_env = priv_per_env[:, layout[name]].flatten()
        feat_flat = priv_flat[:, layout[name]].flatten()
        correlations_between_env[name] = pearson_corr(feat_env, c_per_env)
        correlations_within_episode[name] = pearson_corr(feat_flat, c_flat)
        detail_scalars[name] = {
            "mean": float(feat_env.mean()),
            "std": float(feat_env.std()),
        }

    for name in ["applied_force", "applied_torque", "tracking_err", "base_ang_vel", "com_offset"]:
        if name not in layout:
            continue
        feat_env = priv_per_env[:, layout[name]]
        feat_flat = priv_flat[:, layout[name]]
        norm_env = np.linalg.norm(feat_env, axis=-1)
        norm_flat = np.linalg.norm(feat_flat, axis=-1)
        correlations_between_env[f"|{name}|"] = pearson_corr(norm_env, c_per_env)
        correlations_within_episode[f"|{name}|"] = pearson_corr(norm_flat, c_flat)
        detail_scalars[f"|{name}|"] = {
            "mean": float(norm_env.mean()),
            "std": float(norm_env.std()),
        }

    stats = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "n_envs": int(N),
        "n_rollout_steps": int(S),
        "priv_dim": int(args.priv_dim),
        "c_range": [float(c_lo), float(c_hi)],
        "c_population_mean": float(c_per_env.mean()),
        "c_population_std": float(c_per_env.std()),
        "c_within_env_std_mean": float(c_within_env_std),
        "correlations_with_c_between_env": {
            k: round(v, 4) for k, v in correlations_between_env.items()
        },
        "correlations_with_c_within_episode": {
            k: round(v, 4) for k, v in correlations_within_episode.items()
        },
        "feature_distributions": {
            k: {kk: round(vv, 4) for kk, vv in v.items()}
            for k, v in detail_scalars.items()
        },
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("", flush=True)
    print("=" * 70, flush=True)
    print(f"`c` correlation diagnostic — {args.task}", flush=True)
    print("=" * 70, flush=True)
    print(f"  `c` population: mean={stats['c_population_mean']:+.3f}  "
          f"between-env std={stats['c_population_std']:.3f}  "
          f"within-env std={stats['c_within_env_std_mean']:.3f}", flush=True)
    print(f"  Expected: mean(c) < 0 if policy is correcting shield_v0c "
          f"radius over-estimate.", flush=True)
    print("", flush=True)
    print(f"  Pearson(`c`, feature) WITHIN-episode (per-step):", flush=True)
    sorted_wi = sorted(correlations_within_episode.items(), key=lambda x: -abs(x[1]))
    for name, r in sorted_wi:
        bar_len = int(abs(r) * 40)
        bar = "█" * bar_len
        sign = "+" if r >= 0 else "−"
        marker = ""
        if abs(r) > 0.20:
            marker = "  ←← STRONG"
        elif abs(r) > 0.10:
            marker = "  ← weak"
        print(f"    {name:>22} {sign}{abs(r):.3f}  {bar}{marker}", flush=True)
    print("", flush=True)
    print(f"  Pearson(`c`, feature) BETWEEN-env (per-env mean):", flush=True)
    sorted_be = sorted(correlations_between_env.items(), key=lambda x: -abs(x[1]))
    for name, r in sorted_be:
        sign = "+" if r >= 0 else "−"
        print(f"    {name:>22} {sign}{abs(r):.3f}", flush=True)
    print("", flush=True)
    print(f"  full stats → {out_path}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    import os
    rc = 1
    try:
        rc = main()
    except Exception as e:
        import traceback
        print(f"[diagnose_c_corr] FATAL: {e}", flush=True)
        traceback.print_exc()
    os._exit(rc if rc is not None else 0)
