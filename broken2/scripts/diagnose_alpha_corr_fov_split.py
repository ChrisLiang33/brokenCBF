#!/usr/bin/env python3
"""FOV-split version of diagnose_alpha_corr.py.

The original diagnostic time-averages α over the whole rollout, then
correlates with per-env DR features. That lumps FOV-active steps
(geometric reward fires, dense gradient on α, drowns out env class)
with FOV-idle steps (geometric reward silent, env class is the only
remaining signal source).

This split version:
  - Classifies each (env, step) by FOV-active vs FOV-idle using the
    CBF h value (index 15 of priv obs). In priv_fov mode, out-of-FOV
    obstacles are clamped to sdf=100 → h saturates near 100; any h
    below a small threshold means at least one obstacle is in range.
  - Computes per-env mean α SEPARATELY on each subset.
  - Reports Pearson(α_active, DR) and Pearson(α_idle, DR).

Hypothesis: if env-class adaptation lives in the FOV-idle baseline α
(where there's no dense geometric reward to mask it), the idle subset
should show stronger DR correlation than the active subset.

Usage on lab box:
  cd ~/Desktop/safety-go2/IsaacLab && \\
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr_fov_split.py \\
    --task Isaac-CBF-Go2-RMA-V32-v0 \\
    --checkpoint <path/to/v3.2.1/model_2999.pt> \\
    --num_envs 256 --rollout_steps 100 \\
    --output diagnose_alpha_corr_fov_split_v3_2_1.json \\
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
parser.add_argument("--fov_h_threshold", type=float, default=50.0,
                    help="h below this = FOV-active. priv_fov clamps h to ~100 "
                         "when no obstacle is in range, so 50 is a safe split.")
parser.add_argument("--priv_dim", type=int, default=19,
                    help="Number of priv obs dims. 19 for v3.x (15 DR + 4 cbf_state). "
                         "Layer 2 will be 20 (16 DR + 4 cbf_state).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[fov_split] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[fov_split] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


ALPHA_MIN = 0.1
ALPHA_MAX = 5.0


def get_priv_layout(priv_dim: int) -> dict:
    """Layout of priv obs depending on whether actuation_noise_sigma is included.

    v3.x (priv_dim=19): friction, mass, height, force, torque, tracking_err,
                       com_offset, cbf_state.
    Layer 2 (priv_dim=20): same + actuation_noise_sigma (1 dim) before cbf_state.
    """
    if priv_dim == 19:
        return {
            "friction":      slice(0, 1),
            "base_mass":     slice(1, 2),
            "base_height":   slice(2, 3),
            "applied_force": slice(3, 6),
            "applied_torque": slice(6, 9),
            "tracking_err":  slice(9, 12),
            "com_offset":    slice(12, 15),
            "cbf_state":     slice(15, 19),
        }
    elif priv_dim == 20:
        return {
            "friction":              slice(0, 1),
            "base_mass":             slice(1, 2),
            "base_height":           slice(2, 3),
            "applied_force":         slice(3, 6),
            "applied_torque":        slice(6, 9),
            "tracking_err":          slice(9, 12),
            "com_offset":            slice(12, 15),
            "actuation_noise_sigma": slice(15, 16),
            "cbf_state":             slice(16, 20),
        }
    else:
        raise ValueError(f"Unsupported priv_dim={priv_dim} (expected 19 or 20)")


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float)
    y = y.astype(float)
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 5:
        return float("nan")
    x, y = x[valid], y[valid]
    mx, my = x.mean(), y.mean()
    sx, sy = x.std() + 1e-12, y.std() + 1e-12
    return float(((x - mx) * (y - my)).mean() / (sx * sy))


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    layout = get_priv_layout(args.priv_dim)
    h_idx = layout["cbf_state"].start  # h is the first cbf_state dim

    print(f"[fov_split] task={args.task}, num_envs={args.num_envs}, "
          f"rollout_steps={args.rollout_steps}, priv_dim={args.priv_dim}, "
          f"h_idx={h_idx}, fov_threshold={args.fov_h_threshold}", flush=True)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    print(f"[fov_split] runner loaded.", flush=True)

    obs, _ = env.reset()

    N = args.num_envs
    S = args.rollout_steps
    alpha_history = np.zeros((S, N), dtype=np.float32)
    priv_history = np.zeros((S, N, args.priv_dim), dtype=np.float32)

    for step in range(S):
        with torch.no_grad():
            raw_action = policy(obs)
            alpha_phys = (
                ALPHA_MIN
                + (torch.tanh(raw_action[:, 0]) + 1.0) * 0.5 * (ALPHA_MAX - ALPHA_MIN)
            )
            alpha_history[step] = alpha_phys.cpu().numpy()

            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            priv_history[step] = obs_tensor[:, :args.priv_dim].cpu().numpy()

        step_out = env.step(raw_action)
        obs = step_out[0]
        if step % 20 == 0 or step == S - 1:
            print(f"[fov_split] step {step:>3}/{S}  "
                  f"α mean={alpha_phys.mean().item():.3f}  "
                  f"α std={alpha_phys.std().item():.3f}", flush=True)

    # ── FOV-active mask: h < threshold means at least one obstacle in range ──
    h_history = priv_history[:, :, h_idx]                          # (S, N)
    fov_active_mask = h_history < args.fov_h_threshold              # (S, N) bool

    fov_active_rate_overall = float(fov_active_mask.mean())
    fov_active_rate_per_env = fov_active_mask.mean(axis=0)          # (N,)

    # Per-env α split. NaN if env has no steps in that regime.
    alpha_overall_per_env = alpha_history.mean(axis=0)              # (N,)
    alpha_active_per_env = np.full(N, np.nan, dtype=np.float32)
    alpha_idle_per_env   = np.full(N, np.nan, dtype=np.float32)
    for i in range(N):
        active = fov_active_mask[:, i]
        if active.any():
            alpha_active_per_env[i] = alpha_history[active, i].mean()
        if (~active).any():
            alpha_idle_per_env[i]   = alpha_history[~active, i].mean()

    # Per-env feature values (DR features constant per episode; take step 0)
    priv_per_env = priv_history[0]                                  # (N, priv_dim)

    # Build feature dictionary
    feature_vals = {}
    for name in ["friction", "base_mass", "base_height"]:
        feature_vals[name] = priv_per_env[:, layout[name]].flatten()
    for name in ["applied_force", "applied_torque", "tracking_err", "com_offset"]:
        feat = priv_per_env[:, layout[name]]
        feature_vals[f"|{name}|"] = np.linalg.norm(feat, axis=-1)
        if name == "com_offset":
            for i, axis in enumerate(["com_x", "com_y", "com_z"]):
                feature_vals[axis] = feat[:, i]
    if "actuation_noise_sigma" in layout:
        feature_vals["actuation_noise_sigma"] = priv_per_env[:, layout["actuation_noise_sigma"]].flatten()

    # CBF state features (time-averaged because they vary per step)
    cbf_state_mean_per_env = priv_history[:, :, layout["cbf_state"]].mean(axis=0)  # (N, 4)
    for i, name in enumerate(["h", "Lgh_dot_udes", "Lgh_norm_sq", "slack"]):
        feature_vals[name] = cbf_state_mean_per_env[:, i]

    # Compute Pearson on each subset
    def corr_table(alpha_vec: np.ndarray) -> dict:
        return {name: round(pearson_corr(feat, alpha_vec), 4)
                for name, feat in feature_vals.items()}

    corr_overall = corr_table(alpha_overall_per_env)
    corr_active  = corr_table(alpha_active_per_env)
    corr_idle    = corr_table(alpha_idle_per_env)

    # Diagnostics on how many envs had each regime
    n_envs_with_active = int((~np.isnan(alpha_active_per_env)).sum())
    n_envs_with_idle   = int((~np.isnan(alpha_idle_per_env)).sum())

    stats = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "n_envs": int(N),
        "n_rollout_steps": int(S),
        "fov_h_threshold": float(args.fov_h_threshold),
        "fov_active_rate_overall": fov_active_rate_overall,
        "fov_active_rate_per_env_mean": float(fov_active_rate_per_env.mean()),
        "fov_active_rate_per_env_std": float(fov_active_rate_per_env.std()),
        "n_envs_with_active_steps": n_envs_with_active,
        "n_envs_with_idle_steps":   n_envs_with_idle,
        "alpha": {
            "overall_mean": float(np.nanmean(alpha_overall_per_env)),
            "overall_std":  float(np.nanstd(alpha_overall_per_env)),
            "active_mean":  float(np.nanmean(alpha_active_per_env)),
            "active_std":   float(np.nanstd(alpha_active_per_env)),
            "idle_mean":    float(np.nanmean(alpha_idle_per_env)),
            "idle_std":     float(np.nanstd(alpha_idle_per_env)),
        },
        "correlations_overall": corr_overall,
        "correlations_fov_active": corr_active,
        "correlations_fov_idle":   corr_idle,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Console summary ──
    print("", flush=True)
    print("=" * 78, flush=True)
    print(f"FOV-split α correlation — {args.task}", flush=True)
    print("=" * 78, flush=True)
    print(f"  FOV-active rate:     {fov_active_rate_overall:.3f} "
          f"(per-env: μ={fov_active_rate_per_env.mean():.3f} "
          f"σ={fov_active_rate_per_env.std():.3f})", flush=True)
    print(f"  Envs w/ active:      {n_envs_with_active}/{N}", flush=True)
    print(f"  Envs w/ idle:        {n_envs_with_idle}/{N}", flush=True)
    print("", flush=True)
    print(f"  α distribution:", flush=True)
    print(f"    overall μ={stats['alpha']['overall_mean']:.3f}  σ={stats['alpha']['overall_std']:.3f}", flush=True)
    print(f"    active  μ={stats['alpha']['active_mean']:.3f}  σ={stats['alpha']['active_std']:.3f}", flush=True)
    print(f"    idle    μ={stats['alpha']['idle_mean']:.3f}  σ={stats['alpha']['idle_std']:.3f}", flush=True)
    print("", flush=True)

    # Side-by-side correlation table
    print(f"  Pearson(α, feature) — overall | FOV-active | FOV-idle", flush=True)
    print(f"  {'-'*64}", flush=True)
    all_keys = list(corr_overall.keys())
    for name in all_keys:
        r_o = corr_overall.get(name, float("nan"))
        r_a = corr_active.get(name, float("nan"))
        r_i = corr_idle.get(name, float("nan"))
        marker = ""
        best = max(abs(r_a) if not np.isnan(r_a) else 0,
                   abs(r_i) if not np.isnan(r_i) else 0,
                   abs(r_o) if not np.isnan(r_o) else 0)
        if best > 0.20:
            marker = "  ←← STRONG"
        elif best > 0.10:
            marker = "  ← weak"
        print(f"    {name:>22}  {r_o:+.3f}  |  {r_a:+.3f}  |  {r_i:+.3f}{marker}", flush=True)

    print("", flush=True)
    print(f"  full stats → {out_path}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception as e:
        import traceback
        print(f"[fov_split] FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(rc)
