#!/usr/bin/env python3
"""Caveat-#4 ablation diagnostic — partial correlations of α and φ heads.

The marginal Pearson correlations from diagnose_alpha_corr.py and
diagnose_phi_corr.py both show tracking_err as the dominant signal
(+0.35 for φ, +0.36 for α on V8). But tracking_err is a SYMPTOM that
rises when ANY priv axis is off (σ_act, friction, COM, mass, push).

Question: do α and φ have INDEPENDENT priv-feature sensitivities, or
are they both just gating off tracking_err and looking specialized only
because tracking_err is correlated with everything?

This script does ONE rollout (N envs × S steps) and saves a raw .npz
that lets us run multivariate regression locally. No analysis here —
just data collection. analyze_shared_signal.py runs the regressions.

What gets saved (.npz):
  Per-env (N,) arrays:
    alpha_per_env, phi_per_env       — time-averaged head outputs
    tracking_err_norm_per_env        — time-averaged ‖tracking_err‖
    friction_per_env, base_mass_per_env, base_height_per_env,
    actuation_noise_sigma_per_env, com_norm_per_env, base_ang_vel_norm_per_env
  Per-step per-env (S, N) arrays (for within-ep regression):
    alpha_history, phi_history
    h_history, lgh_norm_sq_history
    tracking_err_norm_history

Usage on lab box (inside tmux):
  cd ~/Desktop/safety-go2/IsaacLab
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_shared_signal.py \\
    --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V8-v0 \\
    --checkpoint logs/rsl_rl/cbf_go2_teacher_rma/2026-05-19_14-18-46/model_2499.pt \\
    --num_envs 256 --rollout_steps 100 \\
    --priv_dim 33 \\
    --alpha_min 0.5 --alpha_max 3.0 \\
    --output diagnose_shared_signal_wk3tight8.npz \\
    --headless
"""

from __future__ import annotations

import argparse
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
parser.add_argument("--priv_dim", type=int, default=33)
parser.add_argument("--alpha_min", type=float, default=0.5,
                    help="α encoding lower bound (V8 default 0.5)")
parser.add_argument("--alpha_max", type=float, default=3.0,
                    help="α encoding upper bound (V8 default 3.0)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print("[diagnose_shared_signal] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[diagnose_shared_signal] AppLauncher ready in {time.time()-t0:.1f}s",
      flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


PHI_MIN = 0.0
PHI_MAX = 5.0


def get_priv_layout(priv_dim: int) -> dict:
    if priv_dim == 33:
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
    raise ValueError(f"Unsupported priv_dim={priv_dim} — only 31, 33 supported here.")


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"
    layout = get_priv_layout(args.priv_dim)

    print(f"[diagnose_shared_signal] task={args.task}, num_envs={args.num_envs}, "
          f"rollout_steps={args.rollout_steps}, priv_dim={args.priv_dim}",
          flush=True)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    print("[diagnose_shared_signal] runner loaded.", flush=True)

    obs, _ = env.reset()

    N = args.num_envs
    S = args.rollout_steps
    alpha_history = np.zeros((S, N), dtype=np.float32)
    phi_history = np.zeros((S, N), dtype=np.float32)
    priv_history = np.zeros((S, N, args.priv_dim), dtype=np.float32)
    h_history = np.zeros((S, N), dtype=np.float32)
    lgh_norm_sq_history = np.zeros((S, N), dtype=np.float32)
    tracking_err_norm_history = np.zeros((S, N), dtype=np.float32)

    a_lo, a_hi = args.alpha_min, args.alpha_max

    for step in range(S):
        with torch.no_grad():
            raw_action = policy(obs)
            alpha_phys = a_lo + (torch.tanh(raw_action[:, 0]) + 1.0) * 0.5 * (a_hi - a_lo)
            phi_phys = PHI_MIN + (torch.tanh(raw_action[:, 1]) + 1.0) * 0.5 * (PHI_MAX - PHI_MIN)
            alpha_history[step] = alpha_phys.cpu().numpy()
            phi_history[step] = phi_phys.cpu().numpy()

            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            priv_history[step] = obs_tensor[:, :args.priv_dim].cpu().numpy()

        step_out = env.step(raw_action)
        obs = step_out[0]

        if hasattr(inner, "last_h_for_obs"):
            h_history[step] = inner.last_h_for_obs.detach().cpu().numpy()
        if hasattr(inner, "last_lgh_for_obs"):
            lgh = inner.last_lgh_for_obs.detach()
            lgh_norm_sq_history[step] = (lgh ** 2).sum(dim=-1).cpu().numpy()

        # Per-step ‖tracking_err‖ — use the most recent 3 of the 15-D history
        # (i.e. tracking_err for the current step).
        te = priv_history[step, :, layout["tracking_err"]]  # (N, 15)
        te_current = te[:, -3:]  # most recent (x, y, z)
        tracking_err_norm_history[step] = np.linalg.norm(te_current, axis=-1)

        if step % 20 == 0 or step == S - 1:
            print(f"[diagnose_shared_signal] step {step:>3}/{S}  "
                  f"α mean={alpha_phys.mean().item():.3f}  "
                  f"φ mean={phi_phys.mean().item():.3f}", flush=True)

    # ── Per-env aggregates ──
    alpha_per_env = alpha_history.mean(axis=0)
    phi_per_env = phi_history.mean(axis=0)

    priv_per_env = priv_history.mean(axis=0)

    def feat_scalar(name):
        return priv_per_env[:, layout[name]].flatten()

    def feat_norm(name):
        v = priv_per_env[:, layout[name]]
        return np.linalg.norm(v, axis=-1)

    friction_per_env = feat_scalar("friction")
    base_mass_per_env = feat_scalar("base_mass")
    base_height_per_env = feat_scalar("base_height")
    sigma_per_env = feat_scalar("actuation_noise_sigma")
    tracking_err_norm_per_env = feat_norm("tracking_err")
    com_norm_per_env = feat_norm("com_offset")
    base_ang_vel_norm_per_env = feat_norm("base_ang_vel")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        task=np.array(args.task),
        checkpoint=np.array(args.checkpoint),
        priv_dim=np.array(args.priv_dim),
        n_envs=np.array(N), n_steps=np.array(S),
        # Per env (N,)
        alpha_per_env=alpha_per_env,
        phi_per_env=phi_per_env,
        friction_per_env=friction_per_env,
        base_mass_per_env=base_mass_per_env,
        base_height_per_env=base_height_per_env,
        actuation_noise_sigma_per_env=sigma_per_env,
        tracking_err_norm_per_env=tracking_err_norm_per_env,
        com_norm_per_env=com_norm_per_env,
        base_ang_vel_norm_per_env=base_ang_vel_norm_per_env,
        # Per step per env (S, N)
        alpha_history=alpha_history,
        phi_history=phi_history,
        h_history=h_history,
        lgh_norm_sq_history=lgh_norm_sq_history,
        tracking_err_norm_history=tracking_err_norm_history,
    )

    print("", flush=True)
    print("=" * 70, flush=True)
    print(f"shared-signal diagnostic — {args.task}", flush=True)
    print("=" * 70, flush=True)
    print(f"  α: pop mean={alpha_per_env.mean():.3f}  "
          f"between-env std={alpha_per_env.std():.3f}  "
          f"within-env std={alpha_history.std(axis=0).mean():.3f}", flush=True)
    print(f"  φ: pop mean={phi_per_env.mean():.3f}  "
          f"between-env std={phi_per_env.std():.3f}  "
          f"within-env std={phi_history.std(axis=0).mean():.3f}", flush=True)
    print(f"  raw .npz written to {out_path}", flush=True)
    print(f"  Run scripts/analyze_shared_signal.py on this file locally.", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    import os
    rc = 1
    try:
        rc = main()
    except Exception as e:
        import traceback
        print(f"[diagnose_shared_signal] FATAL: {e}", flush=True)
        traceback.print_exc()
    os._exit(rc if rc is not None else 0)
