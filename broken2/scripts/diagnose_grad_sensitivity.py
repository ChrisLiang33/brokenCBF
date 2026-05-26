#!/usr/bin/env python3
"""Gradient sensitivity diagnostic — what do α/φ heads actually depend on?

Partial regression (diagnose_alpha_corr/phi_corr + analyze_shared_signal)
is a LINEAR probe of a non-linear policy. It can miss non-linear use of
priv features, and multicollinearity between priv channels can hide
mediated use (e.g., friction's effect routed through tracking_err).

This diagnostic measures ∂α_phys/∂priv_j and ∂φ_phys/∂priv_j directly
via autograd through the policy. Picks up non-linear and mediated use
that partial β misses.

Reports per-channel:
  - Mean |grad| (raw sensitivity magnitude in physical units)
  - Standardized sensitivity = |grad| × std(feature) (apples-to-apples;
    "how much α moves per 1σ of feature")
  - Signed mean grad (directional sensitivity, averaged across envs+steps)

Usage on lab box (after V10 training or when GPU has headroom):
  cd ~/Desktop/safety-go2/IsaacLab
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_grad_sensitivity.py \\
    --task Isaac-CBF-Go2-RMA-Layer3-PhiWin-TightCor-V8-v0 \\
    --checkpoint logs/rsl_rl/cbf_go2_teacher_rma/2026-05-19_14-18-46/model_2499.pt \\
    --num_envs 256 --rollout_steps 50 --priv_dim 33 \\
    --alpha_min 0.5 --alpha_max 3.0 \\
    --output diagnose_grad_sensitivity_wk3tight8.json --headless
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
parser.add_argument("--rollout_steps", type=int, default=50)
parser.add_argument("--priv_dim", type=int, required=True)
parser.add_argument("--priv_layout", type=str, default="auto",
                    choices=["auto", "v8", "v13"],
                    help="V13 uses HIDDEN-first ordering. Auto-detect from task name.")
parser.add_argument("--alpha_min", type=float, default=0.5)
parser.add_argument("--alpha_max", type=float, default=3.0)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[grad_sens] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[grad_sens] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


PHI_MIN = 0.0
PHI_MAX = 5.0


def get_priv_layout(priv_dim: int, layout: str = "auto", task: str = "") -> list[tuple[str, slice]]:
    """Return (name, slice) pairs for grouping the priv vector."""
    # V13 (2026-05-20): two-stream HIDDEN-first ordering, 33-D total.
    if layout == "v13" or (layout == "auto" and "TwoStream" in task):
        return [
            ("friction",              slice(0, 1)),
            ("base_mass",             slice(1, 2)),
            ("|applied_force|",       slice(2, 5)),
            ("|applied_torque|",      slice(5, 8)),
            ("|com_offset|",          slice(8, 11)),
            ("actuation_noise_sigma", slice(11, 12)),
            ("mean_signed_delta_R",   slice(12, 13)),
            ("max_abs_delta_R",       slice(13, 14)),
            ("base_height",           slice(14, 15)),
            ("|tracking_err|",        slice(15, 30)),
            ("|base_ang_vel|",        slice(30, 33)),
        ]
    if priv_dim == 12:
        # V11 (2026-05-20): truly-hidden priv only (V9 minus base_height).
        return [
            ("friction",              slice(0, 1)),
            ("base_mass",             slice(1, 2)),
            ("|applied_force|",       slice(2, 5)),
            ("|applied_torque|",      slice(5, 8)),
            ("|com_offset|",          slice(8, 11)),
            ("actuation_noise_sigma", slice(11, 12)),
        ]
    if priv_dim == 13:
        return [
            ("friction",              slice(0, 1)),
            ("base_mass",             slice(1, 2)),
            ("base_height",           slice(2, 3)),
            ("|applied_force|",       slice(3, 6)),
            ("|applied_torque|",      slice(6, 9)),
            ("|com_offset|",          slice(9, 12)),
            ("actuation_noise_sigma", slice(12, 13)),
        ]
    if priv_dim == 33:
        return [
            ("friction",              slice(0, 1)),
            ("base_mass",             slice(1, 2)),
            ("base_height",           slice(2, 3)),
            ("|applied_force|",       slice(3, 6)),
            ("|applied_torque|",      slice(6, 9)),
            ("|tracking_err|",        slice(9, 24)),   # 5-step history
            ("|base_ang_vel|",        slice(24, 27)),
            ("|com_offset|",          slice(27, 30)),
            ("actuation_noise_sigma", slice(30, 31)),
            ("mean_signed_delta_R",   slice(31, 32)),
            ("max_abs_delta_R",       slice(32, 33)),
        ]
    raise ValueError(f"Unsupported priv_dim={priv_dim} — add the layout.")


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"
    layout = get_priv_layout(args.priv_dim, layout=args.priv_layout, task=args.task)

    print(f"[grad_sens] task={args.task}, num_envs={args.num_envs}, "
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
    print(f"[grad_sens] runner loaded.", flush=True)

    # Locate actor module so we can call it with gradients enabled.
    actor = None
    for path in [("alg", "actor_critic", "actor"),
                 ("alg", "actor_critic"),
                 ("alg", "actor"),
                 ("policy",), ("actor",)]:
        m = runner
        try:
            for p in path:
                m = getattr(m, p)
            if hasattr(m, "forward"):
                actor = m
                break
        except AttributeError:
            continue
    if actor is None:
        raise RuntimeError("could not locate actor module on runner")
    print(f"[grad_sens] actor located: {type(actor).__name__}", flush=True)

    # Locate the inner _SplitRMAMLP. rsl_rl's MlpModel wraps it as `mlp`
    # and expects a dict obs (indexed by group name) at forward(). We bypass
    # the wrapper because `actor.mlp` is the raw _SplitRMAMLP that takes a
    # flat tensor (concat priv + grid) and returns the action mean tensor.
    if not hasattr(actor, "mlp"):
        raise RuntimeError(
            f"expected actor.mlp to exist on {type(actor).__name__}")
    inner_mlp = actor.mlp
    print(f"[grad_sens] inner mlp: {type(inner_mlp).__name__}", flush=True)

    obs, _ = env.reset()

    N = args.num_envs
    S = args.rollout_steps
    P = args.priv_dim

    # Sensitivity accumulators
    a_lo, a_hi = args.alpha_min, args.alpha_max
    grad_alpha_sum = torch.zeros(P, device=device)
    grad_alpha_abs_sum = torch.zeros(P, device=device)
    grad_alpha_sq_sum = torch.zeros(P, device=device)
    grad_phi_sum = torch.zeros(P, device=device)
    grad_phi_abs_sum = torch.zeros(P, device=device)
    grad_phi_sq_sum = torch.zeros(P, device=device)
    priv_sum = torch.zeros(P, device=device)
    priv_sq_sum = torch.zeros(P, device=device)
    # Grid sensitivity: full 8192-D gradient. We track:
    #   - per-env L2 norm of grid grad (one scalar per env, accumulate stats)
    #   - per-channel breakdown (2 channels: current frame, previous frame)
    # Standardized comparison vs priv comes from comparing L2 norms.
    grid_dim = None  # set on first step from obs shape
    grad_grid_alpha_l2_sum = 0.0       # sum of per-env ‖grad_grid_α‖₂
    grad_grid_alpha_l2_sq_sum = 0.0
    grad_grid_phi_l2_sum = 0.0
    grad_grid_phi_l2_sq_sum = 0.0
    grad_grid_alpha_ch_l2_sum = torch.zeros(2, device=device)  # per-channel
    grad_grid_phi_ch_l2_sum = torch.zeros(2, device=device)
    grid_occupancy_mean = 0.0  # how much of the grid is occupied (binary)
    n_samples = 0

    for step in range(S):
        obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
        # Detach + enable grad on a copy. policy.forward typically goes
        # through actor; we'll call actor directly to get gradients.
        x = obs_tensor.detach().clone().requires_grad_(True)

        # Call inner _SplitRMAMLP directly with the flat obs tensor.
        # Returns the action mean (N, action_dim) — bypasses the rsl_rl
        # distribution wrapping that wants a dict obs.
        raw_action = inner_mlp(x)
        if isinstance(raw_action, tuple):
            raw_action = raw_action[0]
        raw_a = raw_action[:, 0]  # α channel raw
        raw_p = raw_action[:, 1]  # φ channel raw

        # Compute physical α, φ via tanh squashing.
        alpha_phys = a_lo + (torch.tanh(raw_a) + 1.0) * 0.5 * (a_hi - a_lo)
        phi_phys = PHI_MIN + (torch.tanh(raw_p) + 1.0) * 0.5 * (PHI_MAX - PHI_MIN)

        # Full gradients over both priv and grid. retain_graph=True for α so
        # we can compute φ next.
        grad_full_alpha = torch.autograd.grad(
            alpha_phys.sum(), x, retain_graph=True, create_graph=False,
        )[0].detach()  # (N, total_dim)
        grad_full_phi = torch.autograd.grad(
            phi_phys.sum(), x, retain_graph=False, create_graph=False,
        )[0].detach()  # (N, total_dim)

        # Priv slice
        grad_alpha = grad_full_alpha[:, :P]  # (N, P)
        grad_phi = grad_full_phi[:, :P]      # (N, P)

        # Accumulate priv stats (sum across N envs).
        grad_alpha_sum   += grad_alpha.sum(dim=0)
        grad_alpha_abs_sum += grad_alpha.abs().sum(dim=0)
        grad_alpha_sq_sum  += (grad_alpha ** 2).sum(dim=0)
        grad_phi_sum   += grad_phi.sum(dim=0)
        grad_phi_abs_sum += grad_phi.abs().sum(dim=0)
        grad_phi_sq_sum  += (grad_phi ** 2).sum(dim=0)

        # Grid slice (8192 = 2 * 64 * 64). L2 norm per env → per-env scalar
        # measuring "how sensitive is α/φ to the entire grid for this env."
        grad_grid_alpha = grad_full_alpha[:, P:]  # (N, 8192)
        grad_grid_phi = grad_full_phi[:, P:]
        if grid_dim is None:
            grid_dim = grad_grid_alpha.shape[1]
        # Per-env L2 norm (one scalar per env)
        grid_alpha_l2 = grad_grid_alpha.norm(dim=1)  # (N,)
        grid_phi_l2 = grad_grid_phi.norm(dim=1)
        grad_grid_alpha_l2_sum += grid_alpha_l2.sum().item()
        grad_grid_alpha_l2_sq_sum += (grid_alpha_l2 ** 2).sum().item()
        grad_grid_phi_l2_sum += grid_phi_l2.sum().item()
        grad_grid_phi_l2_sq_sum += (grid_phi_l2 ** 2).sum().item()
        # Per-channel breakdown: grid is (N, 2, 64, 64)
        grad_grid_alpha_2ch = grad_grid_alpha.view(N, 2, 64, 64)
        grad_grid_phi_2ch = grad_grid_phi.view(N, 2, 64, 64)
        # L2 over spatial dims, sum across envs.
        grad_grid_alpha_ch_l2_sum += grad_grid_alpha_2ch.flatten(2).norm(dim=2).sum(dim=0)
        grad_grid_phi_ch_l2_sum += grad_grid_phi_2ch.flatten(2).norm(dim=2).sum(dim=0)
        # Grid occupancy (how many cells are non-zero) — diagnostic context.
        grid_input = x[:, P:].view(N, 2, 64, 64)
        grid_occupancy_mean += (grid_input != 0).float().mean().item()

        # Priv-feature stats (for standardization).
        priv = x[:, :P].detach()
        priv_sum += priv.sum(dim=0)
        priv_sq_sum += (priv ** 2).sum(dim=0)
        n_samples += N

        # Step environment with the actor's action (no grad needed).
        with torch.no_grad():
            action = policy(obs)
        step_out = env.step(action)
        obs = step_out[0]

        if step % 10 == 0 or step == S - 1:
            print(f"[grad_sens] step {step:>3}/{S}  "
                  f"|∂α/∂priv|_mean={grad_alpha_abs_sum.sum().item()/(n_samples*P):.4f}",
                  flush=True)

    # ── Aggregate ──
    priv_mean = (priv_sum / n_samples).cpu().numpy()
    priv_var = (priv_sq_sum / n_samples - (priv_sum / n_samples) ** 2).cpu().numpy()
    priv_std = np.sqrt(np.maximum(priv_var, 0.0))

    grad_alpha_mean = (grad_alpha_sum / n_samples).cpu().numpy()
    grad_alpha_abs_mean = (grad_alpha_abs_sum / n_samples).cpu().numpy()
    grad_alpha_rms = np.sqrt((grad_alpha_sq_sum / n_samples).cpu().numpy())
    grad_phi_mean = (grad_phi_sum / n_samples).cpu().numpy()
    grad_phi_abs_mean = (grad_phi_abs_sum / n_samples).cpu().numpy()
    grad_phi_rms = np.sqrt((grad_phi_sq_sum / n_samples).cpu().numpy())

    # Group by priv feature (sum or norm per group).
    def group_stat(per_dim_arr, sl):
        # For a vector group, take the L2 norm of the per-dim gradients.
        # For a scalar group, this is just the value.
        return float(np.linalg.norm(per_dim_arr[sl]))

    def group_signed_mean(per_dim_arr, sl):
        # Average of signed values across the slice (preserves direction).
        return float(per_dim_arr[sl].mean())

    def group_priv_std(priv_std_arr, priv_mean_arr, sl):
        # For vector groups, use std of the L2 norm across envs. Approximate
        # by taking the average per-dim std (good enough for "scale").
        return float(priv_std_arr[sl].mean())

    results_alpha = []
    results_phi = []
    for name, sl in layout:
        # |grad| magnitude (group L2 norm)
        ga_mag = group_stat(grad_alpha_abs_mean, sl)
        gp_mag = group_stat(grad_phi_abs_mean, sl)
        # Signed mean (direction)
        ga_signed = group_signed_mean(grad_alpha_mean, sl)
        gp_signed = group_signed_mean(grad_phi_mean, sl)
        # RMS (signal+noise magnitude)
        ga_rms = group_stat(grad_alpha_rms, sl)
        gp_rms = group_stat(grad_phi_rms, sl)
        # Priv-feature scale (for standardization)
        feat_scale = group_priv_std(priv_std, priv_mean, sl)
        # Standardized sensitivity: |grad| × scale = α change per 1σ of feature
        ga_std_sens = ga_mag * feat_scale
        gp_std_sens = gp_mag * feat_scale
        results_alpha.append({
            "feature": name,
            "grad_abs_mean": ga_mag,
            "grad_signed_mean": ga_signed,
            "grad_rms": ga_rms,
            "feature_scale": feat_scale,
            "standardized_sensitivity": ga_std_sens,
        })
        results_phi.append({
            "feature": name,
            "grad_abs_mean": gp_mag,
            "grad_signed_mean": gp_signed,
            "grad_rms": gp_rms,
            "feature_scale": feat_scale,
            "standardized_sensitivity": gp_std_sens,
        })

    # Grid sensitivity aggregates
    n_env_step_samples = float(n_samples)
    grid_alpha_l2_mean = grad_grid_alpha_l2_sum / n_env_step_samples
    grid_alpha_l2_std = (grad_grid_alpha_l2_sq_sum / n_env_step_samples
                         - grid_alpha_l2_mean ** 2) ** 0.5
    grid_phi_l2_mean = grad_grid_phi_l2_sum / n_env_step_samples
    grid_phi_l2_std = (grad_grid_phi_l2_sq_sum / n_env_step_samples
                       - grid_phi_l2_mean ** 2) ** 0.5
    grid_alpha_ch_l2_mean = (grad_grid_alpha_ch_l2_sum / n_env_step_samples).cpu().numpy()
    grid_phi_ch_l2_mean = (grad_grid_phi_ch_l2_sum / n_env_step_samples).cpu().numpy()
    grid_occupancy_mean /= S  # average across steps

    # Priv totals for comparison
    priv_alpha_l2_total = float(np.linalg.norm(grad_alpha_abs_mean))
    priv_phi_l2_total = float(np.linalg.norm(grad_phi_abs_mean))

    grid_summary = {
        "grid_dim": int(grid_dim) if grid_dim is not None else 0,
        "grid_occupancy_mean_fraction": float(grid_occupancy_mean),
        "alpha_grid_l2_per_env_mean": float(grid_alpha_l2_mean),
        "alpha_grid_l2_per_env_std": float(grid_alpha_l2_std),
        "phi_grid_l2_per_env_mean": float(grid_phi_l2_mean),
        "phi_grid_l2_per_env_std": float(grid_phi_l2_std),
        "alpha_grid_l2_per_channel": [float(v) for v in grid_alpha_ch_l2_mean],
        "phi_grid_l2_per_channel": [float(v) for v in grid_phi_ch_l2_mean],
        "alpha_priv_l2_total_abs": priv_alpha_l2_total,
        "phi_priv_l2_total_abs": priv_phi_l2_total,
        "alpha_grid_to_priv_ratio": float(grid_alpha_l2_mean) / max(priv_alpha_l2_total, 1e-9),
        "phi_grid_to_priv_ratio": float(grid_phi_l2_mean) / max(priv_phi_l2_total, 1e-9),
    }

    output = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "n_envs": int(N),
        "n_rollout_steps": int(S),
        "priv_dim": int(P),
        "alpha_head": results_alpha,
        "phi_head": results_phi,
        "grid_sensitivity": grid_summary,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    # Console summary — sorted by standardized sensitivity
    def print_table(rows, head_label):
        print("", flush=True)
        print(f"  ── {head_label} head — sensitivity (sorted by standardized) ──",
              flush=True)
        print(f"  {'feature':<24} {'|grad|':>10} {'signed':>10} "
              f"{'feat σ':>10} {'std·|grad|':>12}", flush=True)
        rows_sorted = sorted(rows, key=lambda r: -r["standardized_sensitivity"])
        for r in rows_sorted:
            bar = "█" * int(min(30, r["standardized_sensitivity"] * 60))
            print(f"  {r['feature']:<24} {r['grad_abs_mean']:>10.4f} "
                  f"{r['grad_signed_mean']:>+10.4f} {r['feature_scale']:>10.4f} "
                  f"{r['standardized_sensitivity']:>12.4f}  {bar}", flush=True)

    print("=" * 78, flush=True)
    print(f"Gradient sensitivity — {args.task}", flush=True)
    print("=" * 78, flush=True)
    print_table(results_alpha, "α")
    print_table(results_phi, "φ")

    print("", flush=True)
    print(f"  ── GRID sensitivity ──", flush=True)
    print(f"  grid input dim: {grid_summary['grid_dim']}  "
          f"avg occupancy: {grid_summary['grid_occupancy_mean_fraction']:.4f}",
          flush=True)
    print(f"  α  ‖∂α/∂grid‖₂ per env: mean={grid_summary['alpha_grid_l2_per_env_mean']:.4f}  "
          f"std={grid_summary['alpha_grid_l2_per_env_std']:.4f}", flush=True)
    print(f"  φ  ‖∂φ/∂grid‖₂ per env: mean={grid_summary['phi_grid_l2_per_env_mean']:.4f}  "
          f"std={grid_summary['phi_grid_l2_per_env_std']:.4f}", flush=True)
    print(f"  per-channel ‖grid_grad‖ (ch0=current frame, ch1=previous):",
          flush=True)
    print(f"     α: ch0={grid_summary['alpha_grid_l2_per_channel'][0]:.4f}  "
          f"ch1={grid_summary['alpha_grid_l2_per_channel'][1]:.4f}", flush=True)
    print(f"     φ: ch0={grid_summary['phi_grid_l2_per_channel'][0]:.4f}  "
          f"ch1={grid_summary['phi_grid_l2_per_channel'][1]:.4f}", flush=True)
    print("", flush=True)
    print(f"  ── PRIV vs GRID head sensitivity ──", flush=True)
    print(f"  ‖priv grad‖₂ total (α):   {grid_summary['alpha_priv_l2_total_abs']:.4f}",
          flush=True)
    print(f"  ‖grid grad‖₂ per env (α): {grid_summary['alpha_grid_l2_per_env_mean']:.4f}",
          flush=True)
    print(f"     ratio grid/priv (α):   {grid_summary['alpha_grid_to_priv_ratio']:.3f}",
          flush=True)
    print(f"  ‖priv grad‖₂ total (φ):   {grid_summary['phi_priv_l2_total_abs']:.4f}",
          flush=True)
    print(f"  ‖grid grad‖₂ per env (φ): {grid_summary['phi_grid_l2_per_env_mean']:.4f}",
          flush=True)
    print(f"     ratio grid/priv (φ):   {grid_summary['phi_grid_to_priv_ratio']:.3f}",
          flush=True)
    print(f"  Read: ratio > 1 → grid carries more weight than priv;",
          flush=True)
    print(f"        ratio < 0.5 → priv dominates;",
          flush=True)
    print(f"        ratio ≈ 0  → policy nearly ignores grid (problem).",
          flush=True)
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
        print(f"[grad_sens] FATAL: {e}", flush=True)
        traceback.print_exc()
    os._exit(rc if rc is not None else 0)
