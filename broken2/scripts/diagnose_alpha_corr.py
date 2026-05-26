#!/usr/bin/env python3
"""Within-distribution state-conditioning diagnostic.

v3.0d showed only Δalpha_mean=0.21 across tasks — below the 0.3 PASS
threshold. BUT HeavyCOM is OOD (training DR doesn't cover those COM
offsets), so cross-task variance is a confounded test. The right test:
does α correlate with privileged features WITHIN the training
distribution? If yes, state-conditioning emerged but doesn't
extrapolate to OOD. If no, the policy ignores DR features entirely
and v3.0d's skip-connection didn't do its job.

For each of N envs (each with different DR sample), capture:
  • mean α output across a rollout
  • per-env privileged features (friction, mass, COM offset, h, ...)

Then compute Pearson correlation between α and each feature.

  |corr| > 0.2  →  meaningful state-conditioning
  |corr| < 0.1  →  policy ignores this feature
  in between    →  weak signal

Usage on lab box:
  cd ~/Desktop/safety-go2/IsaacLab && \\
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \\
    --task Isaac-CBF-Go2-v0 \\
    --checkpoint logs/rsl_rl/cbf_go2_teacher/2026-05-10_23-50-39/model_2999.pt \\
    --num_envs 256 --rollout_steps 100 \\
    --output diagnose_alpha_corr_v3_0d.json \\
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
parser.add_argument("--priv_dim", type=int, default=19,
                    help="Total priv obs dims. 19 for v3.x (15 DR + 4 cbf_state). "
                         "20 for Layer 2 (15 DR + 1 actuation_noise_sigma + 4 cbf_state).")
parser.add_argument("--priv_layout", type=str, default="auto",
                    choices=["auto", "v8", "v13"],
                    help="V13 uses HIDDEN-first ordering (14 hidden + 19 observable). "
                         "Auto-detects from task name (TwoStream → V13).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[diagnose_alpha_corr] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[diagnose_alpha_corr] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


# Same encoding as cbf_go2_env._cbf_filter does for the 0th action dim.
ALPHA_MIN = 0.1
ALPHA_MAX = 5.0

# Layout of priv obs dims (matches TeacherPrivCfg order).
# v3.x:    15 DR + 4 cbf_state = 19
# Layer 2 (v9-v17): 15 DR + 1 actuation_noise_sigma + 4 cbf_state = 20
# Layer 2 (v18+):   15 DR + 1 actuation_noise_sigma          = 16   (cbf_state removed from obs)
def get_priv_layout(priv_dim: int, layout: str = "auto", task: str = "") -> dict:
    # V13 (2026-05-20): two-stream HIDDEN-first ordering, 33-D total.
    if layout == "v13" or (layout == "auto" and "TwoStream" in task):
        return {
            "friction":              slice(0, 1),
            "base_mass":             slice(1, 2),
            "applied_force":         slice(2, 5),
            "applied_torque":        slice(5, 8),
            "com_offset":            slice(8, 11),
            "actuation_noise_sigma": slice(11, 12),
            "mean_signed_delta_R":   slice(12, 13),
            "max_abs_delta_R":       slice(13, 14),
            "base_height":           slice(14, 15),
            "tracking_err":          slice(15, 30),
            "base_ang_vel":          slice(30, 33),
        }
    if priv_dim == 12:
        # V11 (2026-05-20): truly-hidden priv only (V9 minus base_height).
        return {
            "friction":              slice(0, 1),
            "base_mass":             slice(1, 2),
            "applied_force":         slice(2, 5),
            "applied_torque":        slice(5, 8),
            "com_offset":            slice(8, 11),
            "actuation_noise_sigma": slice(11, 12),
        }
    if priv_dim == 13:
        # V9 (2026-05-19): canonical RMA priv only — 13D.
        return {
            "friction":              slice(0, 1),
            "base_mass":             slice(1, 2),
            "base_height":           slice(2, 3),
            "applied_force":         slice(3, 6),
            "applied_torque":        slice(6, 9),
            "com_offset":            slice(9, 12),
            "actuation_noise_sigma": slice(12, 13),
        }
    if priv_dim == 16:
        # v18+: cbf_state removed from obs. No cbf_state slice — α/φ
        # correlations with h/slack/Lgh* are skipped.
        return {
            "friction":              slice(0, 1),
            "base_mass":             slice(1, 2),
            "base_height":           slice(2, 3),
            "applied_force":         slice(3, 6),
            "applied_torque":        slice(6, 9),
            "tracking_err":          slice(9, 12),
            "com_offset":            slice(12, 15),
            "actuation_noise_sigma": slice(15, 16),
        }
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
    elif priv_dim == 31:
        # Wk3 within-episode (2026-05-16): additive symptom-based priv.
        # tracking_err extended to history_length=5 → 15D; base_ang_vel
        # added; applied_force / applied_torque kept (God-mode for teacher).
        return {
            "friction":              slice(0, 1),
            "base_mass":             slice(1, 2),
            "base_height":           slice(2, 3),
            "applied_force":         slice(3, 6),
            "applied_torque":        slice(6, 9),
            "tracking_err":          slice(9, 24),   # 5-step history flattened
            "base_ang_vel":          slice(24, 27),
            "com_offset":            slice(27, 30),
            "actuation_noise_sigma": slice(30, 31),
        }
    elif priv_dim == 33:
        # Wk3 theory bundle (2026-05-17): + mean_signed_delta_R + max_abs_delta_R.
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


PRIV_FEATURE_LAYOUT = get_priv_layout(19)  # back-compat default; overridden in main()


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation between 1-D arrays x and y."""
    x = x.astype(float)
    y = y.astype(float)
    mx, my = x.mean(), y.mean()
    sx, sy = x.std() + 1e-12, y.std() + 1e-12
    return float(((x - mx) * (y - my)).mean() / (sx * sy))


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    global PRIV_FEATURE_LAYOUT
    PRIV_FEATURE_LAYOUT = get_priv_layout(args.priv_dim, layout=args.priv_layout, task=args.task)

    print(f"[diagnose_alpha_corr] task={args.task}, num_envs={args.num_envs}, "
          f"rollout_steps={args.rollout_steps}, priv_dim={args.priv_dim}", flush=True)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped

    # Bug fix (2026-05-16): read α range from env, not hardcoded constants.
    # LAYER3_PUSH_A_C_ACLAMP clamps to (0.5, 3.0); decoding with the legacy
    # (ALPHA_MIN, ALPHA_MAX) gave alpha_phys=4.17 when the env actually
    # outputs 2.54.
    if hasattr(inner, "_alpha_param_range"):
        a_lo, a_hi = inner._alpha_param_range
        a_lo, a_hi = float(a_lo), float(a_hi)
    else:
        a_lo, a_hi = ALPHA_MIN, ALPHA_MAX
    print(f"[diagnose_alpha_corr] env α range: [{a_lo:.4f}, {a_hi:.4f}]", flush=True)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    print(f"[diagnose_alpha_corr] runner loaded.", flush=True)

    obs, _ = env.reset()
    print(f"[diagnose_alpha_corr] resetting env...", flush=True)

    # Storage: per-env per-step
    N = args.num_envs
    S = args.rollout_steps
    alpha_history = np.zeros((S, N), dtype=np.float32)
    priv_history = np.zeros((S, N, args.priv_dim), dtype=np.float32)
    # Wk3 V6 (2026-05-19): also capture h(x) per step so we can compute
    # within-episode Pearson(α_t, h_t). α theoretically responds to
    # obstacle proximity (smaller α = more conservative = constraint
    # binds earlier). Strong negative Pearson(α_t, h_t) = α ramps down
    # as robot approaches obstacles.
    h_history = np.zeros((S, N), dtype=np.float32)

    for step in range(S):
        with torch.no_grad():
            raw_action = policy(obs)  # (N, 5) raw network outputs
            # alpha = a_lo + (tanh(raw) + 1) * 0.5 * (a_hi - a_lo)
            alpha_phys = (
                a_lo
                + (torch.tanh(raw_action[:, 0]) + 1.0) * 0.5 * (a_hi - a_lo)
            )
            alpha_history[step] = alpha_phys.cpu().numpy()

            # Extract priv features from obs (first priv_dim dims of "policy" group)
            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            priv_history[step] = obs_tensor[:, :args.priv_dim].cpu().numpy()

        step_out = env.step(raw_action)
        obs = step_out[0]

        # Wk3 V6: capture h(x) for this step (set in env._cbf_filter as
        # part of env.step). Available iff env stores it.
        if hasattr(inner, "last_h_for_obs"):
            h_history[step] = inner.last_h_for_obs.detach().cpu().numpy()

        if step % 20 == 0 or step == S - 1:
            print(f"[diagnose_alpha_corr] step {step:>3}/{S}  "
                  f"α mean={alpha_phys.mean().item():.3f}  "
                  f"α std={alpha_phys.std().item():.3f}", flush=True)

    # ── Aggregate per env (mean across time) ──
    alpha_per_env = alpha_history.mean(axis=0)        # (N,)
    priv_per_env = priv_history.mean(axis=0)          # (N, 19)

    # Per-env alpha STD (within-env across time) — for context
    alpha_within_env_std = alpha_history.std(axis=0).mean()  # avg over envs

    # Compute correlations per scalar feature
    correlations = {}
    detail_scalars = {}

    # Scalar features (1-D). Includes actuation_noise_sigma if Layer 2 (priv_dim=20).
    scalar_features = ["friction", "base_mass", "base_height"]
    if "actuation_noise_sigma" in PRIV_FEATURE_LAYOUT:
        scalar_features.append("actuation_noise_sigma")
    for name in scalar_features:
        if name not in PRIV_FEATURE_LAYOUT:
            continue
        feat = priv_per_env[:, PRIV_FEATURE_LAYOUT[name]].flatten()
        r = pearson_corr(feat, alpha_per_env)
        correlations[name] = r
        detail_scalars[name] = {
            "mean": float(feat.mean()),
            "std": float(feat.std()),
            "min": float(feat.min()),
            "max": float(feat.max()),
        }

    # Vector features: correlate alpha vs the NORM of the vector.
    # V9 priv may not have tracking_err — skip missing keys.
    for name in ["applied_force", "applied_torque", "tracking_err", "com_offset"]:
        if name not in PRIV_FEATURE_LAYOUT:
            continue
        feat = priv_per_env[:, PRIV_FEATURE_LAYOUT[name]]   # (N, 3)
        feat_norm = np.linalg.norm(feat, axis=-1)            # (N,)
        r = pearson_corr(feat_norm, alpha_per_env)
        correlations[f"|{name}|"] = r
        detail_scalars[f"|{name}|"] = {
            "mean": float(feat_norm.mean()),
            "std": float(feat_norm.std()),
            "min": float(feat_norm.min()),
            "max": float(feat_norm.max()),
        }
        # Also per-axis for COM (most likely state-conditioner)
        if name == "com_offset":
            for i, axis in enumerate(["com_x", "com_y", "com_z"]):
                axis_vals = feat[:, i]
                r_axis = pearson_corr(axis_vals, alpha_per_env)
                correlations[axis] = r_axis

    # cbf_state components (per-dim) — only present for pre-v18 priv layouts.
    if "cbf_state" in PRIV_FEATURE_LAYOUT:
        cbf_state = priv_per_env[:, PRIV_FEATURE_LAYOUT["cbf_state"]]   # (N, 4)
        for i, name in enumerate(["h", "Lgh_dot_udes", "Lgh_norm_sq", "slack"]):
            r = pearson_corr(cbf_state[:, i], alpha_per_env)
            correlations[name] = r

    # Wk3 V6 (2026-05-19): within-episode correlations. The headline
    # question: is α's within-episode variation tracking instantaneous
    # disturbance signals (tracking_err spikes, push symptoms via
    # base_ang_vel, σ_act regime under V5+ within-ep DR, obstacle
    # proximity via h(x)) or just policy-output noise?
    #
    # Theory:
    #   Pearson(α_t, |tracking_err_t|) should be NEGATIVE — bad tracking
    #     → smaller α (more conservative).
    #   Pearson(α_t, |base_ang_vel_t|) should be NEGATIVE — push event →
    #     IMU spikes → smaller α.
    #   Pearson(α_t, σ_act_t) should be NEGATIVE — noisier regime →
    #     smaller α (only meaningful under within-ep σ_act DR, V5+).
    #   Pearson(α_t, h_t) sign depends — could be positive (low h → small α
    #     so constraint binds harder) or zero (α scales (h-c), not driven
    #     by h alone).
    def per_step_norm(layout_key: str) -> np.ndarray:
        """For a vector priv feature, return per-step norm shape (S, N)."""
        sl = PRIV_FEATURE_LAYOUT.get(layout_key)
        if sl is None:
            return None
        vec = priv_history[:, :, sl]  # (S, N, D)
        return np.linalg.norm(vec, axis=-1)  # (S, N)

    def per_step_scalar(layout_key: str) -> np.ndarray:
        sl = PRIV_FEATURE_LAYOUT.get(layout_key)
        if sl is None:
            return None
        return priv_history[:, :, sl].squeeze(-1)  # (S, N)

    within_signals = {}
    te = per_step_norm("tracking_err")
    if te is not None:
        within_signals["|tracking_err|"] = te
    bav = per_step_norm("base_ang_vel")
    if bav is not None:
        within_signals["|base_ang_vel|"] = bav
    sigma = per_step_scalar("actuation_noise_sigma")
    if sigma is not None:
        within_signals["actuation_noise_sigma"] = sigma
    if hasattr(inner, "last_h_for_obs"):
        within_signals["h"] = h_history

    within_pearsons = {}
    for sig_name, sig_steps in within_signals.items():
        per_env = np.zeros(N, dtype=np.float32)
        for e in range(N):
            a_e = alpha_history[:, e]
            s_e = sig_steps[:, e]
            if a_e.std() < 1e-6 or s_e.std() < 1e-6:
                per_env[e] = np.nan
                continue
            per_env[e] = pearson_corr(a_e, s_e)
        valid = ~np.isnan(per_env)
        within_pearsons[sig_name] = {
            "mean": round(float(np.nanmean(per_env)) if valid.any() else float("nan"), 4),
            "std":  round(float(np.nanstd(per_env))  if valid.any() else float("nan"), 4),
            "n_valid_envs": int(valid.sum()),
        }

    stats = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "n_envs": int(N),
        "n_rollout_steps": int(S),
        "alpha_population_mean": float(alpha_per_env.mean()),
        "alpha_population_std": float(alpha_per_env.std()),
        "alpha_within_env_std_mean": float(alpha_within_env_std),
        "correlations_with_alpha": {k: round(v, 4) for k, v in correlations.items()},
        # Wk3 V6: within-episode Pearson averaged across envs.
        "within_episode_pearson_with_alpha": within_pearsons,
        "feature_distributions": {
            k: {kk: round(vv, 4) for kk, vv in v.items()}
            for k, v in detail_scalars.items()
        },
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Console summary
    print("", flush=True)
    print("=" * 70, flush=True)
    print(f"α correlation diagnostic — {args.task}", flush=True)
    print("=" * 70, flush=True)
    print(f"  α population: mean={stats['alpha_population_mean']:.3f}  "
          f"between-env std={stats['alpha_population_std']:.3f}  "
          f"within-env std={stats['alpha_within_env_std_mean']:.3f}", flush=True)
    print("", flush=True)
    print(f"  Pearson(α, feature):", flush=True)
    # Sort by absolute correlation magnitude
    sorted_corr = sorted(correlations.items(), key=lambda x: -abs(x[1]))
    for name, r in sorted_corr:
        bar_len = int(abs(r) * 40)
        bar = "█" * bar_len
        sign = "+" if r >= 0 else "−"
        marker = ""
        if abs(r) > 0.20:
            marker = "  ←← STRONG"
        elif abs(r) > 0.10:
            marker = "  ← weak"
        print(f"    {name:>22} {sign}{abs(r):.3f}  {bar}{marker}", flush=True)

    # Wk3 V6 within-episode summary.
    if within_pearsons:
        print("", flush=True)
        print(f"  Within-episode Pearson(α_t, signal_t) (per env, averaged):", flush=True)
        for sig, vals in within_pearsons.items():
            m, s = vals["mean"], vals["std"]
            marker = ""
            # For tracking_err / base_ang_vel / σ_act, theory says NEGATIVE
            # (more disturbance → smaller α = more conservative).
            if sig in ("|tracking_err|", "|base_ang_vel|", "actuation_noise_sigma"):
                if m < -0.30:   marker = "  ←← STRONG (α adapts down to disturbance)"
                elif m < -0.15: marker = "  ← weak negative (right direction)"
                elif m > 0.15:  marker = "  ← WRONG SIGN"
                else:           marker = "  ← ~0 (no within-ep tracking)"
            elif sig == "h":
                if abs(m) < 0.10: marker = "  ← ~0"
                elif abs(m) > 0.30: marker = "  ←← strong"
            print(f"    α_t vs {sig:<22}  mean={m:+.4f}  std={s:.4f}{marker}", flush=True)

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
        print(f"[diagnose_alpha_corr] FATAL: {e}", flush=True)
        traceback.print_exc()
    # Skip simulation_app.close() — it hangs unreliably (v17 temporal_grid
    # hung 10+ min waiting on close). OS reclaims handles when the process
    # dies via os._exit, so this is safe.
    os._exit(rc if rc is not None else 0)
