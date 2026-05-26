#!/usr/bin/env python3
"""Within-distribution state-conditioning diagnostic for φ (Layer 2).

Mirror of diagnose_alpha_corr.py but for the φ parameter (action index 1).
The headline Layer 2 test: does Pearson(φ, actuation_noise_sigma) > 0.20?

φ multiplies ‖L_g h‖² in the CBF QP rhs (Kolathaya ISSf input-disturbance
margin term). It has a clean physical match to actuation noise — both
are about L_g h estimate uncertainty. The hypothesis Layer 2 tests is
that releasing φ alongside this DR axis will give the policy a parameter
whose physical lever ties to a DR axis the policy observes.

Note on per-episode φ lock: when `USE_PER_EPISODE_PHI=True` (current default
in v2.13+), the env captures the policy's φ output at step 0 of each
episode and replays it for the rest of the episode. The CBF actually
uses the locked value, not the policy's per-step output. We compute φ
from raw_action[:, 1] per step here — for envs that don't reset within
the 100-step rollout, this equals the locked value at every step (because
the policy output may vary per-step but it's overwritten by the lock
inside the env). For envs that DO reset, the time-average will mix
across two episodes. Most envs won't reset given typical 800-step
episodes.

Decision criterion (Layer 2 PASS):
  |Pearson(φ, actuation_noise_sigma)| > 0.20

For each of N envs (each with different DR sample), capture:
  • mean φ output across a rollout
  • per-env privileged features (friction, mass, actuation_noise_sigma, ...)

Then compute Pearson correlation between φ and each feature.

Usage on lab box:
  cd ~/Desktop/safety-go2/IsaacLab && \\
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \\
    --task Isaac-CBF-Go2-RMA-Layer2-v0 \\
    --checkpoint <layer2 checkpoint> \\
    --num_envs 256 --rollout_steps 100 \\
    --priv_dim 20 \\
    --output diagnose_phi_corr_layer2.json \\
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
parser.add_argument("--priv_dim", type=int, default=20,
                    help="Total priv obs dims. Default 20 for Layer 2 "
                         "(15 DR + 1 actuation_noise_sigma + 4 cbf_state). "
                         "Use 19 for v3.x checkpoints with no noise obs.")
parser.add_argument("--priv_layout", type=str, default="auto",
                    choices=["auto", "v8", "v13"],
                    help="V8 (default for priv_dim=33) uses original ordering; "
                         "V13 uses HIDDEN-first ordering (14 hidden + 19 observable).")
parser.add_argument("--use_locked", action="store_true",
                    help="Read env.cbf_phi_locked (the actual CBF-applied "
                         "value) instead of computing from raw_action[:, 1]. "
                         "Matches what the CBF actually uses when "
                         "USE_PER_EPISODE_PHI is on.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[diagnose_phi_corr] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[diagnose_phi_corr] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


# Same encoding as cbf_go2_env._cbf_filter does for action dim 1.
# phi = (tanh(raw[:,1]) + 1) * 0.5 * 5.0  ⇒  range [0.0, 5.0]
PHI_MIN = 0.0
PHI_MAX = 5.0


def get_priv_layout(priv_dim: int, layout: str = "auto", task: str = "") -> dict:
    # V13 (2026-05-20): two-stream HIDDEN-first ordering, 33-D total.
    # First 14 dims = truly hidden env; last 19 = observable proprio.
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
            "tracking_err":          slice(15, 30),  # 5-step history flattened
            "base_ang_vel":          slice(30, 33),
        }
    # V11 (2026-05-20): truly-hidden priv only — 12D (V9 minus base_height).
    if priv_dim == 12:
        return {
            "friction":              slice(0, 1),
            "base_mass":             slice(1, 2),
            "applied_force":         slice(2, 5),
            "applied_torque":        slice(5, 8),
            "com_offset":            slice(8, 11),
            "actuation_noise_sigma": slice(11, 12),
        }
    # V9 (2026-05-19): canonical RMA priv obs only — 13D.
    # friction + mass + base_height + applied_force(3) + applied_torque(3)
    # + com_offset(3) + actuation_noise_sigma.
    # NOTE: no tracking_err, no base_ang_vel, no delta_R features. The
    # diagnostic loops below check for key presence before computing
    # correlations.
    if priv_dim == 13:
        return {
            "friction":              slice(0, 1),
            "base_mass":             slice(1, 2),
            "base_height":           slice(2, 3),
            "applied_force":         slice(3, 6),
            "applied_torque":        slice(6, 9),
            "com_offset":            slice(9, 12),
            "actuation_noise_sigma": slice(12, 13),
        }
    # v18+: priv_dim=16 (cbf_state removed from obs)
    if priv_dim == 16:
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
        # Wk3 theory bundle (2026-05-17): added mean_signed_delta_R + max_abs_delta_R.
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

    layout = get_priv_layout(args.priv_dim, layout=args.priv_layout, task=args.task)

    print(f"[diagnose_phi_corr] task={args.task}, num_envs={args.num_envs}, "
          f"rollout_steps={args.rollout_steps}, priv_dim={args.priv_dim}, "
          f"use_locked={args.use_locked}", flush=True)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    print(f"[diagnose_phi_corr] runner loaded.", flush=True)

    obs, _ = env.reset()

    N = args.num_envs
    S = args.rollout_steps
    phi_history = np.zeros((S, N), dtype=np.float32)
    priv_history = np.zeros((S, N, args.priv_dim), dtype=np.float32)
    # Wk3 V4 (2026-05-18): also capture h and ||L_g h||² each step so we
    # can compute within-episode Pearson(φ_t, h_t). With windowed φ lock
    # (V4+), φ varies within episode; the question is whether that variation
    # tracks obstacle proximity (h_t shrinks → φ_t rises = TISSf-style
    # adaptation) or is just noise.
    h_history = np.zeros((S, N), dtype=np.float32)
    lgh_norm_sq_history = np.zeros((S, N), dtype=np.float32)

    use_locked = args.use_locked
    if use_locked and not hasattr(inner, "cbf_phi_locked"):
        print(f"[diagnose_phi_corr] WARN: --use_locked set but env has no "
              f"cbf_phi_locked attribute. Falling back to per-step computation.",
              flush=True)
        use_locked = False

    for step in range(S):
        with torch.no_grad():
            raw_action = policy(obs)  # (N, 5)
            if use_locked:
                # cbf_phi_locked is the per-episode locked value the CBF
                # actually uses. Constant within an episode after step 0.
                phi_phys = inner.cbf_phi_locked.detach()
            else:
                # Per-step computed φ. With lock on, only step-0 equals the
                # CBF-applied value; subsequent steps' values are ignored
                # by the env (replaced with locked value).
                phi_phys = (
                    PHI_MIN
                    + (torch.tanh(raw_action[:, 1]) + 1.0) * 0.5 * (PHI_MAX - PHI_MIN)
                )
            phi_history[step] = phi_phys.cpu().numpy()

            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            priv_history[step] = obs_tensor[:, :args.priv_dim].cpu().numpy()

        step_out = env.step(raw_action)
        obs = step_out[0]

        # Wk3 V4: capture h and ||L_g h||² this step (set inside env._cbf_filter
        # which ran as part of env.step). Available iff env stores them.
        if hasattr(inner, "last_h_for_obs"):
            h_history[step] = inner.last_h_for_obs.detach().cpu().numpy()
        if hasattr(inner, "last_lgh_for_obs"):
            lgh = inner.last_lgh_for_obs.detach()                   # (N, 2)
            lgh_norm_sq_history[step] = (lgh ** 2).sum(dim=-1).cpu().numpy()

        if step % 20 == 0 or step == S - 1:
            print(f"[diagnose_phi_corr] step {step:>3}/{S}  "
                  f"φ mean={phi_phys.mean().item():.3f}  "
                  f"φ std={phi_phys.std().item():.3f}", flush=True)

    # ── Aggregate per env (mean across time) ──
    phi_per_env = phi_history.mean(axis=0)            # (N,)
    priv_per_env = priv_history.mean(axis=0)          # (N, priv_dim)

    # Per-env phi std across time (zero if locked & env didn't reset)
    phi_within_env_std = phi_history.std(axis=0).mean()

    correlations = {}
    detail_scalars = {}

    # Scalar features
    scalar_features = ["friction", "base_mass", "base_height"]
    if "actuation_noise_sigma" in layout:
        scalar_features.append("actuation_noise_sigma")
    for name in scalar_features:
        if name not in layout:
            continue
        feat = priv_per_env[:, layout[name]].flatten()
        r = pearson_corr(feat, phi_per_env)
        correlations[name] = r
        detail_scalars[name] = {
            "mean": float(feat.mean()),
            "std": float(feat.std()),
            "min": float(feat.min()),
            "max": float(feat.max()),
        }

    # Vector features → use norm. V9 priv may not have tracking_err.
    for name in ["applied_force", "applied_torque", "tracking_err", "com_offset"]:
        if name not in layout:
            continue
        feat = priv_per_env[:, layout[name]]
        feat_norm = np.linalg.norm(feat, axis=-1)
        r = pearson_corr(feat_norm, phi_per_env)
        correlations[f"|{name}|"] = r
        detail_scalars[f"|{name}|"] = {
            "mean": float(feat_norm.mean()),
            "std": float(feat_norm.std()),
            "min": float(feat_norm.min()),
            "max": float(feat_norm.max()),
        }
        if name == "com_offset":
            for i, axis in enumerate(["com_x", "com_y", "com_z"]):
                axis_vals = feat[:, i]
                r_axis = pearson_corr(axis_vals, phi_per_env)
                correlations[axis] = r_axis

    # CBF state components (time-averaged) — only present for pre-v18 priv layouts.
    if "cbf_state" in layout:
        cbf_state = priv_per_env[:, layout["cbf_state"]]   # (N, 4)
        for i, name in enumerate(["h", "Lgh_dot_udes", "Lgh_norm_sq", "slack"]):
            r = pearson_corr(cbf_state[:, i], phi_per_env)
            correlations[name] = r

    # Wk3 V4 (2026-05-18): within-episode correlations between φ_t and
    # obstacle-proximity features. The key question: when φ varies within
    # episode (windowed lock or per-step), is that variation tracking
    # the right signal? Theory says φ should rise as h shrinks (obstacle
    # closer → more defensive margin) and as ||L_g h||² grows. Strong
    # negative Pearson(φ, h) = TISSf-shape adaptation, learned.
    within_corr_phi_h = np.zeros(N, dtype=np.float32)
    within_corr_phi_lgh = np.zeros(N, dtype=np.float32)
    for e in range(N):
        phi_e = phi_history[:, e]
        h_e   = h_history[:, e]
        lgh_e = lgh_norm_sq_history[:, e]
        # Skip envs where phi has zero variance (per-episode lock, no reset
        # during rollout). Pearson is undefined there.
        if phi_e.std() < 1e-6:
            within_corr_phi_h[e] = np.nan
            within_corr_phi_lgh[e] = np.nan
            continue
        within_corr_phi_h[e]   = pearson_corr(phi_e, h_e)
        within_corr_phi_lgh[e] = pearson_corr(phi_e, lgh_e)

    valid = ~np.isnan(within_corr_phi_h)
    n_valid = int(valid.sum())
    within_pearson_phi_h_mean   = float(np.nanmean(within_corr_phi_h)) if n_valid > 0 else float("nan")
    within_pearson_phi_h_std    = float(np.nanstd(within_corr_phi_h))  if n_valid > 0 else float("nan")
    within_pearson_phi_lgh_mean = float(np.nanmean(within_corr_phi_lgh)) if n_valid > 0 else float("nan")
    within_pearson_phi_lgh_std  = float(np.nanstd(within_corr_phi_lgh))  if n_valid > 0 else float("nan")

    stats = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "n_envs": int(N),
        "n_rollout_steps": int(S),
        "priv_dim": int(args.priv_dim),
        "use_locked": bool(use_locked),
        "phi_population_mean": float(phi_per_env.mean()),
        "phi_population_std": float(phi_per_env.std()),
        "phi_within_env_std_mean": float(phi_within_env_std),
        "correlations_with_phi": {k: round(v, 4) for k, v in correlations.items()},
        # Wk3 V4: within-episode Pearson averaged across envs.
        "within_episode_pearson": {
            "phi_vs_h":   {"mean": round(within_pearson_phi_h_mean, 4),
                           "std":  round(within_pearson_phi_h_std, 4)},
            "phi_vs_Lgh_norm_sq": {"mean": round(within_pearson_phi_lgh_mean, 4),
                                   "std":  round(within_pearson_phi_lgh_std, 4)},
            "n_valid_envs": n_valid,
            "n_envs_with_constant_phi": int(N - n_valid),
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

    # ── Console summary ──
    print("", flush=True)
    print("=" * 70, flush=True)
    print(f"φ correlation diagnostic — {args.task}", flush=True)
    print("=" * 70, flush=True)
    print(f"  φ population: mean={stats['phi_population_mean']:.3f}  "
          f"between-env std={stats['phi_population_std']:.3f}  "
          f"within-env std={stats['phi_within_env_std_mean']:.3f}", flush=True)
    print(f"  (within-env std ≈ 0 expected when per-episode lock is on)", flush=True)
    print("", flush=True)
    print(f"  Pearson(φ, feature):", flush=True)
    sorted_corr = sorted(correlations.items(), key=lambda x: -abs(x[1]))
    for name, r in sorted_corr:
        bar_len = int(abs(r) * 40)
        bar = "█" * bar_len
        sign = "+" if r >= 0 else "−"
        marker = ""
        if name == "actuation_noise_sigma":
            if abs(r) > 0.20:
                marker = "  ←← LAYER 2 PASS"
            elif abs(r) > 0.10:
                marker = "  ← weak (AMBIG)"
            else:
                marker = "  ← FAIL"
        elif abs(r) > 0.20:
            marker = "  ←← STRONG"
        elif abs(r) > 0.10:
            marker = "  ← weak"
        print(f"    {name:>22} {sign}{abs(r):.3f}  {bar}{marker}", flush=True)

    # Within-episode adaptation summary (new in V4 diagnostics).
    print("", flush=True)
    print(f"  Within-episode Pearson (per env, averaged across envs):", flush=True)
    wp = stats["within_episode_pearson"]
    for label, key in [("φ_t vs h_t            ", "phi_vs_h"),
                       ("φ_t vs ‖L_g h‖²_t     ", "phi_vs_Lgh_norm_sq")]:
        m = wp[key]["mean"]
        s = wp[key]["std"]
        # Theoretical expectation: NEGATIVE Pearson(φ, h) → φ rises as h shrinks
        # → defensive when close to obstacle = TISSf-style adaptation.
        marker = ""
        if m < -0.30: marker = "  ←← STRONG TISSf-shape"
        elif m < -0.15: marker = "  ← weak negative"
        elif m > 0.15: marker = "  ← WRONG SIGN"
        else: marker = "  ← ~0 (no within-ep adaptation)"
        print(f"    {label}: mean={m:+.4f}  std={s:.4f}{marker}", flush=True)
    print(f"    valid envs (φ had variance): {wp['n_valid_envs']}/{N}", flush=True)

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
        print(f"[diagnose_phi_corr] FATAL: {e}", flush=True)
        traceback.print_exc()
    # Skip simulation_app.close() — it hangs unreliably (v17 temporal_grid
    # hung 10+ min waiting on close). OS reclaims handles when the process
    # dies via os._exit, so this is safe.
    os._exit(rc if rc is not None else 0)
