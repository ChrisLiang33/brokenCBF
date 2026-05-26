#!/usr/bin/env python3
"""Linear probe of encoder latent Z against priv features.

For each priv feature (friction, base_mass, base_height, applied_force xyz,
applied_torque xyz, tracking_err xyz, com_offset xyz — 15 dims total), fits
an OLS linear regression Z → priv_feature on 80% of (env, step) samples and
reports R² on the held-out 20%.

Interpretation:
  - R² > 0.5  → Z encodes that feature linearly (strong signal)
  - 0.1 < R² < 0.5 → partial / noisy encoding
  - R² < 0.1  → feature absent from Z OR only encoded nonlinearly

Run AFTER v3.0e / v3.0f training, using its checkpoint. Uses the same env-setup
+ runner-load pattern as scripts/diagnose_z.py.

Usage on lab box:
  cd ~/Desktop/safety-go2/IsaacLab && \\
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \\
    --task Isaac-CBF-Go2-v0 \\
    --checkpoint logs/rsl_rl/cbf_go2_teacher/<TIMESTAMP>/model_2999.pt \\
    --num_envs 256 --rollout_steps 100 \\
    --output probe_z_linear_v3_0e.json \\
    --headless
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── argparse + AppLauncher must come before sim imports ──
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True, type=str)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rollout_steps", type=int, default=100)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--priv_dim", type=int, default=None,
                    help="Override priv slice size. Default uses module-level "
                         "PRIV_DIM (16, Layer 2+). Set 15 for pre-Layer-2 ckpts.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[probe_z_linear] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[probe_z_linear] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)


# ── post-sim imports ──
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401  -- registers tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


# ── priv-feature layout in the observation tensor ──
# Matches the ObservationManager 'policy' group:
#   dims 0    : friction        (scalar)
#   dim  1    : base_mass       (scalar)
#   dim  2    : base_height     (scalar)
#   dims 3-5  : applied_force   xyz
#   dims 6-8  : applied_torque  xyz
#   dims 9-11 : tracking_err    xyz
#   dims 12-14: com_offset      xyz
#   dims 15-18: cbf_state       (h, Lgh·u_des, ‖Lgh‖², |tracking_err|)
#   dims 19-8210: occupancy     (2 × 64 × 64)
PRIV_FEATURE_NAMES = [
    "friction",
    "base_mass",
    "base_height",
    "force_x", "force_y", "force_z",
    "torque_x", "torque_y", "torque_z",
    "tracking_x", "tracking_y", "tracking_z",
    "com_x", "com_y", "com_z",
    # Layer 2 (2026-05-12): added actuation_noise_sigma at end of priv slice.
    # 16 dims now. For pre-Layer-2 ckpts the obs is 15-D — probe will read
    # the cbf_state h as "actuation_noise_sigma" by mistake; pass --priv_dim 15
    # to override if you need to probe an old ckpt.
    "actuation_noise_sigma",
]
PRIV_DIM = 16  # post-Layer 2; was 15 pre-Layer-2

# V9 (2026-05-19): canonical RMA priv — no tracking_err. Dim order:
#   0: friction, 1: base_mass, 2: base_height, 3-5: force, 6-8: torque,
#   9-11: com, 12: σ_act.
PRIV_FEATURE_NAMES_V9 = [
    "friction",
    "base_mass",
    "base_height",
    "force_x", "force_y", "force_z",
    "torque_x", "torque_y", "torque_z",
    "com_x", "com_y", "com_z",
    "actuation_noise_sigma",
]

# V11 (2026-05-20): V9 minus base_height. 12 dims, truly-hidden priv only.
PRIV_FEATURE_NAMES_V11 = [
    "friction",
    "base_mass",
    "force_x", "force_y", "force_z",
    "torque_x", "torque_y", "torque_z",
    "com_x", "com_y", "com_z",
    "actuation_noise_sigma",
]

# V13 (2026-05-20): two-stream. HIDDEN-first (14) + OBSERVABLE (19).
# 33-D total. Linear-probe both halves to confirm the env encoder learns
# z_env to predict the hidden channels (and z_proprio passes through the
# observable ones identically).
PRIV_FEATURE_NAMES_V13 = [
    # HIDDEN (14)
    "friction",
    "base_mass",
    "force_x", "force_y", "force_z",
    "torque_x", "torque_y", "torque_z",
    "com_x", "com_y", "com_z",
    "actuation_noise_sigma",
    "mean_signed_delta_R",
    "max_abs_delta_R",
    # OBSERVABLE (19)
    "base_height",
    *(f"tracking_{i}" for i in range(15)),  # 5-step history flattened (15)
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
]


def linear_probe_r2(
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_test: np.ndarray, y_test: np.ndarray,
) -> tuple[float, float, float]:
    """OLS linear regression Z → y, returns (train_R², test_R², test_MSE)."""
    # Add bias column.
    A_train = np.column_stack([Z_train, np.ones(len(Z_train))])
    A_test = np.column_stack([Z_test, np.ones(len(Z_test))])
    # Least-squares fit.
    w, *_ = np.linalg.lstsq(A_train, y_train, rcond=None)
    # Predictions + R² on both splits.
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


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    print(f"[probe_z_linear] task={args.task}, num_envs={args.num_envs}, "
          f"rollout_steps={args.rollout_steps}", flush=True)

    # ── build env ──
    print(f"[probe_z_linear] parsing env_cfg...", flush=True)
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    print(f"[probe_z_linear] gym.make({args.task})...", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped
    print(f"[probe_z_linear] env built (num_envs={inner.num_envs}, device={inner.device})",
          flush=True)

    # ── load runner ──
    print(f"[probe_z_linear] loading agent_cfg + runner...", flush=True)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    print(f"[probe_z_linear] runner loaded.", flush=True)

    # ── locate actor (same defensive scan as diagnose_z.py) ──
    actor = None
    candidate_paths = [
        ("alg", "actor_critic", "actor"),
        ("alg", "actor_critic"),
        ("alg", "actor"),
        ("alg", "policy"),
        ("alg", "policy", "actor"),
        ("policy",),
        ("actor_critic",),
        ("actor",),
    ]
    for path in candidate_paths:
        obj = runner
        ok = True
        for a in path:
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok and hasattr(obj, "get_z"):
            actor = obj
            print(f"[probe_z_linear] actor located at runner.{'.'.join(path)}", flush=True)
            break
    if actor is None:
        print(f"[probe_z_linear] ERROR: could not locate actor with get_z()", flush=True)
        return 1

    z_dim = getattr(actor, "z_dim", None)
    print(f"[probe_z_linear] z_dim = {z_dim}", flush=True)

    # ── reset env, collect (Z, priv) over rollout ──
    print(f"[probe_z_linear] resetting env...", flush=True)
    obs, _ = env.reset()

    z_history = []
    priv_history = []
    for step in range(args.rollout_steps):
        with torch.no_grad():
            z = actor.get_z(obs)
            # Extract priv features from observation. obs is a dict-like; the
            # 'policy' group is the flat tensor.
            if isinstance(obs, dict) or hasattr(obs, "keys"):
                obs_tensor = obs["policy"]
            else:
                obs_tensor = obs
            if obs_tensor.dim() > 2:
                obs_tensor = obs_tensor.reshape(-1, obs_tensor.shape[-1])
            priv_dim_runtime = args.priv_dim if args.priv_dim is not None else PRIV_DIM
            priv = obs_tensor[:, :priv_dim_runtime]
            z_history.append(z.detach().cpu().float().numpy())
            priv_history.append(priv.detach().cpu().float().numpy())
            action = policy(obs)
        step_out = env.step(action)
        obs = step_out[0]
        if step % 10 == 0 or step == args.rollout_steps - 1:
            print(f"[probe_z_linear] step {step:>3}/{args.rollout_steps}  "
                  f"z.std={float(z.std()):.3f}", flush=True)

    Z = np.stack(z_history)           # (steps, N, z_dim)
    P = np.stack(priv_history)        # (steps, N, PRIV_DIM)
    Z_flat = Z.reshape(-1, Z.shape[-1])
    P_flat = P.reshape(-1, P.shape[-1])
    N_total = Z_flat.shape[0]

    print(f"[probe_z_linear] collected {N_total} samples (Z: {Z_flat.shape}, "
          f"P: {P_flat.shape})", flush=True)

    # ── train/test split ──
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N_total)
    split = int(N_total * args.train_frac)
    train_idx = perm[:split]
    test_idx = perm[split:]
    Z_train, Z_test = Z_flat[train_idx], Z_flat[test_idx]
    P_train, P_test = P_flat[train_idx], P_flat[test_idx]

    # ── per-feature linear probe ──
    # V9 (priv_dim=13) has a different feature order (no tracking_err);
    # pick the matching name list. Otherwise use 16-D layout (back-compat
    # with 15-D ckpts via truncation).
    # V13 takes priority over V8 since both have 33-D priv (different ordering).
    if P_flat.shape[1] == len(PRIV_FEATURE_NAMES_V13) and "TwoStream" in args.task:
        feature_names_runtime = PRIV_FEATURE_NAMES_V13
    elif P_flat.shape[1] == len(PRIV_FEATURE_NAMES_V11):
        feature_names_runtime = PRIV_FEATURE_NAMES_V11
    elif P_flat.shape[1] == len(PRIV_FEATURE_NAMES_V9):
        feature_names_runtime = PRIV_FEATURE_NAMES_V9
    else:
        feature_names_runtime = PRIV_FEATURE_NAMES[: P_flat.shape[1]]
    results = {}
    for i, name in enumerate(feature_names_runtime):
        y_train = P_train[:, i]
        y_test = P_test[:, i]
        # Skip degenerate features (zero variance — happens if a DR axis
        # was disabled in this task config).
        if y_test.std() < 1e-8:
            results[name] = {
                "r2_train": float("nan"),
                "r2_test": float("nan"),
                "mse_test": float("nan"),
                "feature_std": 0.0,
                "note": "feature has zero variance — DR axis disabled",
            }
            continue
        r2_train, r2_test, mse_test = linear_probe_r2(
            Z_train, y_train, Z_test, y_test,
        )
        results[name] = {
            "r2_train": r2_train,
            "r2_test": r2_test,
            "mse_test": mse_test,
            "feature_std": float(y_test.std()),
        }

    summary = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "n_envs": int(args.num_envs),
        "n_rollout_steps": int(args.rollout_steps),
        "n_samples_total": int(N_total),
        "n_train": int(split),
        "n_test": int(N_total - split),
        "z_dim": int(z_dim) if z_dim is not None else int(Z.shape[-1]),
        "priv_features": PRIV_FEATURE_NAMES,
        "linear_probe": results,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── pretty print ──
    print("", flush=True)
    print("=" * 70, flush=True)
    print(f"Linear probe Z → priv  —  {args.task}", flush=True)
    print("=" * 70, flush=True)
    print(f"  Z dim: {summary['z_dim']}   samples: {N_total}   "
          f"(train {split} / test {N_total-split})", flush=True)
    print("", flush=True)
    print(f"  {'feature':<14}  {'std':>8}  {'R² train':>10}  {'R² test':>10}  {'verdict':<20}",
          flush=True)
    print(f"  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*20}", flush=True)
    for name in feature_names_runtime:
        r = results[name]
        if not np.isfinite(r["r2_test"]):
            verdict = "(zero-variance)"
            print(f"  {name:<14}  {r['feature_std']:>8.4f}  "
                  f"{'n/a':>10}  {'n/a':>10}  {verdict:<20}", flush=True)
            continue
        if r["r2_test"] > 0.5:
            verdict = "STRONG"
        elif r["r2_test"] > 0.1:
            verdict = "partial"
        else:
            verdict = "absent / nonlinear"
        print(f"  {name:<14}  {r['feature_std']:>8.4f}  "
              f"{r['r2_train']:>10.3f}  {r['r2_test']:>10.3f}  {verdict:<20}",
              flush=True)
    print("", flush=True)
    print(f"  full results → {out_path}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    import os
    rc = 1
    try:
        rc = main()
    except Exception as e:
        import traceback
        print(f"[probe_z_linear] FATAL: {e}", flush=True)
        traceback.print_exc()
    # Skip simulation_app.close() — it hangs unreliably (v17 temporal_grid
    # hung 10+ min waiting on close). OS reclaims handles when the process
    # dies via os._exit, so this is safe.
    os._exit(rc if rc is not None else 0)
