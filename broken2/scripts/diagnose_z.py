#!/usr/bin/env python3
"""Z-latent diagnostic: probe the teacher encoder's bottleneck output.

Uses the exact same env-setup + runner-load pattern as eval_baseline.py,
then captures Z via the model's get_z() method instead of running an eval.

Run ONE invocation at a time (don't parallelize until proven stable).

Usage on lab box:
  cd ~/Desktop/safety-go2/IsaacLab && \
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_z.py \
    --task Isaac-CBF-Go2-v0 \
    --checkpoint logs/rsl_rl/cbf_go2_teacher/2026-05-10_16-13-25/model_2999.pt \
    --num_envs 64 --rollout_steps 60 \
    --output diagnose_z_v3_0c_indist.json \
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
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--rollout_steps", type=int, default=60,
                    help="Number of env steps to roll out for Z capture.")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[diagnose_z] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[diagnose_z] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)


# ── post-sim imports ──
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401  -- registers tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    print(f"[diagnose_z] task={args.task}, num_envs={args.num_envs}, "
          f"rollout_steps={args.rollout_steps}", flush=True)

    # ── build env, same pattern as eval_baseline.py ──
    print(f"[diagnose_z] parsing env_cfg...", flush=True)
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)

    print(f"[diagnose_z] gym.make({args.task})...", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped
    print(f"[diagnose_z] env built (num_envs={inner.num_envs}, device={inner.device})",
          flush=True)

    # ── load runner ──
    print(f"[diagnose_z] loading agent_cfg + runner...", flush=True)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    print(f"[diagnose_z] runner loaded.", flush=True)

    # ── locate the actor model (rsl_rl API varies by version) ──
    # Try every plausible path; print structure if we fail.
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
            print(f"[diagnose_z] actor located at runner.{'.'.join(path)}", flush=True)
            break
    if actor is None:
        # Recursive search through runner for any object with get_z
        print(f"[diagnose_z] direct paths failed, scanning runner.alg attributes...",
              flush=True)
        alg = runner.alg
        alg_attrs = [a for a in dir(alg) if not a.startswith('_')]
        print(f"[diagnose_z] runner.alg attrs: {alg_attrs}", flush=True)
        for name in alg_attrs:
            obj = getattr(alg, name, None)
            if obj is not None and hasattr(obj, "get_z"):
                actor = obj
                print(f"[diagnose_z] actor found at runner.alg.{name}", flush=True)
                break
            # one level deeper
            if obj is not None and not callable(obj) and hasattr(obj, "__dict__"):
                for sub in dir(obj):
                    if sub.startswith('_'):
                        continue
                    sub_obj = getattr(obj, sub, None)
                    if sub_obj is not None and hasattr(sub_obj, "get_z"):
                        actor = sub_obj
                        print(f"[diagnose_z] actor found at runner.alg.{name}.{sub}",
                              flush=True)
                        break
                if actor is not None:
                    break
    if actor is None:
        # Last resort: check the policy callable's __self__
        if hasattr(policy, "__self__") and hasattr(policy.__self__, "get_z"):
            actor = policy.__self__
            print(f"[diagnose_z] actor found via policy.__self__", flush=True)
    if actor is None:
        print(f"[diagnose_z] ERROR: could not locate actor with get_z(). "
              f"runner.alg type: {type(runner.alg).__name__}, "
              f"policy type: {type(policy).__name__}", flush=True)
        return 1

    z_dim = getattr(actor, "z_dim", None)
    print(f"[diagnose_z] z_dim = {z_dim}", flush=True)

    # ── reset env, collect Z over rollout ──
    print(f"[diagnose_z] resetting env...", flush=True)
    obs, _ = env.reset()
    print(f"[diagnose_z] obs keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__}",
          flush=True)

    z_history = []
    for step in range(args.rollout_steps):
        with torch.no_grad():
            z = actor.get_z(obs)
            z_history.append(z.detach().cpu().float().numpy())
            action = policy(obs)
        step_out = env.step(action)
        # gymnasium 5-tuple
        obs = step_out[0]
        if step % 10 == 0 or step == args.rollout_steps - 1:
            print(f"[diagnose_z] step {step:>3}/{args.rollout_steps}  "
                  f"z.std={float(z.std()):.4f}  z.mean={float(z.mean()):.4f}",
                  flush=True)

    Z = np.stack(z_history)                            # (steps, N, z_dim)
    Z_flat = Z.reshape(-1, Z.shape[-1])                # (steps*N, z_dim)
    z_dim = Z.shape[-1]

    per_dim_mean = Z_flat.mean(axis=0)
    per_dim_std = Z_flat.std(axis=0)
    per_dim_min = Z_flat.min(axis=0)
    per_dim_max = Z_flat.max(axis=0)
    dead_thresh = 0.05
    per_dim_active = (per_dim_std > dead_thresh).astype(int)

    across_env_std = Z.std(axis=1).mean(axis=0)        # (z_dim,)

    Z_per_env = Z.mean(axis=0)                         # (N, z_dim)
    Z_pop_mean = Z_per_env.mean(axis=0, keepdims=True)
    Z_pop_std = Z_per_env.std(axis=0) + 1e-8
    z_norm_dist = np.linalg.norm((Z_per_env - Z_pop_mean) / Z_pop_std,
                                 axis=1) / np.sqrt(z_dim)
    pct_envs_unique = float((z_norm_dist > 1.0).mean())

    stats = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "n_envs": int(args.num_envs),
        "n_rollout_steps": int(args.rollout_steps),
        "n_samples": int(Z_flat.shape[0]),
        "z_dim": int(z_dim),
        "per_dim_mean": per_dim_mean.tolist(),
        "per_dim_std": per_dim_std.tolist(),
        "per_dim_min": per_dim_min.tolist(),
        "per_dim_max": per_dim_max.tolist(),
        "per_dim_active_mask": per_dim_active.tolist(),
        "n_active_dims": int(per_dim_active.sum()),
        "across_env_std_per_dim": across_env_std.tolist(),
        "across_env_std_mean": float(across_env_std.mean()),
        "pct_envs_far_from_centroid": pct_envs_unique,
        "global_z_std": float(Z_flat.std()),
        "global_z_range": float(Z_flat.max() - Z_flat.min()),
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"Z diagnostic summary  —  {args.task}", flush=True)
    print("=" * 60, flush=True)
    print(f"  z_dim                       : {z_dim}", flush=True)
    print(f"  active dims (std > {dead_thresh})    : {stats['n_active_dims']}/{z_dim}",
          flush=True)
    print(f"  global Z std                : {stats['global_z_std']:.4f}", flush=True)
    print(f"  global Z range              : {stats['global_z_range']:.4f}", flush=True)
    print(f"  across-env std (mean)       : {stats['across_env_std_mean']:.4f}",
          flush=True)
    print(f"  envs far from centroid (>1σ): {pct_envs_unique*100:.1f}%", flush=True)
    print(f"  per-dim std:", flush=True)
    print(f"    {[f'{s:.3f}' for s in per_dim_std.tolist()]}", flush=True)
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
        print(f"[diagnose_z] FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(rc)
