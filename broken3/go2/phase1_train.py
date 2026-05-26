"""Phase 1 -- canonical Isaac Lab + rsl_rl training of a constant (φ, α).

Trains the outer policy on `Isaac-CBF-Adaptive-Go2-v0` using rsl_rl
OnPolicyRunner with `num_envs` parallel envs on the GPU. Observation is
zeroed by the env cfg, so the policy is forced to be state-independent.
After training, performs a grid search over (φ, α) with fixed actions
and prints a PASS/REVIEW verdict comparing the learned constant to the
grid optimum.

Run via Isaac Lab:
    ./isaaclab.sh -p ~/Desktop/cbf_rl_mvp/go2/phase1_train.py \\
        --checkpoint /path/to/locomotion_model_*.pt \\
        --num_envs 64 \\
        --max_iterations 200 \\
        --headless
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# AppLauncher
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True,
                    help="Path to the pretrained Go2 locomotion .pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=64,
                    help="Parallel envs for training. 64 is a reasonable "
                         "default; bump for more throughput if VRAM allows.")
parser.add_argument("--max_iterations", type=int, default=200,
                    help="rsl_rl outer-PPO iterations. With num_steps_per_env=24 "
                         "and num_envs=64, 200 iters ~ 300k rollout steps.")
parser.add_argument("--out_dir", default="phase1_outputs",
                    help="Where to write checkpoints + grid CSV + summary.")
parser.add_argument("--seed", type=int, default=0)
# scenario overrides (default matches what passed Phase 0.6)
parser.add_argument("--goal", type=float, nargs=2, default=[6.0, 0.0])
parser.add_argument("--obstacle", type=float, nargs=2, default=[2.5, 0.3])
parser.add_argument("--obstacle_radius", type=float, default=0.9)
parser.add_argument("--disturbance_force", type=float, default=30.0)
parser.add_argument("--disturbance_resample", type=int, default=50)
# grid search at end
parser.add_argument("--grid_phi", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
parser.add_argument("--grid_alpha", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.5, 4.0])
parser.add_argument("--grid_eps_per_cell", type=int, default=8)
parser.add_argument("--grid_max_steps", type=int, default=1250)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import csv
import importlib.metadata as metadata
import json
import math
import sys as _sys

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# Make our cbf_task package importable when running as a free-standing script.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401  -- registers Isaac-CBF-Adaptive-Go2-v0
from cbf_task.locomotion_loader import load_locomotion_actor


TASK = "Isaac-CBF-Adaptive-Go2-v0"


def build_env_cfg(num_envs: int, locomotion_actor, args, device: str):
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args.seed
    env_cfg.log_dir = None

    # Inject the locomotion policy + scenario into the action term cfg
    at = env_cfg.actions.cbf_param
    at.locomotion_policy_obj = locomotion_actor
    at.goal_xy = tuple(args.goal)
    at.obstacle_xy = tuple(args.obstacle)
    at.obstacle_radius = float(args.obstacle_radius)
    at.disturbance_force = float(args.disturbance_force)
    at.disturbance_resample = int(args.disturbance_resample)
    return env_cfg


# ---------------------------------------------------------------------------
# Grid-search evaluator
# ---------------------------------------------------------------------------
def grid_search(env, action_term_cfg, args) -> list[dict]:
    """Rollout fixed (φ, α) cells in parallel across all envs. Returns a
    list of per-cell aggregate metrics."""
    device = env.unwrapped.device
    N = env.unwrapped.num_envs

    phi_lo, phi_hi = action_term_cfg.phi_bounds
    alpha_lo, alpha_hi = action_term_cfg.alpha_bounds

    def to_norm(phi, alpha):
        a0 = 2.0 * (phi - phi_lo) / (phi_hi - phi_lo) - 1.0
        a1 = 2.0 * (alpha - alpha_lo) / (alpha_hi - alpha_lo) - 1.0
        return torch.tensor([[a0, a1]] * N, device=device, dtype=torch.float32)

    rows = []
    for phi in args.grid_phi:
        for alpha in args.grid_alpha:
            action = to_norm(phi, alpha)
            min_h = torch.full((N,), float("inf"), device=device)
            intervention_sum = torch.zeros(N, device=device)
            env.unwrapped.reset()
            # clear the action term's sticky per-cell flags so the cell
            # starts with no historical reach/collide carried over.
            cbf_term = env.unwrapped.action_manager._terms["cbf_param"]
            cbf_term.episode_reach_any.zero_()
            cbf_term.episode_collide_any.zero_()
            for _ in range(args.grid_max_steps):
                env.step(action)
                # NB: each env may have auto-reset inside step. The sticky
                # flags set inside the termination terms survive that. The
                # per-step buffers (last_intervention, last_h_realized) do
                # NOT survive: post-reset they reflect the new episode.
                min_h = torch.minimum(min_h, cbf_term.last_h_realized)
                intervention_sum = intervention_sum + cbf_term.last_intervention
            # collect cell stats across the N parallel "seeds"
            n_eps = min(args.grid_eps_per_cell, N)
            sel = slice(0, n_eps)
            rows.append({
                "phi": float(phi), "alpha": float(alpha), "n": int(n_eps),
                "collision_rate": float(cbf_term.episode_collide_any[sel].float().mean().item()),
                "reach_rate": float(cbf_term.episode_reach_any[sel].float().mean().item()),
                "min_h_mean": float(min_h[sel].clamp(min=-10.0).mean().item()),
                "intervention_mean": float(intervention_sum[sel].mean().item()),
            })
            r = rows[-1]
            print(f"  phi={phi:.2f} alpha={alpha:.2f}  "
                  f"coll={r['collision_rate']:.2f} reach={r['reach_rate']:.2f} "
                  f"int={r['intervention_mean']:.2f}")
    return rows


def pick_grid_best(rows: list[dict]) -> dict:
    """min φ such that coll==0 AND reach>=0.6; fallback to lex(coll, -reach, int)."""
    ok = [r for r in rows if r["collision_rate"] == 0.0 and r["reach_rate"] >= 0.6]
    if ok:
        return min(ok, key=lambda r: (r["intervention_mean"], r["phi"]))
    return min(rows, key=lambda r: (r["collision_rate"], -r["reach_rate"],
                                     r["intervention_mean"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    # 1) load locomotion actor (no env needed)
    ckpt = retrieve_file_path(args_cli.checkpoint)
    print(f"[phase1] locomotion checkpoint -> {ckpt}")
    locomotion_actor = load_locomotion_actor(ckpt, device)

    # 2) build env cfg with locomotion injected
    env_cfg = build_env_cfg(args_cli.num_envs, locomotion_actor, args_cli, device)
    env = gym.make(TASK, cfg=env_cfg)

    # 3) rsl_rl agent cfg
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    agent_cfg.max_iterations = int(args_cli.max_iterations)
    agent_cfg.seed = int(args_cli.seed)

    # 4) wrap env for rsl_rl
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 5) train
    log_dir = os.path.join(os.path.abspath(args_cli.out_dir), "rsl_rl")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[phase1] training PPO for {agent_cfg.max_iterations} iterations "
          f"({agent_cfg.num_steps_per_env * args_cli.num_envs} rollout steps "
          f"per iter, {args_cli.num_envs} envs) ...")
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                            log_dir=log_dir, device=device)
    import time as _t
    t0 = _t.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=False)
    train_secs = _t.time() - t0
    runner.save(os.path.join(log_dir, "model_final.pt"))
    print(f"[phase1] saved -> {log_dir}/model_final.pt  ({train_secs:.0f}s)")

    # 6) read out the learned constant action by querying the policy on zero obs
    learned_policy = runner.get_inference_policy(device=device)
    zero_obs = {"policy": torch.zeros((1, 7), device=device)}
    # NB: rsl_rl's learn() runs env.step under inference_mode, which marks
    # env state tensors. To re-use the env post-training (for grid search
    # etc.) we need to stay inside inference_mode for any further
    # env.reset / env.step calls.
    with torch.inference_mode():
        learned_action = learned_policy(zero_obs).clamp(-1.0, 1.0)
        at = env_cfg.actions.cbf_param
        phi_lo, phi_hi = at.phi_bounds
        alpha_lo, alpha_hi = at.alpha_bounds
        a0 = float(learned_action[0, 0].item())
        a1 = float(learned_action[0, 1].item())
        learned_phi = phi_lo + (a0 + 1.0) * 0.5 * (phi_hi - phi_lo)
        learned_alpha = alpha_lo + (a1 + 1.0) * 0.5 * (alpha_hi - alpha_lo)
        print(f"[phase1] PPO learned: phi={learned_phi:.3f}  alpha={learned_alpha:.3f}")

        # 7) grid search using all envs in parallel
        print(f"[phase1] grid search over "
              f"{len(args_cli.grid_phi)}x{len(args_cli.grid_alpha)} cells "
              f"({args_cli.grid_eps_per_cell} parallel seeds per cell) ...")
        grid_rows = grid_search(env_wrapped, at, args_cli)
        grid_best = pick_grid_best(grid_rows)
        # also evaluate the learned constant
        print("[phase1] evaluating learned constant ...")
        learned_rows = grid_search(env_wrapped, at,
                                   argparse.Namespace(
                                       grid_phi=[learned_phi],
                                       grid_alpha=[learned_alpha],
                                       grid_eps_per_cell=args_cli.grid_eps_per_cell,
                                       grid_max_steps=args_cli.grid_max_steps,
                                   ))
        learned_m = learned_rows[0]

    # 8) write artifacts
    grid_csv = os.path.join(args_cli.out_dir, "phase1_grid.csv")
    with open(grid_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)

    summary = {
        "ppo": {"phi": learned_phi, "alpha": learned_alpha,
                "train_seconds": train_secs, **{k: v for k, v in learned_m.items()
                                                 if k not in ("phi", "alpha", "n")}},
        "grid_best": grid_best,
    }
    json.dump(summary,
              open(os.path.join(args_cli.out_dir, "phase1_summary.json"), "w"),
              indent=2)

    # 9) verdict
    learned_ok = (learned_m["collision_rate"] == 0.0
                  and learned_m["reach_rate"] >= 0.6)
    gap = abs(learned_m["intervention_mean"] - grid_best["intervention_mean"])
    near_opt = gap <= 0.20 * max(grid_best["intervention_mean"], 1e-6)
    verdict = "PASS" if (learned_ok and near_opt) else "REVIEW"

    print()
    print("=" * 78)
    print("  Phase 1 -- PPO vs grid-search best constant")
    print("=" * 78)
    print(f"  PPO learned : phi={learned_phi:.3f} alpha={learned_alpha:.3f}  "
          f"coll={learned_m['collision_rate']:.2f} reach={learned_m['reach_rate']:.2f} "
          f"int={learned_m['intervention_mean']:.2f}")
    print(f"  grid best   : phi={grid_best['phi']:.3f} alpha={grid_best['alpha']:.3f}  "
          f"coll={grid_best['collision_rate']:.2f} reach={grid_best['reach_rate']:.2f} "
          f"int={grid_best['intervention_mean']:.2f}")
    print(f"  verdict     : {verdict}")
    if verdict != "PASS":
        print("  -> debug reward/training before Phase 2")
    print("=" * 78)
    print(f"\n  saved -> {log_dir}, {grid_csv}, phase1_summary.json")

    env.close()
    simulation_app.close()
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
