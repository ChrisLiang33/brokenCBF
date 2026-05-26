"""Strong α gate -- does the optimal fixed α actually MOVE with terrain
roughness?

This is the test we should have run before training each priv channel:
"perceivable" ≠ "has a state-conditional optimum." If optimal α is flat
across roughness levels, terrain is dead as an α signal and we stop
before wasting a training run.

For each roughness level L in {0, 1, 2, 3, 4}:
  - instantiate the env with that level (all other DR pinned at nominal)
  - sweep fixed α with phi=0 (so the result isolates α's effect)
  - measure collision/reach/fall/stuck/intervention/tracking_err per cell
  - pick the best α per level

Then plot: best α vs. roughness level. **If the curve is flat, drop
terrain.** If it slopes down (lower α = more conservative on rougher
terrain, matching CBF theory), we have a real α signal and can train.

Note the gate uses gym.make WITHOUT the runner -- no learned policy
involved; we manually emit the (phi, alpha) we want.

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase6_alpha_gate.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --num_envs 256 --levels 0 1 2 3 --headless
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-Rough-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--eval_steps", type=int, default=1250)
parser.add_argument("--eval_eps_per_cell", type=int, default=256)
parser.add_argument("--level", type=int, required=True,
                    help="Terrain roughness level (single). Loop over "
                         "levels via the bash wrapper -- Isaac Sim doesn't "
                         "cleanly rebuild a scene within one Python process.")
parser.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
parser.add_argument("--phi_const", type=float, default=0.0,
                    help="Fix phi at this value -- gate isolates alpha.")
parser.add_argument("--out_dir", default="phase6_alpha_gate_outputs")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import gymnasium as gym
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor


def map_to_action(phi_v: torch.Tensor, alpha_v: torch.Tensor,
                  cbf) -> torch.Tensor:
    a0 = 2.0 * (phi_v - cbf._phi_lo) / (cbf._phi_hi - cbf._phi_lo) - 1.0
    a1 = 2.0 * (alpha_v - cbf._alpha_lo) / (cbf._alpha_hi - cbf._alpha_lo) - 1.0
    return torch.stack([a0, a1], dim=-1)


def eval_cell(env, cbf, action, eval_steps, n_eps, device):
    env.unwrapped.reset()
    cbf.episode_reach_any.zero_()
    cbf.episode_collide_any.zero_()
    cbf.episode_fall_any.zero_()
    cbf.episode_stuck_any.zero_()
    N = env.unwrapped.num_envs
    intervention_sum = torch.zeros(N, device=device)
    tracking_err_sum = torch.zeros(N, device=device)
    for step in range(eval_steps):
        env.step(action)
        intervention_sum = intervention_sum + cbf.last_intervention
        # tracking residual: ||u_safe - actual_base_lin_b_xy||
        actual_lin = cbf._robot.data.root_lin_vel_b[:, :2]
        track = torch.linalg.norm(cbf.last_u_safe - actual_lin, dim=-1)
        tracking_err_sum = tracking_err_sum + track
    n_eps = min(n_eps, N)
    sel = slice(0, n_eps)
    return {
        "collision_rate": float(cbf.episode_collide_any[sel].float().mean().item()),
        "reach_rate": float(cbf.episode_reach_any[sel].float().mean().item()),
        "fall_rate": float(cbf.episode_fall_any[sel].float().mean().item()),
        "stuck_rate": float(cbf.episode_stuck_any[sel].float().mean().item()),
        "intervention_mean": float(intervention_sum[sel].mean().item()),
        "tracking_err_mean": float(tracking_err_sum[sel].mean().item() / eval_steps),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)
    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)

    level = args_cli.level
    print()
    print("=" * 84)
    print(f"  TERRAIN LEVEL {level}   (phi pinned at {args_cli.phi_const})")
    print("=" * 84)

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env_cfg.terrain_level = level
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    cbf = env.unwrapped.action_manager._terms["cbf_param"]

    N = args_cli.num_envs
    level_rows = []
    for alpha_v in args_cli.alphas:
        phi_t = torch.full((N,), float(args_cli.phi_const), device=device)
        alpha_t = torch.full((N,), float(alpha_v), device=device)
        action = map_to_action(phi_t, alpha_t, cbf)
        m = eval_cell(env, cbf, action, args_cli.eval_steps,
                      args_cli.eval_eps_per_cell, device)
        row = {"level": level, "alpha": alpha_v, **m}
        level_rows.append(row)
        print(f"  alpha={alpha_v:>4.1f}  "
              f"coll={m['collision_rate']:.2f}  reach={m['reach_rate']:.2f}  "
              f"fall={m['fall_rate']:.2f}  stuck={m['stuck_rate']:.2f}  "
              f"int={m['intervention_mean']:.0f}  "
              f"track_err={m['tracking_err_mean']:.3f}")

    # best alpha for this level
    safe = [r for r in level_rows
            if r["collision_rate"] <= 0.10 and r["reach_rate"] >= 0.80]
    if safe:
        best = min(safe, key=lambda r: r["intervention_mean"])
        tag = "SAFE"
    else:
        best = min(level_rows, key=lambda r: (r["collision_rate"],
                                                -r["reach_rate"]))
        tag = "FALLBACK"
    print()
    print(f"  best alpha @ level {level} [{tag}]: alpha={best['alpha']}  "
          f"(coll={best['collision_rate']:.2f}, reach={best['reach_rate']:.2f}, "
          f"int={best['intervention_mean']:.0f}, "
          f"track={best['tracking_err_mean']:.3f})")

    # per-level CSV + summary JSON (one file per level; merge in wrapper)
    cells_path = os.path.join(args_cli.out_dir,
                              f"alpha_gate_level{level}_cells.csv")
    summ_path = os.path.join(args_cli.out_dir,
                             f"alpha_gate_level{level}_best.json")
    with open(cells_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(level_rows[0].keys()))
        w.writeheader(); w.writerows(level_rows)
    with open(summ_path, "w") as f:
        json.dump({"level": level, "tag": tag, **best}, f, indent=2)
    print(f"  saved -> {cells_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
