"""Strong α gate via commanded velocity (v_max).

Cleanest theoretical test of α: stopping distance scales with v²/(2α),
so doubling v_max should approximately halve optimal α. If the optimum
is flat across v_max, α has no kinematic signal either and we ship the
phi-only story.

One v_max per Python invocation (Isaac Sim won't cleanly rebuild scenes
within one process). Loop via the bash wrapper.
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
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-RMA-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--eval_steps", type=int, default=1250)
parser.add_argument("--eval_eps_per_cell", type=int, default=256)
parser.add_argument("--v_max", type=float, required=True,
                    help="Commanded velocity cap (m/s). Sweep via the bash wrapper.")
parser.add_argument("--obstacle_x", type=float, default=3.0,
                    help="Fixed obstacle position (off-path, feasible).")
parser.add_argument("--obstacle_y", type=float, default=0.5)
parser.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
parser.add_argument("--phi_const", type=float, default=0.0)
parser.add_argument("--out_dir", default="phase6_vmax_gate_outputs")
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


def map_to_action(phi_v, alpha_v, cbf):
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
    for _ in range(eval_steps):
        env.step(action)
        intervention_sum = intervention_sum + cbf.last_intervention
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

    v = args_cli.v_max
    print()
    print("=" * 84)
    print(f"  v_max = {v:.2f}   (obstacle at ({args_cli.obstacle_x:+.2f}, "
          f"{args_cli.obstacle_y:+.2f}); phi pinned at {args_cli.phi_const})")
    print("=" * 84)

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env_cfg.actions.cbf_param.v_max = v
    env_cfg.actions.cbf_param.obstacle_xy = (args_cli.obstacle_x, args_cli.obstacle_y)
    env_cfg.actions.cbf_param.disturbance_force_range = (0.0, 0.0)
    env_cfg.actions.cbf_param.friction_range = (0.6, 0.6)
    env_cfg.actions.cbf_param.base_mass_range = (0.0, 0.0)
    env_cfg.actions.cbf_param.motor_strength_range = (1.0, 1.0)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    cbf = env.unwrapped.action_manager._terms["cbf_param"]

    N = args_cli.num_envs
    rows = []
    for alpha_v in args_cli.alphas:
        phi_t = torch.full((N,), float(args_cli.phi_const), device=device)
        alpha_t = torch.full((N,), float(alpha_v), device=device)
        action = map_to_action(phi_t, alpha_t, cbf)
        m = eval_cell(env, cbf, action, args_cli.eval_steps,
                      args_cli.eval_eps_per_cell, device)
        row = {"v_max": v, "alpha": alpha_v, **m}
        rows.append(row)
        print(f"  alpha={alpha_v:>4.1f}  "
              f"coll={m['collision_rate']:.2f}  reach={m['reach_rate']:.2f}  "
              f"fall={m['fall_rate']:.2f}  stuck={m['stuck_rate']:.2f}  "
              f"int={m['intervention_mean']:.0f}  "
              f"track={m['tracking_err_mean']:.3f}")

    safe = [r for r in rows
            if r["collision_rate"] <= 0.10 and r["reach_rate"] >= 0.80]
    if safe:
        best = min(safe, key=lambda r: r["intervention_mean"])
        verdict_tag = "SAFE"
    else:
        best = min(rows, key=lambda r: (r["collision_rate"], -r["reach_rate"]))
        verdict_tag = "FALLBACK"
    print()
    print(f"  best alpha @ v_max={v} [{verdict_tag}]: alpha={best['alpha']}  "
          f"(coll={best['collision_rate']:.2f}, reach={best['reach_rate']:.2f}, "
          f"int={best['intervention_mean']:.0f}, "
          f"track={best['tracking_err_mean']:.3f})")

    tag = f"v{v:.2f}"
    cells_path = os.path.join(args_cli.out_dir, f"vmax_{tag}_cells.csv")
    summ_path = os.path.join(args_cli.out_dir, f"vmax_{tag}_best.json")
    with open(cells_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(summ_path, "w") as f:
        json.dump({"v_max": v, "tag": verdict_tag, **best}, f, indent=2)
    print(f"  saved -> {cells_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
