"""Per-channel (phi, alpha) response sweep on a trained RMA teacher.

For each priv channel independently, pin it across a range of cell values
while letting the other 3 channels keep their per-episode randomization,
and measure how the teacher's emitted (phi, alpha) shifts. The channels
where (phi or alpha) actually moves are the ones z is driving -- and
therefore the channels the student must recover from history. Channels
that don't move can be deprioritized in student training.

Sweeps (4 channels, ~16-20 cells, ~30s/cell -> ~10 min total):
    disturbance      [0, 15, 30, 45]                N
    friction         [0.3, 0.5, 0.75, 1.0]          coefficient
    base_mass_delta  [-2, -1, 0, +1, +2]            kg
    motor_strength   [0.8, 0.9, 1.0, 1.1, 1.2]      scale

Per channel we compute:
    span(phi_mean)   max - min across cells
    span(alpha_mean) max - min across cells
    drives_phi?      span > 25% of phi bound width
    drives_alpha?    span > 25% of alpha bound width

Run on labbox AFTER teacher training completes:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase5_per_channel_sweep.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --policy_checkpoint phase5_teacher_outputs/rsl_rl/model_final.pt \\
        --num_envs 64 --headless
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True,
                    help="Frozen locomotion .pt.")
parser.add_argument("--policy_checkpoint", required=True,
                    help="Trained RMA teacher .pt to evaluate.")
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-RMA-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--eval_max_steps", type=int, default=1250)
parser.add_argument("--eval_eps_per_cell", type=int, default=256)
parser.add_argument("--out_dir", default="phase5_per_channel_outputs")
parser.add_argument("--drive_threshold", type=float, default=0.25,
                    help="Fraction of bound width that the (phi or alpha) span "
                         "must exceed to call a channel a 'driver'.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import csv
import importlib.metadata as metadata
import json

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.agents import rma_actor_critic  # noqa: F401 -- registers RMAMLPModel
from cbf_task.locomotion_loader import load_locomotion_actor


# (channel_name, action-term lo/hi attribute pair, sweep values)
SWEEPS = [
    ("disturbance",    ("_disturbance_force_lo", "_disturbance_force_hi"),
                       [0.0, 15.0, 30.0, 45.0]),
    ("friction",       ("_friction_lo", "_friction_hi"),
                       [0.30, 0.50, 0.75, 1.00]),
    ("mass_delta",     ("_base_mass_lo", "_base_mass_hi"),
                       [-2.0, -1.0, 0.0, 1.0, 2.0]),
    ("motor_strength", ("_motor_strength_lo", "_motor_strength_hi"),
                       [0.80, 0.90, 1.00, 1.10, 1.20]),
]


def eval_pinned(env_wrapped, runner, cbf, channel_attrs, cell_value,
                eval_steps, n_eps, device, original_ranges):
    """Pin `channel_attrs` (lo, hi) to `cell_value` (others restored from
    `original_ranges`), roll out the policy, collect (phi, alpha) stats
    + safety metrics.
    """
    # restore all channels to their original DR ranges, then pin this one
    for (lo_attr, hi_attr), (lo_val, hi_val) in original_ranges.items():
        setattr(cbf, lo_attr, float(lo_val))
        setattr(cbf, hi_attr, float(hi_val))
    lo_attr, hi_attr = channel_attrs
    setattr(cbf, lo_attr, float(cell_value))
    setattr(cbf, hi_attr, float(cell_value))

    N = env_wrapped.unwrapped.num_envs
    min_h = torch.full((N,), float("inf"), device=device)
    intervention_sum = torch.zeros(N, device=device)
    env_wrapped.unwrapped.reset()
    cbf.episode_reach_any.zero_()
    cbf.episode_collide_any.zero_()
    cbf.episode_fall_any.zero_()
    cbf.episode_stuck_any.zero_()
    obs = env_wrapped.get_observations()
    policy = runner.get_inference_policy(device=device)
    phi_hist, alpha_hist = [], []
    for _ in range(eval_steps):
        action = policy(obs)
        obs, _, _, _ = env_wrapped.step(action)
        min_h = torch.minimum(min_h, cbf.last_h_realized)
        intervention_sum = intervention_sum + cbf.last_intervention
        phi_hist.append(cbf.last_phi.detach().clone())
        alpha_hist.append(cbf.last_alpha.detach().clone())
    n_eps = min(n_eps, N)
    sel = slice(0, n_eps)
    phi_all = torch.stack(phi_hist, dim=0)[:, sel].flatten()
    alpha_all = torch.stack(alpha_hist, dim=0)[:, sel].flatten()
    return {
        "phi_mean": float(phi_all.mean().item()),
        "phi_std": float(phi_all.std().item()),
        "alpha_mean": float(alpha_all.mean().item()),
        "alpha_std": float(alpha_all.std().item()),
        "collision_rate": float(cbf.episode_collide_any[sel].float().mean().item()),
        "reach_rate": float(cbf.episode_reach_any[sel].float().mean().item()),
        "fall_rate": float(cbf.episode_fall_any[sel].float().mean().item()),
        "stuck_rate": float(cbf.episode_stuck_any[sel].float().mean().item()),
        "intervention_mean": float(intervention_sum[sel].mean().item()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env = gym.make(args_cli.task, cfg=env_cfg)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    agent_cfg.max_iterations = 0
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                            log_dir=None, device=device)
    print(f"[sweep] loading teacher -> {args_cli.policy_checkpoint}")
    runner.load(retrieve_file_path(args_cli.policy_checkpoint))

    cbf = env_wrapped.unwrapped.action_manager._terms["cbf_param"]
    # snapshot the original DR ranges so we can restore them between sweeps
    original_ranges = {
        ("_disturbance_force_lo", "_disturbance_force_hi"):
            (cbf._disturbance_force_lo, cbf._disturbance_force_hi),
        ("_friction_lo", "_friction_hi"):
            (cbf._friction_lo, cbf._friction_hi),
        ("_base_mass_lo", "_base_mass_hi"):
            (cbf._base_mass_lo, cbf._base_mass_hi),
        ("_motor_strength_lo", "_motor_strength_hi"):
            (cbf._motor_strength_lo, cbf._motor_strength_hi),
    }

    phi_bounds = env_cfg.actions.cbf_param.phi_bounds
    alpha_bounds = env_cfg.actions.cbf_param.alpha_bounds
    phi_width = phi_bounds[1] - phi_bounds[0]
    alpha_width = alpha_bounds[1] - alpha_bounds[0]
    drive_thr = args_cli.drive_threshold

    all_rows = []
    summary_rows = []
    with torch.inference_mode():
        for channel_name, channel_attrs, cells in SWEEPS:
            print()
            print(f"[sweep] channel = {channel_name}  cells = {cells}")
            per_channel = []
            for c in cells:
                m = eval_pinned(env_wrapped, runner, cbf, channel_attrs, c,
                                args_cli.eval_max_steps,
                                args_cli.eval_eps_per_cell,
                                device, original_ranges)
                row = {"channel": channel_name, "cell_value": float(c), **m}
                per_channel.append(row)
                all_rows.append(row)
                print(f"    cell={c:>+6.2f}  phi={m['phi_mean']:+.3f}+-{m['phi_std']:.3f}  "
                      f"alpha={m['alpha_mean']:.2f}+-{m['alpha_std']:.2f}  "
                      f"coll={m['collision_rate']:.2f}  reach={m['reach_rate']:.2f}  "
                      f"fall={m['fall_rate']:.2f}  stuck={m['stuck_rate']:.2f}")
            phi_span = max(r["phi_mean"] for r in per_channel) \
                       - min(r["phi_mean"] for r in per_channel)
            alpha_span = max(r["alpha_mean"] for r in per_channel) \
                         - min(r["alpha_mean"] for r in per_channel)
            drives_phi = phi_span > drive_thr * phi_width
            drives_alpha = alpha_span > drive_thr * alpha_width
            summary_rows.append({
                "channel": channel_name,
                "phi_span": float(phi_span),
                "phi_span_pct": float(100 * phi_span / phi_width),
                "alpha_span": float(alpha_span),
                "alpha_span_pct": float(100 * alpha_span / alpha_width),
                "drives_phi": drives_phi,
                "drives_alpha": drives_alpha,
            })

    # final per-channel summary
    print()
    print("=" * 96)
    print("  PER-CHANNEL TEACHER RESPONSE  -- which priv channels drive (phi, alpha)?")
    print("=" * 96)
    print(f"  {'channel':>16}  {'phi span':>10}  {'(%phi w)':>10}  "
          f"{'alpha span':>11}  {'(%alpha w)':>11}  {'drives_phi':>11}  {'drives_alpha':>13}")
    for s in summary_rows:
        print(f"  {s['channel']:>16}  {s['phi_span']:>+10.3f}  "
              f"{s['phi_span_pct']:>9.1f}%  "
              f"{s['alpha_span']:>+11.3f}  {s['alpha_span_pct']:>10.1f}%  "
              f"{str(s['drives_phi']):>11}  {str(s['drives_alpha']):>13}")
    print("=" * 96)
    drivers = [s["channel"] for s in summary_rows
               if s["drives_phi"] or s["drives_alpha"]]
    print(f"  channels the policy USES: {drivers if drivers else 'NONE (teacher flat across all 4)'}")

    # write csvs
    with open(os.path.join(args_cli.out_dir, "phase5_per_channel_cells.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    with open(os.path.join(args_cli.out_dir, "phase5_per_channel_summary.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    with open(os.path.join(args_cli.out_dir, "phase5_per_channel_summary.json"),
              "w") as f:
        json.dump({"summary": summary_rows, "drivers": drivers}, f, indent=2)
    print(f"  saved -> {args_cli.out_dir}/")

    env.close()
    simulation_app.close()
    sys.exit(0 if drivers else 1)


if __name__ == "__main__":
    main()
