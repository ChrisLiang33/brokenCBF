"""Phase 0.6 -- the fixed-φ sweep GATE before any RL.

For a SINGLE friction value, sweep fixed φ over a grid, run N seeds per
cell, and log per-rollout results to a CSV. The user invokes this script
once per friction level (see README); analyze_phi_sweep.py then
aggregates the CSVs and renders the optimal-φ-vs-μ gate plot.

The gate's job is one specific thing: confirm that the optimal fixed φ
*moves* with the friction channel. If it doesn't, no policy
architecture can learn to adapt it -- the channel and the parameter are
not paired, and no scene design fixes that.

Single-friction-per-invocation is deliberate. Recreating an Isaac Lab
env in-process between μ values is fragile; this design pays a small
AppLauncher startup cost per friction (~5s) in exchange for a clean
restart each time.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# AppLauncher first; everything else after.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-Play-v0")
parser.add_argument("--checkpoint", required=True,
                    help="rsl_rl checkpoint produced by your stock Go2 training.")
parser.add_argument("--out_csv", required=True,
                    help="Per-rollout output CSV path.")
parser.add_argument("--friction_mu", type=float, default=0.6,
                    help="Static + dynamic friction coefficient. Default 0.6 "
                         "is roughly the locomotion's training-distribution "
                         "default; vary this OR --disturbance_force as the "
                         "channel under test, not both at once.")
parser.add_argument("--disturbance_force", type=float, default=0.0,
                    help="Magnitude (Newtons) of an external horizontal "
                         "force applied to the robot base, mirroring the "
                         "MVP's `d`. Direction is resampled every "
                         "--disturbance_resample steps. 0 disables.")
parser.add_argument("--disturbance_resample", type=int, default=50,
                    help="Steps between disturbance direction resamples "
                         "(50 steps = 1s at dt=0.02).")
parser.add_argument("--phi_values", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
parser.add_argument("--seeds_per_cell", type=int, default=5)
parser.add_argument("--alpha", type=float, default=1.5)
# scenario (kept identical to Phase 0.5 by default so the gate's result is
# directly comparable to a passing Phase 0.5 baseline)
parser.add_argument("--goal", type=float, nargs=2, default=[6.0, 0.0])
# Geometry tightened past Phase 0.5: lateral offset 0.3 + radius 0.9
# (r_safe = 1.2) requires ~0.9 m of lateral deflection vs ~0.7 m at the
# earlier (3.0, 0.4, r=0.8) settings. Combined with high disturbance
# this should finally force fixed-φ=0 into edge cases that collide.
parser.add_argument("--obstacle", type=float, nargs=2, default=[2.5, 0.3])
parser.add_argument("--obstacle_radius", type=float, default=0.9)
parser.add_argument("--robot_radius", type=float, default=0.3)
parser.add_argument("--slack_penalty", type=float, default=1.0e4)
parser.add_argument("--kp", type=float, default=1.0)
parser.add_argument("--v_max", type=float, default=1.3)
parser.add_argument("--goal_tol", type=float, default=0.4)
parser.add_argument("--max_time", type=float, default=25.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import csv
import math

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


# ---------------------------------------------------------------------------
# Closed-form CBF filter (replaces CVXPY for our specific QP).
# See Phase 0.5 for derivation. Half-space project then ball clip; residual
# constraint deficit absorbed as slack. Exact in M -> inf; ~1% off at M=1e4.
# ---------------------------------------------------------------------------
class CBFQPSolver:
    def __init__(self, v_max: float, slack_penalty: float = 1.0e4):
        self.v_max = float(v_max)
        _ = slack_penalty

    def solve(self, grad_h_base: np.ndarray, rhs: float, u_nom_b: np.ndarray):
        g = np.asarray(grad_h_base, dtype=float)
        u = np.asarray(u_nom_b, dtype=float)
        rhs = float(rhs)
        deficit = rhs - float(g @ u)
        if deficit <= 0.0:
            v = u.copy()
        else:
            v = u + deficit * g
        n = float(np.linalg.norm(v))
        if n > self.v_max:
            v = v * (self.v_max / n)
        slack = max(0.0, rhs - float(g @ v))
        return v, slack


def yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def p_world(base_xy: np.ndarray, goal_xy: np.ndarray, kp: float, v_max: float) -> np.ndarray:
    err = goal_xy - base_xy
    cmd = kp * err
    n = float(np.linalg.norm(cmd))
    return cmd * (v_max / n) if n > v_max else cmd


def world_to_base(v_world: np.ndarray, yaw: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([cy * v_world[0] + sy * v_world[1],
                    -sy * v_world[0] + cy * v_world[1]], dtype=float)


def rollout_one(env, policy, qp, cmd_term, robot,
                goal_xy: np.ndarray, obs_xy: np.ndarray, r_safe: float,
                phi: float, alpha: float,
                kp: float, v_max: float, goal_tol: float,
                max_steps: int, dt: float,
                disturbance_force: float, disturbance_resample: int) -> dict:
    """Run one episode under fixed (phi, alpha). Returns aggregate metrics."""
    env.get_observations()

    min_h_realized = float("inf")
    sum_intervention = 0.0
    max_slack = 0.0
    collided = False
    reached = False
    n_steps = 0
    reach_t = float("nan")

    # External-force disturbance state. Resampled every disturbance_resample
    # steps; held constant in between. Mirrors the MVP's `d`.
    dist_theta = float(np.random.uniform(0.0, 2.0 * np.pi)) if disturbance_force > 0 else 0.0
    device = robot.data.root_pos_w.device

    for step in range(max_steps):
        base_xy_w = robot.data.root_pos_w[0, :2].detach().cpu().numpy().astype(float)
        yaw = float(yaw_from_quat_wxyz(robot.data.root_quat_w[0:1]).item())

        diff = base_xy_w - obs_xy
        dist = max(float(np.linalg.norm(diff)), 1e-6)
        grad_h_world = diff / dist
        cy, sy = math.cos(yaw), math.sin(yaw)
        grad_h_base = np.array([cy * grad_h_world[0] + sy * grad_h_world[1],
                               -sy * grad_h_world[0] + cy * grad_h_world[1]])
        h_now = dist - r_safe

        u_nom_w = p_world(base_xy_w, goal_xy, kp, v_max)
        u_nom_b = world_to_base(u_nom_w, yaw)

        rhs = phi - alpha * h_now
        v_safe_b, slack = qp.solve(grad_h_base, rhs, u_nom_b)

        desired_cmd = torch.tensor([[float(v_safe_b[0]), float(v_safe_b[1]), 0.0]],
                                   device=cmd_term.vel_command_b.device,
                                   dtype=cmd_term.vel_command_b.dtype)
        cmd_term.vel_command_b[:] = desired_cmd

        obs_dict = env.unwrapped.observation_manager.compute()
        # Only the policy forward needs inference_mode. Keeping env.step
        # inside it marks env state tensors as inference tensors, which
        # later breaks env.reset() between rollouts.
        with torch.inference_mode():
            action = policy(obs_dict)

        # Apply external-force disturbance to the base BEFORE env.step so
        # the physics tick consumes it. body_ids=[0] = trunk.
        if disturbance_force > 0:
            if step % disturbance_resample == 0:
                dist_theta = float(np.random.uniform(0.0, 2.0 * np.pi))
            fx = disturbance_force * math.cos(dist_theta)
            fy = disturbance_force * math.sin(dist_theta)
            forces = torch.zeros((1, 1, 3), device=device)
            forces[0, 0, 0] = fx
            forces[0, 0, 1] = fy
            torques = torch.zeros_like(forces)
            robot.set_external_force_and_torque(forces, torques, body_ids=[0])

        _, _, dones, _ = env.step(action)

        base_xy_after = robot.data.root_pos_w[0, :2].detach().cpu().numpy().astype(float)
        dist_after = float(np.linalg.norm(base_xy_after - obs_xy))
        h_real = dist_after - r_safe

        intervention = float(np.linalg.norm(v_safe_b - u_nom_b))
        min_h_realized = min(min_h_realized, h_real)
        sum_intervention += intervention
        max_slack = max(max_slack, slack)
        n_steps += 1

        if h_real < 0.0:
            collided = True
        dist_goal = float(np.linalg.norm(goal_xy - base_xy_after))
        if dist_goal < goal_tol and not math.isfinite(reach_t):
            reached = True
            reach_t = (step + 1) * dt
            break
        if bool(dones[0].item()):
            break

    return {
        "collided": int(collided),
        "reached": int(reached),
        "min_h_realized": min_h_realized if math.isfinite(min_h_realized) else 0.0,
        "mean_intervention": sum_intervention / max(n_steps, 1),
        "max_qp_slack": max_slack,
        "n_steps": n_steps,
        "reach_time": reach_t,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mu = float(args_cli.friction_mu)

    # --- build env_cfg with friction pinned ---
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = device
    env_cfg.log_dir = None
    env_cfg.seed = 0

    # Override the stock random physics_material event term to a degenerate
    # (mu, mu) range -- this pins both static + dynamic friction at startup.
    try:
        pm = env_cfg.events.physics_material
        pm.params["static_friction_range"] = (mu, mu)
        pm.params["dynamic_friction_range"] = (mu, mu)
        print(f"[phase0.6] friction pinned at mu={mu:.3f} via events.physics_material")
    except AttributeError:
        print(f"[phase0.6] WARNING: events.physics_material not found on env_cfg; "
              f"friction sweep will be ineffective. Inspect env_cfg.events.", file=sys.stderr)
        raise

    # --- agent cfg ---
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    import importlib.metadata as metadata
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[phase0.6] checkpoint -> {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name != "OnPolicyRunner":
        raise RuntimeError(f"Unsupported runner: {agent_cfg.class_name}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt
    max_steps = int(round(args_cli.max_time / dt))

    goal_xy = np.array(args_cli.goal, dtype=float)
    obs_xy = np.array(args_cli.obstacle, dtype=float)
    r_safe = float(args_cli.obstacle_radius) + float(args_cli.robot_radius)

    qp = CBFQPSolver(args_cli.v_max, args_cli.slack_penalty)

    # --- sweep ---
    out_path = os.path.abspath(args_cli.out_csv)
    fieldnames = ["friction_mu", "disturbance_force", "phi", "alpha", "seed",
                  "collided", "reached", "min_h_realized",
                  "mean_intervention", "max_qp_slack", "n_steps", "reach_time"]
    f = open(out_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    total = len(args_cli.phi_values) * args_cli.seeds_per_cell
    t_start = time.time()
    cell = 0

    for phi in args_cli.phi_values:
        for s in range(args_cli.seeds_per_cell):
            # seed env reset RNG -- gives reproducible per-cell variation
            np.random.seed(s)
            torch.manual_seed(s)
            env.reset()

            m = rollout_one(env, policy, qp, cmd_term, robot,
                            goal_xy, obs_xy, r_safe,
                            phi, args_cli.alpha,
                            args_cli.kp, args_cli.v_max,
                            args_cli.goal_tol, max_steps, dt,
                            float(args_cli.disturbance_force),
                            int(args_cli.disturbance_resample))

            row = {"friction_mu": mu,
                   "disturbance_force": float(args_cli.disturbance_force),
                   "phi": float(phi),
                   "alpha": float(args_cli.alpha), "seed": s, **m}
            writer.writerow(row)
            f.flush()

            cell += 1
            el = time.time() - t_start
            eta = el * (total - cell) / max(cell, 1)
            print(f"[phase0.6] mu={mu:.2f} d={args_cli.disturbance_force:.1f}N "
                  f"phi={phi:.2f} seed={s}  "
                  f"reached={bool(m['reached'])}  collided={bool(m['collided'])}  "
                  f"min_h={m['min_h_realized']:+.3f}  int={m['mean_intervention']:.3f}  "
                  f"({cell}/{total}, eta={eta:.0f}s)")

    f.close()
    print(f"[phase0.6] wrote {cell} rows -> {out_path}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
