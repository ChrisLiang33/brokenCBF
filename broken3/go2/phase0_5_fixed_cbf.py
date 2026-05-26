"""Phase 0.5 -- fixed CBF + one static obstacle.

Adds to Phase 0:
- one static obstacle on the path from start to goal,
- a per-step CBF-QP safety filter with hand-tuned (phi, alpha),
- a check that the REALIZED motion stays in the safe set even though the
  QP can only constrain the COMMANDED velocity.

The load-bearing test is the realized-h check. With a single integrator
realized == commanded so the test is vacuous; on the Go2 the locomotion
policy tracks commanded with error, so realized motion may briefly cross
h = 0 even when the QP is feasible. Quantifying that gap NOW (fixed
filter, one obstacle, no learning) is what determines whether the whole
command-space CBF approach holds on a legged robot.

For a distance barrier h(x) = ||p_base - p_obs|| - r_safe the gradient
is in the WORLD frame; the QP constrains commanded velocity in the BASE
frame; substituting v_world = R(yaw) @ v_base gives one linear
inequality on (vx_b, vy_b). wz_b does not enter the safety constraint
for a 2-D distance barrier and is set to 0 in the nominal -- this is the
deliberate Phase 0.5 simplification.

Run on labbox with Isaac Lab. See README.md.
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# AppLauncher MUST run before any other isaaclab / omni imports.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--goal", type=float, nargs=2, default=[6.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--obstacle", type=float, nargs=2, default=[3.0, 0.0], metavar=("X", "Y"),
                    help="Obstacle center in world frame (meters).")
parser.add_argument("--obstacle_radius", type=float, default=0.8,
                    help="Obstacle physical radius (meters).")
parser.add_argument("--robot_radius", type=float, default=0.3,
                    help="Bounding radius used in the safe radius r_safe.")
parser.add_argument("--phi", type=float, default=0.25,
                    help="Fixed CBF robustness margin.")
parser.add_argument("--alpha", type=float, default=1.5,
                    help="Fixed class-K gain.")
parser.add_argument("--slack_penalty", type=float, default=1.0e4)
parser.add_argument("--kp", type=float, default=1.0)
parser.add_argument("--v_max", type=float, default=1.0)
parser.add_argument("--goal_tol", type=float, default=0.4)
parser.add_argument("--max_time", type=float, default=20.0)
parser.add_argument("--log_csv", type=str, default="phase0_5_log.csv")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Sim is up; rest of the imports.
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

try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except Exception:
    get_published_pretrained_checkpoint = None


# ---------------------------------------------------------------------------
# Closed-form CBF filter (replaces CVXPY for our specific QP).
#
#   min ||v_b - u_nom_b||^2 + M * delta^2
#   s.t.  grad_h_base . v_b + delta >= phi - alpha * h     (unit grad_h)
#         ||v_b||_2 <= v_max
#         delta >= 0
#
# Closed-form (exact in the M -> inf limit; ~1% off at M = 1e4):
#   1. If grad_h.u_nom >= rhs:  v = u_nom.
#   2. Else: half-space project: v = u_nom + (rhs - grad_h.u_nom) * grad_h.
#   3. If ||v|| > v_max: radial clip; residual deficit becomes slack delta.
# ---------------------------------------------------------------------------
class CBFQPSolver:
    def __init__(self, v_max: float, slack_penalty: float = 1.0e4):
        self.v_max = float(v_max)
        _ = slack_penalty   # kept for signature compatibility

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


def p_controller_world(base_xy_w: np.ndarray, goal_xy_w: np.ndarray, kp: float, v_max: float) -> np.ndarray:
    """Returns nominal velocity in WORLD frame (2-vector), clipped to v_max."""
    err = goal_xy_w - base_xy_w
    cmd = kp * err
    n = float(np.linalg.norm(cmd))
    if n > v_max:
        cmd = cmd * (v_max / n)
    return cmd


def world_to_base(v_world: np.ndarray, yaw: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([cy * v_world[0] + sy * v_world[1],
                     -sy * v_world[0] + cy * v_world[1]], dtype=float)


def evaluate_pass_criteria(rows: list[dict], goal_tol: float) -> dict:
    if not rows:
        return {"crashed_or_empty": True}
    reached = any(r["dist_to_goal"] < goal_tol for r in rows)
    final_dist = rows[-1]["dist_to_goal"]
    min_realized_h = min(r["h_realized"] for r in rows)
    min_commanded_h = min(r["h_commanded"] for r in rows)
    realized_violation_steps = sum(1 for r in rows if r["h_realized"] < 0.0)
    max_slack = max(r["qp_slack"] for r in rows)

    tail_start = int(0.2 * len(rows))
    tail = rows[tail_start:]
    if tail:
        cmd_mag = sum(math.hypot(r["vx_cmd"], r["vy_cmd"]) for r in tail) / len(tail)
        err_mag = sum(math.hypot(r["vx_cmd"] - r["vx_real"], r["vy_cmd"] - r["vy_real"])
                      for r in tail) / len(tail)
        tracking_rmse_ratio = err_mag / max(cmd_mag, 1e-6)
    else:
        tracking_rmse_ratio = float("nan")

    passes = {
        "plumbing": True,
        "reached_goal": bool(reached),
        "realized_h_safe": min_realized_h >= 0.0,
        "tracking_ok": tracking_rmse_ratio < 0.30,
    }
    return {
        "passes": passes,
        "all_pass": all(passes.values()),
        "final_dist": final_dist,
        "min_realized_h": min_realized_h,
        "min_commanded_h": min_commanded_h,
        "realized_violation_steps": realized_violation_steps,
        "max_qp_slack": max_slack,
        "tracking_rmse_ratio": tracking_rmse_ratio,
        "n_steps": len(rows),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # ----- env config -----
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = None

    # ----- agent config -----
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    import importlib.metadata as metadata
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.device = device

    # ----- checkpoint -----
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            raise RuntimeError("isaaclab_rl pretrained_checkpoint helper not available.")
        train_task_name = args_cli.task.replace("-Play", "")
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise RuntimeError(f"No published pretrained checkpoint for {train_task_name}.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        raise SystemExit("Pass --checkpoint <path> or --use_pretrained_checkpoint.")
    print(f"[phase0.5] checkpoint -> {resume_path}")

    # ----- env -----
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ----- policy -----
    if agent_cfg.class_name != "OnPolicyRunner":
        raise RuntimeError(f"Unexpected agent runner class: {agent_cfg.class_name}.")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ----- handles + scenario -----
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt
    max_steps = int(round(args_cli.max_time / dt))

    goal_xy_w = np.array(args_cli.goal, dtype=float)
    obs_xy_w = np.array(args_cli.obstacle, dtype=float)
    r_safe = float(args_cli.obstacle_radius) + float(args_cli.robot_radius)

    qp = CBFQPSolver(args_cli.v_max, args_cli.slack_penalty)

    print(f"[phase0.5] dt={dt:.4f}s  goal={tuple(args_cli.goal)}  "
          f"obstacle={tuple(args_cli.obstacle)}  r_safe={r_safe:.2f}  "
          f"phi={args_cli.phi}  alpha={args_cli.alpha}")

    env.get_observations()                # initializes buffers
    rows: list[dict] = []

    for step in range(max_steps):
        # --- read robot state ---
        base_xy_w = robot.data.root_pos_w[0, :2].detach().cpu().numpy().astype(float)
        yaw = float(yaw_from_quat_wxyz(robot.data.root_quat_w[0:1]).item())

        # --- CBF inputs (world frame for grad_h, then rotate to base) ---
        diff = base_xy_w - obs_xy_w
        dist = max(float(np.linalg.norm(diff)), 1e-6)
        h_commanded = dist - r_safe
        grad_h_world = diff / dist
        cy, sy = math.cos(yaw), math.sin(yaw)
        grad_h_base = np.array([cy * grad_h_world[0] + sy * grad_h_world[1],
                               -sy * grad_h_world[0] + cy * grad_h_world[1]], dtype=float)

        # --- nominal command in base frame ---
        u_nom_w = p_controller_world(base_xy_w, goal_xy_w, args_cli.kp, args_cli.v_max)
        u_nom_b = world_to_base(u_nom_w, yaw)

        # --- solve QP: filter the nominal through the CBF constraint ---
        rhs = args_cli.phi - args_cli.alpha * h_commanded
        v_safe_b, slack = qp.solve(grad_h_base, rhs, u_nom_b)

        # --- override the env command (vx_b, vy_b, wz_b=0) and step ---
        desired_cmd = torch.tensor([[float(v_safe_b[0]), float(v_safe_b[1]), 0.0]],
                                   device=cmd_term.vel_command_b.device,
                                   dtype=cmd_term.vel_command_b.dtype)
        cmd_term.vel_command_b[:] = desired_cmd
        # rsl-rl 5.x policies expect the full obs dict, not a tensor.
        obs_dict = env.unwrapped.observation_manager.compute()

        # Only the policy forward needs inference_mode; env.step inside it
        # marks env state tensors as inference tensors and breaks reset.
        with torch.inference_mode():
            action = policy(obs_dict)
        _, _, dones, _ = env.step(action)

        # --- read realized state AFTER step for the load-bearing safety check ---
        base_xy_after = robot.data.root_pos_w[0, :2].detach().cpu().numpy().astype(float)
        dist_after = float(np.linalg.norm(base_xy_after - obs_xy_w))
        h_realized = dist_after - r_safe
        real_lin_b = robot.data.root_lin_vel_b[0, :2].detach().cpu().numpy().astype(float)
        dist_goal = float(np.linalg.norm(goal_xy_w - base_xy_after))

        rows.append({
            "t": (step + 1) * dt,
            "base_x": base_xy_after[0], "base_y": base_xy_after[1], "yaw": yaw,
            "vx_nom_b": float(u_nom_b[0]), "vy_nom_b": float(u_nom_b[1]),
            "vx_cmd": float(v_safe_b[0]), "vy_cmd": float(v_safe_b[1]),
            "vx_real": float(real_lin_b[0]), "vy_real": float(real_lin_b[1]),
            "h_commanded": h_commanded,         # before-step
            "h_realized": h_realized,           # after-step (the load-bearing one)
            "qp_slack": slack,
            "dist_to_goal": dist_goal,
        })

        if dist_goal < args_cli.goal_tol:
            print(f"[phase0.5] goal reached at step {step+1} (t={rows[-1]['t']:.2f}s)")
            break
        if bool(dones[0].item()):
            print(f"[phase0.5] env terminated at step {step+1}")
            break

    # ----- write log + evaluate pass criteria -----
    out_path = os.path.abspath(args_cli.log_csv)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[phase0.5] wrote {len(rows)} rows -> {out_path}")

    s = evaluate_pass_criteria(rows, args_cli.goal_tol)
    print("=" * 72)
    print(f"  reached_goal               : {s['passes']['reached_goal']}")
    print(f"  final dist to goal (m)     : {s['final_dist']:.3f}")
    print(f"  min realized h (m)         : {s['min_realized_h']:+.3f}   "
          f"(pass >= 0)")
    print(f"  min commanded h (m)        : {s['min_commanded_h']:+.3f}")
    print(f"  realized h<0 step count    : {s['realized_violation_steps']}")
    print(f"  max QP slack               : {s['max_qp_slack']:.4f}  "
          f"(large slack = QP straining, late correction failing)")
    print(f"  tracking RMSE ratio        : {s['tracking_rmse_ratio']:.3f}   "
          f"(pass < 0.30)")
    print(f"  Phase 0.5 PASS             : {s['all_pass']}")
    print("=" * 72)

    env.close()
    simulation_app.close()
    sys.exit(0 if s["all_pass"] else 1)


if __name__ == "__main__":
    main()
