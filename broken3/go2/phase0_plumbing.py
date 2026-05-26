"""Phase 0 -- plumbing only.

Drives a stock Isaac Lab Go2 velocity-tracking policy to a fixed waypoint
using a P controller in the base frame. NO obstacle, NO CBF, NO learning.

The point is to surface, on the cheapest possible setup, the one failure
mode the 2D MVP could not see: the gap between the velocity the safety
filter wants the robot to execute and the velocity the locomotion policy
can actually produce. If that gap is large here, every later phase
depends on closing it -- and finding that out now (1 env, no CBF, hours
of work) is much cheaper than at Phase 2 (full training, days).

Run on a Linux + NVIDIA box with Isaac Lab installed. See README.md.
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
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
                    help="Isaac Lab task ID. Must be a Go2 velocity-tracking variant.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel envs. Phase 0 is 1.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to an rsl_rl checkpoint (.pt). Mutually exclusive with "
                         "--use_pretrained_checkpoint.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true",
                    help="Use the Isaac Lab published checkpoint for this task.")
parser.add_argument("--goal", type=float, nargs=2, default=[5.0, 0.0],
                    metavar=("X", "Y"), help="Goal waypoint in world frame (meters).")
parser.add_argument("--kp", type=float, default=1.0,
                    help="P-controller gain on position error.")
parser.add_argument("--v_max", type=float, default=1.0,
                    help="Linear velocity cap (m/s), 2-norm.")
parser.add_argument("--goal_tol", type=float, default=0.4,
                    help="Distance below which the goal is considered reached (meters).")
parser.add_argument("--max_time", type=float, default=15.0,
                    help="Wall-time budget for the run (sim seconds).")
parser.add_argument("--log_csv", type=str, default="phase0_log.csv",
                    help="Output CSV path for per-step telemetry.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# After this point, sim is running and we can import the rest.
# ---------------------------------------------------------------------------
import csv
import math

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401  (registers tasks)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except Exception:
    get_published_pretrained_checkpoint = None


def yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rotation about z) from (w, x, y, z) quaternions. Shape (N,)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def p_controller_command_b(
    base_xy_w: torch.Tensor,    # (N, 2)
    yaw_w: torch.Tensor,        # (N,)
    goal_xy_w: torch.Tensor,    # (2,)
    kp: float,
    v_max: float,
) -> torch.Tensor:
    """Goal-tracking P controller. Returns (N, 3): (vx_b, vy_b, wz). wz is 0 for Phase 0."""
    err_w = goal_xy_w[None, :] - base_xy_w        # (N, 2) in world frame
    cy, sy = torch.cos(yaw_w), torch.sin(yaw_w)
    # rotate world-frame error by -yaw into base frame
    err_b_x = cy * err_w[:, 0] + sy * err_w[:, 1]
    err_b_y = -sy * err_w[:, 0] + cy * err_w[:, 1]
    cmd_xy = torch.stack([kp * err_b_x, kp * err_b_y], dim=-1)
    norm = torch.linalg.norm(cmd_xy, dim=-1, keepdim=True).clamp(min=1e-9)
    scale = torch.where(norm > v_max, v_max / norm, torch.ones_like(norm))
    cmd_xy = cmd_xy * scale
    wz = torch.zeros_like(cmd_xy[:, :1])
    return torch.cat([cmd_xy, wz], dim=-1)


def evaluate_pass_criteria(rows: list[dict], goal_tol: float) -> dict:
    """Compute Phase 0 pass/fail from logged rows. See README.md."""
    if not rows:
        return {"crashed_or_empty": True}

    reached = any(r["dist_to_goal"] < goal_tol for r in rows)
    final_dist = rows[-1]["dist_to_goal"]

    # tracking RMSE over the last 80% of the trajectory (skip transient)
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
        "plumbing": True,                         # got here => no crash
        "reached_goal": bool(reached),
        "tracking_ok": tracking_rmse_ratio < 0.30,
    }
    return {
        "passes": passes,
        "all_pass": all(passes.values()),
        "final_dist": final_dist,
        "tracking_rmse_ratio": tracking_rmse_ratio,
        "n_steps": len(rows),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args_cli.seed)

    # ----- env config -----
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = None

    # ----- agent config (needed for the OnPolicyRunner constructor) -----
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    # rsl-rl version compatibility -- mirrors play.py
    import importlib.metadata as metadata
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.device = device

    # ----- checkpoint resolution -----
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            raise RuntimeError("isaaclab_rl pretrained_checkpoint helper not available; "
                               "pass --checkpoint instead.")
        train_task_name = args_cli.task.replace("-Play", "")
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise RuntimeError(f"No published pretrained checkpoint for {train_task_name}; "
                               "train one with scripts/reinforcement_learning/rsl_rl/train.py "
                               "or pass --checkpoint.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        raise SystemExit("Pass --checkpoint <path> or --use_pretrained_checkpoint.")
    print(f"[phase0] checkpoint -> {resume_path}")

    # ----- env -----
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ----- policy -----
    if agent_cfg.class_name != "OnPolicyRunner":
        raise RuntimeError(f"Unexpected agent runner class: {agent_cfg.class_name}. "
                           "Phase 0 was written against stock OnPolicyRunner Go2 cfg.")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ----- grab references we need to drive the env -----
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    goal_xy_w = torch.tensor(args_cli.goal, device=device, dtype=torch.float32)
    dt = env.unwrapped.step_dt
    max_steps = int(round(args_cli.max_time / dt))

    print(f"[phase0] dt={dt:.4f}s  max_steps={max_steps}  goal={tuple(args_cli.goal)}  "
          f"kp={args_cli.kp}  v_max={args_cli.v_max}")

    obs = env.get_observations()
    rows: list[dict] = []

    for step in range(max_steps):
        # --- compute our P-controller command and overwrite the env's command ---
        base_xy_w = robot.data.root_pos_w[:, :2]
        yaw_w = yaw_from_quat_wxyz(robot.data.root_quat_w)
        desired_cmd_b = p_controller_command_b(
            base_xy_w, yaw_w, goal_xy_w, args_cli.kp, args_cli.v_max,
        )
        # in-place to keep the tensor pointer the obs term reads from
        cmd_term.vel_command_b[:] = desired_cmd_b

        # Recompute obs so the policy sees OUR command, not whatever the
        # default command source put there at the previous step boundary.
        # rsl-rl 5.x policies expect the full obs dict and look up
        # obs["policy"] themselves, so pass the dict, not the tensor.
        obs_dict = env.unwrapped.observation_manager.compute()

        # Only the policy forward needs inference_mode; env.step inside it
        # marks env state tensors as inference tensors and breaks reset.
        with torch.inference_mode():
            action = policy(obs_dict)
        _, _, dones, _ = env.step(action)

        # --- log a representative env=0 row ---
        cmd_now = cmd_term.vel_command_b[0].detach().cpu().tolist()
        real_lin_b = robot.data.root_lin_vel_b[0, :2].detach().cpu().tolist()
        base_now = robot.data.root_pos_w[0, :2].detach().cpu().tolist()
        yaw_now = yaw_w[0].item()
        dist = math.hypot(args_cli.goal[0] - base_now[0], args_cli.goal[1] - base_now[1])

        rows.append({
            "t": (step + 1) * dt,
            "base_x": base_now[0], "base_y": base_now[1], "yaw": yaw_now,
            "vx_cmd": cmd_now[0], "vy_cmd": cmd_now[1], "wz_cmd": cmd_now[2],
            "vx_real": real_lin_b[0], "vy_real": real_lin_b[1],
            "dist_to_goal": dist,
        })

        if dist < args_cli.goal_tol:
            print(f"[phase0] goal reached at step {step+1} (t={rows[-1]['t']:.2f}s)")
            break
        if bool(dones[0].item()):
            print(f"[phase0] env terminated at step {step+1}")
            break

    # ----- write log + evaluate pass criteria -----
    out_path = os.path.abspath(args_cli.log_csv)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[phase0] wrote {len(rows)} rows -> {out_path}")

    summary = evaluate_pass_criteria(rows, args_cli.goal_tol)
    print("=" * 64)
    print(f"  reached_goal           : {summary['passes']['reached_goal']}")
    print(f"  final dist to goal (m) : {summary['final_dist']:.3f}")
    print(f"  tracking RMSE ratio    : {summary['tracking_rmse_ratio']:.3f}   "
          f"(pass < 0.30)")
    print(f"  Phase 0 PASS           : {summary['all_pass']}")
    print("=" * 64)

    env.close()
    simulation_app.close()
    # exit code reflects pass/fail so CI / shell pipelines can react
    sys.exit(0 if summary["all_pass"] else 1)


if __name__ == "__main__":
    main()
