#!/usr/bin/env python3
"""Dump a rollout's trajectory + grid + CBF state to a .npz, so we can
render a matplotlib video locally (bypassing Isaac Sim's broken camera
renderer on the lab box).

Mirrors the diagnose_grad_sensitivity rollout pattern (which is known to
work — uses no camera/RT). Difference: no gradients, just data capture.

Usage on lab box:
  cd ~/Desktop/safety-go2/IsaacLab
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/dump_rollout.py \\
    --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V8-v0 \\
    --checkpoint logs/rsl_rl/cbf_go2_teacher_rma/2026-05-19_14-18-46/model_2499.pt \\
    --num_envs 4 --rollout_steps 800 --priv_dim 33 \\
    --alpha_min 0.5 --alpha_max 3.0 \\
    --output dump_v8_trainmatch.npz --headless
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True, type=str)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--rollout_steps", type=int, default=800)
parser.add_argument("--priv_dim", type=int, required=True)
parser.add_argument("--alpha_min", type=float, default=0.5)
parser.add_argument("--alpha_max", type=float, default=3.0)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--force_corridor", action="store_true",
                    help="Override scene_corridor_prob to 1.0 — every env gets "
                         "a tight-corridor scene. Useful for visualizing the "
                         "specific corridor failure mode.")
parser.add_argument("--force_open", action="store_true",
                    help="Override scene_corridor_prob to 0.0 — every env gets "
                         "an open scene (random obstacle placement).")
parser.add_argument("--force_static_obstacles", action="store_true",
                    help="Override randomize_obstacles_position.params.max_speed "
                         "to 0.0 — obstacles stop drifting. Use with --force_corridor "
                         "for a clean static-corridor visualization.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[dump_rollout] starting AppLauncher (no cameras)...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[dump_rollout] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


PHI_MIN = 0.0
PHI_MAX = 5.0
GRID_H = 64
GRID_W = 64


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    print(f"[dump_rollout] task={args.task}, num_envs={args.num_envs}, "
          f"rollout_steps={args.rollout_steps}, priv_dim={args.priv_dim}",
          flush=True)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)

    # Optional scene-type override — applied BEFORE gym.make so the event
    # picks it up at first _reset_idx. Use to deterministically dump
    # corridor scenes (otherwise scene_corridor_prob in PHIWIN_TIGHTCOR
    # ancestry is 0.50 — half corridor, half open, random per-env).
    if args.force_corridor or args.force_open:
        prob = 1.0 if args.force_corridor else 0.0
        try:
            env_cfg.events.randomize_obstacles_position.params[
                "scene_corridor_prob"
            ] = prob
            tag = "FORCED CORRIDOR" if args.force_corridor else "FORCED OPEN"
            print(f"[dump_rollout] {tag} (scene_corridor_prob = {prob})")
        except (AttributeError, KeyError) as e:
            print(f"[dump_rollout] could not override scene_corridor_prob: {e}")

    if args.force_static_obstacles:
        # Disable kinematic obstacle motion. Useful for clean corridor viz.
        try:
            p = env_cfg.events.randomize_obstacles_position.params
            p["max_speed"] = 0.0
            if "max_speed_range" in p:
                p["max_speed_range"] = (0.0, 0.0)
            if "moving_prob" in p:
                p["moving_prob"] = 0.0
            print(f"[dump_rollout] FORCED STATIC OBSTACLES "
                  f"(max_speed=0, moving_prob=0)")
        except (AttributeError, KeyError) as e:
            print(f"[dump_rollout] could not override obstacle motion: {e}")

    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    print(f"[dump_rollout] runner loaded.", flush=True)

    obs, _ = env.reset()

    N = args.num_envs
    S = args.rollout_steps
    P = args.priv_dim

    # Pre-allocate buffers
    pos_history = np.zeros((S, N, 3), dtype=np.float32)
    yaw_history = np.zeros((S, N), dtype=np.float32)
    grid_history = np.zeros((S, N, 2, GRID_H, GRID_W), dtype=np.float32)
    alpha_history = np.zeros((S, N), dtype=np.float32)
    phi_history = np.zeros((S, N), dtype=np.float32)
    h_history = np.zeros((S, N), dtype=np.float32)
    deflection_history = np.zeros((S, N), dtype=np.float32)
    qp_active_history = np.zeros((S, N), dtype=np.bool_)
    cmd_vel_history = np.zeros((S, N, 3), dtype=np.float32)
    # Active goal position (world frame, xy). Captured from the cmd planner's
    # smooth_goal_pos_w / goal_pos_w / mpc_goal_pos_w (whichever exists).
    # Filled with NaN if no goal is exposed (e.g. uniform-only planner).
    goal_history = np.full((S, N, 2), np.nan, dtype=np.float32)
    # Per-step obstacle positions — populated below after we know K.
    # Shape (S, K, N, 3). 2026-05-22: added because the env runs
    # _advance_obstacle_motion() every step → static t=0 dump misses motion.
    obstacle_history = None  # allocated once K is known

    # Robot asset
    robot = inner.scene["robot"]

    # 2026-05-22 fix v2: obstacles are moving (env calls _advance_obstacle_motion
    # every step). Capture positions PER STEP, not just at t=0.
    # Names from OBSTACLE_NAMES module-level constant (env attribute lookup
    # was the original bug that left every dump with 0 obstacles).
    from isaaclab_tasks.manager_based.safety.cbf_go2.cbf_go2_env_cfg import (
        OBSTACLE_NAMES,
    )
    # Filter to only those actually in the scene
    valid_obs_names = []
    for obs_name in OBSTACLE_NAMES:
        try:
            _ = inner.scene[obs_name].data.root_pos_w
            valid_obs_names.append(obs_name)
        except (KeyError, AttributeError):
            pass
    K = len(valid_obs_names)
    print(f"[dump_rollout] tracking {K} obstacles across {N} envs per step")
    if K > 0:
        obstacle_history = np.zeros((S, K, N, 3), dtype=np.float32)
    else:
        obstacle_history = np.zeros((S, 0, N, 3), dtype=np.float32)

    # Per-env scene type label (set by randomize_obstacles_position at reset).
    # True = tight-corridor scene; False = open (random) scene.
    if hasattr(inner, "cbf_scene_is_corridor"):
        is_corridor = inner.cbf_scene_is_corridor.detach().cpu().numpy().astype(bool)
    else:
        is_corridor = np.zeros(N, dtype=bool)
    print(f"[dump_rollout] scene types: {int(is_corridor.sum())} corridor, "
          f"{int((~is_corridor).sum())} open (out of {N})")

    a_lo, a_hi = args.alpha_min, args.alpha_max

    for step in range(S):
        with torch.no_grad():
            raw_action = policy(obs)
            alpha_phys = a_lo + (torch.tanh(raw_action[:, 0]) + 1.0) * 0.5 * (a_hi - a_lo)
            phi_phys = PHI_MIN + (torch.tanh(raw_action[:, 1]) + 1.0) * 0.5 * (PHI_MAX - PHI_MIN)
            alpha_history[step] = alpha_phys.cpu().numpy()
            phi_history[step] = phi_phys.cpu().numpy()

            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            grid_flat = obs_tensor[:, P:]
            grid_history[step] = grid_flat.reshape(N, 2, GRID_H, GRID_W).cpu().numpy()

            # Robot pose
            pos = robot.data.root_pos_w  # (N, 3)
            pos_history[step] = pos.cpu().numpy()

            # Per-step obstacle positions (moving obstacles tracked here).
            for k, obs_name in enumerate(valid_obs_names):
                p_obs = inner.scene[obs_name].data.root_pos_w  # (N, 3)
                obstacle_history[step, k] = p_obs.detach().cpu().numpy()
            quat = robot.data.root_quat_w  # (N, 4) wxyz
            # Yaw from quaternion (assume wxyz convention)
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            yaw_history[step] = yaw.cpu().numpy()

            # Commanded velocity (planner output)
            try:
                cmd = inner.command_manager.get_command("base_velocity")  # (N, 3)
                cmd_vel_history[step] = cmd.cpu().numpy()
            except Exception:
                pass

            # Active goal position (xy, world frame). Try the three goal
            # tensors the MultiPlannerCommand maintains, in priority order:
            # smooth_goal (V15-Navigation), legacy goal, MPC goal.
            try:
                bv_term = inner.command_manager.get_term("base_velocity")
                for attr in ("smooth_goal_pos_w", "goal_pos_w", "mpc_goal_pos_w"):
                    g = getattr(bv_term, attr, None)
                    if g is not None and g.shape[0] == N:
                        goal_history[step] = g.detach().cpu().numpy()
                        break
            except Exception:
                pass

        step_out = env.step(raw_action)
        obs = step_out[0]

        # CBF state from env after step
        if hasattr(inner, "last_h_for_obs"):
            h_history[step] = inner.last_h_for_obs.detach().cpu().numpy()
        # 2026-05-21: env doesn't expose `last_deflection_for_obs` or
        # `last_qp_active_for_obs` directly — derive both from the env's
        # actual cached attributes:
        #   deflection = ||u_des_xy − u_safe_xy||
        #   qp_active  = (last_slack_for_obs < 0)   (slack<0 means QP fired)
        if hasattr(inner, "last_u_des") and hasattr(inner, "last_u_safe"):
            du = (inner.last_u_des[:, :2] - inner.last_u_safe[:, :2])
            deflection_history[step] = du.norm(dim=-1).detach().cpu().numpy()
        if hasattr(inner, "last_slack_for_obs"):
            qp_active_history[step] = (
                inner.last_slack_for_obs.detach() < 0
            ).cpu().numpy().astype(bool)

        if step % 50 == 0 or step == S - 1:
            print(f"[dump_rollout] step {step:>4}/{S}  "
                  f"α={alpha_history[step].mean():.2f}  "
                  f"φ={phi_history[step].mean():.2f}  "
                  f"h={h_history[step].mean():+.3f}",
                  flush=True)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        task=np.array(args.task),
        checkpoint=np.array(args.checkpoint),
        n_envs=np.array(N), n_steps=np.array(S),
        pos_history=pos_history,
        yaw_history=yaw_history,
        grid_history=grid_history,
        alpha_history=alpha_history,
        phi_history=phi_history,
        h_history=h_history,
        deflection_history=deflection_history,
        qp_active_history=qp_active_history,
        cmd_vel_history=cmd_vel_history,
        # Per-step (S, K, N, 3); animate_rollout reads this preferentially.
        obstacle_history=obstacle_history,
        # Legacy (K, N, 3) static field — t=0 snapshot for backward compat.
        obstacle_positions=obstacle_history[0] if obstacle_history.size > 0
                            else np.zeros((0, N, 3), dtype=np.float32),
        # Per-env scene type — True = corridor, False = open scene.
        is_corridor=is_corridor,
        # Per-step active goal (world frame, xy). NaN where no goal exposed.
        goal_history=goal_history,
    )
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[dump_rollout] DONE. Saved {file_size_mb:.1f} MB to {out_path}",
          flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception as e:
        import traceback
        print(f"[dump_rollout] FATAL: {e}", flush=True)
        traceback.print_exc()
    os._exit(rc if rc is not None else 0)
