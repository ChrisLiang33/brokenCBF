"""Phase 1.5 -- fingerprint sweep.

The pre-Phase-2 gate. Question: with the deployable observation (proprio
+ commanded-velocity + last action -- the locomotion's own 48-dim obs
vector), can a learned encoder *perceive* the disturbance magnitude?

Phase 0.6 proved the env has signal: optimal fixed φ moves with
disturbance. This script asks the next-level question: is that signal
visible from observations the policy can actually consume at deployment?
If yes -> Option 1 (direct training with proprio+action history) for
Phase 2. If no -> Option 2 (RMA-style teacher-student).

Mechanics:
- Sweep one disturbance magnitude per invocation (same shape as Phase 0.6).
- CBF parameters are FIXED at a known-safe (φ, α) so the locomotion is
  operating normally across all disturbance levels (no collisions, no
  navigation failure that would muddy the proprio signal).
- Per step per env, dump the deployable obs vector to a buffer.
- Save buffer + the disturbance magnitude label as an .npz.

A bash loop drives multiple disturbance levels:

    for D in 0 15 30 45; do
        ./isaaclab.sh -p phase1_5_fingerprint_sweep.py \\
            --checkpoint /path/to/loco.pt --disturbance_force $D \\
            --out_npz ~/.../phase1_5_d${D}.npz --headless
    done

Then `analyze_fingerprint.py` loads them and tests whether a small
regressor can predict disturbance level from observations.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True,
                    help="Frozen Go2 locomotion .pt checkpoint.")
parser.add_argument("--out_npz", required=True,
                    help="Output .npz path; contains per-step obs + label.")
parser.add_argument("--disturbance_force", type=float, required=True)
parser.add_argument("--disturbance_resample", type=int, default=50)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--n_episodes", type=int, default=4,
                    help="Sequential episodes per env after first reset. "
                         "Total per-disturbance samples ≈ "
                         "n_episodes × num_envs × ~250 steps.")
parser.add_argument("--episode_max_steps", type=int, default=400)
# Use a fixed safe (phi, alpha) from Phase 1's grid -- 0% coll, 100% reach.
parser.add_argument("--phi", type=float, default=0.3)
parser.add_argument("--alpha", type=float, default=2.5)
# scenario defaults match Phase 1
parser.add_argument("--goal", type=float, nargs=2, default=[6.0, 0.0])
parser.add_argument("--obstacle", type=float, nargs=2, default=[2.5, 0.3])
parser.add_argument("--obstacle_radius", type=float, default=0.9)
parser.add_argument("--v_max", type=float, default=1.3)
parser.add_argument("--friction_mu", type=float, default=0.6)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401  -- registers Isaac-CBF-Adaptive-Go2-v0
from cbf_task.locomotion_loader import load_locomotion_actor


TASK = "Isaac-CBF-Adaptive-Go2-v0"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # 1) locomotion actor (needed for the env's action term)
    ckpt = retrieve_file_path(args_cli.checkpoint)
    print(f"[phase1.5] locomotion -> {ckpt}")
    locomotion_actor = load_locomotion_actor(ckpt, device)

    # 2) env cfg (canonical task, fixed safe CBF, sweep-set disturbance)
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = None
    at = env_cfg.actions.cbf_param
    at.locomotion_policy_obj = locomotion_actor
    at.goal_xy = tuple(args_cli.goal)
    at.obstacle_xy = tuple(args_cli.obstacle)
    at.obstacle_radius = float(args_cli.obstacle_radius)
    at.disturbance_force = float(args_cli.disturbance_force)
    at.disturbance_resample = int(args_cli.disturbance_resample)
    at.v_max = float(args_cli.v_max)
    # Pin friction
    try:
        pm = env_cfg.events.physics_material
        pm.params["static_friction_range"] = (args_cli.friction_mu, args_cli.friction_mu)
        pm.params["dynamic_friction_range"] = (args_cli.friction_mu, args_cli.friction_mu)
    except AttributeError:
        pass

    env = gym.make(TASK, cfg=env_cfg)
    cbf_term = env.unwrapped.action_manager._terms["cbf_param"]
    robot = env.unwrapped.scene["robot"]
    N = env.unwrapped.num_envs

    # 3) prepare fixed action (normalized [-1, 1])
    phi_lo, phi_hi = at.phi_bounds
    alpha_lo, alpha_hi = at.alpha_bounds
    a0 = 2.0 * (args_cli.phi - phi_lo) / (phi_hi - phi_lo) - 1.0
    a1 = 2.0 * (args_cli.alpha - alpha_lo) / (alpha_hi - alpha_lo) - 1.0
    action = torch.tensor([[a0, a1]] * N, device=device, dtype=torch.float32)

    print(f"[phase1.5] sweep d={args_cli.disturbance_force:.1f}N  "
          f"phi={args_cli.phi}  alpha={args_cli.alpha}  "
          f"n_envs={N}  n_eps={args_cli.n_episodes}")

    # 4) drive the env, log per-step deployable obs
    OBS_DIM = 48
    max_steps = args_cli.n_episodes * args_cli.episode_max_steps
    obs_buf = np.zeros((max_steps, N, OBS_DIM), dtype=np.float32)
    step_count = 0
    t_start = time.time()

    # Use env.reset() (the wrapper-respecting one), NOT env.unwrapped.reset.
    # The gym wrappers track a reset_needed flag that .unwrapped bypasses.
    # No inference_mode here -- this is a fresh env without prior training,
    # so there are no poisoned tensors to worry about.
    env.reset()
    cbf_term.episode_reach_any.zero_()
    cbf_term.episode_collide_any.zero_()

    for _ in range(max_steps):
        # Step the env (action term consumes (phi, alpha), runs CBF +
        # locomotion + disturbance, applies joint targets).
        env.step(action)

        # Now read the deployable obs vector PER ENV.
        base_lin_b = robot.data.root_lin_vel_b               # (N, 3)
        base_ang_b = robot.data.root_ang_vel_b               # (N, 3)
        gravity_b = robot.data.projected_gravity_b           # (N, 3)
        joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel
        u_safe = cbf_term.last_u_safe                        # (N, 2)
        cmd3 = torch.cat([u_safe, torch.zeros((N, 1), device=device)], dim=-1)
        last_jt = cbf_term._processed_actions - robot.data.default_joint_pos
        last_jt = last_jt / max(cbf_term._loco_action_scale, 1e-9)

        obs = torch.cat([
            base_lin_b, base_ang_b, gravity_b,
            cmd3,
            joint_pos_rel, joint_vel,
            last_jt,
        ], dim=-1)  # (N, 48)

        obs_buf[step_count] = obs.detach().cpu().numpy()
        step_count += 1

    # 5) save
    out_path = os.path.abspath(args_cli.out_npz)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        obs=obs_buf[:step_count],                   # (T, N, 48)
        disturbance=np.float32(args_cli.disturbance_force),
        phi=np.float32(args_cli.phi),
        alpha=np.float32(args_cli.alpha),
        friction_mu=np.float32(args_cli.friction_mu),
    )
    el = time.time() - t_start
    print(f"[phase1.5] wrote {step_count} x {N} x {OBS_DIM} -> {out_path}  "
          f"({el:.0f}s)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
