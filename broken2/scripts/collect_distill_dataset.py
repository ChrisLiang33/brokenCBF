#!/usr/bin/env python3
"""Wk4 student distillation — dataset collection.

Runs the frozen Teacher in a target env (Layer3_Push_A_C or whichever
wins the 4-param iteration) and saves per-step tuples for offline
supervised training of the Student adaptation module:

    proprio (N_envs, 36)     standard Go2 locomotion features
                             (base_lin_vel, base_ang_vel, projected_gravity,
                              velocity_commands, joint_pos_rel, joint_vel)
    cbf_action (N_envs, 5)   the teacher's CBF param output that step
    z_priv (N_envs, 16)      teacher's priv_encoder output (the regression
                             target the Student will be trained against)

The 48-D locomotion obs the Student will ultimately consume can be
reconstructed at training time by stacking proprio + cbf_action (5-D
replaces locomotion-policy's 12-D last_action). Window length, dropout,
TCN vs MLP are all decisions for the student training script — this
script just gives us the raw stream.

Output: a single .npz with all stored steps concatenated along the
time axis. For T=10000 steps × N=256 envs that's ~700 MB total. Bump
--num_steps as needed for a longer / more diverse dataset.

Usage on lab box (in tmux, after the 4-param iteration trains):
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/collect_distill_dataset.py \\
    --task Isaac-CBF-Go2-RMA-Layer3-Push-A-C-v0 \\
    --checkpoint logs/rsl_rl/cbf_go2_teacher_rma/<run>/model_1499.pt \\
    --num_envs 256 --num_steps 10000 \\
    --output_path datasets/distill_wk3pushac.npz \\
    --headless
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True, type=str)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument(
    "--num_steps",
    type=int,
    default=10000,
    help="Total env steps to collect. Total samples = num_envs * num_steps.",
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Path to write the .npz dataset (e.g., datasets/distill_wk3pushac.npz).",
)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[collect_distill] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[collect_distill] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def build_proprio(inner) -> torch.Tensor:
    """Construct the standard Go2 locomotion proprio vector (36-D, no
    last_action). At student deployment, the same components are exposed
    by the onboard state estimator + IMU + joint encoders.

    Layout (matches velocity_env_cfg `policy` ObsGroup minus actions):
      0–2    base_lin_vel    body frame    (3)   [needs state estimator]
      3–5    base_ang_vel    body frame    (3)   [IMU]
      6–8    projected_gravity            (3)   [IMU]
      9–11   velocity_commands             (3)   [from planner]
      12–23  joint_pos_rel                (12)  [encoders]
      24–35  joint_vel                    (12)  [encoders]
    """
    robot = inner.scene["robot"]
    data = robot.data
    base_lin_vel = data.root_lin_vel_b                       # (N, 3)
    base_ang_vel = data.root_ang_vel_b                       # (N, 3)
    projected_gravity = data.projected_gravity_b             # (N, 3)
    velocity_commands = inner.command_manager.get_command("base_velocity")  # (N, 3)
    joint_pos_rel = data.joint_pos - data.default_joint_pos  # (N, 12)
    joint_vel = data.joint_vel                                # (N, 12)
    return torch.cat(
        [base_lin_vel, base_ang_vel, projected_gravity,
         velocity_commands, joint_pos_rel, joint_vel],
        dim=-1,
    )


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    print(f"[collect_distill] task={args.task}, num_envs={args.num_envs}, "
          f"num_steps={args.num_steps}", flush=True)
    print(f"[collect_distill] checkpoint={args.checkpoint}", flush=True)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    actor = runner.alg.actor  # CbfTeacherRMAModel — exposes get_z and priv_encoder
    print(f"[collect_distill] teacher loaded.", flush=True)

    # Pre-allocate buffers — saves one big copy at the end vs append-and-stack.
    N = args.num_envs
    T = args.num_steps
    PROPRIO_DIM = 36
    CBF_DIM = 5
    Z_DIM = actor.priv_encoder.net[-2].out_features  # last Linear's out → z_priv dim

    proprio_buf = np.zeros((T, N, PROPRIO_DIM), dtype=np.float32)
    cbf_action_buf = np.zeros((T, N, CBF_DIM), dtype=np.float32)
    z_priv_buf = np.zeros((T, N, Z_DIM), dtype=np.float32)

    print(f"[collect_distill] buffers allocated: "
          f"proprio={proprio_buf.nbytes / 1e6:.1f} MB, "
          f"cbf_action={cbf_action_buf.nbytes / 1e6:.1f} MB, "
          f"z_priv={z_priv_buf.nbytes / 1e6:.1f} MB, "
          f"z_dim={Z_DIM}", flush=True)

    obs, _ = env.reset()
    log_every = max(T // 20, 1)
    t_start = time.time()

    for step in range(T):
        with torch.no_grad():
            raw_action = policy(obs)                      # (N, 5)
            # Teacher's z_priv (the regression target).
            z_priv = actor.get_z(obs)                     # (N, z_dim)
            # Standard proprio vector.
            proprio = build_proprio(inner)                # (N, 36)

            proprio_buf[step] = proprio.cpu().numpy()
            cbf_action_buf[step] = raw_action.cpu().numpy()
            z_priv_buf[step] = z_priv.cpu().numpy()

        step_out = env.step(raw_action)
        obs = step_out[0]

        if step % log_every == 0 or step == T - 1:
            dt = time.time() - t_start
            rate = (step + 1) / max(dt, 1e-6)
            eta = (T - step - 1) / max(rate, 1e-6)
            print(f"[collect_distill] step {step:>6}/{T}  "
                  f"|z|_mean={z_priv.abs().mean().item():.3f}  "
                  f"rate={rate:.1f} steps/s  eta={eta:.0f}s", flush=True)

    # Persist
    out_path = Path(args.output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        proprio=proprio_buf,
        cbf_action=cbf_action_buf,
        z_priv=z_priv_buf,
        meta=np.array({
            "task": args.task,
            "checkpoint": args.checkpoint,
            "num_envs": N,
            "num_steps": T,
            "proprio_dim": PROPRIO_DIM,
            "cbf_dim": CBF_DIM,
            "z_priv_dim": Z_DIM,
            "proprio_layout": [
                "base_lin_vel (3)", "base_ang_vel (3)",
                "projected_gravity (3)", "velocity_commands (3)",
                "joint_pos_rel (12)", "joint_vel (12)",
            ],
        }, dtype=object),
    )

    total_samples = N * T
    print("", flush=True)
    print(f"[collect_distill] wrote {total_samples:,} samples to {out_path}", flush=True)
    print(f"[collect_distill] file size: "
          f"{out_path.stat().st_size / 1e6:.1f} MB", flush=True)
    print(f"[collect_distill] elapsed: {time.time() - t_start:.1f}s", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    import os
    rc = 1
    try:
        rc = main()
    except Exception as e:
        import traceback
        print(f"[collect_distill] FATAL: {e}", flush=True)
        traceback.print_exc()
    os._exit(rc if rc is not None else 0)
