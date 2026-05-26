"""Dump teacher rollouts for student distillation.

Records per-step (proprio_observable, prev_action, z_env_target) into a
single .npz file. Designed for V13's two-stream architecture where:
  - priv_obs[:14]  → priv_encoder → z_env (8-D, the distillation target)
  - priv_obs[14:33] → normalize → proprio (19-D, observable, student has at deploy)
  - grid → grid_encoder → z_grid (student also has at deploy from real LiDAR)

The student (Wk4) learns to predict ẑ_env from a temporal history of
(proprio, prev_action). This script generates the (X, y) supervised
dataset for that regression.

Usage on lab box:
  cd ~/Desktop/safety-go2/IsaacLab
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/dump_teacher_rollout_for_student.py \\
    --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-v0 \\
    --checkpoint <path-to-v13-final.pt> \\
    --num_envs 256 --rollout_steps 1000 \\
    --priv_dim 33 --priv_hidden_dim 14 \\
    --output dump_v13_for_student.npz --headless
"""
from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True, type=str)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rollout_steps", type=int, default=1000)
parser.add_argument("--priv_dim", type=int, default=33)
parser.add_argument("--priv_hidden_dim", type=int, default=14,
                    help="V13: 14 (first 14 dims of priv = hidden env). The "
                         "remaining priv_dim - priv_hidden_dim dims are "
                         "observable proprio.")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[dump] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[dump] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    print(f"[dump] task={args.task}, num_envs={args.num_envs}, "
          f"steps={args.rollout_steps}, priv_dim={args.priv_dim}, "
          f"priv_hidden_dim={args.priv_hidden_dim}", flush=True)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)

    # Locate the actor module so we can call priv_encoder directly.
    actor = None
    for path in [("alg", "actor_critic", "actor"),
                 ("alg", "actor_critic"),
                 ("alg", "actor"),
                 ("policy",), ("actor",)]:
        m = runner
        try:
            for p in path:
                m = getattr(m, p)
            if hasattr(m, "mlp"):
                actor = m
                break
        except AttributeError:
            continue
    if actor is None:
        raise RuntimeError("could not locate actor.mlp on runner")
    inner_mlp = actor.mlp
    priv_encoder = inner_mlp[0]  # _PrivEncoder
    print(f"[dump] actor.mlp: {type(inner_mlp).__name__}", flush=True)
    print(f"[dump] priv_encoder: {type(priv_encoder).__name__} "
          f"(input_dim={getattr(priv_encoder, 'input_dim', '?')}, "
          f"z_priv_dim={getattr(priv_encoder, 'z_priv_dim', '?')})", flush=True)

    obs, _ = env.reset()

    N = args.num_envs
    S = args.rollout_steps
    P = args.priv_dim
    H = args.priv_hidden_dim
    obs_proprio_dim = P - H  # 33 - 14 = 19 for V13

    # Determine action dim and z_env dim from a forward pass.
    with torch.no_grad():
        raw_action_0 = policy(obs)
    A = raw_action_0.shape[-1]

    with torch.no_grad():
        obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
        priv_hidden_0 = obs_tensor[:, :H]
        z0 = priv_encoder(priv_hidden_0)
    Z = z0.shape[-1]
    print(f"[dump] action_dim={A}, z_env_dim={Z}, proprio_dim={obs_proprio_dim}", flush=True)

    # Buffers — store everything per env per step. Student training will
    # slide a temporal window over these.
    proprio_history = np.zeros((S, N, obs_proprio_dim), dtype=np.float32)
    action_history = np.zeros((S, N, A), dtype=np.float32)
    z_env_history = np.zeros((S, N, Z), dtype=np.float32)
    # Useful for sanity checks / linear probe — store hidden priv ground truth too.
    priv_hidden_gt_history = np.zeros((S, N, H), dtype=np.float32)
    # Episode reset flags (so student training can mask history across resets).
    reset_history = np.zeros((S, N), dtype=np.bool_)

    prev_action = torch.zeros(N, A, device=device)

    for step in range(S):
        with torch.no_grad():
            raw_action = policy(obs)

            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            priv_hidden = obs_tensor[:, :H]
            priv_proprio = obs_tensor[:, H:P]
            z_env = priv_encoder(priv_hidden)

            proprio_history[step] = priv_proprio.cpu().numpy()
            action_history[step] = prev_action.cpu().numpy()
            z_env_history[step] = z_env.cpu().numpy()
            priv_hidden_gt_history[step] = priv_hidden.cpu().numpy()

            prev_action = raw_action.detach()

        step_out = env.step(raw_action)
        obs = step_out[0]

        # Mark resets if available.
        try:
            dones = step_out[2] | step_out[3]  # terminated | truncated
            reset_history[step] = dones.cpu().numpy()
        except Exception:
            pass

        if step % 100 == 0 or step == S - 1:
            print(f"[dump] step {step:>4}/{S}  z_env mean={z_env.mean().item():+.3f} "
                  f"std={z_env.std().item():.3f}", flush=True)

    print(f"[dump] saving to {args.output}...", flush=True)
    np.savez_compressed(
        args.output,
        proprio_history=proprio_history,
        action_history=action_history,
        z_env_history=z_env_history,
        priv_hidden_gt_history=priv_hidden_gt_history,
        reset_history=reset_history,
        task=args.task,
        checkpoint=args.checkpoint,
        priv_dim=P,
        priv_hidden_dim=H,
        z_env_dim=Z,
        action_dim=A,
        num_envs=N,
        rollout_steps=S,
    )
    print(f"[dump] saved. shapes: proprio={proprio_history.shape}, "
          f"action={action_history.shape}, z_env={z_env_history.shape}", flush=True)

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
