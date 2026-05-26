"""Quick action-smoothness diagnostic on a saved teacher checkpoint.

For each disturbance cell, rolls out the trained teacher and reports
step-to-step |Δphi|, |Δalpha|, and the normalized jitter the action
term caches. Tells us whether the policy's (phi, alpha) is smooth
enough for the locomotion controller below to track without falling.

Healthy ranges (normalized jitter = |Δphi|/1.0 + |Δalpha|/3.8):
   < 0.10   OK
   0.10 - 0.30   noisy (acceptable but suboptimal)
   > 0.30   JITTERY -- locomotion will struggle

Run on labbox AFTER teacher saved:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase5_action_jitter_diag.py \\
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
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy_checkpoint", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-RMA-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--eval_steps", type=int, default=500)
parser.add_argument("--eval_disturbances", type=float, nargs="+",
                    default=[0.0, 15.0, 30.0, 45.0])
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import importlib.metadata as metadata

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
from cbf_task.agents import rma_actor_critic  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    runner.load(retrieve_file_path(args_cli.policy_checkpoint))
    cbf = env_wrapped.unwrapped.action_manager._terms["cbf_param"]
    phi_w = float(cbf._phi_hi - cbf._phi_lo)
    alpha_w = float(cbf._alpha_hi - cbf._alpha_lo)

    print()
    print("=" * 92)
    print("  TEACHER ACTION-SMOOTHNESS DIAGNOSTIC")
    print(f"  phi_width={phi_w:.2f}  alpha_width={alpha_w:.2f}")
    print("=" * 92)
    print(f"  {'d (N)':>6}  {'|Δphi|_mean':>12}  {'|Δphi|_p95':>11}  "
          f"{'|Δalpha|_mean':>13}  {'|Δalpha|_p95':>12}  {'norm jitter':>11}  status")

    rows = []
    with torch.inference_mode():
        for d in args_cli.eval_disturbances:
            cbf._disturbance_force_lo = float(d)
            cbf._disturbance_force_hi = float(d)
            env_wrapped.unwrapped.reset()
            obs = env_wrapped.get_observations()
            policy = runner.get_inference_policy(device=device)
            phi_hist, alpha_hist, jitter_hist = [], [], []
            for _ in range(args_cli.eval_steps):
                action = policy(obs)
                obs, _, _, _ = env_wrapped.step(action)
                phi_hist.append(cbf.last_phi.detach().clone())
                alpha_hist.append(cbf.last_alpha.detach().clone())
                jitter_hist.append(cbf.last_action_jitter.detach().clone())
            phi_stack = torch.stack(phi_hist, dim=0)         # (T, N)
            alpha_stack = torch.stack(alpha_hist, dim=0)
            d_phi = (phi_stack[1:] - phi_stack[:-1]).abs().flatten()
            d_alpha = (alpha_stack[1:] - alpha_stack[:-1]).abs().flatten()
            jitter = torch.stack(jitter_hist, dim=0).flatten()
            j_mean = float(jitter.mean().item())
            status = ("OK" if j_mean < 0.10
                      else "noisy" if j_mean < 0.30
                      else "JITTERY")
            print(f"  {d:>6.1f}  {float(d_phi.mean()):>12.3f}  "
                  f"{float(d_phi.quantile(0.95)):>11.3f}  "
                  f"{float(d_alpha.mean()):>13.3f}  "
                  f"{float(d_alpha.quantile(0.95)):>12.3f}  "
                  f"{j_mean:>11.3f}  {status}")
            rows.append((d, float(d_phi.mean()), float(d_alpha.mean()), j_mean))
    print("=" * 92)
    overall_j = sum(r[3] for r in rows) / len(rows)
    print(f"  overall mean jitter: {overall_j:.3f}  "
          f"({'OK' if overall_j < 0.10 else 'noisy' if overall_j < 0.30 else 'JITTERY -- retrain with smoothness penalty + lower init_noise_std'})")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
