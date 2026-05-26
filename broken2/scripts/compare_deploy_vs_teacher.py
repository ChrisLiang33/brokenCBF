"""Sim-equivalence test: run a rollout, compare per-step actions from the
teacher (rsl_rl, ground-truth z_env via priv_encoder) vs the deploy model
(standalone PyTorch, ẑ_env via student adapter from proprio history).

The only architectural difference is z_env source:
  Teacher : z_env = priv_encoder(priv_hidden_groundtruth)
  Deploy  : ẑ_env = student(history of (proprio, prev_action))

If the student does its job, the two actions should track closely.
This script quantifies the closed-loop equivalence with:
  - per-step MSE between teacher_action and deploy_action
  - per-component (α, φ, a, b, c) correlation
  - aggregate stats

Usage on lab box:
  cd ~/Desktop/safety-go2/IsaacLab
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/compare_deploy_vs_teacher.py \\
    --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-1-v0 \\
    --teacher logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_18-25-01/model_2499.pt \\
    --student ~/Desktop/safety-go2/checkpoints/student_v13_1.pt \\
    --num_envs 1 --rollout_steps 300 --headless
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--teacher", required=True, type=str)
parser.add_argument("--student", required=True, type=str)
parser.add_argument("--num_envs", type=int, default=1,
                    help="Deploy model is single-env; runs over envs sequentially.")
parser.add_argument("--rollout_steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print(f"[compare] starting AppLauncher...", flush=True)
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[compare] AppLauncher ready in {time.time()-t0:.1f}s", flush=True)

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Deploy model.
sys.path.insert(0, str(Path("~/Desktop/safety-go2/deploy").expanduser()))
from cbf_deploy_model import CbfDeployModel


PRIV_HIDDEN_DIM = 14
PRIV_TOTAL_DIM = 33
GRID_FLAT_DIM = 8192


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if hasattr(args, "device") and args.device else "cuda:0"

    print(f"[compare] task={args.task}, num_envs={args.num_envs}, "
          f"steps={args.rollout_steps}", flush=True)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.teacher)
    policy = runner.get_inference_policy(device=device)
    print(f"[compare] teacher loaded.", flush=True)

    deploy = CbfDeployModel(args.teacher, args.student, device=device)
    print(f"[compare] deploy model ready.", flush=True)

    obs, _ = env.reset()
    N = args.num_envs
    S = args.rollout_steps
    A = 5  # action_dim

    teacher_actions = np.zeros((S, N, A), dtype=np.float32)
    deploy_actions = np.zeros((S, N, A), dtype=np.float32)
    teacher_zenv = np.zeros((S, N, deploy.z_priv_dim), dtype=np.float32)
    deploy_zenv = np.zeros((S, N, deploy.z_priv_dim), dtype=np.float32)

    # The deploy model is single-env. We loop sequentially over envs at each
    # step, but reset the history once at start of rollout.
    for e in range(N):
        deploy.reset_history()

    print(f"[compare] running {S} steps...", flush=True)
    for s in range(S):
        with torch.no_grad():
            # Teacher action (via rsl_rl inference policy).
            t_action = policy(obs)  # (N, A)
            teacher_actions[s] = t_action.cpu().numpy()

            # Capture teacher z_env (forward through priv_encoder for the
            # ground-truth hidden slice).
            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            priv_hidden = obs_tensor[:, :PRIV_HIDDEN_DIM]
            priv_observable = obs_tensor[:, PRIV_HIDDEN_DIM:PRIV_TOTAL_DIM]
            grid_flat = obs_tensor[:, PRIV_TOTAL_DIM:PRIV_TOTAL_DIM + GRID_FLAT_DIM]

            # Locate teacher's priv_encoder for z_env capture.
            actor = None
            for path in [("alg", "actor_critic", "actor"),
                         ("alg", "actor_critic"), ("alg", "actor")]:
                m = runner
                try:
                    for p in path:
                        m = getattr(m, p)
                    if hasattr(m, "mlp"):
                        actor = m; break
                except AttributeError:
                    continue
            t_zenv = actor.mlp[0](priv_hidden)
            teacher_zenv[s] = t_zenv.cpu().numpy()

            # Deploy model — run per env (sequential).
            for e in range(N):
                p_np = priv_observable[e].cpu().numpy()
                g_np = grid_flat[e].reshape(2, 64, 64).cpu().numpy()
                out = deploy.infer(p_np, g_np)
                deploy_actions[s, e] = out["raw"]
                # Capture deploy ẑ_env directly from the student.
                # _history is updated inside infer(); call student again.
                with torch.no_grad():
                    dz = deploy.student(deploy._history).cpu().numpy()
                deploy_zenv[s, e] = dz[0]

        # Step env using teacher's action (the "real" rollout).
        step_out = env.step(t_action)
        obs = step_out[0]

        if s % 50 == 0 or s == S - 1:
            cur_act_mse = float(((teacher_actions[s] - deploy_actions[s]) ** 2).mean())
            cur_z_mse = float(((teacher_zenv[s] - deploy_zenv[s]) ** 2).mean())
            print(f"  step {s:>3}/{S}  act_MSE={cur_act_mse:.4f}  "
                  f"z_env_MSE={cur_z_mse:.4f}", flush=True)

    # ─── Aggregate stats ───
    flat_t_a = teacher_actions.reshape(-1, A)
    flat_d_a = deploy_actions.reshape(-1, A)
    flat_t_z = teacher_zenv.reshape(-1, deploy.z_priv_dim)
    flat_d_z = deploy_zenv.reshape(-1, deploy.z_priv_dim)

    print("\n=== Per-component action stats ===")
    names = ["α_raw", "φ_raw", "a_raw", "b_raw", "c_raw"]
    print(f'{"comp":>8s}  {"teacher_μ":>10s}  {"deploy_μ":>10s}  '
          f'{"|Δ|μ":>10s}  {"pearson":>8s}')
    for i, name in enumerate(names):
        t_mean = flat_t_a[:, i].mean()
        d_mean = flat_d_a[:, i].mean()
        abs_diff = float(np.abs(flat_t_a[:, i] - flat_d_a[:, i]).mean())
        # Pearson r
        t_c = flat_t_a[:, i] - t_mean
        d_c = flat_d_a[:, i] - d_mean
        denom = (np.sqrt((t_c**2).sum()) * np.sqrt((d_c**2).sum())) + 1e-12
        r = float((t_c * d_c).sum() / denom)
        print(f'{name:>8s}  {t_mean:>10.3f}  {d_mean:>10.3f}  '
              f'{abs_diff:>10.4f}  {r:>8.3f}')

    # Overall action MSE.
    action_mse = float(((flat_t_a - flat_d_a) ** 2).mean())
    z_mse = float(((flat_t_z - flat_d_z) ** 2).mean())
    print(f"\n  overall action MSE: {action_mse:.5f}")
    print(f"  overall z_env  MSE: {z_mse:.5f}")

    # Per-dim z_env R².
    print(f"\n=== z_env distillation quality (per dim) ===")
    print(f"{'dim':>4s}  {'teacher_var':>11s}  {'mse':>8s}  {'R²':>7s}")
    n_good = 0
    for d in range(deploy.z_priv_dim):
        t_var = float(flat_t_z[:, d].var())
        mse = float(((flat_t_z[:, d] - flat_d_z[:, d]) ** 2).mean())
        r2 = 1.0 - mse / max(t_var, 1e-12)
        if r2 > 0.5: n_good += 1
        print(f"{d:>4d}  {t_var:>11.4f}  {mse:>8.4f}  {r2:>7.3f}")
    print(f"\n  {n_good}/{deploy.z_priv_dim} dims with R² > 0.5")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
