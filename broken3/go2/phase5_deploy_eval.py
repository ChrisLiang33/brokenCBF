"""Step 7 of the RMA build -- end-to-end deployment substitution test.

Load teacher + student. At every env step, replace the privileged slice
of the obs (positions 0:4) with the student's prediction from the
proprio history. Run a disturbance sweep and compare to the teacher's
per-channel sweep numbers -- does the deployed policy adapt similarly
to the privileged teacher?

This is the definitive "RMA works end-to-end" test:
- Teacher achieved alpha span 1.166 (30.7% of bound) across disturbance
- We expect deployment to recover a substantial fraction of that

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase5_deploy_eval.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --policy_checkpoint phase5_teacher_outputs/rsl_rl/model_final.pt \\
        --student_checkpoint phase5_student_outputs/student.pt \\
        --num_envs 64 --headless
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy_checkpoint", required=True)
parser.add_argument("--student_checkpoint", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-RMA-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--eval_max_steps", type=int, default=1250)
parser.add_argument("--eval_eps_per_cell", type=int, default=64)
parser.add_argument("--eval_disturbances", type=float, nargs="+",
                    default=[0.0, 15.0, 30.0, 45.0])
parser.add_argument("--out_dir", default="phase5_deploy_outputs")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import csv
import importlib.metadata as metadata

import gymnasium as gym
import torch
import torch.nn as nn
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.agents import rma_actor_critic  # noqa: F401
from cbf_task.agents.rma_actor_critic import PRIV_SLICE, PROPRIO_DIM
from cbf_task.locomotion_loader import load_locomotion_actor
from cbf_task.mdp import deployable_obs


class StudentMLP(nn.Module):
    """Mirror of the StudentMLP from the trainer."""
    def __init__(self, history_len, proprio_dim, priv_dim, hidden):
        super().__init__()
        in_dim = history_len * proprio_dim
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        layers.append(nn.Linear(last, priv_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_student(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    student = StudentMLP(
        history_len=ckpt["history_len"],
        proprio_dim=ckpt["proprio_dim"],
        priv_dim=ckpt["priv_dim"],
        hidden=ckpt["hidden"],
    ).to(device)
    student.load_state_dict(ckpt["state_dict"])
    student.eval()
    return student, ckpt["history_len"]


def eval_deployed_at_disturbance(env_wrapped, runner, student, cbf,
                                  history_len, disturbance_mag, eval_steps,
                                  n_eps, device):
    """Roll out with the student substituting priv at every step."""
    cbf._disturbance_force_lo = float(disturbance_mag)
    cbf._disturbance_force_hi = float(disturbance_mag)

    policy = runner.get_inference_policy(device=device)
    N = env_wrapped.unwrapped.num_envs
    min_h = torch.full((N,), float("inf"), device=device)
    intervention_sum = torch.zeros(N, device=device)
    env_wrapped.unwrapped.reset()
    cbf.episode_reach_any.zero_()
    cbf.episode_collide_any.zero_()
    cbf.episode_fall_any.zero_()
    obs = env_wrapped.get_observations()

    history_buf = torch.zeros((N, history_len, PROPRIO_DIM), device=device)
    phi_hist, alpha_hist = [], []
    priv_actual_hist, priv_pred_hist = [], []

    for step in range(eval_steps):
        # update history buffer with the latest proprio (computed from
        # the post-step env state, so we use the value from BEFORE we
        # stepped -- it gets shifted in BEFORE the prediction below)
        # Actually order: at the START of each iteration the env state
        # is the "result of last step's action". We compute the history
        # to feed the student, get prediction, substitute, step env.
        unwr = env_wrapped.unwrapped
        proprio_now = deployable_obs(unwr)
        history_buf = torch.roll(history_buf, shifts=-1, dims=1)
        history_buf[:, -1, :] = proprio_now

        # student inference -- replace priv slice in obs
        with torch.no_grad():
            hist_flat = history_buf.reshape(N, -1)
            priv_pred = student(hist_flat)            # (N, 4)

        # obs is a TensorDict; substitute the priv slice of the "policy"
        # group with the student's prediction
        obs_modified = obs.clone() if hasattr(obs, "clone") else obs
        # mutate the policy tensor in-place safely
        pol_tensor = obs_modified["policy"].clone()
        pol_tensor[..., PRIV_SLICE] = priv_pred
        obs_modified["policy"] = pol_tensor

        action = policy(obs_modified)
        obs, _, _, _ = env_wrapped.step(action)

        min_h = torch.minimum(min_h, cbf.last_h_realized)
        intervention_sum = intervention_sum + cbf.last_intervention
        phi_hist.append(cbf.last_phi.detach().clone())
        alpha_hist.append(cbf.last_alpha.detach().clone())
        if step % 50 == 0:
            priv_actual_hist.append(cbf._disturbance_force.detach().clone())
            priv_pred_hist.append(priv_pred[:, 0].detach().clone())

    n_eps = min(n_eps, N)
    sel = slice(0, n_eps)
    phi_all = torch.stack(phi_hist, dim=0)[:, sel].flatten()
    alpha_all = torch.stack(alpha_hist, dim=0)[:, sel].flatten()
    # disturbance prediction error stat (only the channel that matters)
    if priv_actual_hist:
        act = torch.stack(priv_actual_hist, dim=0)[:, sel].flatten()
        prd = torch.stack(priv_pred_hist, dim=0)[:, sel].flatten()
        dist_pred_err = float((prd - act).abs().mean().item())
        dist_pred_mean = float(prd.mean().item())
    else:
        dist_pred_err = float("nan")
        dist_pred_mean = float("nan")
    return {
        "phi_mean": float(phi_all.mean().item()),
        "phi_std": float(phi_all.std().item()),
        "alpha_mean": float(alpha_all.mean().item()),
        "alpha_std": float(alpha_all.std().item()),
        "collision_rate": float(cbf.episode_collide_any[sel].float().mean().item()),
        "reach_rate": float(cbf.episode_reach_any[sel].float().mean().item()),
        "fall_rate": float(cbf.episode_fall_any[sel].float().mean().item()),
        "intervention_mean": float(intervention_sum[sel].mean().item()),
        "dist_pred_mean": dist_pred_mean,
        "dist_pred_err": dist_pred_err,
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
    runner.load(retrieve_file_path(args_cli.policy_checkpoint))
    print(f"[deploy] teacher loaded -> {args_cli.policy_checkpoint}")

    student, history_len = load_student(args_cli.student_checkpoint, device)
    print(f"[deploy] student loaded -> {args_cli.student_checkpoint} "
          f"(history_len={history_len})")

    cbf = env_wrapped.unwrapped.action_manager._terms["cbf_param"]
    phi_bounds = env_cfg.actions.cbf_param.phi_bounds
    alpha_bounds = env_cfg.actions.cbf_param.alpha_bounds
    phi_width = phi_bounds[1] - phi_bounds[0]
    alpha_width = alpha_bounds[1] - alpha_bounds[0]

    rows = []
    print()
    with torch.inference_mode():
        for d in args_cli.eval_disturbances:
            print(f"[deploy] eval @ d={d:>5.1f}N ...")
            m = eval_deployed_at_disturbance(
                env_wrapped, runner, student, cbf, history_len, d,
                args_cli.eval_max_steps, args_cli.eval_eps_per_cell, device,
            )
            rows.append({"disturbance_force": float(d), **m})
            print(f"    coll={m['collision_rate']:.2f}  reach={m['reach_rate']:.2f}  "
                  f"int={m['intervention_mean']:.0f}")
            print(f"    phi={m['phi_mean']:+.3f}+-{m['phi_std']:.3f}  "
                  f"alpha={m['alpha_mean']:.2f}+-{m['alpha_std']:.2f}")
            print(f"    student disturbance pred: mean={m['dist_pred_mean']:.2f}N "
                  f"  |err|={m['dist_pred_err']:.2f}N (actual={d}N)")

    # compare adaptation to the teacher's
    phi_span = max(r["phi_mean"] for r in rows) - min(r["phi_mean"] for r in rows)
    alpha_span = max(r["alpha_mean"] for r in rows) - min(r["alpha_mean"] for r in rows)
    TEACHER_PHI_SPAN = 0.140    # from per-channel sweep
    TEACHER_ALPHA_SPAN = 1.166

    print()
    print("=" * 80)
    print("  DEPLOYMENT TEST  -- does the policy adapt when priv comes from student?")
    print("=" * 80)
    print(f"  phi   span: deployed {phi_span:.3f}  vs teacher {TEACHER_PHI_SPAN:.3f}  "
          f"({100*phi_span/max(TEACHER_PHI_SPAN, 1e-9):.0f}% retention)")
    print(f"  alpha span: deployed {alpha_span:.3f}  vs teacher {TEACHER_ALPHA_SPAN:.3f}  "
          f"({100*alpha_span/max(TEACHER_ALPHA_SPAN, 1e-9):.0f}% retention)")
    alpha_retention = alpha_span / max(TEACHER_ALPHA_SPAN, 1e-9)
    if alpha_retention >= 0.5:
        verdict = "PASS -- deployed policy retains substantial adaptation"
    elif alpha_retention >= 0.25:
        verdict = "PARTIAL -- adaptation reduced but present"
    else:
        verdict = "FAIL -- student doesn't deliver adaptation"
    print(f"  verdict: {verdict}")

    # write outputs
    with open(os.path.join(args_cli.out_dir, "phase5_deploy_eval.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args_cli.out_dir, "phase5_deploy_summary.json"),
              "w") as f:
        json.dump({"phi_span": phi_span, "alpha_span": alpha_span,
                   "teacher_phi_span": TEACHER_PHI_SPAN,
                   "teacher_alpha_span": TEACHER_ALPHA_SPAN,
                   "alpha_retention": alpha_retention,
                   "verdict": verdict, "rows": rows}, f, indent=2)
    print(f"  saved -> {args_cli.out_dir}/")

    env.close()
    simulation_app.close()
    sys.exit(0 if alpha_retention >= 0.5 else 1)


if __name__ == "__main__":
    main()
