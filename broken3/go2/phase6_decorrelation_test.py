"""Decorrelation test: is teacher alpha modulation driven by lidar
(real adaptation), or by dist_to_goal (a slalom-specific proxy)?

The slalom train env has obstacles aligned with the start->goal line.
That structurally couples (dist_to_goal) and (min_lidar): when robot
is close to goal, it's also past the obstacles, so min_lidar is large.
A policy that conditions alpha on dist_to_goal alone would APPEAR
adaptive in slalom but fail on any layout where obstacles aren't on
the path.

This test runs the teacher in a DECORRELATED env (single obstacle
uniformly jittered across +/-4m x and y, goal fixed at (7,0)) and
runs three checks:

  1. Marginal Pearson:
       corr(alpha, min_lidar)   <- if dominant, lidar is doing work
       corr(alpha, dist_to_goal)<- if dominant, proxy is doing work

  2. Partial linear regression of alpha on BOTH:
       alpha = a + b*min_lidar + c*dist_to_goal
     Compare |b| vs |c|. Larger means dominant cause.

  3. 2D bin table: alpha mean across (min_lidar_bin, dist_to_goal_bin).
     If alpha varies along the lidar axis WITHIN a fixed goal column
     -> real lidar use.  If alpha only varies along the goal axis ->
     proxy.

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase6_decorrelation_test.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --policy_checkpoint phase6_slalom_intervention0_teacher_outputs/rsl_rl/model_final.pt \\
        --task Isaac-CBF-Adaptive-Go2-Decorr-v0 \\
        --num_envs 256 --rollout_steps 600 --headless
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy_checkpoint", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-Decorr-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rollout_steps", type=int, default=600)
parser.add_argument("--disturbance", type=float, default=0.0)
parser.add_argument("--out_dir", default="phase6_decorrelation_test_outputs")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import importlib.metadata as metadata

import gymnasium as gym
import numpy as np
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
from cbf_task.mdp import lidar_obs


def _pearson(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _standardize(x):
    return (x - x.mean()) / (x.std() + 1e-9)


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
    cbf = env_wrapped.unwrapped.action_manager._terms["cbf_param"]

    cbf._disturbance_force_lo = float(args_cli.disturbance)
    cbf._disturbance_force_hi = float(args_cli.disturbance)

    env_wrapped.unwrapped.reset()
    obs = env_wrapped.get_observations()
    policy = runner.get_inference_policy(device=device)

    # ============================================================
    # ROLLOUT  --  collect per-step (alpha, phi, min_lidar, dist_to_goal)
    # ============================================================
    print(f"[decorr] rolling {args_cli.rollout_steps} steps x {args_cli.num_envs} envs ...")
    phi_hist, alpha_hist, ml_hist, gd_hist = [], [], [], []
    with torch.inference_mode():
        for _ in range(args_cli.rollout_steps):
            action = policy(obs)
            obs, _, _, _ = env_wrapped.step(action)
            phi_hist.append(cbf.last_phi.detach().clone())
            alpha_hist.append(cbf.last_alpha.detach().clone())
            ml_hist.append(lidar_obs(env_wrapped.unwrapped).min(dim=-1).values
                            .detach().clone())
            gd_hist.append(cbf.last_dist_to_goal.detach().clone())

    phi = torch.stack(phi_hist).flatten().cpu().numpy()
    alpha = torch.stack(alpha_hist).flatten().cpu().numpy()
    ml = torch.stack(ml_hist).flatten().cpu().numpy()
    gd = torch.stack(gd_hist).flatten().cpu().numpy()

    # filter to "in-play" samples (robot not at goal, obstacle in lidar range)
    mask = (gd > 0.4) & (ml < 10.0)
    phi, alpha, ml, gd = phi[mask], alpha[mask], ml[mask], gd[mask]
    print(f"[decorr] {len(alpha)} valid samples after filtering")

    # ============================================================
    # 1.  MARGINAL PEARSON
    # ============================================================
    print()
    print("=" * 90)
    print("  1. MARGINAL PEARSON  --  which signal does alpha track?")
    print("=" * 90)
    c_a_ml = _pearson(alpha, ml)
    c_a_gd = _pearson(alpha, gd)
    c_p_ml = _pearson(phi, ml)
    c_p_gd = _pearson(phi, gd)
    # decorrelation sanity check: are the two predictors themselves
    # uncorrelated in this env?  (They were ~ -1 in slalom.)
    c_ml_gd = _pearson(ml, gd)
    print(f"  corr(alpha, min_lidar)    = {c_a_ml:+.3f}")
    print(f"  corr(alpha, dist_to_goal) = {c_a_gd:+.3f}")
    print(f"  corr(phi,   min_lidar)    = {c_p_ml:+.3f}")
    print(f"  corr(phi,   dist_to_goal) = {c_p_gd:+.3f}")
    print(f"  --- predictor decorrelation check ---")
    print(f"  corr(min_lidar, dist_to_goal) = {c_ml_gd:+.3f}  "
          f"(want close to 0 in this env)")

    # ============================================================
    # 2.  PARTIAL LINEAR REGRESSION
    # ============================================================
    print()
    print("=" * 90)
    print("  2. PARTIAL LINEAR REGRESSION  --  alpha = a + b*ml + c*gd")
    print("     (standardized -- |b| and |c| are comparable)")
    print("=" * 90)
    ml_z, gd_z, alpha_z = _standardize(ml), _standardize(gd), _standardize(alpha)
    # least squares solve
    X = np.stack([ml_z, gd_z, np.ones_like(ml_z)], axis=1)         # (N, 3)
    beta, *_ = np.linalg.lstsq(X, alpha_z, rcond=None)
    b_ml, b_gd, b_const = float(beta[0]), float(beta[1]), float(beta[2])
    print(f"  alpha (z) = {b_const:+.4f} + ({b_ml:+.4f})*min_lidar(z) "
          f"+ ({b_gd:+.4f})*dist_to_goal(z)")
    print()
    print(f"  |b_lidar| = {abs(b_ml):.4f}")
    print(f"  |b_goal|  = {abs(b_gd):.4f}")
    if abs(b_ml) > 1e-9 or abs(b_gd) > 1e-9:
        ratio = abs(b_ml) / (abs(b_ml) + abs(b_gd))
        print(f"  lidar fraction of total weight: {100*ratio:.1f}%")

    # ============================================================
    # 3.  2D BIN TABLE  --  alpha across (min_lidar_bin, dist_to_goal_bin)
    # ============================================================
    print()
    print("=" * 90)
    print("  3. 2D BIN TABLE  --  alpha mean per (lidar_bin, goal_bin)")
    print("     reads horizontally: does alpha change along the LIDAR axis")
    print("     within each fixed-goal column?  if yes -> real lidar use.")
    print("=" * 90)
    ml_edges = np.quantile(ml, [0, 0.33, 0.67, 1.0])
    gd_edges = np.quantile(gd, [0, 0.33, 0.67, 1.0])
    # header
    print(f"  {'min_lidar_bin':>20}    {'gd_low':>9} {'gd_mid':>9} {'gd_high':>9}")
    rows_tbl = []
    for i in range(3):
        ml_lo, ml_hi = ml_edges[i], ml_edges[i + 1]
        row_str = f"  [{ml_lo:>+5.2f}, {ml_hi:>+5.2f}]    "
        row_data = {"ml_low": float(ml_lo), "ml_high": float(ml_hi), "by_gd": []}
        for j in range(3):
            gd_lo, gd_hi = gd_edges[j], gd_edges[j + 1]
            m = (ml >= ml_lo) & (ml < ml_hi) & (gd >= gd_lo) & (gd < gd_hi)
            if m.sum() == 0:
                row_str += f"{'-':>9} "
                row_data["by_gd"].append(None)
            else:
                row_str += f"{alpha[m].mean():>+9.3f} "
                row_data["by_gd"].append({"gd_low": float(gd_lo),
                                          "gd_high": float(gd_hi),
                                          "n": int(m.sum()),
                                          "alpha_mean": float(alpha[m].mean())})
        print(row_str)
        rows_tbl.append(row_data)

    # ============================================================
    # VERDICT
    # ============================================================
    print()
    print("=" * 90)
    print("  DECORRELATION VERDICT")
    print("=" * 90)
    abs_decorr = abs(c_ml_gd) < 0.3
    if not abs_decorr:
        print(f"  WARNING: min_lidar and dist_to_goal still correlated "
              f"({c_ml_gd:+.3f}). Decorrelation env may not be wide enough.")
    if abs(b_ml) > 1e-9 or abs(b_gd) > 1e-9:
        ratio = abs(b_ml) / (abs(b_ml) + abs(b_gd))
        if ratio > 0.7 and abs(c_a_ml) > 0.2:
            verdict = (f"PASS -- alpha is primarily driven by lidar "
                       f"({100*ratio:.0f}% of weight on lidar in partial "
                       f"regression; marginal corr {c_a_ml:+.3f}). "
                       f"Modulation is real adaptive use of lidar.")
        elif ratio < 0.3 and abs(c_a_gd) > 0.2:
            verdict = (f"FAIL -- alpha is primarily driven by dist_to_goal "
                       f"({100*(1-ratio):.0f}% of weight on goal in partial "
                       f"regression; lidar corr {c_a_ml:+.3f}). The slalom "
                       f"modulation was a goal-proxy crutch, not lidar use. "
                       f"Redesign training scenario (this Decorr env is a "
                       f"good candidate) or add aux obstacle-prediction loss.")
        else:
            verdict = (f"MIXED -- both signals contribute "
                       f"(lidar {100*ratio:.0f}%, goal {100*(1-ratio):.0f}%; "
                       f"lidar corr {c_a_ml:+.3f}, goal corr {c_a_gd:+.3f}). "
                       f"Lidar is being used but not as the dominant cue. "
                       f"Annealing should still work; consider adding lidar-"
                       f"only aux loss to strengthen lidar dependence.")
    else:
        verdict = "INSUFFICIENT -- both coefficients are zero. Policy is constant."
    print(f"  {verdict}")
    print("=" * 90)

    # save
    with open(os.path.join(args_cli.out_dir, "phase6_decorrelation_test.json"), "w") as f:
        json.dump({
            "marginal_pearson": {
                "corr_alpha_min_lidar": c_a_ml,
                "corr_alpha_dist_to_goal": c_a_gd,
                "corr_phi_min_lidar": c_p_ml,
                "corr_phi_dist_to_goal": c_p_gd,
                "corr_predictors": c_ml_gd,
            },
            "partial_regression": {
                "b_lidar": b_ml, "b_goal": b_gd, "b_const": b_const,
                "lidar_weight_fraction": (abs(b_ml) / (abs(b_ml) + abs(b_gd))
                                           if abs(b_ml) + abs(b_gd) > 0 else None),
            },
            "bin_table": rows_tbl,
            "n_samples": int(len(alpha)),
            "verdict": verdict,
        }, f, indent=2)
    with open(os.path.join(args_cli.out_dir, "phase6_decorrelation_raw.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi", "alpha", "min_lidar", "dist_to_goal"])
        for i in range(len(alpha)):
            w.writerow([float(phi[i]), float(alpha[i]),
                        float(ml[i]), float(gd[i])])
    print(f"  saved -> {args_cli.out_dir}/")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
