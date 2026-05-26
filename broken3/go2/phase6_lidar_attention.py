"""Lidar attention diagnostic. Two tests:

  PART A -- OBSERVATIONAL (rollout-based):
    Roll out the trained teacher, log per-step (phi, alpha, h_realized,
    min_lidar) tuples. Compute Pearson correlation between (phi, alpha)
    and obstacle proximity. Bin by h and report mean (phi, alpha) per
    bin. Answers: under the policy's natural state distribution, does
    (phi, alpha) covary with obstacle distance?

  PART B -- INTERVENTIONAL (network probing):
    Construct a canonical obs and inject varied lidar values (one
    forward ray set to different "obstacle distances", others at max).
    Forward through the actor, record output (phi, alpha). No env step
    -- just network probing, essentially instant.
    Answers: holding everything else fixed, does the actor's output
    change with the lidar input? (sharper test than correlation -- if
    A is zero but B shows response, the policy uses lidar but its
    state distribution didn't visit varied h enough to expose it.)

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase6_lidar_attention.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --policy_checkpoint phase5_teacher_outputs/rsl_rl/model_final.pt \\
        --num_envs 256 --headless
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy_checkpoint", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-RMA-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rollout_steps", type=int, default=500)
parser.add_argument("--disturbance", type=float, default=0.0)
parser.add_argument("--out_dir", default="phase6_lidar_attention_outputs")
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
from cbf_task.agents.rma_actor_critic import (
    PRIV_SLICE, PROPRIO_SLICE, PREV_ACT_SLICE,
    LIDAR_PREV_SLICE, LIDAR_SLICE, EXPECTED_OBS_DIM,
)
from cbf_task.locomotion_loader import load_locomotion_actor


def _decode_action(action: torch.Tensor, cbf) -> tuple[torch.Tensor, torch.Tensor]:
    """[-1, 1]^2 -> (phi, alpha) in their bound spaces."""
    a = action.clamp(-1.0, 1.0)
    phi = cbf._phi_lo + (a[..., 0] + 1.0) * 0.5 * (cbf._phi_hi - cbf._phi_lo)
    alpha = cbf._alpha_lo + (a[..., 1] + 1.0) * 0.5 * (cbf._alpha_hi - cbf._alpha_lo)
    return phi, alpha


def part_a_observational(env_wrapped, runner, cbf, n_steps, disturbance, device):
    """Roll out trained teacher; collect (phi, alpha, h, min_lidar).
    Also returns a real obs sample (env 0 at step n_steps//2) to use
    as the canonical baseline for Part B's interventional probe.
    """
    from cbf_task.mdp import lidar_obs
    cbf._disturbance_force_lo = float(disturbance)
    cbf._disturbance_force_hi = float(disturbance)
    env_wrapped.unwrapped.reset()
    obs = env_wrapped.get_observations()
    policy = runner.get_inference_policy(device=device)

    phi_hist, alpha_hist, h_hist, ml_hist = [], [], [], []
    canonical_obs = None
    mid_step = n_steps // 2
    for step in range(n_steps):
        action = policy(obs)
        obs, _, _, _ = env_wrapped.step(action)
        phi_hist.append(cbf.last_phi.detach().clone())
        alpha_hist.append(cbf.last_alpha.detach().clone())
        h_hist.append(cbf.last_h_realized.detach().clone())
        ml_hist.append(lidar_obs(env_wrapped.unwrapped).min(dim=-1).values
                       .detach().clone())
        if step == mid_step:
            # snapshot env-0 obs as canonical baseline for Part B
            pol = obs["policy"] if hasattr(obs, "keys") else obs
            canonical_obs = pol[0:1].detach().clone()        # (1, 90)

    phi = torch.stack(phi_hist, dim=0).flatten().cpu().numpy()
    alpha = torch.stack(alpha_hist, dim=0).flatten().cpu().numpy()
    h = torch.stack(h_hist, dim=0).flatten().cpu().numpy()
    ml = torch.stack(ml_hist, dim=0).flatten().cpu().numpy()
    return {"phi": phi, "alpha": alpha, "h": h, "min_lidar": ml,
            "canonical_obs": canonical_obs}


def _pearson(a, b):
    """Pearson correlation, robust to constant inputs."""
    a = np.asarray(a); b = np.asarray(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def part_b_interventional(actor, cbf, canonical_obs, device):
    """Probe the actor with synthetic obs: vary only the lidar slice (one
    forward ray to "obstacle distance D", others at max=20m).

    `canonical_obs` is a real (1, 90) obs sampled from Part A's rollout
    -- the actor saw inputs in this distribution during training, so
    the response we get is in-distribution. We then modify ONLY the
    lidar slice and call actor.forward through the full pipeline
    (obs_normalizer -> mlp -> distribution deterministic output) so
    normalization is applied correctly.
    """
    from tensordict import TensorDict

    # 72 rays at 5 deg starting -180; bearing 0 deg -> idx 36
    n_rays = LIDAR_SLICE.stop - LIDAR_SLICE.start
    FORWARD_IDX = n_rays // 2
    LIDAR_OFFSET = LIDAR_SLICE.start
    MAX_RANGE = 20.0
    distances = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, MAX_RANGE]

    rows = []
    for d in distances:
        obs = canonical_obs.clone()
        # Set BOTH lidar frames (t-1 and t) to the same configuration:
        # all rays at MAX, one forward ray at distance d. Zero temporal
        # change isolates the spatial response of the CNN.
        obs[..., LIDAR_PREV_SLICE] = MAX_RANGE
        obs[..., LIDAR_PREV_SLICE.start + FORWARD_IDX] = d
        obs[..., LIDAR_SLICE] = MAX_RANGE
        obs[..., LIDAR_OFFSET + FORWARD_IDX] = d

        td = TensorDict({"policy": obs}, batch_size=[obs.shape[0]])
        with torch.no_grad():
            # actor.forward(td) goes through obs_normalizer -> mlp ->
            # distribution.deterministic_output -- so we get the
            # POST-distribution action mean in [-1, 1] space.
            action = actor.forward(td)
            phi, alpha = _decode_action(action, cbf)
        rows.append({
            "obstacle_distance_m": d,
            "phi": float(phi.mean().item()),
            "alpha": float(alpha.mean().item()),
        })
    return rows


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
    actor = runner.alg.actor
    cbf = env_wrapped.unwrapped.action_manager._terms["cbf_param"]

    # ============================================================
    # PART A -- observational
    # ============================================================
    print()
    print("=" * 88)
    print(f"  PART A  --  observational rollout @ d={args_cli.disturbance}N "
          f"({args_cli.rollout_steps} steps x {args_cli.num_envs} envs)")
    print("=" * 88)
    with torch.inference_mode():
        a_data = part_a_observational(env_wrapped, runner, cbf,
                                       args_cli.rollout_steps,
                                       args_cli.disturbance, device)

    corr_phi_h = _pearson(a_data["phi"], a_data["h"])
    corr_phi_ml = _pearson(a_data["phi"], a_data["min_lidar"])
    corr_alpha_h = _pearson(a_data["alpha"], a_data["h"])
    corr_alpha_ml = _pearson(a_data["alpha"], a_data["min_lidar"])

    print(f"  pearson(phi,    h_realized) = {corr_phi_h:+.3f}")
    print(f"  pearson(phi,    min_lidar)  = {corr_phi_ml:+.3f}")
    print(f"  pearson(alpha,  h_realized) = {corr_alpha_h:+.3f}")
    print(f"  pearson(alpha,  min_lidar)  = {corr_alpha_ml:+.3f}")

    # bin by h
    h = a_data["h"]
    valid = (h > -0.5) & (h < 6.0)
    h_v = h[valid]
    phi_v = a_data["phi"][valid]
    alpha_v = a_data["alpha"][valid]
    bins = np.linspace(h_v.min(), h_v.max(), 6)
    print(f"\n  binned by h_realized:")
    print(f"  {'h_bin':>14}  {'count':>7}  {'phi_mean':>10}  {'alpha_mean':>11}")
    bin_rows = []
    for i in range(len(bins) - 1):
        mask = (h_v >= bins[i]) & (h_v < bins[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        phi_mu = float(phi_v[mask].mean())
        alpha_mu = float(alpha_v[mask].mean())
        bin_rows.append({"h_low": float(bins[i]), "h_high": float(bins[i + 1]),
                         "count": n, "phi_mean": phi_mu, "alpha_mean": alpha_mu})
        print(f"  [{bins[i]:>+5.2f}, {bins[i+1]:>+5.2f}]  {n:>7d}  "
              f"{phi_mu:>+10.3f}  {alpha_mu:>+11.3f}")

    # ============================================================
    # PART B -- interventional
    # ============================================================
    print()
    print("=" * 88)
    print("  PART B  --  interventional: synthetic obs, vary forward-ray distance")
    print("=" * 88)
    with torch.inference_mode():
        b_rows = part_b_interventional(actor, cbf, a_data["canonical_obs"], device)
    print(f"  {'obstacle_dist (m)':>18}  {'phi':>8}  {'alpha':>8}")
    for r in b_rows:
        print(f"  {r['obstacle_distance_m']:>18.2f}  "
              f"{r['phi']:>+8.3f}  {r['alpha']:>+8.3f}")
    phi_b_span = max(r["phi"] for r in b_rows) - min(r["phi"] for r in b_rows)
    alpha_b_span = max(r["alpha"] for r in b_rows) - min(r["alpha"] for r in b_rows)
    print(f"\n  phi span across distances:   {phi_b_span:.3f}  "
          f"({100*phi_b_span/(cbf._phi_hi - cbf._phi_lo):.1f}% of bound)")
    print(f"  alpha span across distances: {alpha_b_span:.3f}  "
          f"({100*alpha_b_span/(cbf._alpha_hi - cbf._alpha_lo):.1f}% of bound)")

    # ---- verdict ----
    print()
    print("=" * 88)
    print("  LIDAR ATTENTION VERDICT")
    print("=" * 88)
    obs_strong = abs(corr_phi_h) > 0.3 or abs(corr_alpha_h) > 0.3
    int_strong = (phi_b_span > 0.10 or alpha_b_span > 0.5)
    if obs_strong and int_strong:
        verdict = "PASS -- policy attends to lidar (both observational + interventional)"
    elif int_strong and not obs_strong:
        verdict = ("PARTIAL -- interventional test fires, but observational corr is "
                   "weak. Policy CAN use lidar but its trained state distribution "
                   "doesn't visit varied h enough to expose it. Suggests lidar is "
                   "wired but under-used in the current scenario.")
    elif obs_strong and not int_strong:
        verdict = ("PARTIAL -- observational corr exists but synthetic probe is flat. "
                   "Possibly the correlation is mediated by another channel (e.g., "
                   "deployable proprio reacting to nearby obstacle).")
    else:
        verdict = "FAIL -- policy is essentially ignoring lidar"
    print(f"  {verdict}")
    print("=" * 88)

    # save
    with open(os.path.join(args_cli.out_dir, "phase6_lidar_attention.json"), "w") as f:
        json.dump({
            "observational": {
                "corr_phi_h": corr_phi_h, "corr_phi_min_lidar": corr_phi_ml,
                "corr_alpha_h": corr_alpha_h, "corr_alpha_min_lidar": corr_alpha_ml,
                "bins": bin_rows,
            },
            "interventional": {
                "rows": b_rows,
                "phi_span": phi_b_span, "alpha_span": alpha_b_span,
            },
            "verdict": verdict,
        }, f, indent=2)
    with open(os.path.join(args_cli.out_dir, "phase6_lidar_attention_raw.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi", "alpha", "h_realized", "min_lidar"])
        for i in range(len(a_data["phi"])):
            w.writerow([float(a_data["phi"][i]), float(a_data["alpha"][i]),
                        float(a_data["h"][i]), float(a_data["min_lidar"][i])])
    print(f"  saved -> {args_cli.out_dir}/")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
