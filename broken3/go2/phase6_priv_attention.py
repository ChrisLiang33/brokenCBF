"""Privileged-channel attention diagnostic. Mirror of `phase6_lidar_attention.py`
but for priv obs channels. Tests whether the trained teacher actually uses
each priv channel to modulate (phi, alpha).

For each priv channel:
  - PART A (observational): correlate (phi, alpha) with the channel
    value over a natural rollout.
  - PART B (interventional): pin the channel at each value in its sweep
    grid (other channels at nominal), forward through the actor, record
    output (phi, alpha) span.

Output per channel:
  - Part A Pearson correlation
  - Part B (phi, alpha) span across the sweep
  - PASS / FAIL based on Part B span thresholds

This is the actual gate for "does the teacher use this priv channel?".
If a channel fails, the student has nothing to recover from history (RMA
breaks). Drop the failing channel and retrain.

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase6_priv_attention.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --policy_checkpoint phase6_unified_teacher_outputs/rsl_rl/model_final.pt \\
        --task Isaac-CBF-Adaptive-Go2-Unified-v0 \\
        --num_envs 256 --headless
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
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-Unified-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rollout_steps", type=int, default=500)
parser.add_argument("--out_dir", default="phase6_priv_attention_outputs")
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
from tensordict import TensorDict

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.agents import rma_actor_critic  # noqa: F401
from cbf_task.agents.rma_actor_critic import PRIV_SLICE
from cbf_task.locomotion_loader import load_locomotion_actor


# Per-channel index (matches mdp.priv_obs layout):
#   0 disturbance, 1 friction, 2 base_mass_delta, 3 motor_strength,
#   4 actuation_noise_std, 5 com_offset, 6 v_max
PRIV_CHANNELS = [
    # (name, idx, sweep_values, nominal, theoretical_axis)
    ("disturbance",      0, [0.0, 7.5, 15.0, 22.5, 30.0],     0.0,   "α/φ"),
    ("friction",         1, [0.3, 0.45, 0.6, 0.75, 1.0],      0.6,   "φ"),
    ("base_mass_delta",  2, [-3.0, -1.5, 0.0, 1.5, 3.0],      0.0,   "α"),
    ("motor_strength",   3, [0.7, 0.85, 1.0, 1.15, 1.3],      1.0,   "φ"),
    ("actuation_noise",  4, [0.0, 0.0125, 0.025, 0.0375, 0.05], 0.0, "φ"),
    ("com_offset",       5, [-0.05, -0.025, 0.0, 0.025, 0.05], 0.0, "α"),
    ("v_max",            6, [1.0, 1.25, 1.5, 1.75, 2.0],      1.3,   "α"),
]

# Pass thresholds (chosen to match lidar_attention's "10% bound" rule)
PHI_BOUND = 1.0
ALPHA_BOUND = 3.8
PHI_PASS_SPAN = 0.10 * PHI_BOUND
ALPHA_PASS_SPAN = 0.10 * ALPHA_BOUND


def _decode_action(action: torch.Tensor, cbf):
    a = action.clamp(-1.0, 1.0)
    phi = cbf._phi_lo + (a[..., 0] + 1.0) * 0.5 * (cbf._phi_hi - cbf._phi_lo)
    alpha = cbf._alpha_lo + (a[..., 1] + 1.0) * 0.5 * (cbf._alpha_hi - cbf._alpha_lo)
    return phi, alpha


def _pearson(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def part_a_observational(env_wrapped, runner, cbf, n_steps, device):
    """Rollout. Per step, record (phi, alpha) AND every priv channel value."""
    env_wrapped.unwrapped.reset()
    obs = env_wrapped.get_observations()
    policy = runner.get_inference_policy(device=device)
    phi_hist, alpha_hist = [], []
    priv_hist = [[] for _ in PRIV_CHANNELS]
    canonical_obs = None
    mid = n_steps // 2
    for step in range(n_steps):
        action = policy(obs)
        obs, _, _, _ = env_wrapped.step(action)
        phi_hist.append(cbf.last_phi.detach().clone())
        alpha_hist.append(cbf.last_alpha.detach().clone())
        for i, (_, idx, _, _, _) in enumerate(PRIV_CHANNELS):
            # read directly from action term -- these are the *true* values
            # the priv_obs would have used this step
            attr = ["_disturbance_force", "_friction_coef", "_base_mass_delta",
                    "_motor_strength", "_actuation_noise_std", "_com_offset",
                    "_v_max"][idx]
            priv_hist[i].append(getattr(cbf, attr).detach().clone())
        if step == mid:
            pol = obs["policy"] if hasattr(obs, "keys") else obs
            canonical_obs = pol[0:1].detach().clone()      # (1, 200)
    phi = torch.stack(phi_hist).flatten().cpu().numpy()
    alpha = torch.stack(alpha_hist).flatten().cpu().numpy()
    privs = [torch.stack(h).flatten().cpu().numpy() for h in priv_hist]
    return {"phi": phi, "alpha": alpha, "privs": privs,
            "canonical_obs": canonical_obs}


def part_b_interventional(actor, cbf, canonical_obs, device):
    """For each priv channel, sweep its value across its grid (others at
    nominal), forward through actor, record (phi, alpha) span.
    """
    results = []
    for name, idx, values, nominal, axis in PRIV_CHANNELS:
        rows = []
        for v in values:
            obs = canonical_obs.clone()
            # set entire priv slice to nominals first
            for _, jj, _, j_nom, _ in PRIV_CHANNELS:
                obs[..., PRIV_SLICE.start + jj] = j_nom
            # then override this channel
            obs[..., PRIV_SLICE.start + idx] = v
            td = TensorDict({"policy": obs}, batch_size=[obs.shape[0]])
            with torch.no_grad():
                action = actor.forward(td)
                phi, alpha = _decode_action(action, cbf)
            rows.append({"value": v,
                         "phi": float(phi.mean().item()),
                         "alpha": float(alpha.mean().item())})
        phi_span = max(r["phi"] for r in rows) - min(r["phi"] for r in rows)
        alpha_span = max(r["alpha"] for r in rows) - min(r["alpha"] for r in rows)
        results.append({"name": name, "idx": idx, "axis": axis,
                        "rows": rows,
                        "phi_span": phi_span, "alpha_span": alpha_span})
    return results


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
    # PART A
    # ============================================================
    print()
    print("=" * 96)
    print(f"  PART A -- observational rollout ({args_cli.rollout_steps} steps x {args_cli.num_envs} envs)")
    print("=" * 96)
    with torch.inference_mode():
        a_data = part_a_observational(env_wrapped, runner, cbf,
                                       args_cli.rollout_steps, device)

    print(f"  {'channel':>18}  {'idx':>3}  {'axis':>4}  {'corr(phi,c)':>11}  {'corr(alpha,c)':>13}")
    a_rows = []
    for i, (name, idx, _, _, axis) in enumerate(PRIV_CHANNELS):
        cv = a_data["privs"][i]
        cp = _pearson(a_data["phi"], cv)
        ca = _pearson(a_data["alpha"], cv)
        print(f"  {name:>18}  {idx:>3}  {axis:>4}  {cp:>+11.3f}  {ca:>+13.3f}")
        a_rows.append({"name": name, "idx": idx, "axis": axis,
                       "corr_phi": cp, "corr_alpha": ca})

    # ============================================================
    # PART B
    # ============================================================
    print()
    print("=" * 96)
    print("  PART B -- interventional: vary each priv channel, others at nominal")
    print("=" * 96)
    with torch.inference_mode():
        b_results = part_b_interventional(actor, cbf, a_data["canonical_obs"], device)

    for r in b_results:
        print(f"\n  {r['name']:>18}  (idx {r['idx']}, axis {r['axis']})")
        print(f"    {'value':>10}  {'phi':>8}  {'alpha':>8}")
        for row in r["rows"]:
            print(f"    {row['value']:>10.4f}  {row['phi']:>+8.3f}  {row['alpha']:>+8.3f}")
        phi_pct = 100 * r["phi_span"] / PHI_BOUND
        alpha_pct = 100 * r["alpha_span"] / ALPHA_BOUND
        print(f"    --> phi span {r['phi_span']:.3f} ({phi_pct:.1f}%)  "
              f"alpha span {r['alpha_span']:.3f} ({alpha_pct:.1f}%)")

    # ============================================================
    # VERDICT
    # ============================================================
    print()
    print("=" * 96)
    print("  PRIV ATTENTION VERDICT  (per-channel)")
    print("=" * 96)
    print(f"  {'channel':>18}  {'phi_span':>10}  {'alpha_span':>12}  verdict")
    overall_pass = 0
    for r in b_results:
        used_phi = r["phi_span"] >= PHI_PASS_SPAN
        used_alpha = r["alpha_span"] >= ALPHA_PASS_SPAN
        if used_phi and used_alpha:
            verdict = "BOTH (used by both phi and alpha)"
        elif used_phi:
            verdict = "phi only"
        elif used_alpha:
            verdict = "alpha only"
        else:
            verdict = "DEAD (policy ignores this channel)"
        if used_phi or used_alpha:
            overall_pass += 1
        print(f"  {r['name']:>18}  {r['phi_span']:>10.3f}  {r['alpha_span']:>12.3f}  {verdict}")
    print(f"\n  {overall_pass}/{len(PRIV_CHANNELS)} priv channels are actively used by the policy.")
    if overall_pass < 2:
        print("  WARNING: very few priv channels used; RMA student will have nothing"
              " meaningful to distill from history. Consider scenario redesign.")
    print("=" * 96)

    # save
    with open(os.path.join(args_cli.out_dir, "phase6_priv_attention.json"), "w") as f:
        json.dump({
            "observational": a_rows,
            "interventional": [{**r, "rows": r["rows"]} for r in b_results],
        }, f, indent=2)
    print(f"  saved -> {args_cli.out_dir}/")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
