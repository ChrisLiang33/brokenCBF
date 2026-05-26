"""Phase 2 -- state-conditional (φ, α) policy with deployable obs.

Trains the outer policy on `Isaac-CBF-Adaptive-Go2-Phase2-v0`:
- Observation: 20-step window of the 48-dim deployable obs (Phase 1.5 showed
  R²=0.955 for inferring disturbance from this).
- Disturbance: per-episode random magnitude in [0, 45] N -- the OOD signal
  the policy must learn to perceive and adapt to.
- Reward shape unchanged from Phase 1 (progress, intervention=-0.1,
  collision=-100, goal=+50).

After training, evaluates:
1. The learned policy across a sweep of test disturbance levels.
2. A grid of fixed (φ, α) constants at the same disturbance levels --
   the fixed-parameter Pareto baseline.

Pass criterion: the learned policy is non-dominated -- i.e. at each test
disturbance level, no fixed (φ, α) cell achieves both lower intervention
AND higher reach AND zero collisions.

Run via Isaac Lab:
    ./isaaclab.sh -p ~/Desktop/cbf_rl_mvp/go2/phase2_train.py \\
        --checkpoint /path/to/loco.pt --num_envs 64 \\
        --max_iterations 500 --headless
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-Phase2-v0",
                    help="Task ID. Use Isaac-CBF-Adaptive-Go2-Phase3-v0 for the "
                         "multi-obstacle scenario.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=500,
                    help="PPO outer-loop iterations. Set to 0 to skip training "
                         "(useful for gate-checking a new scenario before "
                         "committing GPU time).")
parser.add_argument("--policy_checkpoint", default=None,
                    help="If set, load this trained outer-policy .pt and skip "
                         "training (max_iterations is forced to 0). Use this "
                         "to re-run the eval + (phi, alpha) diagnostic on an "
                         "existing checkpoint without retraining.")
parser.add_argument("--out_dir", default="phase2_outputs")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--goal", type=float, nargs=2, default=[6.0, 0.0])
parser.add_argument("--obstacle", type=float, nargs=2, default=[2.5, 0.3])
parser.add_argument("--obstacle_radius", type=float, default=0.9)
parser.add_argument("--disturbance_range", type=float, nargs=2, default=[0.0, 45.0],
                    help="Training-time disturbance magnitude range (Newtons).")
# evaluation sweep -- test the trained policy and grid baseline at fixed
# magnitudes across the training range
parser.add_argument("--eval_disturbances", type=float, nargs="+",
                    default=[0.0, 15.0, 30.0, 45.0])
parser.add_argument("--grid_phi", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
parser.add_argument("--grid_alpha", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.5, 4.0])
parser.add_argument("--eval_eps_per_cell", type=int, default=64,
                    help="Seeds per cell. We run num_envs=64 in parallel "
                         "anyway -- using all of them widens the binary-"
                         "outcome CIs from ~±35pp (n=8) to ~±13pp (n=64).")
parser.add_argument("--eval_max_steps", type=int, default=1250)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import csv
import importlib.metadata as metadata
import json
import time

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
from cbf_task.locomotion_loader import load_locomotion_actor


TASK = args_cli.task


def build_env_cfg(num_envs, locomotion_actor, args, device):
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args.seed
    env_cfg.log_dir = None
    at = env_cfg.actions.cbf_param
    at.locomotion_policy_obj = locomotion_actor
    at.goal_xy = tuple(args.goal)
    at.obstacle_xy = tuple(args.obstacle)
    at.obstacle_radius = float(args.obstacle_radius)
    at.disturbance_force_range = tuple(args.disturbance_range)
    return env_cfg


def to_norm(phi, alpha, phi_bounds, alpha_bounds):
    plo, phi_hi = phi_bounds
    alo, ahi = alpha_bounds
    return np.array([2.0 * (phi - plo) / (phi_hi - plo) - 1.0,
                     2.0 * (alpha - alo) / (ahi - alo) - 1.0],
                    dtype=np.float32)


def eval_action_at_disturbance(env_wrapped, action_per_env, at_cfg,
                                disturbance_mag, eval_steps, n_eps):
    """Roll out `action_per_env` (N, 2) at a fixed disturbance, return
    cell-level aggregate metrics."""
    cbf_term = env_wrapped.unwrapped.action_manager._terms["cbf_param"]
    # pin the disturbance range to a degenerate single-value during eval
    cbf_term._disturbance_force_lo = float(disturbance_mag)
    cbf_term._disturbance_force_hi = float(disturbance_mag)

    N = env_wrapped.unwrapped.num_envs
    device = env_wrapped.unwrapped.device
    min_h = torch.full((N,), float("inf"), device=device)
    intervention_sum = torch.zeros(N, device=device)
    env_wrapped.unwrapped.reset()
    cbf_term.episode_reach_any.zero_()
    cbf_term.episode_collide_any.zero_()
    cbf_term.episode_fall_any.zero_()
    for _ in range(eval_steps):
        env_wrapped.step(action_per_env)
        min_h = torch.minimum(min_h, cbf_term.last_h_realized)
        intervention_sum = intervention_sum + cbf_term.last_intervention
    n_eps = min(n_eps, N)
    sel = slice(0, n_eps)
    return {
        "n": int(n_eps),
        "collision_rate": float(cbf_term.episode_collide_any[sel].float().mean().item()),
        "reach_rate": float(cbf_term.episode_reach_any[sel].float().mean().item()),
        "fall_rate": float(cbf_term.episode_fall_any[sel].float().mean().item()),
        "min_h_mean": float(min_h[sel].clamp(min=-10.0).mean().item()),
        "intervention_mean": float(intervention_sum[sel].mean().item()),
    }


def eval_learned_policy_at_disturbance(env_wrapped, runner, at_cfg,
                                        disturbance_mag, eval_steps, n_eps,
                                        device):
    """Same as eval_action_at_disturbance but pulls action from the
    learned policy each step instead of using a fixed action.

    Also accumulates per-step (phi, alpha) the policy actually emits, so
    we can tell whether the policy collapsed to a constant or is
    state-conditional. Summary stats (mean/std/p05/p50/p95) are added to
    the returned dict.
    """
    cbf_term = env_wrapped.unwrapped.action_manager._terms["cbf_param"]
    cbf_term._disturbance_force_lo = float(disturbance_mag)
    cbf_term._disturbance_force_hi = float(disturbance_mag)

    policy = runner.get_inference_policy(device=device)

    N = env_wrapped.unwrapped.num_envs
    min_h = torch.full((N,), float("inf"), device=device)
    intervention_sum = torch.zeros(N, device=device)
    obs, _ = env_wrapped.unwrapped.reset()
    cbf_term.episode_reach_any.zero_()
    cbf_term.episode_collide_any.zero_()
    cbf_term.episode_fall_any.zero_()
    obs = env_wrapped.get_observations()
    phi_hist = []
    alpha_hist = []
    for _ in range(eval_steps):
        action = policy(obs)
        obs, _, _, _ = env_wrapped.step(action)
        min_h = torch.minimum(min_h, cbf_term.last_h_realized)
        intervention_sum = intervention_sum + cbf_term.last_intervention
        phi_hist.append(cbf_term.last_phi.detach().clone())
        alpha_hist.append(cbf_term.last_alpha.detach().clone())
    n_eps = min(n_eps, N)
    sel = slice(0, n_eps)
    phi_all = torch.stack(phi_hist, dim=0)[:, sel].flatten()       # (steps*n_eps,)
    alpha_all = torch.stack(alpha_hist, dim=0)[:, sel].flatten()
    q = torch.tensor([0.05, 0.50, 0.95], device=device)
    phi_q = torch.quantile(phi_all, q).tolist()
    alpha_q = torch.quantile(alpha_all, q).tolist()
    return {
        "n": int(n_eps),
        "collision_rate": float(cbf_term.episode_collide_any[sel].float().mean().item()),
        "reach_rate": float(cbf_term.episode_reach_any[sel].float().mean().item()),
        "fall_rate": float(cbf_term.episode_fall_any[sel].float().mean().item()),
        "min_h_mean": float(min_h[sel].clamp(min=-10.0).mean().item()),
        "intervention_mean": float(intervention_sum[sel].mean().item()),
        "phi_mean": float(phi_all.mean().item()),
        "phi_std": float(phi_all.std().item()),
        "phi_p05": float(phi_q[0]),
        "phi_p50": float(phi_q[1]),
        "phi_p95": float(phi_q[2]),
        "alpha_mean": float(alpha_all.mean().item()),
        "alpha_std": float(alpha_all.std().item()),
        "alpha_p05": float(alpha_q[0]),
        "alpha_p50": float(alpha_q[1]),
        "alpha_p95": float(alpha_q[2]),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    # 1) load locomotion actor
    ckpt = retrieve_file_path(args_cli.checkpoint)
    print(f"[phase2] locomotion -> {ckpt}")
    locomotion_actor = load_locomotion_actor(ckpt, device)

    # 2) env cfg + env
    env_cfg = build_env_cfg(args_cli.num_envs, locomotion_actor, args_cli, device)
    env = gym.make(TASK, cfg=env_cfg)

    # 3) rsl_rl agent
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    agent_cfg.max_iterations = int(args_cli.max_iterations)
    agent_cfg.seed = int(args_cli.seed)

    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_dir = os.path.join(os.path.abspath(args_cli.out_dir), "rsl_rl")
    os.makedirs(log_dir, exist_ok=True)
    if args_cli.policy_checkpoint is not None:
        # eval-only mode: load existing weights, skip training entirely
        agent_cfg.max_iterations = 0
        runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                                log_dir=None, device=device)
        policy_ckpt = retrieve_file_path(args_cli.policy_checkpoint)
        print(f"[phase2] eval-only mode: loading policy -> {policy_ckpt}")
        runner.load(policy_ckpt)
        train_secs = 0.0
    else:
        print(f"[phase2] training PPO for {agent_cfg.max_iterations} iterations "
              f"({args_cli.num_envs} envs) ...")
        runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                                log_dir=log_dir, device=device)
        t0 = time.time()
        runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                     init_at_random_ep_len=False)
        train_secs = time.time() - t0
        runner.save(os.path.join(log_dir, "model_final.pt"))
        print(f"[phase2] saved -> {log_dir}/model_final.pt  ({train_secs:.0f}s)")

    # 4) post-training eval: learned policy + fixed-param grid, each at
    # multiple disturbance levels. NB: stay in inference_mode for any
    # post-training env interaction; rsl_rl marked env tensors during learn.
    at = env_cfg.actions.cbf_param
    phi_bounds = at.phi_bounds
    alpha_bounds = at.alpha_bounds
    N = env_wrapped.unwrapped.num_envs

    learned_rows = []
    grid_rows = []
    with torch.inference_mode():
        for d in args_cli.eval_disturbances:
            print(f"[phase2] eval learned @ d={d}N ...")
            m = eval_learned_policy_at_disturbance(
                env_wrapped, runner, at, d,
                args_cli.eval_max_steps, args_cli.eval_eps_per_cell, device,
            )
            learned_rows.append({"disturbance_force": float(d), **m})
            print(f"    coll={m['collision_rate']:.2f}  reach={m['reach_rate']:.2f}  "
                  f"int={m['intervention_mean']:.2f}")
            print(f"    policy outputs at d={d}N: "
                  f"phi  mean={m['phi_mean']:+.3f} std={m['phi_std']:.3f} "
                  f"p05/50/95=[{m['phi_p05']:+.2f},{m['phi_p50']:+.2f},{m['phi_p95']:+.2f}]  "
                  f"alpha mean={m['alpha_mean']:.2f} std={m['alpha_std']:.2f} "
                  f"p05/50/95=[{m['alpha_p05']:.2f},{m['alpha_p50']:.2f},{m['alpha_p95']:.2f}]")

            for phi in args_cli.grid_phi:
                for alpha in args_cli.grid_alpha:
                    a_norm = to_norm(phi, alpha, phi_bounds, alpha_bounds)
                    action = torch.tensor([a_norm] * N, device=device,
                                           dtype=torch.float32)
                    g = eval_action_at_disturbance(
                        env_wrapped, action, at, d,
                        args_cli.eval_max_steps, args_cli.eval_eps_per_cell,
                    )
                    grid_rows.append({"disturbance_force": float(d),
                                       "phi": float(phi), "alpha": float(alpha),
                                       **g})
                    print(f"    grid d={d} phi={phi:.2f} alpha={alpha:.2f}  "
                          f"coll={g['collision_rate']:.2f}  reach={g['reach_rate']:.2f}  "
                          f"int={g['intervention_mean']:.2f}")

    # 5) write outputs
    with open(os.path.join(args_cli.out_dir, "phase2_learned_eval.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(learned_rows[0].keys()))
        w.writeheader()
        w.writerows(learned_rows)
    with open(os.path.join(args_cli.out_dir, "phase2_grid_eval.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        w.writeheader()
        w.writerows(grid_rows)

    # 6) Verdict: aggregate-over-disturbance comparison.
    #
    # Per-disturbance "best fixed" lets the fixed baseline cheat by
    # implicitly adapting (picking a different (phi, alpha) at each
    # disturbance). In deployment you commit to ONE setting. So the fair
    # baseline is: the single (phi, alpha) that's safe across the FULL
    # disturbance range and minimizes mean intervention there.
    #
    # Then compare learned's aggregate to that single fixed's aggregate.
    from collections import defaultdict

    COLL_THR = 0.10        # max acceptable worst-case collision rate
    REACH_THR = 0.80       # min acceptable worst-case reach rate
    INT_TOL = 1.5          # max int relative to best deployable fixed

    def aggregate(rows: list[dict]) -> dict:
        return {
            "worst_coll": max(r["collision_rate"] for r in rows),
            "worst_reach": min(r["reach_rate"] for r in rows),
            "mean_int": float(np.mean([r["intervention_mean"] for r in rows])),
        }

    by_cell = defaultdict(list)
    for g in grid_rows:
        by_cell[(g["phi"], g["alpha"])].append(g)
    fixed_aggs = []
    for (phi, alpha), cells in by_cell.items():
        a = aggregate(cells)
        a.update({"phi": phi, "alpha": alpha})
        fixed_aggs.append(a)

    # filter to "deployable safe" fixed cells: worst_coll == 0 AND
    # worst_reach >= REACH_THR across the full disturbance range.
    safe_fixed = [a for a in fixed_aggs
                  if a["worst_coll"] == 0.0 and a["worst_reach"] >= REACH_THR]
    best_fixed = min(safe_fixed, key=lambda a: a["mean_int"]) if safe_fixed else None

    learned_agg = aggregate(learned_rows)

    print()
    print("=" * 90)
    print("  Phase 2 -- aggregate-over-disturbance verdict")
    print(f"  Each (phi, alpha) is run across d in {args_cli.eval_disturbances}; "
          f"compared as a single deployable setting.")
    print("=" * 90)
    print(f"  Learned aggregate:  worst_coll={learned_agg['worst_coll']:.2f}  "
          f"worst_reach={learned_agg['worst_reach']:.2f}  "
          f"mean_int={learned_agg['mean_int']:.0f}")
    if best_fixed is not None:
        print(f"  Best deployable fixed (phi={best_fixed['phi']:.2f}, "
              f"alpha={best_fixed['alpha']:.2f}):")
        print(f"     worst_coll={best_fixed['worst_coll']:.2f}  "
              f"worst_reach={best_fixed['worst_reach']:.2f}  "
              f"mean_int={best_fixed['mean_int']:.0f}")
    else:
        print("  No fixed (phi, alpha) is safe across the full disturbance range.")
        print("  Listing the SAFEST single fixed cell (by worst-case collision):")
        safest = min(fixed_aggs, key=lambda a: (a["worst_coll"], -a["worst_reach"]))
        print(f"     phi={safest['phi']:.2f} alpha={safest['alpha']:.2f}  "
              f"worst_coll={safest['worst_coll']:.2f}  "
              f"worst_reach={safest['worst_reach']:.2f}  "
              f"mean_int={safest['mean_int']:.0f}")
        best_fixed = safest  # fallback for efficiency comparison

    # also print per-d learned breakdown for transparency
    print("  Per-disturbance learned breakdown:")
    for L in learned_rows:
        print(f"     d={L['disturbance_force']:>5.1f}N  coll={L['collision_rate']:.2f}  "
              f"reach={L['reach_rate']:.2f}  fall={L.get('fall_rate', 0):.2f}  "
              f"int={L['intervention_mean']:.0f}")

    # --- policy-collapse diagnostic ---
    # If the policy is genuinely state-conditional, the (phi, alpha) it
    # emits should shift with disturbance. If it collapsed to a constant,
    # the per-d means will be ~identical and per-d std will be ~0.
    print()
    print("  POLICY-OUTPUT DIAGNOSTIC (does (phi,alpha) shift with d?)")
    print(f"     {'d (N)':>6}  {'phi_mean':>10}  {'phi_std':>8}  "
          f"{'alpha_mean':>11}  {'alpha_std':>9}")
    for L in learned_rows:
        print(f"     {L['disturbance_force']:>6.1f}  "
              f"{L['phi_mean']:>+10.3f}  {L['phi_std']:>8.3f}  "
              f"{L['alpha_mean']:>11.2f}  {L['alpha_std']:>9.2f}")
    phi_range = (max(L["phi_mean"] for L in learned_rows)
                 - min(L["phi_mean"] for L in learned_rows))
    alpha_range = (max(L["alpha_mean"] for L in learned_rows)
                   - min(L["alpha_mean"] for L in learned_rows))
    phi_std_mean = float(np.mean([L["phi_std"] for L in learned_rows]))
    alpha_std_mean = float(np.mean([L["alpha_std"] for L in learned_rows]))
    print(f"     across-d range:  phi_mean spans {phi_range:.3f},  "
          f"alpha_mean spans {alpha_range:.3f}")
    print(f"     within-d spread: phi_std avg {phi_std_mean:.3f},   "
          f"alpha_std avg {alpha_std_mean:.3f}")
    # heuristic: a "collapsed" policy has both range AND within-d std < ~5%
    # of its bound width. phi width = phi_hi - phi_lo (typ. 1.0);
    # alpha width = alpha_hi - alpha_lo (typ. 3.8).
    phi_width = at.phi_bounds[1] - at.phi_bounds[0]
    alpha_width = at.alpha_bounds[1] - at.alpha_bounds[0]
    collapsed = (phi_range < 0.05 * phi_width
                 and alpha_range < 0.05 * alpha_width
                 and phi_std_mean < 0.05 * phi_width
                 and alpha_std_mean < 0.05 * alpha_width)
    if collapsed:
        print("     -> POLICY COLLAPSED to near-constant output. PPO didn't "
              "find a state-conditional solution; check exploration / "
              "reward shape / scenario.")
    else:
        print("     -> policy is varying across d (state-conditional). "
              "If learned still loses to fixed, the issue is which "
              "(phi,alpha) it picks, not whether it adapts.")

    safe_ok = learned_agg["worst_coll"] <= COLL_THR
    reach_ok = learned_agg["worst_reach"] >= REACH_THR
    eff_ok = learned_agg["mean_int"] <= INT_TOL * best_fixed["mean_int"]
    per_d_pass = [safe_ok, reach_ok, eff_ok]

    flags = []
    if not safe_ok:
        flags.append(f"UNSAFE worst_coll={learned_agg['worst_coll']:.2f}>{COLL_THR}")
    if not reach_ok:
        flags.append(f"REACH worst_reach={learned_agg['worst_reach']:.2f}<{REACH_THR}")
    if not eff_ok:
        flags.append(f"INEFFICIENT mean_int={learned_agg['mean_int']:.0f}>"
                     f"{INT_TOL}x{best_fixed['mean_int']:.0f}")
    print(f"\n  checks: {' | '.join(flags) if flags else 'all pass'}")

    verdict = "PASS" if all(per_d_pass) else "REVIEW"

    # --- detailed per-disturbance comparison table ---
    def _bestsafe_at_d(d):
        cells = [g for g in grid_rows if g["disturbance_force"] == d
                                      and g["collision_rate"] == 0.0
                                      and g["reach_rate"] >= REACH_THR]
        if not cells:
            cells = [g for g in grid_rows if g["disturbance_force"] == d]
            return min(cells, key=lambda c: (c["collision_rate"],
                                              -c["reach_rate"],
                                              c["intervention_mean"]))
        return min(cells, key=lambda c: c["intervention_mean"])

    table_lines = []
    table_lines.append("")
    table_lines.append("=" * 96)
    table_lines.append("  PER-DISTURBANCE COMPARISON     learned (state-conditional) vs best safe fixed (at that d)")
    table_lines.append("=" * 96)
    table_lines.append(
        f"  {'d (N)':>6} | {'LEARNED':<32} | {'BEST SAFE FIXED at this d':<48}"
    )
    table_lines.append(
        f"  {'':>6} | {'coll':>6} {'reach':>7} {'int':>8} {'min_h':>7} | "
        f"{'(phi,alpha)':>12} {'coll':>6} {'reach':>7} {'int':>8}"
    )
    table_lines.append("  " + "-" * 94)
    for L in learned_rows:
        d = L["disturbance_force"]
        bf = _bestsafe_at_d(d)
        delta_int = L["intervention_mean"] - bf["intervention_mean"]
        delta_sign = "+" if delta_int >= 0 else ""
        table_lines.append(
            f"  {d:>6.1f} | "
            f"{L['collision_rate']:>5.2f}  {L['reach_rate']:>6.2f}  "
            f"{L['intervention_mean']:>7.0f}  {L['min_h_mean']:>+6.2f} | "
            f"({bf['phi']:.1f},{bf['alpha']:.1f})    "
            f"{bf['collision_rate']:>5.2f}  {bf['reach_rate']:>6.2f}  "
            f"{bf['intervention_mean']:>7.0f}   "
            f"(Δint = {delta_sign}{delta_int:.0f})"
        )
    table_lines.append("=" * 96)
    for line in table_lines:
        print(line)

    # write the table as a human-readable .txt next to the JSON
    txt_path = os.path.join(args_cli.out_dir, "phase2_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"Task: {args_cli.task}\n")
        f.write(f"Train range: {args_cli.disturbance_range}\n")
        f.write(f"Iterations: {args_cli.max_iterations}\n")
        f.write(f"Train seconds: {train_secs:.0f}\n")
        f.write(f"Verdict: {verdict}\n")
        f.write("\n".join(table_lines))
        f.write("\n\nPOLICY-OUTPUT DIAGNOSTIC\n")
        f.write(f"{'d (N)':>6}  {'phi_mean':>10}  {'phi_std':>8}  "
                f"{'alpha_mean':>11}  {'alpha_std':>9}\n")
        for L in learned_rows:
            f.write(f"{L['disturbance_force']:>6.1f}  "
                    f"{L['phi_mean']:>+10.3f}  {L['phi_std']:>8.3f}  "
                    f"{L['alpha_mean']:>11.2f}  {L['alpha_std']:>9.2f}\n")
        f.write(f"\nacross-d range:  phi_mean spans {phi_range:.3f},  "
                f"alpha_mean spans {alpha_range:.3f}\n")
        f.write(f"within-d spread: phi_std avg {phi_std_mean:.3f},   "
                f"alpha_std avg {alpha_std_mean:.3f}\n")
        f.write(f"collapsed: {collapsed}\n")

    # --- video recording instructions (separate script; needs --enable_cameras) ---
    model_path = os.path.join(log_dir, "model_final.pt")
    print()
    print("To record videos of the trained policy at chosen disturbance levels:")
    print("  ./isaaclab.sh -p ~/Desktop/cbf_rl_mvp/go2/phase2_play.py \\")
    print(f"      --loco_checkpoint {args_cli.checkpoint} \\")
    print(f"      --policy_checkpoint {model_path} \\")
    print(f"      --task {args_cli.task} \\")
    print(f"      --disturbance_force <0|20|40> --num_envs 4 --n_steps 500 \\")
    print(f"      --out_dir {os.path.abspath(args_cli.out_dir)}/videos --enable_cameras")
    print(f"\n  verdict : {verdict}")
    print("=" * 78)

    summary = {"learned": learned_rows, "verdict": verdict,
               "per_d_pass": per_d_pass,
               "train_seconds": train_secs}
    with open(os.path.join(args_cli.out_dir, "phase2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved -> {log_dir}, phase2_learned_eval.csv, phase2_grid_eval.csv, "
          f"phase2_summary.json")

    env.close()
    simulation_app.close()
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
