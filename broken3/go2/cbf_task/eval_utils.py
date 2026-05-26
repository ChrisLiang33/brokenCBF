"""Shared eval helpers for phase5_baselines.py and phase6_eval_scenes.py.

Extracted here so phase6_eval_scenes (driver+worker) can import the
helpers without triggering phase5_baselines's top-level argparse +
AppLauncher side effects (which require --checkpoint and tried to spin
up a second Isaac sim context inside the worker).

These functions assume:
  - Isaac sim app is already alive (the caller has initialized
    AppLauncher before importing this module).
  - The `cbf` object is an instance of the CBFActionTerm and the `env`
    is the unwrapped Isaac Lab ManagerBasedRLEnv.

No top-level side effects -- only function defs + one constant.
"""
from __future__ import annotations

import torch


# h_realized below this counts as "in the unsafe zone" for the
# `time_in_unsafe_frac` metric. Matches both eval pipelines.
UNSAFE_THR = 0.2

EVAL_COLS = ["collision_rate", "reach_rate", "fall_rate", "stuck_rate",
             "intervention_mean", "jitter_mean", "min_h_mean",
             "time_in_unsafe_frac", "time_to_goal_mean"]


def map_to_action(phi_v: torch.Tensor, alpha_v: torch.Tensor,
                  cbf) -> torch.Tensor:
    """Map (phi, alpha) in their native units to the policy's
    normalized [-1, 1]^2 action space via the same linear decode the
    action term uses."""
    a0 = 2.0 * (phi_v - cbf._phi_lo) / (cbf._phi_hi - cbf._phi_lo) - 1.0
    a1 = 2.0 * (alpha_v - cbf._alpha_lo) / (cbf._alpha_hi - cbf._alpha_lo) - 1.0
    return torch.stack([a0, a1], dim=-1)


def eval_cell(env, cbf, action_fn, disturbance, eval_steps, n_eps, device):
    """Run `action_fn(cbf, step) -> (N, 2)` for eval_steps steps at a
    pinned disturbance magnitude, collect aggregate metrics across the
    first `n_eps` envs.

    NOTE: `episode_*_any` flags are sticky-within-episode and the env
    auto-reset path inside cbf_action_term.py does not clear them across
    auto-resets within a single eval window. So `collision_rate` here is
    "fraction of envs that ever collided during the eval window" NOT
    "% of episodes that ended in collision". Consistent across teacher
    and baselines so comparisons are valid; absolute numbers should be
    read as ever-occurred rates, not per-episode rates.
    """
    cbf._disturbance_force_lo = float(disturbance)
    cbf._disturbance_force_hi = float(disturbance)
    env.unwrapped.reset()
    cbf.episode_reach_any.zero_()
    cbf.episode_collide_any.zero_()
    cbf.episode_fall_any.zero_()
    cbf.episode_stuck_any.zero_()
    N = env.unwrapped.num_envs
    min_h = torch.full((N,), float("inf"), device=device)
    intervention_sum = torch.zeros(N, device=device)
    unsafe_steps = torch.zeros(N, device=device)
    goal_step = torch.full((N,), -1, device=device, dtype=torch.long)
    jitter_hist = []
    for step in range(eval_steps):
        action = action_fn(cbf, step)
        env.step(action)
        min_h = torch.minimum(min_h, cbf.last_h_realized)
        intervention_sum = intervention_sum + cbf.last_intervention
        unsafe_steps = unsafe_steps + (cbf.last_h_realized < UNSAFE_THR).float()
        newly_reached = cbf.episode_reach_any & (goal_step == -1)
        goal_step[newly_reached] = step
        jitter_hist.append(cbf.last_action_jitter.detach().clone())
    n_eps = min(n_eps, N)
    sel = slice(0, n_eps)
    jitter_all = torch.stack(jitter_hist, dim=0)[:, sel].flatten()
    reached_mask = goal_step[sel] >= 0
    time_to_goal_mean = (float(goal_step[sel][reached_mask].float().mean().item())
                         if reached_mask.any() else float("nan"))
    return {
        "collision_rate": float(cbf.episode_collide_any[sel].float().mean().item()),
        "reach_rate": float(cbf.episode_reach_any[sel].float().mean().item()),
        "fall_rate": float(cbf.episode_fall_any[sel].float().mean().item()),
        "stuck_rate": float(cbf.episode_stuck_any[sel].float().mean().item()),
        "intervention_mean": float(intervention_sum[sel].mean().item()),
        "jitter_mean": float(jitter_all.mean().item()),
        "min_h_mean": float(min_h[sel].clamp(min=-10.0).mean().item()),
        "time_in_unsafe_frac": float((unsafe_steps[sel] / eval_steps).mean().item()),
        "time_to_goal_mean": time_to_goal_mean,
    }


def aggregate_across_d(rows):
    """Worst-case aggregates across a disturbance sweep."""
    return {
        "worst_coll": max(r["collision_rate"] for r in rows),
        "worst_reach": min(r["reach_rate"] for r in rows),
        "mean_int": sum(r["intervention_mean"] for r in rows) / len(rows),
        "mean_jitter": sum(r["jitter_mean"] for r in rows) / len(rows),
    }


def best_safe_cell(cells_agg, reach_thr=0.80):
    """Pick the deployable cell per family.

    Preferred path: among cells with `worst_coll == 0` AND
    `worst_reach >= reach_thr`, pick the lowest `mean_int`.

    Fallback: if no cell is "safe", pick by (worst_coll, -worst_reach)
    -- lowest collision, ties broken by highest reach. NOTE: this can
    pick a degenerate freeze configuration (zero reach, zero collision)
    if no cell has any reach; downstream consumers should sanity-check
    the picked cell's worst_reach > 0.
    """
    safe = [c for c in cells_agg if c["agg"]["worst_coll"] == 0.0
            and c["agg"]["worst_reach"] >= reach_thr]
    if safe:
        return min(safe, key=lambda c: c["agg"]["mean_int"]), True
    fallback = min(cells_agg, key=lambda c: (c["agg"]["worst_coll"],
                                              -c["agg"]["worst_reach"]))
    return fallback, False
