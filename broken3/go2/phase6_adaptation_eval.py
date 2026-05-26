"""Adaptation plot: does the trained teacher modulate (phi, alpha) as a
function of one DR axis, with everything else pinned at nominal?

The disturbance sweep we've been showing in §12 confuses things because:
  - alpha hedges TRACKING ERROR (which v_max produces cleanly via
    faster commanded speed -> harder to brake). v_max is the only
    validated alpha-channel ([[alpha_channel_search]]).
  - phi hedges CONTROL-EFFECTIVENESS (friction/motor strength per
    [[cbf_parameter_theory]]). Neither has passed a strong gate yet,
    but they're the theoretical right axes.
  - Disturbance is a hard-to-classify perturbation that gets smoothed
    through the locomotion controller. Flat phi vs disturbance is NOT
    evidence the policy doesn't use phi.

This script sweeps ONE DR axis, holds all others at nominal, and
reports per-bin mean (phi, alpha) plus the standard cell metrics. Use
it to answer "does the policy modulate X channel in response to Y
priv signal?"

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase6_adaptation_eval.py \\
        --teacher_ckpt phase6_shield_v7_1_stuck_teacher_outputs/rsl_rl/model_final.pt \\
        --locomotion_ckpt /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --axis v_max --levels 1.0 1.25 1.5 1.75 2.0 \\
        --headless

Axes supported (with theory-paired CBF param in brackets):
    v_max          [alpha]  -- VALIDATED. Expected to see clear modulation.
    friction       [phi]    -- theoretical phi-channel; not gate-validated.
    motor_strength [phi]    -- theoretical phi-channel; not gate-validated.
    base_mass      [alpha]  -- weak alpha-channel; not gate-validated.
    disturbance    [???]    -- the old axis; included for back-compat
                              (and as a "should NOT cleanly map to
                              either param" reference).
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


AXIS_TO_ATTRS = {
    "v_max":          ("_v_max_lo",          "_v_max_hi",          None),
    "friction":       ("_friction_lo",       "_friction_hi",       "_apply_friction"),
    "motor_strength": ("_motor_strength_lo", "_motor_strength_hi", None),
    "base_mass":      ("_base_mass_lo",      "_base_mass_hi",      "_apply_mass"),
    "disturbance":    ("_disturbance_force_lo", "_disturbance_force_hi", None),
}

# Default sweep ranges per axis. Picked to span the training DR range.
DEFAULT_LEVELS = {
    "v_max":          [1.0, 1.25, 1.5, 1.75, 2.0],
    "friction":       [0.3, 0.5, 0.7, 1.0],
    "motor_strength": [0.7, 0.85, 1.0, 1.15, 1.3],
    "base_mass":      [-3.0, -1.5, 0.0, 1.5, 3.0],
    "disturbance":    [0.0, 15.0, 30.0],
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher_ckpt", required=True)
parser.add_argument("--locomotion_ckpt", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-UnifiedLidarSDF-v0",
                    help="Env that has all DR channels active. SHIELD env "
                         "is the right default for v7+; older teachers may "
                         "need their training task.")
parser.add_argument("--axis", required=True, choices=list(AXIS_TO_ATTRS.keys()),
                    help="Which DR axis to sweep. v_max for alpha test; "
                         "friction or motor_strength for phi test.")
parser.add_argument("--levels", type=float, nargs="+", default=None,
                    help="Levels to sweep along the axis. Defaults to "
                         "axis-specific range (see DEFAULT_LEVELS).")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--eval_eps_per_cell", type=int, default=256)
parser.add_argument("--eval_steps", type=int, default=1250)
parser.add_argument("--out_dir", default="phase6_adaptation_eval_outputs")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

# ---------------------------------------------------------------------------
import csv
import importlib.metadata as metadata
import json
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa
from cbf_task.agents import rma_actor_critic  # noqa
from cbf_task.locomotion_loader import load_locomotion_actor
from cbf_task.eval_utils import UNSAFE_THR


# --- pinning all DR EXCEPT the swept axis at nominal ---
# Nominal values match the "no-DR" centerpoint of each channel.
NOMINAL = {
    "_v_max_lo": 1.3,                   "_v_max_hi": 1.3,
    "_friction_lo": 0.7,                "_friction_hi": 0.7,
    "_motor_strength_lo": 1.0,          "_motor_strength_hi": 1.0,
    "_base_mass_lo": 0.0,               "_base_mass_hi": 0.0,
    "_disturbance_force_lo": 0.0,       "_disturbance_force_hi": 0.0,
    "_actuation_noise_std_lo": 0.0,     "_actuation_noise_std_hi": 0.0,
    "_com_offset_lo": 0.0,              "_com_offset_hi": 0.0,
}


def _pin_all_dr_nominal(cbf):
    """Set every DR knob to its nominal centerpoint. Caller then overrides
    the ONE swept axis after this."""
    for attr, val in NOMINAL.items():
        setattr(cbf, attr, val)


def _force_physics_apply(cbf, env_ids):
    """The action term's reset code gates friction/mass/com_offset PhysX
    writes on `_hi > _lo` (line 461-466 of cbf_action_term.py). When we
    pin lo=hi for an eval, the priv obs reflects the level but the
    PhysX material/mass stays at default. Force the apply methods so
    the eval physics actually matches the priv signal.
    """
    if hasattr(cbf, "_apply_friction"):
        try: cbf._apply_friction(env_ids)
        except Exception as e: print(f"  [WARN] _apply_friction failed: {e}")
    if hasattr(cbf, "_apply_mass"):
        try: cbf._apply_mass(env_ids)
        except Exception as e: print(f"  [WARN] _apply_mass failed: {e}")
    if hasattr(cbf, "_apply_com_offset"):
        try: cbf._apply_com_offset(env_ids)
        except Exception as e: print(f"  [WARN] _apply_com_offset failed: {e}")


def eval_at_level(env, env_wrapped, cbf, policy, axis, level, eval_steps,
                   n_eps, device):
    """Pin all DR at nominal, then set `axis` to `level`. Reset, roll out
    the policy, return per-bin aggregate.

    Returns a dict with mean policy outputs (phi/alpha mean+std) and
    per-episode rates (counted from terminations, NOT sticky flags).
    """
    _pin_all_dr_nominal(cbf)
    lo_attr, hi_attr, _ = AXIS_TO_ATTRS[axis]
    setattr(cbf, lo_attr, float(level))
    setattr(cbf, hi_attr, float(level))

    env.unwrapped.reset()
    # Now physics needs the pinned values applied (the reset path skips
    # apply when lo==hi). Force it.
    N = env.unwrapped.num_envs
    env_ids = torch.arange(N, device=device)
    _force_physics_apply(cbf, env_ids)

    # Per-episode termination counting. Each step we read the manager's
    # per-term done masks and accumulate counts. At end of eval, rates =
    # counts / total_episodes (mutually exclusive, sums to 1 within
    # numerical noise).
    term_counts = {"reach": 0, "collision": 0, "fall": 0, "time_out": 0}
    n_episodes = 0
    phi_hist = []
    alpha_hist = []
    intervention_sum = torch.zeros(N, device=device)
    jitter_hist = []
    min_h = torch.full((N,), float("inf"), device=device)
    unsafe_steps = torch.zeros(N, device=device)
    stuck_seen = torch.zeros(N, dtype=torch.bool, device=device)

    obs = env_wrapped.get_observations().to(device)
    for step in range(eval_steps):
        with torch.inference_mode():
            action = policy(obs)
        obs_next, _, _, _ = env_wrapped.step(action)
        obs = obs_next.to(device)

        # Per-step caches (set by action term)
        phi_hist.append(cbf.last_phi.detach().clone())
        alpha_hist.append(cbf.last_alpha.detach().clone())
        intervention_sum = intervention_sum + cbf.last_intervention
        jitter_hist.append(cbf.last_action_jitter.detach().clone())
        min_h = torch.minimum(min_h, cbf.last_h_realized)
        unsafe_steps = unsafe_steps + (cbf.last_h_realized < UNSAFE_THR).float()

        # Per-step termination check (the manager exposes per-term dones
        # AFTER the step). Count any new fires.
        term_mgr = env.unwrapped.termination_manager
        try:
            t_reach = term_mgr.get_term("goal_reached")
            t_coll = term_mgr.get_term("collision")
            t_fall = term_mgr.get_term("fall")
            t_to   = term_mgr.get_term("time_out")
        except Exception:
            # Fallback: only count timeouts via episode_length_buf
            t_reach = t_coll = t_fall = t_to = torch.zeros(N, dtype=torch.bool, device=device)
        term_counts["reach"]     += int(t_reach.sum().item())
        term_counts["collision"] += int(t_coll.sum().item())
        term_counts["fall"]      += int(t_fall.sum().item())
        term_counts["time_out"]  += int(t_to.sum().item())
        n_episodes += int((t_reach | t_coll | t_fall | t_to).sum().item())

        # Stuck (sticky-any across the window, since it's a single per-
        # env flag without a clean terminal counterpart).
        stuck_seen |= cbf.episode_stuck_any

    sel = slice(0, min(n_eps, N))
    phi_all = torch.stack(phi_hist, dim=0)[:, sel].flatten()
    alpha_all = torch.stack(alpha_hist, dim=0)[:, sel].flatten()
    jitter_all = torch.stack(jitter_hist, dim=0)[:, sel].flatten()

    total = max(n_episodes, 1)
    return {
        "axis": axis,
        "level": float(level),
        "n_episodes_seen": int(n_episodes),
        # per-episode termination rates (mutually exclusive, sum to 1)
        "reach_rate":     term_counts["reach"]     / total,
        "collision_rate": term_counts["collision"] / total,
        "fall_rate":      term_counts["fall"]      / total,
        "time_out_rate":  term_counts["time_out"]  / total,
        # policy output stats
        "phi_mean":   float(phi_all.mean().item()),
        "phi_std":    float(phi_all.std().item()),
        "alpha_mean": float(alpha_all.mean().item()),
        "alpha_std":  float(alpha_all.std().item()),
        # safety/diagnostic
        "intervention_mean": float(intervention_sum[sel].mean().item()),
        "jitter_mean":  float(jitter_all.mean().item()),
        "min_h_mean":   float(min_h[sel].clamp(min=-10.0).mean().item()),
        "time_in_unsafe_frac": float((unsafe_steps[sel] / eval_steps).mean().item()),
        # stuck flag (still per-env sticky; documented limitation)
        "stuck_rate_sticky": float(stuck_seen[sel].float().mean().item()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    levels = args_cli.levels or DEFAULT_LEVELS[args_cli.axis]
    print(f"\n  axis={args_cli.axis}  levels={levels}")
    print(f"  task={args_cli.task}  num_envs={args_cli.num_envs}  "
          f"eval_steps={args_cli.eval_steps}")

    # ----- build env, load teacher -----
    loco = load_locomotion_actor(retrieve_file_path(args_cli.locomotion_ckpt), device)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env = gym.make(args_cli.task, cfg=env_cfg)
    cbf = env.unwrapped.action_manager._terms["cbf_param"]

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                             log_dir=None, device=device)
    runner.load(retrieve_file_path(args_cli.teacher_ckpt))
    policy = runner.get_inference_policy(device=device)
    print(f"  loaded teacher: {args_cli.teacher_ckpt}\n")

    # ----- sweep -----
    rows = []
    for L in levels:
        t0 = time.time()
        row = eval_at_level(env, env_wrapped, cbf, policy,
                             args_cli.axis, L,
                             args_cli.eval_steps,
                             args_cli.eval_eps_per_cell, device)
        wall = time.time() - t0
        rows.append(row)
        print(f"  level={L:>6.2f}  "
              f"phi={row['phi_mean']:+.3f}±{row['phi_std']:.2f}  "
              f"alpha={row['alpha_mean']:.2f}±{row['alpha_std']:.2f}  "
              f"reach={row['reach_rate']:.2f}  coll={row['collision_rate']:.2f}  "
              f"fall={row['fall_rate']:.2f}  time_out={row['time_out_rate']:.2f}  "
              f"n_eps={row['n_episodes_seen']}  (wall={wall:.1f}s)")

    # ----- save + summary -----
    csv_path = os.path.join(args_cli.out_dir,
                             f"adaptation_{args_cli.axis}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    phi_vals = [r["phi_mean"] for r in rows]
    alpha_vals = [r["alpha_mean"] for r in rows]
    phi_range = max(phi_vals) - min(phi_vals)
    alpha_range = max(alpha_vals) - min(alpha_vals)
    phi_w = float(cbf._phi_hi - cbf._phi_lo)
    alpha_w = float(cbf._alpha_hi - cbf._alpha_lo)

    print()
    print("=" * 88)
    print(f"  ADAPTATION SUMMARY  --  axis={args_cli.axis}")
    print("=" * 88)
    print(f"  phi modulation:   {phi_range:.3f}  "
          f"({100 * phi_range / phi_w:.1f}% of phi bound width)")
    print(f"  alpha modulation: {alpha_range:.3f}  "
          f"({100 * alpha_range / alpha_w:.1f}% of alpha bound width)")
    print(f"  saved -> {csv_path}")
    print("=" * 88)

    summ_path = os.path.join(args_cli.out_dir,
                              f"adaptation_{args_cli.axis}_summary.json")
    with open(summ_path, "w") as f:
        json.dump({
            "axis": args_cli.axis,
            "levels": list(levels),
            "phi_means": phi_vals,
            "alpha_means": alpha_vals,
            "phi_modulation_pct_of_width": 100 * phi_range / phi_w,
            "alpha_modulation_pct_of_width": 100 * alpha_range / alpha_w,
            "teacher_ckpt": args_cli.teacher_ckpt,
            "task": args_cli.task,
        }, f, indent=2)

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
