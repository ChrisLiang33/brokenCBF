"""Curriculum eval: sweep difficulty from trivial -> extreme on a single
teacher checkpoint. Each level bumps obstacle count AND DR magnitude
together so the result is one-dimensional ("does the policy degrade
gracefully as scene difficulty grows").

Subprocess per level because obstacle count K changes the action term's
tensor shapes at env init time -- can't mutate K mid-process. Within
each worker, DR ranges are mutated on the action term to scale DR
magnitude, and the disturbance sweep runs in-process (cheap).

Levels (each is a (K obstacles, DR magnitude %) tuple):
    1 trivial   K=1  DR=  0%   (nominal only, single obstacle)
    2 easy      K=2  DR= 25%
    3 medium    K=3  DR= 50%   (about the training distribution centerpoint)
    4 hard      K=4  DR= 75%
    5 extreme   K=5  DR=100%   (full training DR + most obstacles)

DR magnitude X% means linear interpolation from nominal centerpoint
(X=0) to full training range (X=1) on each priv channel.

Workflow:
    1. Train an RMA-static teacher -> model_final.pt
    2. (Optional) Tune baselines on the RMAStatic env via phase5_baselines.py
       -> phase5_baselines_summary.json
    3. Run THIS script:
         ~/IsaacLab/isaaclab.sh -p phase7_curriculum_eval.py \\
             --teacher_ckpt phase7_rma_static_teacher_outputs/rsl_rl/model_final.pt \\
             --locomotion_ckpt /home/.../model_299.pt \\
             [--baselines_summary path/to/summary.json] \\
             --headless
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


# (K obstacles, DR magnitude 0..1, label)
DEFAULT_LEVELS = [
    (1, 0.00, "trivial"),
    (2, 0.25, "easy"),
    (3, 0.50, "medium"),
    (4, 0.75, "hard"),
    (5, 1.00, "extreme"),
]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher_ckpt", required=True)
parser.add_argument("--locomotion_ckpt", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-RMAStatic-v0",
                    help="Env that has all DR channels active AND supports "
                         "random_topology. RMAStatic is the right default.")
parser.add_argument("--baselines_summary", default=None,
                    help="Optional path to phase5_baselines_summary.json. "
                         "If set, B-trivial + B0/B1/B2-best are also eval'd.")
parser.add_argument("--disturbances", type=float, nargs="+",
                    default=[0.0, 15.0, 30.0])
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--eval_eps_per_cell", type=int, default=256)
parser.add_argument("--eval_steps", type=int, default=1250)
parser.add_argument("--out_dir", default="phase7_curriculum_eval_outputs")
parser.add_argument("--seed", type=int, default=0)
# WORKER mode (driver sets these when spawning)
parser.add_argument("--level_idx", type=int, default=None,
                    help="WORKER mode: which level index to run (0-based).")
parser.add_argument("--json_out", default=None,
                    help="WORKER mode: where to write the per-level JSON.")
# AppLauncher args are only needed in WORKER. Driver doesn't touch Isaac.
args_known, _ = parser.parse_known_args()


# DR channel nominal centerpoints (X=0) and full training ranges (X=1).
# Linear interpolation: lo(X) = nom - X*(nom - lo_full); hi(X) = nom + X*(hi_full - nom).
# For asymmetric ranges (disturbance: 0..30), we just scale the upper bound.
DR_RANGES = {
    # name: (nominal, full_lo, full_hi)
    "friction":        (0.7,  0.3, 1.0),
    "base_mass":       (0.0, -3.0, 3.0),
    "motor_strength":  (1.0,  0.7, 1.3),
    "disturbance":     (0.0,  0.0, 30.0),    # lower bound stays 0
}


def _scaled_range(nom, full_lo, full_hi, frac):
    """Linearly interpolate (lo, hi) from (nom, nom) at frac=0 to
    (full_lo, full_hi) at frac=1."""
    lo = nom + frac * (full_lo - nom)
    hi = nom + frac * (full_hi - nom)
    return (lo, hi)


def run_driver():
    os.makedirs(args_known.out_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    isaaclab_sh = os.path.expanduser("~/IsaacLab/isaaclab.sh")
    if not os.path.exists(isaaclab_sh):
        print(f"  [ERROR] isaaclab.sh not found at {isaaclab_sh}")
        sys.exit(1)

    per_level_results = []
    for i, (K, dr_frac, label) in enumerate(DEFAULT_LEVELS):
        json_out = os.path.join(args_known.out_dir,
                                 f"level_{i+1}_{label}_results.json")
        cmd = [
            isaaclab_sh, "-p", os.path.join(here, "phase7_curriculum_eval.py"),
            "--teacher_ckpt", args_known.teacher_ckpt,
            "--locomotion_ckpt", args_known.locomotion_ckpt,
            "--task", args_known.task,
            "--level_idx", str(i),
            "--json_out", json_out,
            "--num_envs", str(args_known.num_envs),
            "--eval_eps_per_cell", str(args_known.eval_eps_per_cell),
            "--eval_steps", str(args_known.eval_steps),
            "--seed", str(args_known.seed),
            "--out_dir", args_known.out_dir,
            "--disturbances", *[str(d) for d in args_known.disturbances],
            "--headless",
        ]
        if args_known.baselines_summary:
            cmd += ["--baselines_summary", args_known.baselines_summary]
        print()
        print("=" * 96)
        print(f"  DRIVER  --  spawning worker for level {i+1} "
              f"({label}: K={K}, DR={dr_frac:.0%})")
        print("=" * 96)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  [ERROR] worker for level {i+1} exited with code {rc}")
            continue
        if not os.path.exists(json_out):
            print(f"  [ERROR] worker for level {i+1} produced no JSON")
            continue
        with open(json_out) as f:
            per_level_results.append(json.load(f))

    # ----- aggregate + print summary -----
    import csv as _csv
    all_rows = []
    summary_rows = []
    for r in per_level_results:
        all_rows.extend(r["cells"])
        summary_rows.extend(r["summary"])
    cells_path = os.path.join(args_known.out_dir,
                                "phase7_curriculum_cells.csv")
    summ_path = os.path.join(args_known.out_dir,
                              "phase7_curriculum_summary.csv")
    if all_rows:
        with open(cells_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
    if summary_rows:
        with open(summ_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)

    print()
    print("=" * 96)
    print(f"  CURRICULUM SUMMARY  (worst across {args_known.disturbances} N)")
    print("=" * 96)
    print(f"  {'level':<14}  {'policy':<48}  {'wcoll':>6}  {'wreach':>7}  "
          f"{'mint':>7}  {'mjit':>6}")
    for r in summary_rows:
        print(f"  {r['level']:<14}  {r['policy']:<48}  "
              f"{r['worst_coll']:>6.2f}  {r['worst_reach']:>7.2f}  "
              f"{r['mean_int']:>7.0f}  {r['mean_jitter']:>6.3f}")
    print()
    print(f"  saved -> {cells_path}")
    print(f"  saved -> {summ_path}")


def run_worker(level_idx: int, json_out: str):
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--locomotion_ckpt", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--baselines_summary", default=None)
    p.add_argument("--level_idx", type=int, required=True)
    p.add_argument("--json_out", required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--eval_eps_per_cell", type=int, default=256)
    p.add_argument("--eval_steps", type=int, default=1250)
    p.add_argument("--disturbances", type=float, nargs="+",
                   default=[0.0, 15.0, 30.0])
    p.add_argument("--out_dir", default=".")
    p.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(p)
    wargs, _ = p.parse_known_args()
    wargs.headless = True if not hasattr(wargs, "headless") else wargs.headless

    app_launcher = AppLauncher(wargs)
    sim_app = app_launcher.app

    import time
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
    import cbf_task  # noqa
    from cbf_task.agents import rma_actor_critic  # noqa
    from cbf_task.agents import rma_classic_actor_critic  # noqa
    from cbf_task.locomotion_loader import load_locomotion_actor
    from cbf_task.eval_utils import eval_cell, aggregate_across_d, map_to_action

    K, dr_frac, label = DEFAULT_LEVELS[level_idx]
    print(f"\n  WORKER  --  level {level_idx+1} ({label}): K={K}  DR={dr_frac:.0%}")

    # Build env with overridden K. Loaded cfg will go through __post_init__
    # which sets random_topology_K=3 by default; we override AFTER that.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loco = load_locomotion_actor(retrieve_file_path(wargs.locomotion_ckpt), device)
    env_cfg = load_cfg_from_registry(wargs.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = wargs.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = wargs.seed + level_idx   # different seed per level
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    # Override the obstacle count BEFORE env init. Also need to swap the
    # `obstacles` fallback list to have the right K entries (action term's
    # init reads cfg.obstacles to size its tensors).
    env_cfg.actions.cbf_param.random_topology_K = K
    env_cfg.actions.cbf_param.obstacles = [
        # spread along x to give the init shape; positions immediately
        # overwritten by random_topology sampling at first reset
        (2.0 + 0.8 * i, 0.0, 0.5) for i in range(K)
    ]

    env = gym.make(wargs.task, cfg=env_cfg)
    cbf = env.unwrapped.action_manager._terms["cbf_param"]

    # Override DR ranges in-place to scale magnitude. Friction/mass/motor
    # are read from cfg in __init__; the action term has both the
    # _<name>_lo/_hi attrs (mutable) and the cfg fields. We mutate the
    # ATTRS since that's what reset() reads.
    for name in ("friction", "base_mass", "motor_strength", "disturbance_force"):
        cfg_key = name if name != "disturbance_force" else "disturbance"
        nom, full_lo, full_hi = DR_RANGES[cfg_key]
        lo, hi = _scaled_range(nom, full_lo, full_hi, dr_frac)
        setattr(cbf, f"_{name}_lo", float(lo))
        setattr(cbf, f"_{name}_hi", float(hi))
    print(f"    DR ranges (frac={dr_frac:.2f}):  "
          f"fric={cbf._friction_lo:.2f}..{cbf._friction_hi:.2f}  "
          f"mass={cbf._base_mass_lo:+.2f}..{cbf._base_mass_hi:+.2f}  "
          f"motor={cbf._motor_strength_lo:.2f}..{cbf._motor_strength_hi:.2f}  "
          f"dist={cbf._disturbance_force_lo:.1f}..{cbf._disturbance_force_hi:.1f}")

    agent_cfg = load_cfg_from_registry(wargs.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    teacher_ckpt = retrieve_file_path(wargs.teacher_ckpt)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                             log_dir=None, device=device)
    runner.load(teacher_ckpt)
    print(f"    loaded teacher: {teacher_ckpt}")

    # ----- assemble policies (teacher + B-trivial + optional baselines) -----
    def teacher_af(_cbf, _step):
        obs = env_wrapped.get_observations().to(device)
        with torch.inference_mode():
            return runner.get_inference_policy(device=device)(obs)

    def make_const_af(phi_v, alpha_v):
        N = wargs.num_envs
        phi_t = torch.full((N,), float(phi_v), device=device)
        alpha_t = torch.full((N,), float(alpha_v), device=device)
        action = map_to_action(phi_t, alpha_t, cbf)
        def af(_cbf, _step, _a=action): return _a
        return af

    def make_b2_af(alpha_v, eps0, lam):
        N = wargs.num_envs
        alpha_t = torch.full((N,), float(alpha_v), device=device)
        def af(_cbf, _step):
            h = _cbf.last_h_realized.clamp(min=0.0)
            phi_t = ((1.0 / float(eps0)) * torch.exp(-float(lam) * h)
                     ).clamp(min=_cbf._phi_lo, max=_cbf._phi_hi)
            return map_to_action(phi_t, alpha_t, _cbf)
        return af

    policies = [("teacher", teacher_af),
                ("B-trivial(phi=0,alpha=2.5)", make_const_af(0.0, 2.5))]

    if wargs.baselines_summary:
        with open(wargs.baselines_summary) as f:
            baselines = json.load(f)
        for entry in baselines:
            name = entry["baseline"]
            params = entry.get("params")
            if params is None:
                continue
            if name == "B0":
                policies.append((f"B0-best(a={params['alpha']:.2f})",
                                 make_const_af(0.0, params["alpha"])))
            elif name == "B1":
                policies.append(
                    (f"B1-best(a={params['alpha']:.2f},phi={params['phi']:.2f})",
                     make_const_af(params["phi"], params["alpha"])))
            elif name == "B2":
                policies.append(
                    (f"B2-best(a={params['alpha']:.2f},e={params['eps0']:.2f},l={params['lam']:.2f})",
                     make_b2_af(params["alpha"], params["eps0"], params["lam"])))

    # ----- run all (policy, disturbance) cells -----
    level_label = f"L{level_idx+1}_{label}"
    cell_rows = []
    summary_rows = []
    for pol_name, af in policies:
        per_pol_rows = []
        print(f"  policy: {pol_name}")
        for d in wargs.disturbances:
            t0 = time.time()
            m = eval_cell(env, cbf, af, d, wargs.eval_steps,
                          wargs.eval_eps_per_cell, device)
            wall = time.time() - t0
            row = {"level": level_label, "K": K, "dr_frac": dr_frac,
                   "policy": pol_name, "disturbance": float(d), **m}
            per_pol_rows.append(row)
            cell_rows.append(row)
            print(f"    d={d:>5.1f}  coll={m['collision_rate']:.2f}  "
                  f"reach={m['reach_rate']:.2f}  fall={m['fall_rate']:.2f}  "
                  f"stuck={m['stuck_rate']:.2f}  "
                  f"int={m['intervention_mean']:.0f}  (wall={wall:.1f}s)")
        agg = aggregate_across_d(per_pol_rows)
        summary_rows.append({"level": level_label, "K": K, "dr_frac": dr_frac,
                             "policy": pol_name, **agg})

    with open(json_out, "w") as f:
        json.dump({"cells": cell_rows, "summary": summary_rows}, f, indent=2)
    print(f"  wrote -> {json_out}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    if args_known.level_idx is not None:
        if args_known.json_out is None:
            print("--level_idx requires --json_out", file=sys.stderr)
            sys.exit(2)
        run_worker(args_known.level_idx, args_known.json_out)
    else:
        run_driver()
