"""Cross-scene comparison of the learned teacher vs train-tuned baselines.

Isaac Lab allows only ONE sim context per Python process, so we can't
loop scenes in a single process (env.close() does not release the
context fully). Architecture:

  - DRIVER mode (default): spawn one CHILD process per scene. Each child
    runs ONE scene's full disturbance sweep across all policies, writes
    a per-scene JSON. Driver aggregates the JSONs into a combined CSV
    plus a printed summary table.
  - WORKER mode (--scene X --json_out PATH): does one scene end-to-end
    and writes the JSON. Invoked by the driver; can also be invoked
    manually for debugging a single scene.

Workflow:
    1. Train the teacher on the SHIELD env -> model_final.pt
    2. Run baselines on the SHIELD env to pick the best per family:
         ~/IsaacLab/isaaclab.sh -p phase5_baselines.py \\
             --task Isaac-CBF-Adaptive-Go2-UnifiedLidarSDF-v0 \\
             --out_dir phase6_shield_baselines_outputs --headless
    3. Run THIS script in DRIVER mode:
         ~/IsaacLab/isaaclab.sh -p phase6_eval_scenes.py \\
             --teacher_ckpt phase6_shield_v7_teacher_outputs/rsl_rl/model_final.pt \\
             --locomotion_ckpt /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
             --baselines_summary phase6_shield_baselines_outputs/phase5_baselines_summary.json \\
             --headless

Scenes evaluated (held-out from training; teacher trained on E2-equivalent
geometry only):
    E1 SINGLE       one cylinder on the path; basic CBF
    E2 SLALOM       the training geometry; in-distribution regression check
    E3 DENSE FIELD  5 obstacles; multi-obstacle SDF stress
    E4 NARROW GAP   two cylinders, ~1.2m corridor; tight tolerance

E0 (empty) was dropped because the action term's K=0 path crashes (see
cbf_task/cbf_adaptive_env_cfg.py comment). B-trivial (phi=0, alpha=2.5)
in the comparison serves as the no-tuning baseline.

Default disturbance grid is {0, 15, 30} N -- d=45 dropped because it's
out-of-DR extrapolation where every controller fails (IMPLEMENTATION.md
S11).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

DEFAULT_SCENES = ["E1", "E2", "E3", "E4"]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher_ckpt", required=True,
                    help="Path to the trained teacher's model_final.pt")
parser.add_argument("--locomotion_ckpt", required=True,
                    help="Frozen locomotion checkpoint (Go2 stock).")
parser.add_argument("--baselines_summary", default=None,
                    help="Path to phase5_baselines_summary.json (from a "
                         "phase5_baselines.py run on the SHIELD env). If "
                         "omitted, only the teacher + B-trivial are evaluated.")
parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES,
                    choices=DEFAULT_SCENES,
                    help="Subset of held-out scenes to evaluate.")
parser.add_argument("--scene_prefix", default="Eval",
                    help="Task name prefix. Default 'Eval' resolves to "
                         "Isaac-CBF-Adaptive-Go2-Eval{X}-v0 (7-priv SHIELD "
                         "obs, 198-dim). Use 'EvalNoPriv' for 4-priv "
                         "RMA-classic obs (195-dim) compatible with NoPriv "
                         "teachers.")
parser.add_argument("--disturbances", type=float, nargs="+",
                    default=[0.0, 15.0, 30.0])
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--eval_eps_per_cell", type=int, default=256,
                    help="Capped at num_envs.")
parser.add_argument("--eval_steps", type=int, default=1250)
parser.add_argument("--out_dir", default="phase6_eval_scenes_outputs")
parser.add_argument("--seed", type=int, default=0)
# WORKER mode flags (set by the driver when it spawns a child):
parser.add_argument("--scene", default=None,
                    help="WORKER mode: run only this single scene and write "
                         "results to --json_out. If unset, run as DRIVER.")
parser.add_argument("--json_out", default=None,
                    help="WORKER mode: where the worker writes its per-scene "
                         "JSON results.")
# AppLauncher flags are only needed in WORKER mode -- the driver doesn't
# touch Isaac. We add them conditionally below.

# Peek at args to decide driver-vs-worker without launching Isaac in driver mode.
args_known, _ = parser.parse_known_args()


def run_driver():
    """Orchestrate one worker subprocess per scene. Aggregate results."""
    os.makedirs(args_known.out_dir, exist_ok=True)

    # Use this script's own path; isaaclab.sh handles the python launch.
    here = os.path.dirname(os.path.abspath(__file__))
    isaaclab_sh = os.path.expanduser("~/IsaacLab/isaaclab.sh")
    if not os.path.exists(isaaclab_sh):
        print(f"  [ERROR] isaaclab.sh not found at {isaaclab_sh}")
        sys.exit(1)

    per_scene_results = []
    for scene in args_known.scenes:
        json_out = os.path.join(args_known.out_dir, f"scene_{scene}_results.json")
        cmd = [
            isaaclab_sh, "-p", os.path.join(here, "phase6_eval_scenes.py"),
            "--teacher_ckpt", args_known.teacher_ckpt,
            "--locomotion_ckpt", args_known.locomotion_ckpt,
            "--scene", scene,
            "--scene_prefix", args_known.scene_prefix,
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
        print(f"  DRIVER  --  spawning worker for scene {scene}")
        print("=" * 96)
        # Stream worker stdout/stderr directly to the driver's terminal
        # so the user sees per-cell rates live. Worker still writes JSON
        # for the driver to aggregate.
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  [ERROR] worker for {scene} exited with code {rc}; "
                  f"continuing with remaining scenes")
            continue
        if not os.path.exists(json_out):
            print(f"  [ERROR] worker for {scene} produced no JSON; skipping")
            continue
        with open(json_out) as f:
            per_scene_results.append(json.load(f))

    # ===== aggregate =====
    import csv as _csv
    all_rows = []
    summary_rows = []
    for r in per_scene_results:
        all_rows.extend(r["cells"])
        summary_rows.extend(r["summary"])
    cells_path = os.path.join(args_known.out_dir, "phase6_eval_scenes_cells.csv")
    summ_path = os.path.join(args_known.out_dir, "phase6_eval_scenes_summary.csv")
    if all_rows:
        with open(cells_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
    if summary_rows:
        with open(summ_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)

    # ===== printed summary =====
    print()
    print("=" * 96)
    print("  CROSS-SCENE SUMMARY  (worst across "
          f"{args_known.disturbances} N)")
    print("=" * 96)
    print(f"  {'scene':<6}  {'policy':<48}  {'wcoll':>6}  {'wreach':>7}  "
          f"{'mint':>7}  {'mjit':>6}")
    for r in summary_rows:
        print(f"  {r['scene']:<6}  {r['policy']:<48}  "
              f"{r['worst_coll']:>6.2f}  {r['worst_reach']:>7.2f}  "
              f"{r['mean_int']:>7.0f}  {r['mean_jitter']:>6.3f}")
    print()
    print(f"  saved -> {cells_path}")
    print(f"  saved -> {summ_path}")


def run_worker(scene: str, json_out: str):
    """Evaluate one scene end-to-end and write a JSON with cells + summary."""
    # Worker needs Isaac. Defer all heavy imports until here so the driver
    # branch above stays Isaac-free and starts instantly.
    from isaaclab.app import AppLauncher

    # Re-parse args INCLUDING AppLauncher flags (the driver passes
    # --headless through subprocess argv).
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--locomotion_ckpt", required=True)
    p.add_argument("--baselines_summary", default=None)
    p.add_argument("--scene", required=True)
    p.add_argument("--scene_prefix", default="Eval")
    p.add_argument("--json_out", required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--eval_eps_per_cell", type=int, default=256)
    p.add_argument("--eval_steps", type=int, default=1250)
    p.add_argument("--disturbances", type=float, nargs="+",
                   default=[0.0, 15.0, 30.0])
    p.add_argument("--out_dir", default=".")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)  # unused; from driver
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
    from cbf_task.locomotion_loader import load_locomotion_actor
    from cbf_task.eval_utils import eval_cell, aggregate_across_d, map_to_action

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loco = load_locomotion_actor(retrieve_file_path(wargs.locomotion_ckpt), device)

    task = f"Isaac-CBF-Adaptive-Go2-{wargs.scene_prefix}{scene}-v0"
    print(f"  WORKER  --  scene {scene}  task={task}")

    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = wargs.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = wargs.seed
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env = gym.make(task, cfg=env_cfg)
    cbf = env.unwrapped.action_manager._terms["cbf_param"]

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    teacher_ckpt = retrieve_file_path(wargs.teacher_ckpt)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                             log_dir=None, device=device)
    runner.load(teacher_ckpt)
    print(f"    loaded teacher ckpt: {teacher_ckpt}")

    # ----- assemble policy list (teacher + B-trivial + train-tuned baselines) -----
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
                print(f"    [WARN] {name} has no `params` field; "
                      "regenerate with updated phase5_baselines.py")
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
            row = {"scene": scene, "policy": pol_name,
                   "disturbance": float(d), **m}
            per_pol_rows.append(row)
            cell_rows.append(row)
            print(f"    d={d:>5.1f}  coll={m['collision_rate']:.2f}  "
                  f"reach={m['reach_rate']:.2f}  fall={m['fall_rate']:.2f}  "
                  f"stuck={m['stuck_rate']:.2f}  "
                  f"int={m['intervention_mean']:.0f}  (wall={wall:.1f}s)")
        agg = aggregate_across_d(per_pol_rows)
        summary_rows.append({"scene": scene, "policy": pol_name, **agg})

    with open(json_out, "w") as f:
        json.dump({"cells": cell_rows, "summary": summary_rows}, f, indent=2)
    print(f"  wrote -> {json_out}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    if args_known.scene is not None:
        if args_known.json_out is None:
            print("--scene requires --json_out", file=sys.stderr)
            sys.exit(2)
        run_worker(args_known.scene, args_known.json_out)
    else:
        run_driver()
