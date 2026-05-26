"""Non-learned CBF-parameterization baselines for the RMA teacher (BR)
to beat. All three are evaluated on the same env + disturbance sweep +
priv DR; the best per-family config is reported.

- **B0**  Exponential CBF (Ames et al. 2017): α_t = α_const, φ_t = 0.
          Sweep α. 3 configs.
- **B1**  ECBF + fixed ISSf: α_t = α_const, φ_t = φ_const.
          Sweep (α, φ). 6 configs.
- **B2**  TISSf-CBF (Cohen et al. 2024 / Molnar et al. 2023):
          α_t = α_const, φ_t = (1/ε₀) · exp(-λ · h_t).
          φ maxes at 1/ε₀ near the obstacle (h→0), decays as the
          system becomes safer (h grows). Sweep (α, ε₀, λ). 6 configs.

BR has to beat the best of these aggregates to claim the learned
adaptive parameterization is worth it.

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase5_baselines.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --num_envs 64 --headless
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-RMA-v0")
parser.add_argument("--num_envs", type=int, default=256,
                    help="Match teacher training (256). 64 gives ~±12pp CI on "
                         "collision rate, 256 narrows to ~±6pp.")
parser.add_argument("--eval_steps", type=int, default=1250)
parser.add_argument("--eval_eps_per_cell", type=int, default=256,
                    help="Cap at num_envs. With 256 envs we get up to 256 "
                         "episodes per (config, disturbance) cell.")
parser.add_argument("--eval_disturbances", type=float, nargs="+",
                    default=[0.0, 15.0, 30.0, 45.0])
parser.add_argument("--out_dir", default="phase5_baseline_outputs")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import csv
import json
from collections import defaultdict

import gymnasium as gym
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor


# ---- baseline sweep grids ----
# B0: Exponential CBF -- 3 alpha values, phi pinned at 0
B0_ALPHAS = [1.0, 2.5, 4.0]                               # 3 cells

# B1: ECBF + fixed ISSf -- 3 alpha values * 2 phi values
B1_ALPHAS = [1.0, 2.5, 4.0]
B1_PHIS = [0.3, 1.0]                                       # 6 cells

# B2: TISSf -- phi(h) = (1/eps0) * exp(-lambda * h), alpha constant.
# (eps0, lam) pairs picked to cover (max margin, slow decay) ->
# (medium margin, fast decay).
B2_ALPHAS = [1.0, 2.5]
B2_EPS0_LAM = [
    (1.0, 0.5),   # phi_max=1.0, slow decay  -- aggressive hedging
    (1.0, 1.0),   # phi_max=1.0, medium decay
    (2.0, 1.0),   # phi_max=0.5, medium decay -- more conservative
]                                                         # 6 cells

# Eval helpers extracted to cbf_task/eval_utils.py so other scripts
# (phase6_eval_scenes.py) can import them without triggering this
# module's top-level argparse + AppLauncher side effects.
from cbf_task.eval_utils import (
    EVAL_COLS, UNSAFE_THR, map_to_action, eval_cell,
    aggregate_across_d, best_safe_cell,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    cbf = env.unwrapped.action_manager._terms["cbf_param"]

    N = args_cli.num_envs

    # ===== B0: const alpha, phi=0 =====
    print("\n" + "=" * 90)
    print("  B0  --  const alpha, phi=0")
    print("=" * 90)
    b0_cells = []
    for alpha_v in B0_ALPHAS:
        cell_rows = []
        print(f"  alpha={alpha_v}")
        for d in args_cli.eval_disturbances:
            phi_t = torch.zeros(N, device=device)
            alpha_t = torch.full((N,), alpha_v, device=device)
            action = map_to_action(phi_t, alpha_t, cbf)
            def af(cbf, step, _a=action): return _a
            m = eval_cell(env, cbf, af, d, args_cli.eval_steps,
                          args_cli.eval_eps_per_cell, device)
            cell_rows.append({"d": d, **m})
            print(f"    d={d:>5.1f}  coll={m['collision_rate']:.2f}  "
                  f"reach={m['reach_rate']:.2f}  fall={m['fall_rate']:.2f}  "
                  f"stuck={m['stuck_rate']:.2f}  int={m['intervention_mean']:.0f}")
        b0_cells.append({
            "baseline": "B0", "phi": 0.0, "alpha": alpha_v,
            "rows": cell_rows, "agg": aggregate_across_d(cell_rows),
        })

    # ===== B1: const (phi, alpha) =====
    print("\n" + "=" * 90)
    print("  B1  --  const (phi, alpha)")
    print("=" * 90)
    b1_cells = []
    for phi_v in B1_PHIS:
        for alpha_v in B1_ALPHAS:
            cell_rows = []
            print(f"  phi={phi_v}  alpha={alpha_v}")
            for d in args_cli.eval_disturbances:
                phi_t = torch.full((N,), phi_v, device=device)
                alpha_t = torch.full((N,), alpha_v, device=device)
                action = map_to_action(phi_t, alpha_t, cbf)
                def af(cbf, step, _a=action): return _a
                m = eval_cell(env, cbf, af, d, args_cli.eval_steps,
                              args_cli.eval_eps_per_cell, device)
                cell_rows.append({"d": d, **m})
                print(f"    d={d:>5.1f}  coll={m['collision_rate']:.2f}  "
                      f"reach={m['reach_rate']:.2f}  fall={m['fall_rate']:.2f}  "
                      f"stuck={m['stuck_rate']:.2f}  int={m['intervention_mean']:.0f}")
            b1_cells.append({
                "baseline": "B1", "phi": phi_v, "alpha": alpha_v,
                "rows": cell_rows, "agg": aggregate_across_d(cell_rows),
            })

    # ===== B2: TISSf-CBF (Cohen 2024 / Molnar 2023) =====
    # alpha constant. phi(h) = (1/eps0) * exp(-lambda * h).
    # phi is clamped to the env's phi_bounds. h read from cbf.last_h_realized.
    print("\n" + "=" * 90)
    print("  B2  --  TISSf-CBF: phi(h) = (1/eps0) * exp(-lambda * h), alpha const")
    print("=" * 90)
    b2_cells = []
    for alpha_v in B2_ALPHAS:
        for eps0, lam in B2_EPS0_LAM:
            cell_rows = []
            print(f"  alpha={alpha_v}  eps0={eps0}  lambda={lam}  "
                  f"(phi_max={1.0/eps0:.2f})")
            def make_action_fn(_alpha=alpha_v, _eps0=eps0, _lam=lam):
                def af(cbf, step):
                    h = cbf.last_h_realized.clamp(min=0.0)
                    phi_t = ((1.0 / _eps0) * torch.exp(-_lam * h)
                             ).clamp(min=cbf._phi_lo, max=cbf._phi_hi)
                    alpha_t = torch.full((N,), _alpha, device=device)
                    return map_to_action(phi_t, alpha_t, cbf)
                return af
            action_fn = make_action_fn()
            for d in args_cli.eval_disturbances:
                m = eval_cell(env, cbf, action_fn, d, args_cli.eval_steps,
                              args_cli.eval_eps_per_cell, device)
                cell_rows.append({"d": d, **m})
                print(f"    d={d:>5.1f}  coll={m['collision_rate']:.2f}  "
                      f"reach={m['reach_rate']:.2f}  fall={m['fall_rate']:.2f}  "
                      f"stuck={m['stuck_rate']:.2f}  int={m['intervention_mean']:.0f}")
            b2_cells.append({
                "baseline": "B2",
                "alpha": alpha_v, "eps0": eps0, "lam": lam,
                "rows": cell_rows, "agg": aggregate_across_d(cell_rows),
            })

    # ===== summary: best per baseline =====
    print("\n" + "=" * 96)
    print("  BASELINE SUMMARY  --  best deployable setting per family")
    print("=" * 96)
    summary = []
    for name, cells in [("B0", b0_cells), ("B1", b1_cells), ("B2", b2_cells)]:
        best, was_safe = best_safe_cell(cells)
        if name == "B0":
            cfg_desc = f"alpha={best['alpha']}, phi=0"
        elif name == "B1":
            cfg_desc = f"alpha={best['alpha']}, phi={best['phi']}"
        else:
            cfg_desc = (f"alpha={best['alpha']}, eps0={best['eps0']}, "
                        f"lambda={best['lam']}  (phi_max={1.0/best['eps0']:.2f})")
        a = best["agg"]
        tag = "SAFE" if was_safe else "FALLBACK"
        print(f"  {name}  [{tag}]  {cfg_desc}")
        print(f"     worst_coll={a['worst_coll']:.2f}  "
              f"worst_reach={a['worst_reach']:.2f}  "
              f"mean_int={a['mean_int']:.0f}  "
              f"mean_jitter={a['mean_jitter']:.3f}")
        # also export the structured params (alpha, phi, eps0, lam) so
        # downstream scripts (phase6_eval_scenes.py) can load the "best
        # train-tuned" baseline params without regex-parsing cfg_desc.
        params = {}
        if name == "B0":
            params = {"alpha": float(best["alpha"]), "phi": 0.0}
        elif name == "B1":
            params = {"alpha": float(best["alpha"]),
                      "phi": float(best["phi"])}
        else:
            params = {"alpha": float(best["alpha"]),
                      "eps0": float(best["eps0"]),
                      "lam": float(best["lam"])}
        summary.append({"baseline": name, "config": cfg_desc, "safe": was_safe,
                        "params": params, **a})
    print("=" * 96)
    print("  BR (learned adaptive) must beat the best of these aggregates on the same env.")

    # write cells + summary
    os.makedirs(args_cli.out_dir, exist_ok=True)
    all_rows = []
    for cells in (b0_cells, b1_cells, b2_cells):
        for c in cells:
            for r in c["rows"]:
                row = {k: c.get(k) for k in
                       ("baseline", "phi", "alpha", "eps0", "lam")}
                row.update(r)
                all_rows.append(row)
    fieldnames = sorted({k for r in all_rows for k in r.keys()})
    with open(os.path.join(args_cli.out_dir, "phase5_baselines_cells.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(all_rows)
    with open(os.path.join(args_cli.out_dir, "phase5_baselines_summary.json"),
              "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved -> {args_cli.out_dir}/")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
