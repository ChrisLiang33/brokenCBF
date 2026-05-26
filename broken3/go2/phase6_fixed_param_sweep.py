"""Fixed (phi, alpha) sweep across obstacle distance.

Phase 0.6-style gate: does the OPTIMAL fixed (phi, alpha) actually move
with obstacle distance? If yes -> there IS a signal for an adaptive
policy to learn (and the current pegged-at-max teacher is genuinely
failing). If no -> scenario gives lidar no signal, the pegged policy
is correct, and the bug is the scenario/reward, not the policy.

For each (phi, alpha) on a coarse grid:
  - inject fixed action (phi, alpha) into the cbf action term for every
    env, every step (the trained policy is NOT loaded)
  - roll out N steps, log per-step:
        cbf.last_h_realized        (signed distance, negative = collision)
        cbf.last_intervention      (||u_safe - u_nom||)
        prev_dist_to_goal - last_dist_to_goal     (per-step progress)
  - bucket per-step samples by `h_realized` bin and aggregate

Then for each obstacle-distance bin:
  - compute (collision_rate, mean_intervention, mean_progress) per
    (phi, alpha)
  - pick the "best" (phi, alpha) = argmax(progress) s.t. coll_rate <= thr
  - if the best (phi, alpha) is THE SAME across all bins -> FLAT.
    Means no signal. Stop tuning policy. Redesign scenario.
  - if the best (phi, alpha) MOVES with distance -> SIGNAL. Means the
    policy is genuinely under-using lidar.

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase6_fixed_param_sweep.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --task Isaac-CBF-Adaptive-Go2-RandObs-v0 \\
        --num_envs 256 --steps_per_combo 200 --headless
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True,
                    help="Frozen locomotion-policy checkpoint (Go2 stock).")
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-RandObs-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps_per_combo", type=int, default=200)
parser.add_argument("--disturbance", type=float, default=0.0,
                    help="Disturbance force (N). Set to a moderate value if "
                         "you want to test signal *under* disturbance.")
parser.add_argument("--phi_grid", default="0.0,0.25,0.5,0.75,1.0")
parser.add_argument("--alpha_grid", default="1.0,2.0,3.0,4.0")
parser.add_argument("--coll_rate_thr", type=float, default=0.05,
                    help="Per-step collision-rate ceiling for 'best' lookup.")
parser.add_argument("--out_dir", default="phase6_fixed_param_sweep_outputs")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor


# --- distance bins (m) for h_realized. Tuned to span typical obstacle
# distances under RandObs jitter (~0-5m). Wider top bin catches "no
# obstacle nearby" samples.
H_BINS = np.array([-0.5, 0.3, 0.8, 1.5, 3.0, 8.0])
H_BIN_LABELS = [
    "[h<0.3]  (very close)",
    "[0.3-0.8]",
    "[0.8-1.5]",
    "[1.5-3.0]",
    "[h>3.0]  (far)",
]


def _phi_alpha_to_action(phi: float, alpha: float, cbf, n: int, device):
    """Map a single (phi, alpha) point to the [-1, 1]^2 action that
    decodes to it inside CBFParamActionTerm.process_actions."""
    a_phi = 2.0 * (phi - cbf._phi_lo) / (cbf._phi_hi - cbf._phi_lo) - 1.0
    a_alpha = 2.0 * (alpha - cbf._alpha_lo) / (cbf._alpha_hi - cbf._alpha_lo) - 1.0
    a = torch.tensor([a_phi, a_alpha], device=device, dtype=torch.float32)
    return a.unsqueeze(0).expand(n, 2).clone()


def _run_combo(env, cbf, phi, alpha, n_steps, device):
    """Roll out n_steps with fixed (phi, alpha). Returns per-step arrays
    (concatenated over envs) of (h_realized, intervention, progress,
    collided_this_step)."""
    action = _phi_alpha_to_action(phi, alpha, cbf, env.num_envs, device)
    env.reset()
    h_buf, inter_buf, prog_buf, coll_buf = [], [], [], []
    for _ in range(n_steps):
        # cache pre-step dist-to-goal for progress calc
        prev_dist = cbf.last_dist_to_goal.detach().clone()
        env.step(action)
        h = cbf.last_h_realized.detach().clone()              # (N,)
        intervention = cbf.last_intervention.detach().clone()  # (N,)
        progress = (prev_dist - cbf.last_dist_to_goal).detach().clone()  # (N,)
        coll = (h < 0.0).float().detach().clone()             # (N,)
        h_buf.append(h)
        inter_buf.append(intervention)
        prog_buf.append(progress)
        coll_buf.append(coll)
    h = torch.cat(h_buf).cpu().numpy()
    inter = torch.cat(inter_buf).cpu().numpy()
    prog = torch.cat(prog_buf).cpu().numpy()
    coll = torch.cat(coll_buf).cpu().numpy()
    return h, inter, prog, coll


def _aggregate(h, inter, prog, coll):
    """For each h bin, return (count, coll_rate, mean_inter, mean_prog)."""
    rows = []
    for i in range(len(H_BINS) - 1):
        mask = (h >= H_BINS[i]) & (h < H_BINS[i + 1])
        n = int(mask.sum())
        if n == 0:
            rows.append({"count": 0, "coll_rate": float("nan"),
                         "mean_inter": float("nan"),
                         "mean_prog": float("nan")})
            continue
        rows.append({
            "count": n,
            "coll_rate": float(coll[mask].mean()),
            "mean_inter": float(inter[mask].mean()),
            "mean_prog": float(prog[mask].mean()),
        })
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    phi_grid = [float(x) for x in args_cli.phi_grid.split(",")]
    alpha_grid = [float(x) for x in args_cli.alpha_grid.split(",")]
    n_combos = len(phi_grid) * len(alpha_grid)
    print(f"[sweep] grid: {len(phi_grid)} phi x {len(alpha_grid)} alpha = "
          f"{n_combos} combos x {args_cli.steps_per_combo} steps")

    # --- env setup ---
    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    cbf = env.action_manager._terms["cbf_param"]

    # pin disturbance
    cbf._disturbance_force_lo = float(args_cli.disturbance)
    cbf._disturbance_force_hi = float(args_cli.disturbance)

    # --- run sweep ---
    # results[(phi, alpha)] -> list of per-bin dicts
    results = {}
    for phi in phi_grid:
        for alpha in alpha_grid:
            print(f"[sweep] phi={phi:.2f}  alpha={alpha:.2f} ...")
            with torch.inference_mode():
                h, inter, prog, coll = _run_combo(
                    env, cbf, phi, alpha, args_cli.steps_per_combo, device,
                )
            rows = _aggregate(h, inter, prog, coll)
            results[(phi, alpha)] = rows

    # ============================================================
    # PER-BIN TABLES
    # ============================================================
    print()
    print("=" * 100)
    print("  PER-BIN METRICS (one block per obstacle-distance bin)")
    print("=" * 100)

    per_bin_best = []   # (best_phi, best_alpha, best_prog) per bin
    for bi, label in enumerate(H_BIN_LABELS):
        print(f"\n  BIN {bi}: {label}")
        print(f"  {'phi':>5}  {'alpha':>5}  {'count':>7}  "
              f"{'coll%':>7}  {'inter':>7}  {'prog/step':>10}")
        # find best (phi, alpha) for this bin -- max progress s.t.
        # coll_rate <= thr; if nothing safe, fall back to min coll_rate.
        rows_for_bin = []
        for phi in phi_grid:
            for alpha in alpha_grid:
                r = results[(phi, alpha)][bi]
                rows_for_bin.append((phi, alpha, r))
                cnt = r["count"]
                cr = r["coll_rate"]
                it = r["mean_inter"]
                pr = r["mean_prog"]
                print(f"  {phi:>5.2f}  {alpha:>5.2f}  {cnt:>7d}  "
                      f"{100*cr:>6.2f}%  {it:>7.3f}  {pr:>+10.5f}"
                      if cnt > 0 else
                      f"  {phi:>5.2f}  {alpha:>5.2f}  {cnt:>7d}  "
                      f"{'-':>7}  {'-':>7}  {'-':>10}")
        # rank: prefer (coll_rate <= thr); among those, max progress
        safe_rows = [t for t in rows_for_bin
                     if t[2]["count"] > 0 and t[2]["coll_rate"] <= args_cli.coll_rate_thr]
        if safe_rows:
            best = max(safe_rows, key=lambda t: t[2]["mean_prog"])
            note = f"argmax progress | coll<={100*args_cli.coll_rate_thr:.0f}%"
        else:
            # fall back: min coll_rate
            with_data = [t for t in rows_for_bin if t[2]["count"] > 0]
            best = min(with_data, key=lambda t: t[2]["coll_rate"]) if with_data else None
            note = f"NO SAFE COMBO -- argmin coll_rate"
        if best is not None:
            print(f"  -> BEST: phi={best[0]:.2f} alpha={best[1]:.2f}  ({note})")
            per_bin_best.append((bi, label, best[0], best[1], best[2]))
        else:
            print(f"  -> BIN EMPTY (no samples)")
            per_bin_best.append((bi, label, None, None, None))

    # ============================================================
    # VERDICT
    # ============================================================
    print()
    print("=" * 100)
    print("  SIGNAL VERDICT  --  does optimal (phi, alpha) move with obstacle distance?")
    print("=" * 100)
    valid = [(bi, lbl, p, a, r) for (bi, lbl, p, a, r) in per_bin_best if p is not None]
    if len(valid) < 2:
        print("  INSUFFICIENT DATA -- not enough non-empty bins to compare")
        verdict = "INSUFFICIENT"
    else:
        unique_pairs = {(p, a) for (_, _, p, a, _) in valid}
        # also check: did MAX (phi, alpha) win every bin?
        phi_max, alpha_max = max(phi_grid), max(alpha_grid)
        max_dominant = all((p == phi_max and a == alpha_max)
                           for (_, _, p, a, _) in valid)
        print(f"  best (phi, alpha) per bin:")
        for bi, lbl, p, a, r in valid:
            print(f"    bin {bi} {lbl:>25}:  phi={p:.2f}  alpha={a:.2f}  "
                  f"prog={r['mean_prog']:+.5f}  coll={100*r['coll_rate']:.2f}%")
        print()
        if len(unique_pairs) == 1:
            pair = next(iter(unique_pairs))
            if max_dominant:
                verdict = (f"FLAT -- max (phi={pair[0]:.2f}, alpha={pair[1]:.2f}) wins "
                           f"EVERY bin. Scenario incentivizes max conservatism "
                           f"everywhere; lidar has no signal to drive adaptation. "
                           f"The pegged policy is CORRECT for this scenario. "
                           f"Stop tuning policy -- redesign scenario.")
            else:
                verdict = (f"FLAT -- same (phi={pair[0]:.2f}, alpha={pair[1]:.2f}) "
                           f"wins every bin (not the max corner). Optimal is "
                           f"constant across distance -> no signal for adaptation. "
                           f"Redesign scenario.")
        else:
            phis = {p for (_, _, p, _, _) in valid}
            alphas = {a for (_, _, _, a, _) in valid}
            verdict = (f"SIGNAL EXISTS -- optimal (phi, alpha) shifts across bins "
                       f"({len(unique_pairs)} unique winners; phi values "
                       f"{sorted(phis)}, alpha values {sorted(alphas)}). "
                       f"An adaptive policy should outperform any fixed combo. "
                       f"Current pegged teacher is genuinely under-using lidar. "
                       f"Justifies aux-loss / annealed-intervention fix.")
        print(f"  {verdict}")
    print("=" * 100)

    # --- save ---
    json_out = {
        "config": {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "steps_per_combo": args_cli.steps_per_combo,
            "disturbance": args_cli.disturbance,
            "phi_grid": phi_grid,
            "alpha_grid": alpha_grid,
            "h_bins": H_BINS.tolist(),
            "coll_rate_thr": args_cli.coll_rate_thr,
        },
        "results": {
            f"phi={p:.2f},alpha={a:.2f}": rows
            for (p, a), rows in results.items()
        },
        "per_bin_best": [
            {"bin_idx": bi, "bin_label": lbl,
             "phi": p, "alpha": a,
             "row": r}
            for (bi, lbl, p, a, r) in per_bin_best
        ],
        "verdict": verdict,
    }
    with open(os.path.join(args_cli.out_dir, "phase6_fixed_param_sweep.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    with open(os.path.join(args_cli.out_dir, "phase6_fixed_param_sweep_grid.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi", "alpha", "bin_idx", "bin_label",
                    "count", "coll_rate", "mean_intervention", "mean_progress"])
        for (p, a), rows in results.items():
            for bi, r in enumerate(rows):
                w.writerow([p, a, bi, H_BIN_LABELS[bi],
                            r["count"], r["coll_rate"],
                            r["mean_inter"], r["mean_prog"]])
    print(f"  saved -> {args_cli.out_dir}/")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
