"""Phase 0.6 analyzer -- aggregate per-(channel-value) CSVs, compute
optimal-φ per channel value, render the gate plot, print pass/fail.

Run *after* you've finished `phase0_6_phi_sweep.py` for each value of
whichever channel you're testing (friction_mu OR disturbance_force).
The analyzer is channel-agnostic: pass `--channel friction_mu` or
`--channel disturbance_force` to pick which column to use as the x-axis.

Aggregates over seeds; finds optimal-φ lexicographically (minimize
collision_rate, then mean_intervention).

Gate decision:
- PASS if optimal-φ range across the channel > 0.15.
  → φ has signal; advance to Phase 1.
- FAIL if optimal-φ is flat across the channel.
  → φ has no signal on this channel under this geometry/locomotion.
    Either tighten the scenario, push the channel further, or switch
    channels before any RL.

Usage:
    python analyze_phi_sweep.py phase0_6_d*.csv \\
        --channel disturbance_force -o phase0_6_gate.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(csv_paths: list[str], channel: str) -> dict:
    """rows[(channel_val, phi)] = list of per-seed row dicts."""
    rows = defaultdict(list)
    for path in csv_paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                if channel not in r:
                    raise KeyError(f"Channel column '{channel}' not in {path}. "
                                   f"Available: {list(r.keys())}")
                key = (round(float(r[channel]), 4),
                       round(float(r["phi"]), 4))
                rows[key].append({
                    "collided": int(r["collided"]),
                    "reached": int(r["reached"]),
                    "min_h_realized": float(r["min_h_realized"]),
                    "mean_intervention": float(r["mean_intervention"]),
                    "max_qp_slack": float(r["max_qp_slack"]),
                })
    return rows


def aggregate(rows: dict) -> list[dict]:
    """One aggregate dict per (channel_val, phi) cell."""
    out = []
    for (cv, phi), seeds in sorted(rows.items()):
        coll = np.array([r["collided"] for r in seeds], dtype=float)
        reach = np.array([r["reached"] for r in seeds], dtype=float)
        min_h = np.array([r["min_h_realized"] for r in seeds])
        intr = np.array([r["mean_intervention"] for r in seeds])
        out.append({
            "channel_val": cv, "phi": phi, "n": len(seeds),
            "collision_rate": float(coll.mean()),
            "reach_rate": float(reach.mean()),
            "mean_min_h": float(min_h.mean()),
            "std_min_h": float(min_h.std(ddof=0)),
            "mean_intervention": float(intr.mean()),
            "std_intervention": float(intr.std(ddof=0)),
        })
    return out


def optimal_phi_per_channel(cells: list[dict],
                            coll_thresh: float = 0.20,
                            reach_thresh: float = 0.60) -> dict:
    """For each channel value, pick the smallest φ that achieves
    collision_rate ≤ coll_thresh AND reach_rate ≥ reach_thresh.
    If no φ in the grid satisfies both, fall back to lexicographic:
    (collision_rate, -reach_rate, mean_intervention).

    The smallest-φ-that-meets-the-bar framing avoids the artifact we hit
    on Go2 Phase 0.6: at high disturbance, the robot gets pushed off
    course and never reaches the obstacle, which made
    "lowest-intervention among zero-collision cells" pick degenerate
    "didn't engage" episodes instead of competent-safe ones. Requiring
    reach_rate ≥ threshold filters those out.
    """
    by_cv = defaultdict(list)
    for c in cells:
        by_cv[c["channel_val"]].append(c)
    out = {}
    for cv, group in by_cv.items():
        ok = [c for c in group
              if c["collision_rate"] <= coll_thresh
              and c["reach_rate"] >= reach_thresh]
        if ok:
            out[cv] = min(ok, key=lambda c: c["phi"])
        else:
            out[cv] = min(group, key=lambda c: (c["collision_rate"],
                                                 -c["reach_rate"],
                                                 c["mean_intervention"]))
    return out


def render(cells: list[dict], best: dict, channel: str, out_png: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes

    cvs = sorted({c["channel_val"] for c in cells})
    phis = sorted({c["phi"] for c in cells})

    # --- panel 1: optimal-φ vs channel ---
    opt_phi = [best[cv]["phi"] for cv in cvs]
    opt_coll = [best[cv]["collision_rate"] for cv in cvs]
    ax1.plot(cvs, opt_phi, "o-", lw=2.5, color="#534AB7",
             label="argmin (collision, intervention)")
    ax1.set_xlabel(channel)
    ax1.set_ylabel("optimal fixed φ")
    ax1.set_title(f"THE GATE: does optimal φ move with {channel}?\n"
                  "flat = no φ signal, sloped = signal present")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="best")
    for cv, p, c in zip(cvs, opt_phi, opt_coll):
        if c > 0.0:
            ax1.annotate(f"coll={c:.2f}", (cv, p), textcoords="offset points",
                         xytext=(5, -12), fontsize=8, color="red")

    # --- panel 2: intervention(φ) curves, one line per channel value ---
    by_cv = defaultdict(list)
    for c in cells:
        by_cv[c["channel_val"]].append(c)
    cmap = plt.cm.viridis
    for i, cv in enumerate(cvs):
        group = sorted(by_cv[cv], key=lambda c: c["phi"])
        xs = [g["phi"] for g in group]
        ys = [g["mean_intervention"] for g in group]
        es = [g["std_intervention"] for g in group]
        color = cmap(i / max(len(cvs) - 1, 1))
        ax2.errorbar(xs, ys, yerr=es, marker="o", color=color,
                     label=f"{channel}={cv:g}", capsize=2)
    ax2.set_xlabel("fixed φ")
    ax2.set_ylabel("mean intervention cost (per step)")
    ax2.set_title(f"Intervention vs φ (one curve per {channel})")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="best")

    # --- panel 3: collision-rate heatmap ---
    grid = np.full((len(cvs), len(phis)), np.nan)
    cv_to_i = {cv: i for i, cv in enumerate(cvs)}
    phi_to_j = {phi: j for j, phi in enumerate(phis)}
    for c in cells:
        grid[cv_to_i[c["channel_val"]], phi_to_j[c["phi"]]] = c["collision_rate"]
    im = ax3.imshow(grid, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
                    origin="lower",
                    extent=[min(phis) - 0.05, max(phis) + 0.05,
                            min(cvs) - 0.05, max(cvs) + 0.05])
    ax3.set_xticks(phis)
    ax3.set_yticks(cvs)
    ax3.set_xlabel("fixed φ")
    ax3.set_ylabel(channel)
    ax3.set_title("Collision rate (red = collides)")
    fig.colorbar(im, ax=ax3, label="collision rate")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def gate_decision(best: dict) -> dict:
    cvs = sorted(best.keys())
    opt_phi = [best[cv]["phi"] for cv in cvs]
    phi_range = max(opt_phi) - min(opt_phi)
    deltas = np.diff(opt_phi)
    mono = bool(len(deltas) == 0 or (np.all(deltas >= 0) or np.all(deltas <= 0)))
    return {
        "channel_vals": cvs,
        "optimal_phi": opt_phi,
        "phi_range": float(phi_range),
        "monotone": mono,
        "pass": bool(phi_range > 0.15),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="+",
                    help="One or more per-channel-value CSV paths from "
                         "phase0_6_phi_sweep.py (globs ok).")
    ap.add_argument("--channel", default="friction_mu",
                    choices=["friction_mu", "disturbance_force"],
                    help="Which CSV column to use as the x-axis channel.")
    ap.add_argument("--coll_threshold", type=float, default=0.20,
                    help="Cells with collision_rate above this are deemed "
                         "unsafe and excluded from the optimum.")
    ap.add_argument("--reach_threshold", type=float, default=0.60,
                    help="Cells with reach_rate below this are deemed "
                         "degenerate (robot didn't engage) and excluded.")
    ap.add_argument("-o", "--out_png", default="phase0_6_gate.png")
    args = ap.parse_args()

    csv_paths: list[str] = []
    for p in args.csvs:
        matches = sorted(glob.glob(p))
        csv_paths.extend(matches if matches else [p])
    if not csv_paths:
        print("No CSVs found.", file=sys.stderr)
        sys.exit(2)
    print(f"[analyze] channel = {args.channel}")
    print(f"[analyze] reading {len(csv_paths)} CSV(s):")
    for p in csv_paths:
        print(f"    {p}")

    rows = load_runs(csv_paths, args.channel)
    cells = aggregate(rows)
    best = optimal_phi_per_channel(cells, args.coll_threshold, args.reach_threshold)
    render(cells, best, args.channel, args.out_png)
    decision = gate_decision(best)

    print()
    print("=" * 78)
    print(f"  Phase 0.6 GATE -- optimal fixed φ vs {args.channel}")
    print(f"  Optimum := smallest φ with coll≤{args.coll_threshold:.2f} "
          f"AND reach≥{args.reach_threshold:.2f}")
    print("=" * 78)
    for cv, p in zip(decision["channel_vals"], decision["optimal_phi"]):
        b = best[cv]
        flag = ""
        if b["collision_rate"] > args.coll_threshold or b["reach_rate"] < args.reach_threshold:
            flag = "  [fallback: no cell met thresholds]"
        print(f"    {args.channel}={cv:<6.3g}  φ*={p:.2f}   "
              f"coll={b['collision_rate']:.2f}   "
              f"reach={b['reach_rate']:.2f}   "
              f"int={b['mean_intervention']:.3f}{flag}")
    print()
    print(f"    optimal-φ range across channel : {decision['phi_range']:.3f}   "
          f"(pass > 0.15)")
    print(f"    monotone trend?                : {decision['monotone']}")
    print(f"    GATE PASS                      : {decision['pass']}")
    print("=" * 72)
    print(f"\n  saved -> {args.out_png}")


if __name__ == "__main__":
    main()
