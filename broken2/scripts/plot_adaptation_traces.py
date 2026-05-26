"""Render α(t)/φ(t)/h(t) adaptation traces from a dump_rollout.npz.

Use as the paper/talk "the model is doing something interesting" visual.
Highlights moments where the CBF intervened (qp_active spikes) and where
the policy's α/φ moved in response.

Usage:
  python3 scripts/plot_adaptation_traces.py \\
    data_from_lab/dump_v13_1_trainmatch.npz \\
    --env_idx 0 \\
    --output docs/viz/adapt_trace_v13_1.png
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz_path", type=str)
    p.add_argument("--env_idx", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--title", type=str, default=None,
                   help="Plot supertitle. Defaults to task name.")
    p.add_argument("--max_steps", type=int, default=600)
    args = p.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    e = args.env_idx
    S_full = data["pos_history"].shape[0]
    S = min(args.max_steps, S_full)

    pos = data["pos_history"][:S, e]            # (S, 3)
    alpha = data["alpha_history"][:S, e]         # (S,)
    phi = data["phi_history"][:S, e]             # (S,)
    h = data["h_history"][:S, e]                 # (S,)
    defl = data["deflection_history"][:S, e]     # (S,)
    qp_active = data["qp_active_history"][:S, e] # (S,) bool
    cmd = data["cmd_vel_history"][:S, e]         # (S, 3)
    obstacles = data["obstacle_positions"]

    # Detect CBF intervention episodes (qp_active runs).
    qp_starts = []
    qp_ends = []
    in_run = False
    for s in range(S):
        if qp_active[s] and not in_run:
            qp_starts.append(s); in_run = True
        elif not qp_active[s] and in_run:
            qp_ends.append(s); in_run = False
    if in_run:
        qp_ends.append(S)

    t = np.arange(S) * 0.02  # sim dt = 0.02s

    # ── Layout: 4 panels, left = trajectory (square), right = stacked traces ──
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 2.0])
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_params = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 1], sharex=ax_params)
    ax_defl = fig.add_subplot(gs[2, 1], sharex=ax_params)

    # ── Trajectory ──
    px, py = pos[:, 0], pos[:, 1]
    margin = 1.5
    ax_traj.set_xlim(px.min() - margin, px.max() + margin)
    ax_traj.set_ylim(py.min() - margin, py.max() + margin)
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_title("Trajectory (top-down)", fontsize=11)
    ax_traj.set_xlabel("x [m]"); ax_traj.set_ylabel("y [m]")

    # Static obstacles (radius 0.3 m per SHIELD convention).
    for ob in obstacles:
        if np.linalg.norm(ob[:2]) > 50: continue
        ax_traj.add_patch(patches.Circle(
            (ob[0], ob[1]), 0.30, fc="#b91c1c", ec="#7f1d1d",
            alpha=0.4, zorder=2))

    # Path. Color = α (warmer = higher α). Single LineCollection-equivalent.
    from matplotlib.collections import LineCollection
    points = np.array([px, py]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis",
                        norm=plt.Normalize(0.5, 3.0))
    lc.set_array(alpha[:-1])
    lc.set_linewidth(2.0)
    ax_traj.add_collection(lc)
    fig.colorbar(lc, ax=ax_traj, label="α (CBF gain)", shrink=0.6)

    # CBF intervention markers.
    if qp_active.any():
        ax_traj.scatter(px[qp_active], py[qp_active],
                        c="#ff7f0e", marker="*", s=40, zorder=5,
                        label="CBF active")
        ax_traj.legend(loc="upper right", fontsize=9)

    # Robot start + end markers.
    ax_traj.plot(px[0], py[0], "go", ms=10, zorder=6)
    ax_traj.plot(px[-1], py[-1], "ro", ms=10, zorder=6)

    # ── α and φ over time ──
    ax_params.set_title("Adaptive CBF parameters", fontsize=11)
    ax_params.plot(t, alpha, color="#1f77b4", lw=1.8, label="α (recovery rate)")
    ax_params.plot(t, phi, color="#d62728", lw=1.8, label="φ (ISSf margin)")
    ax_params.set_ylabel("param value")
    ax_params.legend(loc="upper right", fontsize=9)
    ax_params.grid(True, alpha=0.3)
    # Shade qp-active intervals.
    for s_, e_ in zip(qp_starts, qp_ends):
        ax_params.axvspan(s_ * 0.02, e_ * 0.02, color="#ff7f0e",
                          alpha=0.15, zorder=0)

    # ── h(x) over time ──
    ax_h.set_title("Safety margin h(x)", fontsize=11)
    ax_h.plot(t, h, color="#2ca02c", lw=1.5)
    ax_h.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax_h.set_ylabel("h(x)")
    ax_h.grid(True, alpha=0.3)
    for s_, e_ in zip(qp_starts, qp_ends):
        ax_h.axvspan(s_ * 0.02, e_ * 0.02, color="#ff7f0e", alpha=0.15)

    # ── Deflection ──
    ax_defl.set_title("|u_des − u_safe| (CBF deflection magnitude)", fontsize=11)
    ax_defl.plot(t, defl, color="#9467bd", lw=1.5)
    ax_defl.set_ylabel("|deflection|")
    ax_defl.set_xlabel("time [s]")
    ax_defl.grid(True, alpha=0.3)
    for s_, e_ in zip(qp_starts, qp_ends):
        ax_defl.axvspan(s_ * 0.02, e_ * 0.02, color="#ff7f0e", alpha=0.15)

    title = args.title or f"{data.get('task', '?')}  (env {e})"
    fig.suptitle(title, fontsize=12)

    out = args.output or str(Path(args.npz_path).with_suffix(".png").name)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[plot] saved {out}")

    # Print summary stats too.
    print(f"\n[stats] α range: {alpha.min():.2f}–{alpha.max():.2f}  "
          f"mean={alpha.mean():.2f}  std={alpha.std():.3f}")
    print(f"[stats] φ range: {phi.min():.2f}–{phi.max():.2f}  "
          f"mean={phi.mean():.2f}  std={phi.std():.3f}")
    print(f"[stats] CBF active fraction: {qp_active.mean():.1%}  "
          f"({len(qp_starts)} intervention episodes)")
    print(f"[stats] min h(x): {h.min():.3f}  (negative = boundary breach)")


if __name__ == "__main__":
    main()
