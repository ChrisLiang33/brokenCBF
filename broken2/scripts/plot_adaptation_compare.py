"""Side-by-side α/φ/h trace comparison across multiple envs in a dump.

Shows that the same trained policy produces qualitatively different
α/φ patterns depending on the scenario (different obstacles, different
DR samples, different push events). Paper figure for "adaptive behavior
is regime-aware."

Usage:
  python3 scripts/plot_adaptation_compare.py \\
    data_from_lab/dump_v13_1_stressor_v2.npz \\
    --envs 0,1,2,3 \\
    --output docs/viz/adapt_compare_v13_1_stressor.png
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz_path", type=str)
    p.add_argument("--envs", type=str, default="0,1,2,3",
                   help="Comma-separated env indices to compare.")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=600)
    args = p.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    env_idxs = [int(x) for x in args.envs.split(",")]
    S_full = data["pos_history"].shape[0]
    S = min(args.max_steps, S_full)

    alpha_all = data["alpha_history"][:S]    # (S, N)
    phi_all = data["phi_history"][:S]
    h_all = data["h_history"][:S]
    qp_all = data["qp_active_history"][:S]
    t = np.arange(S) * 0.02

    n_envs = len(env_idxs)
    fig, axes = plt.subplots(3, n_envs, figsize=(4 * n_envs, 8),
                              sharex=True, constrained_layout=True)
    if n_envs == 1:
        axes = axes.reshape(3, 1)

    for col, e in enumerate(env_idxs):
        alpha = alpha_all[:, e]
        phi = phi_all[:, e]
        h = h_all[:, e]
        qp = qp_all[:, e]

        # CBF intervention spans
        spans = []
        in_run = False
        for s in range(S):
            if qp[s] and not in_run:
                s0 = s; in_run = True
            elif not qp[s] and in_run:
                spans.append((s0, s)); in_run = False
        if in_run:
            spans.append((s0, S))

        # Row 0: α
        ax = axes[0, col]
        ax.plot(t, alpha, color="#1f77b4", lw=1.5)
        for s0, s1 in spans:
            ax.axvspan(s0 * 0.02, s1 * 0.02, color="#ff7f0e", alpha=0.15)
        ax.set_ylim(0.3, 3.2)
        ax.grid(True, alpha=0.3)
        cbf_pct = qp.mean() * 100
        ax.set_title(f"env {e}  (CBF active {cbf_pct:.0f}%)", fontsize=10)
        if col == 0: ax.set_ylabel("α", fontsize=11)

        # Row 1: φ
        ax = axes[1, col]
        ax.plot(t, phi, color="#d62728", lw=1.5)
        for s0, s1 in spans:
            ax.axvspan(s0 * 0.02, s1 * 0.02, color="#ff7f0e", alpha=0.15)
        ax.set_ylim(-0.2, 5.2)
        ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel("φ", fontsize=11)

        # Row 2: h
        ax = axes[2, col]
        ax.plot(t, h, color="#2ca02c", lw=1.5)
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
        for s0, s1 in spans:
            ax.axvspan(s0 * 0.02, s1 * 0.02, color="#ff7f0e", alpha=0.15)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time [s]")
        if col == 0: ax.set_ylabel("h(x)", fontsize=11)

    title = args.title or f"{data.get('task', '?')}  α/φ/h across envs"
    fig.suptitle(title, fontsize=12)

    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    print(f"[compare] saved {args.output}")

    # Summary
    print(f"\n  env  α_mean  α_std   φ_mean  φ_std   h_min  CBF%")
    for e in env_idxs:
        a = alpha_all[:, e]; p_ = phi_all[:, e]; h_ = h_all[:, e]; q_ = qp_all[:, e]
        print(f"  {e:>3d}  {a.mean():6.3f}  {a.std():5.3f}   "
              f"{p_.mean():6.3f}  {p_.std():5.3f}   {h_.min():+6.3f}  {q_.mean()*100:5.1f}")


if __name__ == "__main__":
    main()
