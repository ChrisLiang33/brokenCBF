"""Demo: c sweep across three perception-bias regimes.

c shifts the effective boundary: constraint is L_g h_perc . u + alpha(h_perc - c) >= ...
The CBF treats h_perc = c as the "boundary" (where alpha term vanishes).

Derivation of analytical optimum:
  perceived h:  h_perc(x) = ||x - obs|| - (r_true + delta_R)
  At the true boundary, h_true = 0, h_perc = -delta_R.
  For the CBF to allow exactly the true boundary (no more, no less),
  we need (h_perc - c) = 0 there, i.e., -delta_R - c = 0 -> c = -delta_R.

So:
  delta_R > 0 (perception OVER-estimates obstacle size):  c* = -delta_R  (negative c)
  delta_R = 0 (perception correct):                       c* = 0
  delta_R < 0 (perception UNDER-estimates obstacle size): c* = -delta_R  (positive c)

This sweep verifies that empirically and shows the failure modes:
  - c too positive when delta_R > 0:  overly conservative, slow
  - c too negative when delta_R < 0:  COLLISION (robot allowed inside true obstacle)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from sandbox import CBFParams, Obstacle, simulate


def run_grid(delta_Rs, cs, n_trials=1, alpha=2.0):
    """Returns (delta_R, c) -> {min_h_true, collided, t_to_goal, frac_reached}.

    n_trials = 1 by default since c has no stochastic component (deterministic
    given perception_bias, no actuation noise).
    """
    obs = Obstacle(pos=np.array([0.0, 0.0]), radius=0.5)
    x0 = np.array([-3.0, 0.05])
    goal = np.array([3.0, 0.0])

    results = {}
    for delta_R in delta_Rs:
        for c in cs:
            params = CBFParams(alpha=alpha, phi=0.0, a=0.0, c=c)
            collided = 0
            ttg_list = []
            min_h_list = []
            for seed in range(n_trials):
                traj = simulate(
                    x0=x0, goal=goal, obs=obs, params=params,
                    T=15.0, dt=0.05, u_max=2.0,
                    perception_bias=delta_R, seed=seed,
                )
                if traj["collided"]:
                    collided += 1
                if traj["reached_goal"]:
                    ttg_list.append(traj["time_to_goal"])
                min_h_list.append(traj["min_h_true"])
            results[(delta_R, c)] = {
                "col_rate": collided / n_trials,
                "t_to_goal": float(np.median(ttg_list)) if ttg_list else float("inf"),
                "mean_min_h": float(np.mean(min_h_list)),
                "frac_reached": len(ttg_list) / n_trials,
            }
    return results


def find_c_star_min(results, delta_R, cs, target_margin=0.0):
    """Smallest c (most permissive) where min_h_true >= target_margin."""
    safe_cs = [c for c in cs if results[(delta_R, c)]["mean_min_h"] >= target_margin]
    return min(safe_cs) if safe_cs else None


def main():
    delta_Rs = [-0.10, -0.05, 0.0, 0.05, 0.10]
    cs = np.linspace(-0.20, 0.20, 41)

    print(f"Running {len(delta_Rs)} x {len(cs)} = {len(delta_Rs)*len(cs)} sims...")
    results = run_grid(delta_Rs, cs, n_trials=1, alpha=2.0)
    print("Done.\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.cm.RdBu(np.linspace(0.1, 0.9, len(delta_Rs)))

    # Panel 1: min_h_true vs c per delta_R
    ax = axes[0]
    for color, delta_R in zip(cmap, delta_Rs):
        ys = [results[(delta_R, c)]["mean_min_h"] for c in cs]
        ax.plot(cs, ys, "-o", color=color, label=f"delta_R={delta_R:+.2f}",
                markersize=4)
        c_star = -delta_R
        ax.axvline(c_star, color=color, linestyle=":", alpha=0.5)
    ax.axhline(0.0, color="r", linestyle="--", alpha=0.5, label="collision threshold")
    ax.set_xlabel("c")
    ax.set_ylabel("min h_true")
    ax.set_title("Safety margin vs c per perception bias\n(dotted: analytical c* = -delta_R)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: time_to_goal vs c per delta_R
    ax = axes[1]
    for color, delta_R in zip(cmap, delta_Rs):
        ys = [results[(delta_R, c)]["t_to_goal"] for c in cs]
        ys = [y if np.isfinite(y) else np.nan for y in ys]
        ax.plot(cs, ys, "-o", color=color, label=f"delta_R={delta_R:+.2f}",
                markersize=4)
    ax.set_xlabel("c")
    ax.set_ylabel("time to goal (s)")
    ax.set_title("Time-to-goal vs c\n(higher c = more conservative = slower)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: empirical c* (minimal safe c) vs delta_R
    ax = axes[2]
    empirical_c_stars = []
    delta_R_with_star = []
    for delta_R in delta_Rs:
        c_star = find_c_star_min(results, delta_R, cs, target_margin=0.0)
        if c_star is not None:
            delta_R_with_star.append(delta_R)
            empirical_c_stars.append(c_star)
    ax.plot(delta_R_with_star, empirical_c_stars, "ko-",
            label="empirical c* (smallest safe)", markersize=9)
    delta_fine = np.linspace(min(delta_Rs), max(delta_Rs), 100)
    ax.plot(delta_fine, -delta_fine, "b--", alpha=0.7,
            label="analytical c* = -delta_R")
    ax.set_xlabel("delta_R (perception bias)")
    ax.set_ylabel("c*")
    ax.set_title("Empirical minimal safe c vs perception bias")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("c sweep across perception bias regimes — alpha=2.0, no noise")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "out_c_sweep.png")
    plt.savefig(out, dpi=110)
    print(f"Saved {out}\n")

    print(f"{'delta_R':>8} {'c*_anal':>9} {'c*_emp':>9} {'min_h@c*':>10} {'t_goal@c*':>11}")
    print("-" * 55)
    for delta_R in delta_Rs:
        c_star_anal = -delta_R
        c_star_emp = find_c_star_min(results, delta_R, cs, target_margin=0.0)
        if c_star_emp is None:
            print(f"{delta_R:>+8.2f} {c_star_anal:>+9.3f} {'none safe':>9}")
            continue
        r = results[(delta_R, c_star_emp)]
        print(f"{delta_R:>+8.2f} {c_star_anal:>+9.3f} {c_star_emp:>+9.3f} "
              f"{r['mean_min_h']:>10.3f} {r['t_to_goal']:>11.2f}")

    print()
    print("Safety failure modes:")
    print(f"{'delta_R':>8} {'col_rate(c=0)':>14} {'min_h(c=0)':>11}")
    print("-" * 42)
    for delta_R in delta_Rs:
        # find c closest to 0
        c0 = min(cs, key=lambda c: abs(c))
        r = results[(delta_R, c0)]
        print(f"{delta_R:>+8.2f} {r['col_rate']:>14.3f} {r['mean_min_h']:>11.3f}")


if __name__ == "__main__":
    main()
