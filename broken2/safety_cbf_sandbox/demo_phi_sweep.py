"""Demo: sweep phi under actuation noise, find empirical optimum vs sigma.

phi multiplies ||L_g h||^2 in the CBF constraint. It's intended to absorb
actuation uncertainty: u_applied = u_safe + epsilon where epsilon ~ N(0, sigma^2 I).
Higher sigma should demand higher phi to keep the true h positive in expectation.

This experiment:
  - Fixes alpha = 2.0, a = 0, c = 0
  - Sweeps sigma_act in {0, 0.05, 0.10, 0.15, 0.20}
  - Sweeps phi in linspace(0, 5, 21)
  - Runs N_trials = 30 seeds per (sigma, phi)
  - For each (sigma, phi): computes collision_rate and median time_to_goal

Outputs:
  - out_phi_sweep.png: 3 panels (collision_rate vs phi, t_to_goal vs phi,
    empirical phi* vs sigma)
  - Console table of (sigma, phi*, collision_rate at phi*, t_to_goal at phi*)

phi* defined as: smallest phi where collision_rate <= 0.05 (5% target).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from sandbox import CBFParams, Obstacle, simulate


def run_grid(sigmas, phis, n_trials=30, alpha=0.5, start_y=0.05):
    """Returns dict mapping (sigma, phi) -> {col_rate, t_to_goal, mean_min_h}.

    alpha kept LOW so the robot approaches the obstacle boundary (where
    actuation noise can actually push it into collision). start_y close
    to zero forces the trajectory to pass near the obstacle.
    """
    obs = Obstacle(pos=np.array([0.0, 0.0]), radius=0.5)
    x0 = np.array([-3.0, start_y])
    goal = np.array([3.0, 0.0])

    results = {}
    for sigma in sigmas:
        for phi in phis:
            params = CBFParams(alpha=alpha, phi=phi, a=0.0, c=0.0)
            collided = 0
            ttg_list = []
            min_h_list = []
            for seed in range(n_trials):
                traj = simulate(
                    x0=x0, goal=goal, obs=obs, params=params,
                    T=15.0, dt=0.05, u_max=2.0,
                    actuation_noise_sigma=sigma, seed=seed,
                )
                if traj["collided"]:
                    collided += 1
                if traj["reached_goal"]:
                    ttg_list.append(traj["time_to_goal"])
                min_h_list.append(traj["min_h_true"])
            col_rate = collided / n_trials
            t_to_goal = float(np.median(ttg_list)) if ttg_list else float("inf")
            results[(sigma, phi)] = {
                "col_rate": col_rate,
                "t_to_goal": t_to_goal,
                "mean_min_h": float(np.mean(min_h_list)),
                "frac_reached": len(ttg_list) / n_trials,
            }
    return results


def find_phi_star(results, sigma, phis, col_target=0.05):
    """Smallest phi where col_rate <= col_target. Returns None if no phi meets it."""
    for phi in phis:
        if results[(sigma, phi)]["col_rate"] <= col_target:
            return phi
    return None


def main():
    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20]
    phis = np.linspace(0.0, 5.0, 21)
    n_trials = 30

    print(f"Running {len(sigmas)} sigmas x {len(phis)} phis x {n_trials} trials = "
          f"{len(sigmas) * len(phis) * n_trials} simulations...")
    results = run_grid(sigmas, phis, n_trials=n_trials)
    print("Done.")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(sigmas)))

    # Panel 1: collision rate vs phi
    ax = axes[0]
    for color, sigma in zip(cmap, sigmas):
        ys = [results[(sigma, phi)]["col_rate"] for phi in phis]
        ax.plot(phis, ys, "-o", color=color, label=f"sigma={sigma:.2f}",
                markersize=4)
    ax.axhline(0.05, color="r", linestyle="--", alpha=0.5, label="5% target")
    ax.set_xlabel("phi")
    ax.set_ylabel("collision rate (true h < 0)")
    ax.set_title("Collision rate vs phi")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: time-to-goal vs phi
    ax = axes[1]
    for color, sigma in zip(cmap, sigmas):
        ys = [results[(sigma, phi)]["t_to_goal"] for phi in phis]
        ys = [y if np.isfinite(y) else np.nan for y in ys]
        ax.plot(phis, ys, "-o", color=color, label=f"sigma={sigma:.2f}",
                markersize=4)
    ax.set_xlabel("phi")
    ax.set_ylabel("median time to goal (s)")
    ax.set_title("Time-to-goal vs phi  (higher phi = more conservative = slower)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: empirical phi* vs sigma
    ax = axes[2]
    phi_stars = []
    sigma_with_star = []
    for sigma in sigmas:
        phi_star = find_phi_star(results, sigma, phis, col_target=0.05)
        if phi_star is not None:
            sigma_with_star.append(sigma)
            phi_stars.append(phi_star)
    ax.plot(sigma_with_star, phi_stars, "ko-", label="empirical phi* (col<=5%)",
            markersize=8)

    # Kolathaya-style reference curves (different scaling assumptions)
    sigma_fine = np.linspace(0.01, 0.25, 100)
    # 3-sigma worst-case bound: phi ~ 3*sigma / ||L_g h||, with ||L_g h||=1
    ax.plot(sigma_fine, 3 * sigma_fine, "b--", alpha=0.6,
            label="3-sigma worst-case  phi = 3 sigma")
    # Quadratic-tolerance: phi ~ (3 sigma)^2 = 9 sigma^2 (if you square the bound)
    ax.plot(sigma_fine, 9 * sigma_fine ** 2, "g--", alpha=0.6,
            label="quadratic bound  phi = 9 sigma^2")

    ax.set_xlabel("actuation noise sigma")
    ax.set_ylabel("phi*")
    ax.set_title("Empirical phi* vs sigma  (vs analytical references)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("phi sweep with actuation noise — alpha=2.0, single obstacle")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "out_phi_sweep.png")
    plt.savefig(out, dpi=110)
    print(f"\nSaved {out}")

    # Console table
    print()
    print(f"{'sigma':>6} {'phi*':>6} {'col_rate@phi*':>14} {'t_to_goal@phi*':>16}")
    print("-" * 50)
    for sigma in sigmas:
        phi_star = find_phi_star(results, sigma, phis, col_target=0.05)
        if phi_star is not None:
            r = results[(sigma, phi_star)]
            print(f"{sigma:>6.2f} {phi_star:>6.2f} {r['col_rate']:>14.3f} "
                  f"{r['t_to_goal']:>16.2f}")
        else:
            print(f"{sigma:>6.2f} {'-':>6} {'no phi <= 5% in sweep':>14}")

    # min_h as a function of phi for each sigma (margin behavior)
    print()
    print("mean min_h(true) per (sigma, phi):")
    phi_show = [phis[0], phis[len(phis)//4], phis[len(phis)//2],
                phis[3*len(phis)//4], phis[-1]]
    header = f"{'sigma':>6}" + "".join(f"{f'phi={p:.2f}':>11}" for p in phi_show)
    print(header)
    print("-" * len(header))
    for sigma in sigmas:
        row = f"{sigma:>6.2f}"
        for phi in phi_show:
            row += f"{results[(sigma, phi)]['mean_min_h']:>11.3f}"
        print(row)


if __name__ == "__main__":
    main()
