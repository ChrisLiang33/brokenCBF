"""Demo: how does φ* depend on collision_cost?

For each (sigma, phi, collision_cost):
  - Run N trials with adversarial actuation noise at that sigma
  - Compute average time-to-goal and collision rate
  - Treat them as a reward: E[R] = -time_to_goal - collision_cost * collision_rate

For each (sigma, collision_cost):
  - argmax_phi E[R] gives the "optimal phi" PPO would converge to if its
    gradient estimator were perfectly unbiased

Plot phi*(sigma) curves for different collision_costs:
  - Low collision_cost: curve has positive slope (phi varies with sigma -> adaptation)
  - High collision_cost: curve flattens out near worst-case phi (hedging)
  - Intermediate: somewhere in between

This is the analytical demonstration that the magnitude of the collision
penalty *relative* to the time-to-goal penalty determines whether the
"optimal" policy is adaptive or hedged.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from sandbox import CBFParams, Obstacle, simulate


def run_grid(sigmas, phis, n_trials=15, alpha=0.5):
    obs = Obstacle(pos=np.array([0.0, 0.0]), radius=0.5)
    x0 = np.array([-3.0, 0.05])
    goal = np.array([3.0, 0.0])

    results = {}  # (sigma, phi) -> (col_rate, mean_ttg)
    for sigma in sigmas:
        for phi in phis:
            params = CBFParams(alpha=alpha, phi=phi, a=0.0, c=0.0)
            n_col = 0
            ttg_list = []
            for seed in range(n_trials):
                traj = simulate(
                    x0=x0, goal=goal, obs=obs, params=params,
                    T=20.0, dt=0.05, u_max=2.0,
                    actuation_noise_sigma=sigma,
                    actuation_noise_mode="adversarial",
                    seed=seed,
                )
                if traj["collided"]:
                    n_col += 1
                # Penalty: if didn't reach goal, count as max time
                ttg = traj["time_to_goal"] if traj["reached_goal"] else 20.0
                ttg_list.append(ttg)
            results[(sigma, phi)] = {
                "col_rate": n_col / n_trials,
                "mean_ttg": float(np.mean(ttg_list)),
            }
    return results


def expected_reward(results, sigma, phi, collision_cost):
    """E[R] = -time_to_goal - collision_cost * collision_rate. Higher = better."""
    r = results[(sigma, phi)]
    return -r["mean_ttg"] - collision_cost * r["col_rate"]


def find_phi_star(results, sigma, phis, collision_cost):
    """argmax_phi E[R | sigma, collision_cost]."""
    rewards = [expected_reward(results, sigma, phi, collision_cost) for phi in phis]
    return phis[int(np.argmax(rewards))]


def main():
    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    phis = np.linspace(0.0, 2.0, 41)
    n_trials = 15

    print(f"Running {len(sigmas)} x {len(phis)} x {n_trials} = "
          f"{len(sigmas)*len(phis)*n_trials} sims...")
    results = run_grid(sigmas, phis, n_trials=n_trials)
    print("Done.\n")

    collision_costs = [1.0, 5.0, 20.0, 100.0, 1000.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: phi* vs sigma per collision_cost
    ax = axes[0]
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(collision_costs)))
    for color, cc in zip(cmap, collision_costs):
        phi_stars = [find_phi_star(results, s, phis, cc) for s in sigmas]
        ax.plot(sigmas, phi_stars, "-o", color=color,
                label=f"C_coll = {cc:.0f}", markersize=8)
    # Reference: analytical adversarial bound
    sigma_fine = np.linspace(0, max(sigmas), 100)
    ax.plot(sigma_fine, 2 * sigma_fine, "k--", alpha=0.5,
            label="analytical 2σ worst-case bound", linewidth=1)
    ax.set_xlabel("σ_act (adversarial)")
    ax.set_ylabel("φ* (argmax E[R])")
    ax.set_title("φ* depends strongly on collision_cost\n"
                 "Low C_coll → adaptive curve, high C_coll → flat/hedged")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Right panel: heatmap of E[R] for each (sigma, phi) at one collision_cost
    # to show the landscape PPO is climbing
    ax = axes[1]
    cc_show = 100.0
    Z = np.zeros((len(sigmas), len(phis)))
    for i, s in enumerate(sigmas):
        for j, p in enumerate(phis):
            Z[i, j] = expected_reward(results, s, p, cc_show)
    im = ax.imshow(Z, aspect="auto", origin="lower",
                   extent=[phis[0], phis[-1], sigmas[0], sigmas[-1]],
                   cmap="viridis")
    plt.colorbar(im, ax=ax, label="E[R]")
    # Overlay phi* curve
    phi_stars = [find_phi_star(results, s, phis, cc_show) for s in sigmas]
    ax.plot(phi_stars, sigmas, "r-o", markersize=8, label=f"φ* @ C_coll={cc_show:.0f}")
    ax.set_xlabel("φ")
    ax.set_ylabel("σ_act")
    ax.set_title(f"E[R] landscape, C_coll = {cc_show:.0f}\n"
                 "φ* curve overlaid (red)")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle("Optimal φ depends on collision_cost — the lever that determines"
                 " whether adaptation pays")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "out_phi_collision_cost.png")
    plt.savefig(out, dpi=110)
    print(f"Saved {out}\n")

    # Console table
    print(f"{'σ':>5} " + "  ".join(f"C={cc:>5.0f}" for cc in collision_costs))
    print("-" * (8 + 9 * len(collision_costs)))
    for s in sigmas:
        row = f"{s:>5.2f} "
        for cc in collision_costs:
            row += f"  φ*={find_phi_star(results, s, phis, cc):>4.2f}"
        print(row)

    print()
    print("How to read the table:")
    print("  - Each column = a different collision_cost value")
    print("  - Each row = a different σ_act level")
    print("  - Entries = the φ that maximizes expected return at (σ, C_coll)")
    print()
    print("Look at how the φ* values VARY across σ for each column:")
    print("  - Low C_coll: φ* varies (low σ → low φ, high σ → high φ) = adaptation")
    print("  - High C_coll: φ* flat or hedged = same φ regardless of σ (no adaptation)")


if __name__ == "__main__":
    main()
