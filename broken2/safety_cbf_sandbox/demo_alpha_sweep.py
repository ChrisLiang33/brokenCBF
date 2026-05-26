"""Demo: sweep alpha and visualize trajectories around a single obstacle.

Sanity-check that the closed-form CBF QP behaves as expected:
  - High alpha = aggressive constraint enforcement, sharp turns at the boundary
  - Low alpha = gentle / late braking
  - alpha = 0 = no enforcement past h=0 (collision)

Runs through five alpha values, no disturbance, plots a 1x5 grid + a
summary table of min_h, deflection, and time-to-goal per alpha.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from sandbox import CBFParams, Obstacle, simulate, plot_trajectory


def main():
    # Scene
    x0 = np.array([-3.0, 0.3])
    goal = np.array([3.0, 0.0])
    obs = Obstacle(pos=np.array([0.0, 0.0]), radius=0.5)

    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    rows = []
    for ax, alpha in zip(axes, alphas):
        params = CBFParams(alpha=alpha, phi=0.0, a=0.0, c=0.0)
        traj = simulate(
            x0=x0, goal=goal, obs=obs, params=params,
            T=15.0, dt=0.05, u_max=2.0,
        )
        title = (f"alpha={alpha:.1f}\n"
                 f"min_h={traj['min_h_true']:.2f}  "
                 f"collided={traj['collided']}\n"
                 f"defl={traj['mean_deflection']:.2f}  "
                 f"t_goal={traj['time_to_goal']:.1f}s")
        plot_trajectory(traj, obs, goal, ax=ax, title=title)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-2, 2)
        rows.append((alpha, traj["min_h_true"], traj["collided"],
                     traj["mean_deflection"], traj["time_to_goal"]))

    fig.suptitle("alpha sweep — single obstacle, no disturbance, h_lin = ||x|| - r")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "out_alpha_sweep.png")
    plt.savefig(out, dpi=110)
    print(f"Saved {out}")

    print()
    print(f"{'alpha':>6} {'min_h':>7} {'collided':>9} {'mean_defl':>10} {'time_to_goal':>13}")
    print("-" * 50)
    for a, mh, col, defl, ttg in rows:
        print(f"{a:>6.1f} {mh:>7.3f} {str(col):>9} {defl:>10.3f} {ttg:>13.2f}")


if __name__ == "__main__":
    main()
