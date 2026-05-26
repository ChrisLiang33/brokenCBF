"""Phase 2 -- state-conditioned policy + the in-distribution Pareto plot.

Validates the core claim's first half: an adaptive parameterization
beats every fixed one. We sweep fixed (phi, alpha) to trace a frontier
in (intervention cost, safety) space, then drop the learned policy onto
the same axes. A win = the learned point sits up and/or left of the
fixed-parameter frontier (same safety for less intervention, or more
safety for the same intervention).

Run:  python phase2_train_adaptive.py [--quick] [--timesteps N]
"""
import argparse
import numpy as np

from config import Config
from utils import evaluate, plot_pareto, policy_from_model, to_action, train_ppo


DISTURBANCE = 0.25


def fixed_param_sweep(cfg, n_episodes=25):
    phis = np.linspace(cfg.phi_bounds[0] + 0.05, cfg.phi_bounds[1], 6)
    alphas = np.linspace(cfg.alpha_bounds[0], cfg.alpha_bounds[1], 5)
    points = []
    for phi in phis:
        for alpha in alphas:
            m = evaluate(cfg, to_action(cfg, phi, alpha),
                         DISTURBANCE, n_episodes=n_episodes)
            points.append({"phi": phi, "alpha": alpha, **m})
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--timesteps", type=int, default=None)
    args = ap.parse_args()
    timesteps = args.timesteps or (10_000 if args.quick else 200_000)
    eps = 12 if args.quick else 25

    cfg = Config(state_conditioned=True, disturbance=DISTURBANCE)

    print("=" * 64)
    print("Phase 2  --  state-conditioned policy + Pareto plot")
    print("=" * 64)

    print(f"  training adaptive PPO for {timesteps} steps ...")
    model = train_ppo(cfg, timesteps)
    model.save("phase2_adaptive_ppo")

    print("  sweeping fixed (phi, alpha) grid for the frontier ...")
    fixed_points = fixed_param_sweep(cfg, n_episodes=eps)

    learned = evaluate(cfg, policy_from_model(model),
                       DISTURBANCE, n_episodes=eps * 2)

    print()
    print(f"  learned adaptive | collisions={learned['collision_rate']:.2f} "
          f"min_h={learned['min_h']:+.3f} intervention={learned['intervention']:.2f}")
    best_fixed = min(fixed_points,
                     key=lambda p: (p["collision_rate"], p["intervention"]))
    print(f"  best fixed       | collisions={best_fixed['collision_rate']:.2f} "
          f"min_h={best_fixed['min_h']:+.3f} "
          f"intervention={best_fixed['intervention']:.2f} "
          f"(phi={best_fixed['phi']:.2f}, alpha={best_fixed['alpha']:.2f})")

    plot_pareto(fixed_points, learned, "phase2_pareto.png", DISTURBANCE)
    print("\n  saved -> phase2_pareto.png, phase2_adaptive_ppo.zip")


if __name__ == "__main__":
    main()
