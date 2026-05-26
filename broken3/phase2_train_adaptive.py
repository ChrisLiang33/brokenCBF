"""Phase 2 -- state-conditioned policy + the in-distribution Pareto plot.

Validates the core claim's first half: an adaptive parameterization
beats every fixed one. We sweep fixed (phi, alpha) to trace a frontier
in (intervention cost, safety) space, then drop the learned policy onto
the same axes. A win = the learned point sits up and/or left of the
fixed-parameter frontier (same safety for less intervention, or more
safety for the same intervention).

The disturbance level matters: too low and the corridor gap is easy for
any moderate fixed phi, so the fixed points collapse to a near-vertical
line (intervention barely varies) and adaptivity has nothing to win.
Default is 0.40 -- high enough that low phi genuinely collides in the
gap, opening a real intervention/safety spread. Sweep it with
--disturbance to see the frontier widen.

Run:  python phase2_train_adaptive.py [--quick] [--timesteps N]
                                      [--disturbance D]
"""
import argparse
import json
import numpy as np

from config import Config
from env import CBFParamEnv
from utils import (evaluate, plot_pareto, plot_trajectory, policy_from_model,
                   rollout, to_action, train_ppo)


def fixed_param_sweep(cfg, disturbance, n_episodes=25):
    phis = np.linspace(cfg.phi_bounds[0] + 0.05, cfg.phi_bounds[1], 6)
    alphas = np.linspace(cfg.alpha_bounds[0], cfg.alpha_bounds[1], 5)
    points = []
    for phi in phis:
        for alpha in alphas:
            m = evaluate(cfg, to_action(cfg, phi, alpha),
                         disturbance, n_episodes=n_episodes)
            points.append({"phi": float(phi), "alpha": float(alpha), **m})
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--disturbance", type=float, default=0.40)
    args = ap.parse_args()
    timesteps = args.timesteps or (10_000 if args.quick else 200_000)
    eps = 12 if args.quick else 25
    D = args.disturbance

    cfg = Config(state_conditioned=True, disturbance=D)

    print("=" * 64)
    print(f"Phase 2  --  state-conditioned policy + Pareto plot  (D={D})")
    print("=" * 64)

    print(f"  training adaptive PPO for {timesteps} steps ...")
    model = train_ppo(cfg, timesteps)
    model.save("phase2_adaptive_ppo")

    print("  sweeping fixed (phi, alpha) grid for the frontier ...")
    fixed_points = fixed_param_sweep(cfg, D, n_episodes=eps)

    learned = evaluate(cfg, policy_from_model(model), D, n_episodes=eps * 2)

    # cache so the plots can be regenerated without re-sweeping
    json.dump({"pts": fixed_points, "learned": learned, "D": D},
              open("phase2_sweep_cache.json", "w"))

    print()
    print(f"  learned adaptive | collisions={learned['collision_rate']:.2f} "
          f"min_h={learned['min_h']:+.3f} intervention={learned['intervention']:.2f}")
    best_fixed = min(fixed_points,
                     key=lambda p: (p["collision_rate"], p["intervention"]))
    print(f"  best fixed       | collisions={best_fixed['collision_rate']:.2f} "
          f"min_h={best_fixed['min_h']:+.3f} "
          f"intervention={best_fixed['intervention']:.2f} "
          f"(phi={best_fixed['phi']:.2f}, alpha={best_fixed['alpha']:.2f})")

    plot_pareto(fixed_points, learned, "phase2_pareto.png", D)

    # qualitative showcase: learned adaptive vs the best fixed parameter,
    # same seed -> the phi-over-time panel should show the learned policy
    # staying cheap on the open approach and spiking inside the gap.
    env = CBFParamEnv(cfg)
    r_learned = rollout(env, policy_from_model(model), disturbance=D, seed=7)
    r_fixed = rollout(env, to_action(cfg, best_fixed["phi"], best_fixed["alpha"]),
                      disturbance=D, seed=7)
    plot_trajectory(
        cfg,
        [("learned adaptive", r_learned),
         (f"best fixed (phi={best_fixed['phi']:.2f})", r_fixed)],
        "phase2_trajectory.png",
    )

    print("\n  saved -> phase2_pareto.png, phase2_trajectory.png, "
          "phase2_sweep_cache.json, phase2_adaptive_ppo.zip")


if __name__ == "__main__":
    main()
