"""Phase 1 -- PPO learns *constant* parameters.

Validates: the reward design and the PPO loop wiring, in isolation.
The env observation is zeroed (state_conditioned=False), so the policy
can only learn a single state-independent (phi, alpha). We then check
that PPO recovers something close to the best constant found by a brute
grid search. If it does not, a bad Phase 2 result would be ambiguous --
this step removes that ambiguity for almost no extra code.

Run:  python phase1_train_constant.py [--quick] [--timesteps N]
"""
import argparse
import numpy as np

from config import Config
from env import CBFParamEnv
from utils import evaluate, to_action, train_ppo


DISTURBANCE = 0.25   # nonzero, so the safety/intervention tradeoff is real


def grid_search(cfg):
    """Brute-force the best constant (phi, alpha) as an honest baseline."""
    phis = np.linspace(cfg.phi_bounds[0] + 0.05, cfg.phi_bounds[1], 7)
    alphas = np.linspace(cfg.alpha_bounds[0], cfg.alpha_bounds[1], 7)
    best = None
    for phi in phis:
        for alpha in alphas:
            action = to_action(cfg, phi, alpha)
            m = evaluate(cfg, action, DISTURBANCE, n_episodes=20)
            # objective: feasible (no collisions) then minimal intervention
            score = (m["collision_rate"], m["intervention"])
            if best is None or score < best[0]:
                best = (score, phi, alpha, m)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--timesteps", type=int, default=None)
    args = ap.parse_args()
    timesteps = args.timesteps or (8_000 if args.quick else 120_000)

    cfg = Config(state_conditioned=False, disturbance=DISTURBANCE)

    print("=" * 64)
    print("Phase 1  --  PPO learns a constant (phi, alpha)")
    print("=" * 64)

    print(f"  training PPO for {timesteps} steps ...")
    model = train_ppo(cfg, timesteps)

    # state_conditioned=False -> obs is always zeros -> policy is constant
    obs_dim = CBFParamEnv(cfg).observation_space.shape[0]
    zero_obs = np.zeros(obs_dim, dtype=np.float32)
    action, _ = model.predict(zero_obs, deterministic=True)
    learned_phi, learned_alpha = CBFParamEnv(cfg)._map_action(action)
    learned_m = evaluate(cfg, action, DISTURBANCE, n_episodes=40)

    print("  grid-searching the best constant baseline ...")
    (_, g_phi, g_alpha, grid_m) = grid_search(cfg)

    print()
    print(f"  grid-search best : phi={g_phi:.3f} alpha={g_alpha:.3f} "
          f"| collisions={grid_m['collision_rate']:.2f} "
          f"intervention={grid_m['intervention']:.2f}")
    print(f"  PPO learned      : phi={learned_phi:.3f} alpha={learned_alpha:.3f} "
          f"| collisions={learned_m['collision_rate']:.2f} "
          f"intervention={learned_m['intervention']:.2f}")
    print()
    print("  PASS if PPO lands near the grid optimum -> reward + loop are sound.")
    print("  (If not, fix this before trusting Phase 2.)")

    model.save("phase1_constant_ppo")
    print("\n  saved -> phase1_constant_ppo.zip")


if __name__ == "__main__":
    main()
