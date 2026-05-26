"""Phase 3 -- out-of-distribution robustness test (the core claim).

Training disturbance is randomized over [0, D_TRAIN_MAX] each episode, so
the policy actually learns the mapping  disturbance estimate -> phi. We
also pick the best *fixed* parameter over that same training range. Then
we evaluate BOTH across a sweep of test disturbances, including levels
ABOVE the training ceiling. The fixed parameter should start colliding
once the disturbance exceeds what it was tuned for; the adaptive policy
reads the disturbance estimate, raises phi, and degrades more gracefully
-- and extrapolates somewhat past the training range.

Run:  python phase3_ood.py [--quick] [--timesteps N]
"""
import argparse
import numpy as np

from config import Config
from utils import evaluate, plot_ood, policy_from_model, to_action, train_ppo

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


D_TRAIN_MAX = 0.30                       # training disturbance ~ U(0, 0.30)
TEST_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def best_fixed_over_range(cfg, levels, n_episodes=15):
    """Best single fixed (phi, alpha) averaged over the training range."""
    phis = np.linspace(cfg.phi_bounds[0] + 0.05, cfg.phi_bounds[1], 7)
    alphas = np.linspace(cfg.alpha_bounds[0], cfg.alpha_bounds[1], 5)
    best = None
    for phi in phis:
        for alpha in alphas:
            action = to_action(cfg, phi, alpha)
            ms = [evaluate(cfg, action, d, n_episodes=n_episodes)
                  for d in levels]
            score = (float(np.mean([m["collision_rate"] for m in ms])),
                     float(np.mean([m["intervention"] for m in ms])))
            if best is None or score < best[0]:
                best = (score, phi, alpha)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--timesteps", type=int, default=None)
    args = ap.parse_args()
    timesteps = args.timesteps or (10_000 if args.quick else 200_000)
    eps = 20 if args.quick else 50

    cfg = Config(state_conditioned=True, disturbance_range=(0.0, D_TRAIN_MAX))

    print("=" * 64)
    print(f"Phase 3  --  OOD robustness (train disturbance ~ U(0, {D_TRAIN_MAX}))")
    print("=" * 64)

    # adaptive policy: reuse a Phase 3 model if present, else train
    model = None
    if PPO is not None:
        try:
            model = PPO.load("phase3_adaptive_ppo")
            print("  loaded adaptive policy from phase3_adaptive_ppo.zip")
        except Exception:
            model = None
    if model is None:
        print(f"  training adaptive PPO for {timesteps} steps ...")
        model = train_ppo(cfg, timesteps)
        model.save("phase3_adaptive_ppo")

    train_levels = [0.0, 0.1, 0.2, 0.3]
    print(f"  picking best fixed (phi, alpha) over the training range ...")
    g_phi, g_alpha = best_fixed_over_range(cfg, train_levels,
                                           n_episodes=eps // 3)
    print(f"  best fixed: phi={g_phi:.3f}, alpha={g_alpha:.3f}")

    fixed_action = to_action(cfg, g_phi, g_alpha)
    adaptive = policy_from_model(model)

    fixed_rates, adaptive_rates = [], []
    print("\n  test disturbance |  fixed collision  |  adaptive collision")
    print("  " + "-" * 56)
    for d in TEST_LEVELS:
        mf = evaluate(cfg, fixed_action, d, n_episodes=eps)
        ma = evaluate(cfg, adaptive, d, n_episodes=eps)
        fixed_rates.append(mf["collision_rate"])
        adaptive_rates.append(ma["collision_rate"])
        print(f"       {d:4.2f}        |      {mf['collision_rate']:4.2f}       "
              f"|        {ma['collision_rate']:4.2f}")

    plot_ood(
        TEST_LEVELS,
        {f"fixed (tuned on [0, {D_TRAIN_MAX}])": fixed_rates,
         "learned adaptive": adaptive_rates},
        "phase3_ood.png",
        D_TRAIN_MAX,
    )
    print()
    print("  PASS if the adaptive curve stays below the fixed curve as the")
    print("  test disturbance grows past the training ceiling -- that is")
    print("  the whole thesis.")
    print("\n  saved -> phase3_ood.png")


if __name__ == "__main__":
    main()
