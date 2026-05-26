"""Phase 0 -- environment + QP, no RL.

Validates: the dynamics, the QP, and that a *fixed* hand-tuned CBF
actually keeps the single integrator out of the obstacle. If this is
broken, nothing downstream means anything. No learning here.

Run:  python phase0_fixed_cbf.py
"""
from config import Config
from env import CBFParamEnv
from utils import rollout, to_action, plot_trajectory


def main():
    cfg = Config(state_conditioned=False)
    env = CBFParamEnv(cfg)
    action = to_action(cfg, cfg.fixed_phi, cfg.fixed_alpha)

    r_clean = rollout(env, action, disturbance=0.0, seed=1)
    r_dist = rollout(env, action, disturbance=0.30, seed=1)

    print("=" * 64)
    print(f"Phase 0  --  fixed CBF (phi={cfg.fixed_phi}, alpha={cfg.fixed_alpha})")
    print("=" * 64)
    for name, r in [("no disturbance ", r_clean), ("disturbance 0.30", r_dist)]:
        print(f"  {name} | reached={str(r['reached']):5s} "
              f"collided={str(r['collided']):5s} "
              f"min_h={r['min_h']:+.3f}  intervention={r['intervention']:6.2f}")
    print()
    print("  Expect: reaches the goal, min_h stays >= 0 with no disturbance.")
    print("  With disturbance the fixed margin may be too thin -- that gap")
    print("  is exactly what the learned policy will later close.")

    plot_trajectory(
        cfg,
        [("no disturbance", r_clean), ("disturbance 0.30", r_dist)],
        "phase0_fixed_cbf.png",
    )
    print("\n  saved -> phase0_fixed_cbf.png")


if __name__ == "__main__":
    main()
