"""Evaluation matching the 2D MVP: same scenes, same metrics, same baselines.

Runs `n_seeds × len(starts)` trajectories per (scene, controller, regime)
and reports:
  SAFETY:    collision_rate, time_unsafe_frac, min_h, mean_min_h
  PERF:      success_rate, mean_time, mean_path_length, mean_detour_ratio

Controllers compared:
  - ISSf  ε₀=10 (small φ)
  - ISSf  ε₀=1  (big φ)
  - TISSf ε₀=1, λ=3
  - Ours fullsetup (priv)      — checkpoint --fullsetup_ckpt
  - Ours nopriv (no priv)      — checkpoint --nopriv_ckpt

Regimes (matching MVP):
  static / drift / adversarial / worst_case

Outputs a results CSV + per-regime grid PNG.
"""
from __future__ import annotations

import argparse
import torch

from policy.networks import CBFPolicy, log_params_to_cbf
from core.scenes import SCENES, TRAIN_SCENES, TEST_SCENES


# Same TISSf parameters used in the MVP for fair comparison
def phi_issf(eps_0):       return lambda h: torch.full_like(h, 1.0 / eps_0)
def phi_tissf(eps_0, lam): return lambda h: torch.exp(-lam * h) / eps_0


REGIMES = [
    # (label, sigma, drift, g_eps, adv_prob)
    ("static",      0.6, 0.0,  0.0,  0.0),
    ("drift",       0.6, 0.15, 0.10, 0.0),
    ("adversarial", 0.6, 0.0,  0.0,  0.20),
    ("worst_case",  0.6, 0.20, 0.15, 0.20),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fullsetup_ckpt", type=str)
    p.add_argument("--nopriv_ckpt", type=str)
    p.add_argument("--n_seeds", type=int, default=20)
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--task", type=str, default="Isaac-Go2-CBF-v0")
    return p.parse_args()


def load_policy(ckpt_path: str, use_priv: bool, env) -> CBFPolicy:
    policy = CBFPolicy(
        proprio_dim=env.observation_manager.group_obs_term_dim["proprio"],
        past_action_dim=4 * 5,
        priv_dim=env.observation_manager.group_obs_term_dim.get("priv_obs", 0),
        occ_hw=(32, 32),
        use_priv=use_priv,
    )
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return policy.to(env.device).eval()


def run_episode(env, controller, regime, n_seeds: int):
    """Run trajectories and collect metrics.
    `controller` is either:
       - dict {alpha, phi, a, b, c} of constants/callables (for baselines)
       - a CBFPolicy (for ours) — called per-step to produce log_params

    Per env we track: collided, time_unsafe_count, min_h, reached_t, path_len,
    initial_dist_to_goal. Aggregated at the end.
    """
    # IMPLEMENTATION NOTES:
    # - Set env.cfg's randomization to (regime sigma/drift/g_eps/adv) deterministically
    # - For each seed, reset env; step until done; track metrics
    # - Aggregate across (n_seeds × num_envs) trajectories
    # See the 2D MVP's evaluate() for the metric definitions.
    raise NotImplementedError("Fill in once env wiring is in place.")


def main():
    args = parse_args()

    # Boot env (same as train.py); load policies
    # env = build_env(args)
    # fullsetup = load_policy(args.fullsetup_ckpt, use_priv=True,  env=env)
    # nopriv    = load_policy(args.nopriv_ckpt,    use_priv=False, env=env)

    # Build the 5 controllers
    # controllers = {
    #     "ISSf eps0=10":  {"alpha": 3.0, "phi": phi_issf(10.0)},
    #     "ISSf eps0=1":   {"alpha": 3.0, "phi": phi_issf(1.0)},
    #     "TISSf":         {"alpha": 3.0, "phi": phi_tissf(1.0, 3.0)},
    #     "Ours fullsetup": fullsetup,
    #     "Ours nopriv":    nopriv,
    # }

    # for label, sigma, drift, geps, adv in REGIMES:
    #     for scene_name in TEST_SCENES:
    #         set_env_scene(env, scene_name)
    #         set_env_uncertainty(env, sigma, drift, geps, adv)
    #         for cname, ctrl in controllers.items():
    #             metrics = run_episode(env, ctrl, ..., args.n_seeds)
    #             print(label, scene_name, cname, metrics)

    print("[Skeleton] eval pipeline ready — wire env interfaces and run.")


if __name__ == "__main__":
    main()
