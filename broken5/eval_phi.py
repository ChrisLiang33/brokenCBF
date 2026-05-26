"""Sweep kick_magnitude and check whether the trained policy raises φ.

If the policy adapts, mean φ should rise monotonically with the expected
kick magnitude, and crash rate should stay low across the range. Compares
against a fixed-φ baseline so you can see how much adaptation buys you.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from phi_env import KICK_MAX, PHI_MAX, PHI_MIN, AdaptivePhiEnv

HERE = Path(__file__).parent

KICKS = np.linspace(0.0, KICK_MAX, 9)
EPISODES_PER_KICK = 25


def phi_to_action(phi: float) -> np.ndarray:
    a = 2.0 * (phi - PHI_MIN) / (PHI_MAX - PHI_MIN) - 1.0
    return np.array([a], dtype=np.float32)


def rollout(env: AdaptivePhiEnv, kick: float, rng: np.random.Generator,
            policy=None, fixed_phi: float | None = None) -> dict:
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    env.set_scenario(kick_mag=kick)
    obs = env._obs()
    phis = []
    while True:
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = phi_to_action(float(fixed_phi))
        obs, _, term, trunc, info = env.step(action)
        phis.append(info["phi"])
        if term or trunc:
            return {"phi_mean": float(np.mean(phis)),
                    "reached": info["reached"], "crashed": info["crashed"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str,
                        default=str(HERE / "runs" / "adaptive_phi" / "policy.zip"))
    parser.add_argument("--out", type=str, default=str(HERE / "phi_vs_kick.png"))
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_KICK)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    policy = PPO.load(args.policy)
    env = AdaptivePhiEnv()
    rng = np.random.default_rng(args.seed)

    print(f"{'kick (m)':>9} {'mean φ':>8} {'reach%':>7} {'crash%':>7}")
    policy_phi, policy_reach, policy_crash = [], [], []
    for k in KICKS:
        results = [rollout(env, float(k), rng, policy=policy)
                   for _ in range(args.episodes)]
        m = float(np.mean([r["phi_mean"] for r in results]))
        rr = float(np.mean([r["reached"] for r in results]))
        cr = float(np.mean([r["crashed"] for r in results]))
        policy_phi.append(m); policy_reach.append(rr); policy_crash.append(cr)
        print(f"{k:9.3f} {m:8.3f} {100*rr:6.0f}% {100*cr:6.0f}%")

    # Fixed-φ baselines for comparison
    fixed_levels = [0.05, 0.5, 1.5]
    fixed_results = {phi: {"reach": [], "crash": []} for phi in fixed_levels}
    for phi in fixed_levels:
        for k in KICKS:
            rs = [rollout(env, float(k), rng, fixed_phi=phi)
                  for _ in range(args.episodes)]
            fixed_results[phi]["reach"].append(np.mean([r["reached"] for r in rs]))
            fixed_results[phi]["crash"].append(np.mean([r["crashed"] for r in rs]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(KICKS, policy_phi, "o-", color="crimson", lw=2, ms=8,
                 label="learned policy")
    for phi in fixed_levels:
        axes[0].axhline(phi, color="0.7", ls=":", lw=1)
        axes[0].text(KICKS[-1], phi, f" fixed φ={phi}", color="0.5",
                     va="center", fontsize=8)
    axes[0].axhline(PHI_MIN, color="0.5", ls="--", lw=1)
    axes[0].axhline(PHI_MAX, color="0.5", ls="--", lw=1)
    axes[0].set_xlabel("kick magnitude [m]")
    axes[0].set_ylabel("mean φ (policy output)")
    axes[0].set_title("Does φ rise with kick magnitude?")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(KICKS, 100 * np.array(policy_reach), "o-", color="seagreen",
                 lw=2, ms=8, label="learned")
    for phi in fixed_levels:
        axes[1].plot(KICKS, 100 * np.array(fixed_results[phi]["reach"]),
                     "x--", alpha=0.6, label=f"fixed φ={phi}")
    axes[1].set_xlabel("kick magnitude [m]")
    axes[1].set_ylabel("reach %")
    axes[1].set_title("Reach rate vs kick")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(-5, 105)

    axes[2].plot(KICKS, 100 * np.array(policy_crash), "o-", color="firebrick",
                 lw=2, ms=8, label="learned")
    for phi in fixed_levels:
        axes[2].plot(KICKS, 100 * np.array(fixed_results[phi]["crash"]),
                     "x--", alpha=0.6, label=f"fixed φ={phi}")
    axes[2].set_xlabel("kick magnitude [m]")
    axes[2].set_ylabel("crash %")
    axes[2].set_title("Crash rate vs kick")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(-5, 105)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nSaved plot -> {args.out}")


if __name__ == "__main__":
    main()
