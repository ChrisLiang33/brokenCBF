"""Stage 3 eval: 2-D heatmap of α over (σ, goal-clearance).

For each (σ, goal-clearance) bucket we sample a few episodes (different
noise samples & goal positions inside the bucket) and average the mean α
the policy used. If the policy learned the 2-axis decision, the heatmap
should be brightest in the (low σ, low clearance) corner — high α to thread
a tight goal you can confidently see — and darkest along the high-σ
column — drop α once you don't trust the sensor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from env import (ALPHA_MAX, ALPHA_MIN, AdaptiveCBFEnv, GOAL_MAX_CLEAR,
                 GOAL_MIN_CLEAR, GOAL_X_RANGE, GOAL_Y_RANGE, SIGMA_MAX)

HERE = Path(__file__).parent

SIGMAS = np.array([0.0, 0.025, 0.05, 0.075, 0.10])
CLEAR_BINS = np.linspace(GOAL_MIN_CLEAR, GOAL_MAX_CLEAR, 6)
EPISODES_PER_CELL = 6


def sample_goal_at_clearance(env: AdaptiveCBFEnv, clear_lo: float,
                             clear_hi: float, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(400):
        g = np.array([rng.uniform(*GOAL_X_RANGE), rng.uniform(*GOAL_Y_RANGE)])
        c = env._goal_to_nearest_true(g)
        if clear_lo <= c <= clear_hi:
            return g
    return None


def rollout(policy: PPO, env: AdaptiveCBFEnv, sigma: float,
            goal: np.ndarray, rng: np.random.Generator) -> dict:
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    noise = rng.normal(0.0, max(sigma, 1e-12), size=env._true_centers.shape)
    env.set_scenario(sigma=sigma, goal=tuple(goal), noise=noise)
    alphas = []
    obs = env._obs()
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        alphas.append(info["alpha"])
        if term or trunc:
            return {"alpha_mean": float(np.mean(alphas)),
                    "reached": info["reached"], "crashed": info["crashed"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str,
                        default=str(HERE / "runs" / "adaptive_alpha_combined" / "policy.zip"))
    parser.add_argument("--out", type=str, default=str(HERE / "alpha_2d.png"))
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_CELL)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    policy = PPO.load(args.policy)
    env = AdaptiveCBFEnv()
    rng = np.random.default_rng(args.seed)

    n_sig = len(SIGMAS)
    n_clr = len(CLEAR_BINS) - 1
    alpha_grid = np.full((n_sig, n_clr), np.nan)
    reach_grid = np.full((n_sig, n_clr), np.nan)
    crash_grid = np.full((n_sig, n_clr), np.nan)

    print(f"{'σ':>6} {'clear-bin':>14}  {'mean α':>7}  {'reach%':>6}  {'crash%':>6}")
    for i, sigma in enumerate(SIGMAS):
        for j in range(n_clr):
            clo, chi = CLEAR_BINS[j], CLEAR_BINS[j + 1]
            alphas, reached, crashed = [], 0, 0
            for _ in range(args.episodes):
                g = sample_goal_at_clearance(env, clo, chi, rng)
                if g is None:
                    continue
                r = rollout(policy, env, float(sigma), g, rng)
                alphas.append(r["alpha_mean"])
                reached += int(r["reached"])
                crashed += int(r["crashed"])
            if alphas:
                alpha_grid[i, j] = float(np.mean(alphas))
                reach_grid[i, j] = reached / len(alphas) * 100
                crash_grid[i, j] = crashed / len(alphas) * 100
                print(f"{sigma:6.3f} [{clo:.2f}–{chi:.2f}]  "
                      f"{alpha_grid[i,j]:7.2f}  {reach_grid[i,j]:5.0f}%  {crash_grid[i,j]:5.0f}%")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    clear_centers = 0.5 * (CLEAR_BINS[:-1] + CLEAR_BINS[1:])
    extent = (CLEAR_BINS[0], CLEAR_BINS[-1], SIGMAS[0] - 0.012, SIGMAS[-1] + 0.012)

    for ax, grid, title, cmap, vmin, vmax, cbar_label in [
        (axes[0], alpha_grid, "Mean α  (high = aggressive)", "magma",
         ALPHA_MIN, ALPHA_MAX, "α"),
        (axes[1], reach_grid, "Reach rate %", "Greens", 0, 100, "%"),
        (axes[2], crash_grid, "Crash rate %", "Reds", 0, 100, "%"),
    ]:
        im = ax.imshow(grid, origin="lower", aspect="auto", extent=extent,
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("goal clearance from nearest obstacle [m]")
        ax.set_ylabel("sensor noise σ [m]")
        ax.set_title(title)
        ax.set_xticks(clear_centers)
        ax.set_xticklabels([f"{c:.2f}" for c in clear_centers], fontsize=8)
        ax.set_yticks(SIGMAS)
        fig.colorbar(im, ax=ax, label=cbar_label)
        # annotate cells
        for i, sigma in enumerate(SIGMAS):
            for j, cx in enumerate(clear_centers):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.1f}" if grid is alpha_grid else f"{int(v)}"
                ax.text(cx, sigma, txt, ha="center", va="center",
                        color="white" if (grid is alpha_grid and v < 2.5)
                        or (grid is reach_grid and v < 50)
                        or (grid is crash_grid and v < 50)
                        else "black", fontsize=8)

    fig.suptitle("Stage 3: does a single α-policy adapt across both axes?", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nSaved plot -> {args.out}")


if __name__ == "__main__":
    main()
