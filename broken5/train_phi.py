"""Train PPO on AdaptivePhiEnv (kick magnitude → φ adaptation)."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from phi_env import AdaptivePhiEnv

HERE = Path(__file__).parent
RUNS_DIR = HERE / "runs"
DEFAULT_NAME = "adaptive_phi"


def make_env(seed: int):
    def _f():
        env = AdaptivePhiEnv()
        env.reset(seed=seed)
        return env
    return _f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--name", type=str, default=DEFAULT_NAME)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = RUNS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    vec_env = SubprocVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
    vec_env = VecMonitor(vec_env, filename=str(out_dir / "monitor.csv"))

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=str(out_dir / "tb"),
        seed=args.seed,
    )

    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(out_dir / "policy")
    print(f"Saved policy -> {out_dir / 'policy.zip'}")


if __name__ == "__main__":
    main()
