"""PPO training via Isaac Lab's rsl_rl integration (4.x format)."""
from __future__ import annotations

import argparse
import os
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["fullsetup", "nopriv"], required=True)
    p.add_argument("--num_envs", type=int, default=2048)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loco_ckpt", type=str,
                   default="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/"
                           "2026-05-26_09-47-24/exported/policy.pt")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--no_dr", action="store_true",
                   help="Zero out all DR ranges (diagnostic: isolates whether "
                        "falls come from DR or from the CBF wrapper itself).")
    return p.parse_args()


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher
    _ = AppLauncher(headless=True).app

    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,
        RslRlMLPModelCfg, RslRlPpoAlgorithmCfg,
    )

    from env.go2_cbf_env import Go2CbfFlatEnvCfg, Go2CbfRLEnv

    # ----- Build env -----
    env_cfg = Go2CbfFlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = 8.0
    env_cfg.episode_length_s = 12.0

    if args.no_dr:
        # Diagnostic mode: turn off all randomization so the loco sees a clean
        # world. Physics DR is via EventCfg now (mass/friction/motor), so we
        # collapse those ranges too — events still fire but do nothing.
        env_cfg.rand_lidar_noise   = (0.0, 0.0)
        env_cfg.rand_sigma_pose    = (0.0, 0.0)
        env_cfg.rand_drift_std     = (0.0, 0.0)
        env_cfg.rand_adv_prob      = (0.0, 0.0)
        env_cfg.rand_tracking_err  = (0.0, 0.0)
        env_cfg.rand_mass_scale    = (1.0, 1.0)
        env_cfg.rand_motor_strength= (1.0, 1.0)
        env_cfg.rand_friction      = (1.0, 1.0)
        env_cfg.rand_v_max         = (1.0, 1.0)
        print("[train] --no_dr: all DR disabled.")

    env = Go2CbfRLEnv(cfg=env_cfg)
    env.priv_masked = (args.arch == "nopriv")
    env.locomotion = torch.jit.load(args.loco_ckpt, map_location=env.device).eval()
    for p in env.locomotion.parameters():
        p.requires_grad_(False)

    env_wrapped = RslRlVecEnvWrapper(env)
    env_wrapped.num_actions = 5
    # Force wrapper to expose our outer obs (1085-D) instead of std 48-D
    # locomotion obs. Must be a TensorDict (has both .keys() and .to()).
    from tensordict import TensorDict
    _outer_obs, _ = env.reset()
    _outer_td = TensorDict(_outer_obs, batch_size=[env.num_envs])
    env_wrapped.get_observations = lambda: _outer_td

    log_root = os.path.join(args.save_dir, args.arch)
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    runner_cfg = RslRlOnPolicyRunnerCfg(
        seed=args.seed,
        device=str(env.device),
        num_steps_per_env=24,
        max_iterations=args.iters,
        empirical_normalization=False,
        save_interval=100,
        experiment_name=args.arch,
        run_name="",
        logger="tensorboard",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=RslRlMLPModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=True,
            # Required-but-deprecated fields (filtered out below); kept to
            # satisfy the cfg dataclass.
            stochastic=True, init_noise_std=0.3, noise_std_type="scalar",
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
                init_std=0.3, std_type="scalar",
            ),
        ),
        critic=RslRlMLPModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=True,
            stochastic=False, init_noise_std=0.0, noise_std_type="scalar",
            distribution_cfg=None,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.1,         # was 0.2 — tighter PPO update prevents blowups
            entropy_coef=0.001,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-4,   # was 3e-4 — slower learning, less likely to spike
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.005,       # was 0.01 — tighter KL trust region
            max_grad_norm=1.0,
        ),
    )

    # Filter out fields rsl-rl 5.x's MLPModel rejects (kept in cfg dataclass
    # only because Isaac Lab still has them as MISSING).
    DEPRECATED = ("stochastic", "init_noise_std", "noise_std_type",
                  "state_dependent_std")
    cfg_dict = runner_cfg.to_dict()
    for k in ("actor", "critic"):
        for d in DEPRECATED:
            cfg_dict.get(k, {}).pop(d, None)
    # Also drop the legacy 'policy' key if present
    cfg_dict.pop("policy", None)
    # Older runner cfg also keeps 'empirical_normalization' which the new runner
    # may not expect; drop if present.
    cfg_dict.pop("empirical_normalization", None)

    runner = OnPolicyRunner(
        env_wrapped, cfg_dict, log_dir=log_dir, device=str(env.device),
    )

    # ----- Bias-init actor head to MVP-known-good CBF params -----
    # Without this, stock MLP starts with random output bias → initial params
    # are roughly (α=1, φ=1, a=1, b=1, c=1). b=1 is huge and crushes u_safe
    # to ~0 (robot crawls, can't reach goal in 8s). MVP defaults work much
    # better; PPO can then fine-tune from there.
    #   alpha=3.0, phi=0.1, a=b=c=0.05  →  log(...) = bias init
    init_bias = torch.tensor([1.099, -2.302, -2.996, -2.996, -2.996])
    # The actor lives inside rsl_rl's algorithm; path differs by version.
    actor_obj = None
    for candidate in (
        lambda: runner.alg.actor,
        lambda: runner.alg.policy.actor,
        lambda: runner.alg.actor_critic.actor,
    ):
        try:
            actor_obj = candidate()
            break
        except AttributeError:
            continue
    if actor_obj is None:
        print("[train] WARNING: could not locate actor module — skipping bias init")
    else:
        # Walk to the last nn.Linear layer of whatever the actor's MLP is.
        final_linear = None
        for module in actor_obj.modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == 5:
                final_linear = module
        if final_linear is None:
            print("[train] WARNING: no Linear(out=5) layer found — skipping bias init")
        else:
            with torch.no_grad():
                # Shrink weights so initial output ≈ bias (i.e., MVP defaults)
                # regardless of obs. Without this, random W·features dominates.
                final_linear.weight.mul_(0.01)
                final_linear.bias.copy_(init_bias.to(final_linear.bias.device))
            print(f"[train] actor head: bias = log-MVP, weight × 0.01 "
                  f"(initial output ≈ α=3, φ=0.1, a=b=c=0.05)")

    print(f"[train] arch={args.arch} num_envs={args.num_envs} iters={args.iters}")
    print(f"[train] log_dir = {log_dir}")
    runner.learn(num_learning_iterations=args.iters)
    runner.save(os.path.join(log_dir, f"model_{args.iters}.pt"))


if __name__ == "__main__":
    main()
