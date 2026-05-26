"""Record a video of the Go2-CBF env running with hardcoded MVP params.

Usage:
    ~/IsaacLab/isaaclab.sh -p play_video.py --num_envs 16 \
        --video --video_length 600 --enable_cameras --headless

Output: a .mp4 in ./videos/<timestamp>/rl-video-*.mp4
"""
from __future__ import annotations

import argparse
import os
import math


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--loco_ckpt", type=str,
                   default="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/"
                           "2026-05-26_09-47-24/exported/policy.pt")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--episode_s", type=float, default=12.0)
    p.add_argument("--v_max", type=float, default=0.5)
    p.add_argument("--log_alpha", type=float, default=1.099)
    p.add_argument("--log_phi",   type=float, default=-2.302)
    p.add_argument("--log_a",     type=float, default=-2.996)
    p.add_argument("--log_b",     type=float, default=-2.996)
    p.add_argument("--log_c",     type=float, default=-2.996)
    p.add_argument("--rec_length", type=int, default=600,
                   help="number of policy steps to record")
    p.add_argument("--rec_dir", type=str, default="./videos")
    p.add_argument("--policy_ckpt", type=str, default="",
                   help="Path to trained policy .pt (e.g., checkpoints/fullsetup/"
                        "<ts>/model_500.pt). If empty, use hardcoded MVP log params.")
    # AppLauncher will add: --headless, --enable_cameras, --video, etc.
    return p


def main():
    parser = parse_args()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import torch
    import gymnasium as gym
    from env.go2_cbf_env import Go2CbfFlatEnvCfg, Go2CbfRLEnv

    env_cfg = Go2CbfFlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = 8.0
    env_cfg.episode_length_s = args.episode_s
    # DR off — clean visualization
    env_cfg.rand_lidar_noise    = (0.0, 0.0)
    env_cfg.rand_sigma_pose     = (0.0, 0.0)
    env_cfg.rand_drift_std      = (0.0, 0.0)
    env_cfg.rand_adv_prob       = (0.0, 0.0)
    env_cfg.rand_tracking_err   = (0.0, 0.0)
    env_cfg.rand_mass_scale     = (1.0, 1.0)
    env_cfg.rand_motor_strength = (1.0, 1.0)
    env_cfg.rand_friction       = (1.0, 1.0)
    env_cfg.rand_v_max          = (args.v_max, args.v_max)

    env = Go2CbfRLEnv(cfg=env_cfg, render_mode="rgb_array")
    env.locomotion = torch.jit.load(args.loco_ckpt, map_location=env.device).eval()
    for p in env.locomotion.parameters():
        p.requires_grad_(False)

    os.makedirs(args.rec_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args.rec_dir,
        step_trigger=lambda step: step == 0,
        video_length=args.rec_length,
        name_prefix="go2-cbf",
        disable_logger=True,
    )
    print(f"[play] recording {args.rec_length} steps to {args.rec_dir}/")

    # Load trained policy if a ckpt path was provided, else use hardcoded params
    use_trained_policy = bool(args.policy_ckpt)
    if use_trained_policy:
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import (
            RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,
            RslRlMLPModelCfg, RslRlPpoAlgorithmCfg,
        )
        # We don't need full training cfg; just build a runner and load weights.
        env_wrap = RslRlVecEnvWrapper(env.unwrapped)
        env_wrap.num_actions = 5
        from tensordict import TensorDict
        _outer_obs, _ = env.unwrapped.reset()
        _outer_td = TensorDict(_outer_obs, batch_size=[env.unwrapped.num_envs])
        env_wrap.get_observations = lambda: _outer_td

        # Build minimal cfg to instantiate the runner so we can load the ckpt.
        runner_cfg = RslRlOnPolicyRunnerCfg(
            seed=0, device=str(env.unwrapped.device),
            num_steps_per_env=24, max_iterations=1,
            empirical_normalization=False,
            save_interval=100, experiment_name="play", run_name="",
            logger="tensorboard",
            obs_groups={"actor": ["policy"], "critic": ["policy"]},
            actor=RslRlMLPModelCfg(
                hidden_dims=[512, 256, 128], activation="elu",
                obs_normalization=True,
                stochastic=True, init_noise_std=0.3, noise_std_type="scalar",
                distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
                    init_std=0.3, std_type="scalar"),
            ),
            critic=RslRlMLPModelCfg(
                hidden_dims=[512, 256, 128], activation="elu",
                obs_normalization=True,
                stochastic=False, init_noise_std=0.0, noise_std_type="scalar",
                distribution_cfg=None,
            ),
            algorithm=RslRlPpoAlgorithmCfg(
                value_loss_coef=1.0, use_clipped_value_loss=True,
                clip_param=0.1, entropy_coef=0.001,
                num_learning_epochs=5, num_mini_batches=4,
                learning_rate=1.0e-4, schedule="adaptive",
                gamma=0.99, lam=0.95, desired_kl=0.005, max_grad_norm=1.0,
            ),
        )
        cfg_dict = runner_cfg.to_dict()
        for k in ("actor", "critic"):
            for d in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
                cfg_dict.get(k, {}).pop(d, None)
        cfg_dict.pop("policy", None)
        cfg_dict.pop("empirical_normalization", None)
        runner = OnPolicyRunner(env_wrap, cfg_dict, log_dir=None,
                                device=str(env.unwrapped.device))
        runner.load(args.policy_ckpt)
        # Get a deterministic inference function from the loaded actor
        policy_fn = runner.get_inference_policy(device=env.unwrapped.device)
        print(f"[play] loaded trained policy: {args.policy_ckpt}")
    else:
        fixed_log = torch.tensor(
            [args.log_alpha, args.log_phi, args.log_a, args.log_b, args.log_c],
            device=env.unwrapped.device,
        ).unsqueeze(0).expand(env.unwrapped.num_envs, -1).contiguous()
        print(f"[play] hardcoded log params: α={math.exp(args.log_alpha):.2f}, "
              f"φ={math.exp(args.log_phi):.2f}, a={math.exp(args.log_a):.3f}, "
              f"b={math.exp(args.log_b):.3f}, c={math.exp(args.log_c):.3f}")
    print(f"[play] running {args.rec_length} steps...")

    obs, _ = env.reset()
    with torch.no_grad():
        for t in range(args.rec_length):
            if use_trained_policy:
                # MLPModel expects a TensorDict keyed by obs_group name ("policy")
                from tensordict import TensorDict
                obs_td = TensorDict(
                    {"policy": obs["policy"]},
                    batch_size=[env.unwrapped.num_envs],
                )
                action = policy_fn(obs_td)
            else:
                action = fixed_log
            obs, _, _, _, _ = env.step(action)

    print(f"[play] done. Videos in {args.rec_dir}/")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
