"""Smoke-test the frozen Go2 locomotion policy in a CLEAN env.

No CBF, no obstacles, no DR. Just send the loco a constant forward velocity
command and see how often the robot falls in an episode.

Usage:
    python test_loco_alone.py \
        --loco_ckpt /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/<run>/exported/policy.pt \
        --num_envs 512 --episode_s 8.0 --cmd_vx 0.5
"""
from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--loco_ckpt", type=str,
                   default="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/"
                           "2026-05-26_09-47-24/exported/policy.pt")
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--episode_s", type=float, default=8.0)
    p.add_argument("--cmd_vx", type=float, default=0.5,
                   help="constant forward velocity command (m/s)")
    return p.parse_args()


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher
    _ = AppLauncher(headless=True).app

    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )

    cfg = UnitreeGo2FlatEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.episode_s
    env = ManagerBasedRLEnv(cfg=cfg)

    loco = torch.jit.load(args.loco_ckpt, map_location=env.device).eval()
    for p in loco.parameters():
        p.requires_grad_(False)

    env.reset()

    # Inject a constant forward velocity command for all envs.
    cmd = torch.zeros(env.num_envs, 3, device=env.device)
    cmd[:, 0] = args.cmd_vx
    env.command_manager._terms["base_velocity"].vel_command_b[:] = cmd

    # Run one episode; count terminations by reason.
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    timed_out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    n_policy_steps = int(args.episode_s / env.cfg.sim.dt / env.cfg.decimation)
    print(f"[loco-test] num_envs={env.num_envs}, "
          f"cmd_vx={args.cmd_vx} m/s, episode={args.episode_s}s "
          f"({n_policy_steps} policy steps)")

    with torch.no_grad():
        for t in range(n_policy_steps):
            # Re-inject the command each step — auto-resets blow it away.
            env.command_manager._terms["base_velocity"].vel_command_b[:] = cmd
            obs = env.observation_manager.compute()["policy"]
            action = loco(obs)
            _, _, term, trunc, _ = env.step(action)
            # Mark first-time terminations
            new_fell  = term & ~timed_out & ~fell
            new_tout  = trunc & ~timed_out & ~fell
            fell      |= new_fell
            timed_out |= new_tout

    n = env.num_envs
    n_fell = int(fell.sum().item())
    n_tout = int(timed_out.sum().item())
    n_alive = n - n_fell - n_tout
    print("---- Results ----")
    print(f"  fell (base_contact):  {n_fell}/{n}  ({100*n_fell/n:.1f}%)")
    print(f"  timed out (full ep):  {n_tout}/{n}  ({100*n_tout/n:.1f}%)")
    print(f"  still alive at end:   {n_alive}/{n}  ({100*n_alive/n:.1f}%)")
    print(f"  TOTAL fall rate over 1 episode: {100*n_fell/n:.1f}%")


if __name__ == "__main__":
    main()
