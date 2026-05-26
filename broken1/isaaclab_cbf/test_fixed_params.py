"""Diagnostic: run Go2CbfRLEnv with HARDCODED CBF params — no policy.

Purpose: verify the CBF math + wrapper can solve the navigation task at all
when fed MVP-known-good params (α=3, φ=0.1, a=b=c=0.05). If success rate is
high, the problem is purely on the training side. If success rate is low, the
problem is in the env / CBF / wrapper and no PPO will fix it.

Usage:
    python test_fixed_params.py --num_envs 512 --episode_s 12.0
"""
from __future__ import annotations

import argparse
import math


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--loco_ckpt", type=str,
                   default="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/"
                           "2026-05-26_09-47-24/exported/policy.pt")
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--episode_s", type=float, default=12.0,
                   help="give it generous time; default 12s vs training 8s")
    p.add_argument("--no_dr", action="store_true", default=True,
                   help="diagnostic mode — DR off so we test the math alone")
    p.add_argument("--log_alpha", type=float, default=1.099,
                   help="default = log(3.0)")
    p.add_argument("--log_phi",   type=float, default=-2.302,
                   help="default = log(0.1)")
    p.add_argument("--log_a",     type=float, default=-2.996,
                   help="default = log(0.05)")
    p.add_argument("--log_b",     type=float, default=-2.996,
                   help="default = log(0.05)")
    p.add_argument("--log_c",     type=float, default=-2.996,
                   help="default = log(0.05)")
    p.add_argument("--v_max",     type=float, default=0.5,
                   help="speed cap (m/s). Loco is comfortable at 0.5; 1.0 is edge.")
    # Turn-and-go controller knobs (per friend's review)
    p.add_argument("--k_yaw",      type=float, default=2.0)
    p.add_argument("--omega_max",  type=float, default=1.0)
    p.add_argument("--v_fwd_max",  type=float, default=1.0)
    p.add_argument("--v_lat_max",  type=float, default=0.25)
    p.add_argument("--debug_print", action="store_true",
                   help="Every 50 steps, print cmd + robot pose for env 0")
    return p.parse_args()


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher
    _ = AppLauncher(headless=True).app

    import torch
    from env.go2_cbf_env import Go2CbfFlatEnvCfg, Go2CbfRLEnv

    env_cfg = Go2CbfFlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = 8.0
    env_cfg.episode_length_s = args.episode_s

    if args.no_dr:
        env_cfg.rand_lidar_noise    = (0.0, 0.0)
        env_cfg.rand_sigma_pose     = (0.0, 0.0)
        env_cfg.rand_drift_std      = (0.0, 0.0)
        env_cfg.rand_adv_prob       = (0.0, 0.0)
        env_cfg.rand_tracking_err   = (0.0, 0.0)
        env_cfg.rand_mass_scale     = (1.0, 1.0)
        env_cfg.rand_motor_strength = (1.0, 1.0)
        env_cfg.rand_friction       = (1.0, 1.0)
        env_cfg.rand_v_max          = (args.v_max, args.v_max)
        print(f"[test] DR disabled, v_max={args.v_max} — pure CBF math test")

    env = Go2CbfRLEnv(cfg=env_cfg)
    # Wire turn-and-go controller knobs onto the env
    env.K_YAW     = args.k_yaw
    env.OMEGA_MAX = args.omega_max
    env.V_FWD_MAX = args.v_fwd_max
    env.V_LAT_MAX = args.v_lat_max
    print(f"[test] controller: K_YAW={args.k_yaw}, OMEGA_MAX={args.omega_max}, "
          f"V_FWD_MAX={args.v_fwd_max}, V_LAT_MAX={args.v_lat_max}")
    env.locomotion = torch.jit.load(args.loco_ckpt, map_location=env.device).eval()
    for p in env.locomotion.parameters():
        p.requires_grad_(False)

    # Hardcoded log params (same for every env, every step)
    fixed_log_params = torch.tensor(
        [args.log_alpha, args.log_phi, args.log_a, args.log_b, args.log_c],
        device=env.device,
    ).unsqueeze(0).expand(env.num_envs, -1).contiguous()

    print(f"[test] log params: α={args.log_alpha:.3f} (→{math.exp(args.log_alpha):.3f}), "
          f"φ={args.log_phi:.3f} (→{math.exp(args.log_phi):.3f}), "
          f"a={args.log_a:.3f} (→{math.exp(args.log_a):.3f}), "
          f"b={args.log_b:.3f} (→{math.exp(args.log_b):.3f}), "
          f"c={args.log_c:.3f} (→{math.exp(args.log_c):.3f})")
    print(f"[test] num_envs={env.num_envs}, episode={args.episode_s}s")

    obs, _ = env.reset()

    n = env.num_envs
    reached     = torch.zeros(n, dtype=torch.bool, device=env.device)
    collided    = torch.zeros(n, dtype=torch.bool, device=env.device)
    fell        = torch.zeros(n, dtype=torch.bool, device=env.device)
    timed_out   = torch.zeros(n, dtype=torch.bool, device=env.device)
    # Distances at end (or at termination)
    final_dist  = torch.full((n,), float("nan"), device=env.device)
    init_dist   = (env.scene["robot"].data.root_pos_w[:, :2]
                   - env.scene.env_origins[:, :2]
                   - env.goal_xy).norm(dim=-1)
    print(f"[test] initial avg distance to goal: {init_dist.mean().item():.2f} m, "
          f"min={init_dist.min().item():.2f}, max={init_dist.max().item():.2f}")

    n_steps = int(args.episode_s / 0.02)
    print(f"[test] running {n_steps} policy steps...")
    with torch.no_grad():
        for t in range(n_steps):
            outer_obs, reward, term, trunc, info = env.step(fixed_log_params)
            if args.debug_print and t % 50 == 0:
                cmd = env.command_manager._terms["base_velocity"].vel_command_b[0]
                xy0 = (env.scene["robot"].data.root_pos_w[0, :2]
                       - env.scene.env_origins[0, :2])
                quat0 = env.scene["robot"].data.root_quat_w[0]
                yaw0 = torch.atan2(
                    2.0 * (quat0[0]*quat0[3] + quat0[1]*quat0[2]),
                    1.0 - 2.0 * (quat0[2]**2 + quat0[3]**2),
                )
                print(f"  t={t:3d}  pos=({xy0[0]:.2f},{xy0[1]:.2f})  "
                      f"yaw={yaw0.item():.2f}  "
                      f"cmd=({cmd[0].item():.2f},{cmd[1].item():.2f},{cmd[2].item():.2f})")
            # Snapshot terminations as they first fire
            # term flag includes std_term | reached_flag | collided_obs
            log = info.get("log", {})
            # We can't tell from outside which OF (reached/collided/fell) fired,
            # so we use the env's last-step internal state.
            # Robot xy NOW
            xy = (env.scene["robot"].data.root_pos_w[:, :2]
                  - env.scene.env_origins[:, :2])
            cur_dist = (xy - env.goal_xy).norm(dim=-1)
            new_reached   = (cur_dist < 0.4) & ~reached & ~collided & ~fell & ~timed_out
            # for collided we'd need the obstacle distance computation. Use env state:
            diff_obs = env.true_obs_centers - xy.unsqueeze(1)
            d_obs = diff_obs.norm(dim=-1)
            new_coll = ((d_obs < env.true_obs_radii + 0.05) & (env.true_obs_mask > 0)).any(dim=-1) \
                       & ~reached & ~collided & ~fell & ~timed_out
            new_fell = term & ~new_reached & ~new_coll & ~reached & ~collided & ~fell & ~timed_out
            new_tout = trunc & ~new_reached & ~new_coll & ~new_fell & ~reached & ~collided & ~fell & ~timed_out

            # Record final distance at first termination
            first_term = new_reached | new_coll | new_fell | new_tout
            final_dist = torch.where(first_term & final_dist.isnan(), cur_dist, final_dist)

            reached   |= new_reached
            collided  |= new_coll
            fell      |= new_fell
            timed_out |= new_tout

    # For envs that never terminated, fill final_dist with current dist
    xy = (env.scene["robot"].data.root_pos_w[:, :2]
          - env.scene.env_origins[:, :2])
    cur_dist = (xy - env.goal_xy).norm(dim=-1)
    final_dist = torch.where(final_dist.isnan(), cur_dist, final_dist)

    def pct(x): return 100 * int(x.sum().item()) / n

    print("\n========== RESULTS ==========")
    print(f"  reached:    {int(reached.sum().item())}/{n}  ({pct(reached):.1f}%)")
    print(f"  collided:   {int(collided.sum().item())}/{n}  ({pct(collided):.1f}%)")
    print(f"  fell:       {int(fell.sum().item())}/{n}  ({pct(fell):.1f}%)")
    print(f"  timed out:  {int(timed_out.sum().item())}/{n}  ({pct(timed_out):.1f}%)")
    print(f"  no term:    {n - int((reached|collided|fell|timed_out).sum().item())}/{n}")
    print(f"  initial avg dist: {init_dist.mean().item():.2f} m")
    print(f"  final   avg dist: {final_dist.mean().item():.2f} m")
    print(f"  best  final dist: {final_dist.min().item():.2f} m")
    print(f"  worst final dist: {final_dist.max().item():.2f} m")


if __name__ == "__main__":
    main()
