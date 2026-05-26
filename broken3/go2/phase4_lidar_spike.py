"""Lidar pipeline spike -- retire the risk that Isaac Lab's warp-based
raycaster works headless on labbox before we commit to the RMA rebuild.

What this does:
1. Loads the Phase 2 env (single obstacle Go2 scene).
2. Attaches a `RayCasterCfg` 2D ring sensor to the robot base (36 rays,
   tilted ~5 deg downward so they hit the ground -- sanity-check signal).
3. Steps with zero (phi, alpha) for `n_steps` and dumps per-step lidar
   ranges + base xy to a CSV.

Pass criterion:
- No crashes (the actual risk).
- CSV has 36 range columns with non-degenerate values (i.e., the rays are
  hitting *something*, not all max_distance / all zero).

This is intentionally a minimal spike: no obstacles in the USD scene yet
(our obstacles are logical-only in the action term), so rays will only
hit the ground plane. That's fine -- the goal is "does the sensor pipeline
work", not "can we navigate by lidar yet".

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase4_lidar_spike.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --num_envs 1 --n_steps 20 --out_csv phase4_lidar_spike.csv \\
        --headless
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True,
                    help="Frozen Go2 locomotion .pt (needed for env init).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--n_steps", type=int, default=20)
parser.add_argument("--out_csv", default="phase4_lidar_spike.csv")
parser.add_argument("--n_rays", type=int, default=36,
                    help="Horizontal resolution: 360 / n_rays degrees per ray.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import csv

import gymnasium as gym
import torch

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor


TASK = "Isaac-CBF-Adaptive-Go2-Phase2-v0"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) loco actor (required to build the env's action term)
    loco_ckpt = retrieve_file_path(args_cli.checkpoint)
    print(f"[spike] locomotion -> {loco_ckpt}")
    locomotion_actor = load_locomotion_actor(loco_ckpt, device)

    # 2) env cfg
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = locomotion_actor

    # 3) attach a 2D lidar ring to the robot base. Pattern: single channel,
    # tilted ~5 deg DOWN so the rays actually hit the ground plane (a fully-
    # horizontal ring goes to infinity over flat terrain and tells us nothing).
    deg_per_ray = 360.0 / args_cli.n_rays
    env_cfg.scene.lidar = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=0.02,                      # match sim step
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.30)),  # ~base height
        attach_yaw_only=True,                    # rotate with robot yaw only
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-5.0, -5.0),     # single line, tilted down
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=deg_per_ray,
        ),
        max_distance=20.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],       # only the ground plane exists
    )

    print(f"[spike] lidar: {args_cli.n_rays} rays at {deg_per_ray:.1f} deg apart")

    # 4) build env
    env = gym.make(TASK, cfg=env_cfg)
    print("[spike] env built. resetting ...")
    obs, _ = env.reset()

    lidar = env.unwrapped.scene.sensors["lidar"]

    # 5) step with zero action; dump lidar ranges + base xy
    rows = []
    for step in range(args_cli.n_steps):
        action = torch.zeros((args_cli.num_envs, 2), device=device)
        env.step(action)

        # ray_hits_w: (num_envs, num_rays, 3) world-frame hit positions
        # pos_w:      (num_envs, 3) sensor world position
        hits = lidar.data.ray_hits_w               # (N, R, 3)
        pos = lidar.data.pos_w                     # (N, 3)
        ranges = torch.linalg.norm(hits - pos.unsqueeze(1), dim=-1)  # (N, R)

        base_xy = env.unwrapped.scene["robot"].data.root_pos_w[0, :2].cpu().tolist()
        r0 = ranges[0].cpu().tolist()
        row = {"step": step, "base_x": base_xy[0], "base_y": base_xy[1]}
        row.update({f"r_{i:02d}": v for i, v in enumerate(r0)})
        rows.append(row)

    # 6) summary + write
    last = rows[-1]
    ray_vals = [v for k, v in last.items() if k.startswith("r_")]
    print(f"[spike] step {args_cli.n_steps-1}: base=({last['base_x']:.2f},{last['base_y']:.2f})")
    print(f"[spike] sample ranges (first 8): "
          f"{[f'{v:.2f}' for v in ray_vals[:8]]}")
    print(f"[spike] range stats: min={min(ray_vals):.2f}  max={max(ray_vals):.2f}  "
          f"mean={sum(ray_vals)/len(ray_vals):.2f}")

    with open(args_cli.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[spike] wrote {len(rows)} rows -> {args_cli.out_csv}")

    # pass/fail heuristic: ranges should be finite, non-zero, and vary
    all_ok = (min(ray_vals) > 0.0
              and max(ray_vals) < 20.0
              and (max(ray_vals) - min(ray_vals)) > 0.1)
    print(f"\n[spike] verdict: {'PASS -- raycaster pipeline works' if all_ok else 'REVIEW -- ranges look degenerate'}")

    env.close()
    simulation_app.close()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
