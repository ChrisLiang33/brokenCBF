"""Step 2 + 3 of the RMA build: verify the analytic 'lidar' returns
sensible per-ray ranges to the action term's obstacle.

NB on the design pivot: Isaac Lab's warp raycaster's `mesh_prim_paths`
doesn't accept the `env_.*` regex needed for per-env-replicated obstacle
prims, so we compute lidar analytically (ray-cylinder intersection)
from the action term's cached obstacle positions. Same (N, R) output,
multi-obstacle out of the box, faster, no Isaac Lab sensor pipeline
needed. See [[lidar-strategy]].

Phase 2 inheritance puts the obstacle at local (2.5, 0.3) with physical
radius 0.9. Robot spawns at (0, 0) with yaw ~0. Obstacle bearing from
robot's +x = atan2(0.3, 2.5) ~= +6.84 deg. With 36 rays at 10 deg
spacing starting at -180 deg, the ray closest to that bearing is
ray index round((6.84 + 180) / 10) = 19 (bearing +10 deg). That ray
should read approximately sqrt(2.5^2 + 0.3^2) - 0.9 ~= 1.62 m.

PASS criteria:
- lidar_obs returns shape (N, 36) with all values in [0, 20]
- min range < 3 m (cylinder hit)
- closest-ray bearing within ~30 deg of +6.84 deg
- median range == max (most rays miss the obstacle)

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase5_lidar_obstacle_spike.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --headless
"""
from __future__ import annotations

import argparse
import math
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--n_steps", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import gymnasium as gym
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor
from cbf_task.mdp import lidar_obs


TASK = "Isaac-CBF-Adaptive-Go2-RMA-v0"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco

    print(f"[lidar_spike] building {TASK} ({args_cli.num_envs} envs) ...")
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()

    for _ in range(args_cli.n_steps):
        action = torch.zeros((args_cli.num_envs, 2), device=device)
        env.step(action)

    # call analytic lidar directly
    ranges = lidar_obs(env.unwrapped)                  # (N, R)
    print(f"[lidar_spike] lidar_obs shape: {tuple(ranges.shape)}")

    n_rays = ranges.shape[1]
    r0 = ranges[0].cpu().tolist()
    print(f"[lidar_spike] env 0  min={min(r0):.2f}  max={max(r0):.2f}  "
          f"mean={sum(r0)/len(r0):.2f}")
    sorted_r = sorted(r0)
    median_r = sorted_r[n_rays // 2]
    print(f"[lidar_spike] env 0  median={median_r:.2f} m")

    indexed = sorted(enumerate(r0), key=lambda x: x[1])[:5]
    print(f"[lidar_spike] env 0  5 shortest rays (idx, bearing_deg, range_m):")
    for idx, r in indexed:
        bearing = -180.0 + idx * (360.0 / n_rays)
        print(f"    ray {idx:2d}  bearing={bearing:+6.1f} deg  range={r:.2f} m")

    closest_idx = indexed[0][0]
    closest_bearing = -180.0 + closest_idx * (360.0 / n_rays)
    obs_local = env_cfg.actions.cbf_param.obstacle_xy
    expected_bearing = math.degrees(math.atan2(obs_local[1], obs_local[0]))
    bearing_err = abs(((closest_bearing - expected_bearing + 180) % 360) - 180)

    print(f"\n[lidar_spike] expected obstacle bearing: {expected_bearing:+.2f} deg")
    print(f"[lidar_spike] closest-ray bearing:        {closest_bearing:+.2f} deg")
    print(f"[lidar_spike] bearing error:              {bearing_err:.2f} deg")

    shape_ok = ranges.shape == (args_cli.num_envs, 36)
    range_ok = (ranges.min().item() >= 0.0
                and ranges.max().item() <= 20.0)
    obstacle_seen = (
        min(r0) < 3.0
        and median_r > 15.0
        and bearing_err < 30.0
    )

    print(f"\n[lidar_spike] shape_ok: {shape_ok} | range_ok: {range_ok} "
          f"| obstacle_seen: {obstacle_seen}")
    ok = shape_ok and range_ok and obstacle_seen
    print(f"\n[lidar_spike] verdict: "
          f"{'PASS -- analytic lidar works' if ok else 'REVIEW'}")

    env.close()
    simulation_app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
