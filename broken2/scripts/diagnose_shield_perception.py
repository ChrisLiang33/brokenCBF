"""Diagnose shield_v0c perception bias on known obstacle configurations.

Three iterations of fixes haven't dropped collision_rate (broken Wk3 v1 →
rfit → gr60: all ~0.70). Time to stop theorizing about what shield_v0c
produces and actually look. This script sets up synthetic obstacle scenes
with known (position, radius), runs the full LiDAR → cluster → radius-fit
pipeline, and prints the bias per obstacle.

Key metric: PERCEIVED NEAR-SURFACE DISTANCE vs TRUE NEAR-SURFACE DISTANCE
along the robot→obstacle ray. The QP brakes when h(robot, perceived) = 0,
i.e. when robot crosses perceived near-surface. If perceived near-surface
is CLOSER to robot than true near-surface, the QP brakes early (over-
protective, h_min > 0). If perceived near-surface is FARTHER from robot
than true near-surface, the QP brakes late and the robot drives past true
near-surface before realizing — collision.

Run on the lab box:
    python ~/Desktop/safety-go2/scripts/diagnose_shield_perception.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch


def _load_perception_module():
    here = os.path.dirname(os.path.abspath(__file__))
    perception_py = os.path.normpath(os.path.join(
        here, "..", "IsaacLab", "source", "isaaclab_tasks",
        "isaaclab_tasks", "manager_based", "safety", "cbf_go2",
        "cbf_go2_perception.py",
    ))
    spec = importlib.util.spec_from_file_location(
        "cbf_go2_perception_standalone", perception_py,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_p = _load_perception_module()
synthetic_lidar_raycast = _p.synthetic_lidar_raycast
cluster_points_grid_v = _p.cluster_points_grid_v
SHIELD_GRID_RES_DEFAULT = _p.SHIELD_GRID_RES_DEFAULT
SHIELD_K_MAX_DEFAULT = _p.SHIELD_K_MAX_DEFAULT
SHIELD_N_RAYS_DEFAULT = _p.SHIELD_N_RAYS_DEFAULT
SHIELD_SENSOR_RANGE = _p.SHIELD_SENSOR_RANGE


def perceive_one_env(
    robot_pos: torch.Tensor,    # (2,)
    obs_centers: torch.Tensor,  # (K, 2) world positions
    obs_radii: torch.Tensor,    # (K,)
):
    """Run the full perception pipeline on a single env, return per-obstacle
    perception outputs aligned with the true obstacle list. For each true
    obstacle, find the perceived cluster whose centroid is closest and
    report that as the perceived (center, radius) for this obstacle.
    Returns a list of dicts."""
    robot_pos_b = robot_pos.unsqueeze(0)                        # (1, 2)
    obs_centers_b = obs_centers.unsqueeze(0)                    # (1, K, 2)
    obs_radii_b = obs_radii.unsqueeze(0)                        # (1, K)

    hit_xy, hit_valid = synthetic_lidar_raycast(
        robot_pos_b, obs_centers_b, obs_radii_b,
        n_rays=SHIELD_N_RAYS_DEFAULT, max_range=SHIELD_SENSOR_RANGE,
    )                                                            # (1, n_rays, 2), (1, n_rays)
    centers, radii, valid = cluster_points_grid_v(
        hit_xy, hit_valid,
        grid_res=SHIELD_GRID_RES_DEFAULT, K_max=SHIELD_K_MAX_DEFAULT,
        return_radii=True,
    )                                                            # (1, K_max, 2), (1, K_max), (1, K_max)
    centers = centers[0]
    radii = radii[0]
    valid = valid[0]
    n_clusters = int(valid.sum().item())

    results = []
    for k in range(obs_centers.shape[0]):
        true_c = obs_centers[k]
        true_r = float(obs_radii[k].item())

        # Find the perceived cluster closest to this true obstacle.
        valid_centers = centers[valid]                          # (n_valid, 2)
        if valid_centers.shape[0] == 0:
            results.append(dict(
                true_center=true_c, true_radius=true_r,
                perceived_center=None, perceived_radius=None,
                matched=False,
                n_hits_visible=0,
                true_dist=float((true_c - robot_pos).norm().item()),
                perceived_dist=None,
                true_near_surface=None,
                perceived_near_surface=None,
                h_gap=None,
            ))
            continue
        dists = (valid_centers - true_c).norm(dim=-1)           # (n_valid,)
        j = int(dists.argmin().item())
        valid_radii = radii[valid]
        perc_c = valid_centers[j]
        perc_r = float(valid_radii[j].item())

        true_dist = float((true_c - robot_pos).norm().item())
        perc_dist = float((perc_c - robot_pos).norm().item())
        true_near = true_dist - true_r
        perc_near = perc_dist - perc_r
        h_gap = true_near - perc_near  # >0 = perceived surface closer to robot (over-protect)
                                       # <0 = perceived surface farther from robot (UNDER-protect → collision)

        # Count hits that actually hit this specific obstacle (for diagnostic).
        # synthetic_lidar_raycast returns the closest hit per ray; we approximate
        # "hits on obstacle k" as hits within max(true_r, perc_r) + 0.10m of true_c.
        hit_xy_v = hit_xy[0][hit_valid[0]]                       # (n_hits, 2)
        d_to_true = (hit_xy_v - true_c).norm(dim=-1)
        n_hits_visible = int((d_to_true < true_r + 0.10).sum().item())

        results.append(dict(
            true_center=true_c, true_radius=true_r,
            perceived_center=perc_c, perceived_radius=perc_r,
            matched=True,
            n_hits_visible=n_hits_visible,
            true_dist=true_dist,
            perceived_dist=perc_dist,
            true_near_surface=true_near,
            perceived_near_surface=perc_near,
            h_gap=h_gap,
        ))

    return results, n_clusters


def print_table_header():
    print(f"{'scenario':<30} {'R_true':>6} {'d_true':>6} {'n_hits':>6} "
          f"{'R_perc':>6} {'d_perc':>6} {'near_true':>9} {'near_perc':>9} "
          f"{'h_gap':>7} {'direction':>12}")
    print("-" * 110)


def print_row(label: str, r: dict):
    if not r["matched"]:
        print(f"{label:<30} {r['true_radius']:>6.3f} {r['true_dist']:>6.3f} "
              f"{r['n_hits_visible']:>6d} "
              f"{'---':>6} {'---':>6} {r['true_near_surface']:>9.3f} {'---':>9} "
              f"{'---':>7} {'NO_CLUSTER':>12}")
        return
    direction = "OVER-protect" if r["h_gap"] > 0 else "UNDER-protect"
    print(f"{label:<30} {r['true_radius']:>6.3f} {r['true_dist']:>6.3f} "
          f"{r['n_hits_visible']:>6d} "
          f"{r['perceived_radius']:>6.3f} {r['perceived_dist']:>6.3f} "
          f"{r['true_near_surface']:>9.3f} {r['perceived_near_surface']:>9.3f} "
          f"{r['h_gap']:>+7.3f} {direction:>12}")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, grid_res: {SHIELD_GRID_RES_DEFAULT}, "
          f"n_rays: {SHIELD_N_RAYS_DEFAULT}")
    print()

    # Robot at origin facing +x.
    robot_pos = torch.tensor([0.0, 0.0], device=device)

    print("=" * 110)
    print("SCENARIO A: head-on obstacles at d=2.0m, varying radius")
    print("=" * 110)
    print_table_header()
    for R in [0.10, 0.20, 0.30, 0.40, 0.50]:
        obs_c = torch.tensor([[2.0, 0.0]], device=device)
        obs_r = torch.tensor([R], device=device)
        results, _ = perceive_one_env(robot_pos, obs_c, obs_r)
        print_row(f"R={R:.2f}, d=2.0, head-on", results[0])

    print()
    print("=" * 110)
    print("SCENARIO B: grazing-angle obstacles at d=2.0m, varying angle")
    print("(R=0.30 fixed; angle measured from robot's forward direction)")
    print("=" * 110)
    print_table_header()
    for angle_deg in [0, 30, 60, 80, 90]:
        angle = torch.tensor(angle_deg * 3.14159 / 180.0)
        obs_c = torch.tensor([[2.0 * torch.cos(angle).item(),
                               2.0 * torch.sin(angle).item()]], device=device)
        obs_r = torch.tensor([0.30], device=device)
        results, _ = perceive_one_env(robot_pos, obs_c, obs_r)
        print_row(f"angle={angle_deg}°, R=0.30, d=2.0", results[0])

    print()
    print("=" * 110)
    print("SCENARIO C: large obstacle at various distances")
    print("(R=0.50 fixed)")
    print("=" * 110)
    print_table_header()
    for d in [1.0, 1.5, 2.0, 3.0, 4.0]:
        obs_c = torch.tensor([[d, 0.0]], device=device)
        obs_r = torch.tensor([0.50], device=device)
        results, _ = perceive_one_env(robot_pos, obs_c, obs_r)
        print_row(f"R=0.50, d={d:.1f}, head-on", results[0])

    print()
    print("=" * 110)
    print("SCENARIO D: two close obstacles (cluster-merge risk)")
    print("=" * 110)
    print_table_header()
    for sep in [0.4, 0.6, 0.8, 1.2, 2.0]:
        # Two obstacles symmetrically placed at x=2.0, y=±sep/2
        obs_c = torch.tensor([
            [2.0, +sep / 2],
            [2.0, -sep / 2],
        ], device=device)
        obs_r = torch.tensor([0.30, 0.30], device=device)
        results, n_clusters = perceive_one_env(robot_pos, obs_c, obs_r)
        for i, r in enumerate(results):
            print_row(f"2x R=0.30, sep={sep:.2f}, [{i}]", r)
        print(f"   → n_clusters detected = {n_clusters} (expected 2)")

    print()
    print("=" * 110)
    print("SCENARIO E: small obstacle in clutter (the corridor case)")
    print("Two rails at y=±0.55, x in [2.5, 3.5] every 0.5m, R=0.30")
    print("=" * 110)
    print_table_header()
    obs_c = torch.tensor([
        [2.5, +0.55], [3.0, +0.55], [3.5, +0.55],
        [2.5, -0.55], [3.0, -0.55], [3.5, -0.55],
    ], device=device)
    obs_r = torch.tensor([0.30] * 6, device=device)
    results, n_clusters = perceive_one_env(robot_pos, obs_c, obs_r)
    for i, r in enumerate(results):
        side = "+y" if i < 3 else "-y"
        print_row(f"corridor [{side}, x={obs_c[i,0]:.1f}]", r)
    print(f"   → n_clusters detected = {n_clusters} (expected 2-6 depending on merge)")

    print()
    print("=" * 110)
    print("READING THE TABLE")
    print("=" * 110)
    print("h_gap > 0  → perceived surface CLOSER to robot than true surface → over-protective (safe)")
    print("h_gap < 0  → perceived surface FARTHER from robot than true surface → UNDER-PROTECTIVE → collision risk")
    print("Look at rows with negative h_gap. Those are the cases where the policy/QP will drive")
    print("the robot toward what it perceives as safe distance but actually collides.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
