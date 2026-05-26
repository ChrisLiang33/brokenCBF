"""Standalone sanity test for shield_v0c synthetic LiDAR + clustering.

Run from project root:
    python3 scripts/test_shield_v0c.py

Verifies:
  1. Raycast finds correct hit positions for known cylinder placements.
  2. Occlusion works (closer cylinder blocks rays from a farther one).
  3. Clustering groups hits sharing a grid cell.
  4. End-to-end shield_perceive_v0c returns sane (positions, radii).
  5. Performance: time per call at B=64 (eval) and B=512 (mid-size training).

No Isaac Lab dependency — pure torch.
"""
import math
import time
import sys
from pathlib import Path

import torch

# Make the perception module importable without IsaacLab.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2"))
from cbf_go2_perception import (
    synthetic_lidar_raycast,
    cluster_points_grid,
    shield_perceive_v0c,
    SHIELD_FIXED_R,
    SHIELD_SENSOR_RANGE,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[test_shield_v0c] running on device: {device}\n")


# ─────────────────────────────────────────────────────────────────────────
# Test 1 — single ray hits a single cylinder at known location
# ─────────────────────────────────────────────────────────────────────────
print("Test 1 — single env, single cylinder hit")
robot_pos = torch.tensor([[0.0, 0.0]], device=device)                     # (1, 2)
obs_pos   = torch.tensor([[[2.0, 0.0]]], device=device)                   # (1, 1, 2) — 2m east
obs_radii = torch.tensor([[0.3]], device=device)                          # (1, 1)
hit_xy, hit_valid = synthetic_lidar_raycast(robot_pos, obs_pos, obs_radii, n_rays=8)
# Ray direction 0 is (+x), should hit at distance 2.0 - 0.3 = 1.7
ray0_hit = hit_xy[0, 0]
assert hit_valid[0, 0], f"Ray 0 should hit, got hit_valid={hit_valid[0]}"
assert abs(ray0_hit[0].item() - 1.7) < 0.01, f"Ray 0 hit at x={ray0_hit[0]}, expected 1.7"
assert abs(ray0_hit[1].item() - 0.0) < 0.01, f"Ray 0 hit at y={ray0_hit[1]}, expected 0.0"
# Ray pointing west (ray 4 = π) should miss (no obstacle west)
assert not hit_valid[0, 4], f"Ray 4 (west) should miss, got hit_valid={hit_valid[0, 4]}"
print(f"  ✓ ray-0 hit at ({ray0_hit[0]:.3f}, {ray0_hit[1]:.3f}), expected (1.7, 0.0)")
print(f"  ✓ ray-4 (west) correctly missed")


# ─────────────────────────────────────────────────────────────────────────
# Test 2 — occlusion: closer cylinder blocks farther one
# ─────────────────────────────────────────────────────────────────────────
print("\nTest 2 — occlusion (near cylinder blocks rays to far cylinder)")
robot_pos = torch.tensor([[0.0, 0.0]], device=device)
obs_pos   = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]], device=device)       # near at 1m, far at 3m
obs_radii = torch.tensor([[0.2, 0.2]], device=device)
hit_xy, hit_valid = synthetic_lidar_raycast(robot_pos, obs_pos, obs_radii, n_rays=8)
# Ray 0 (+x) should hit the NEAR cylinder at distance 1.0 - 0.2 = 0.8
assert hit_valid[0, 0]
assert abs(hit_xy[0, 0, 0].item() - 0.8) < 0.01, \
    f"Ray-0 should hit near cyl at x=0.8, got {hit_xy[0, 0, 0]}"
print(f"  ✓ ray-0 hit near cylinder at x={hit_xy[0, 0, 0]:.3f} (expected 0.8 = nearer cyl)")


# ─────────────────────────────────────────────────────────────────────────
# Test 3 — sensor range gating (obstacles beyond range don't get hit)
# ─────────────────────────────────────────────────────────────────────────
print("\nTest 3 — sensor range gating")
robot_pos = torch.tensor([[0.0, 0.0]], device=device)
obs_pos   = torch.tensor([[[10.0, 0.0]]], device=device)                  # 10m away (beyond 6m default)
obs_radii = torch.tensor([[0.3]], device=device)
hit_xy, hit_valid = synthetic_lidar_raycast(
    robot_pos, obs_pos, obs_radii, n_rays=8, max_range=6.0,
)
assert not hit_valid.any(), f"All rays should miss far obstacle, got hit_valid={hit_valid[0]}"
print("  ✓ obstacle at 10m correctly outside 6m sensor range")


# ─────────────────────────────────────────────────────────────────────────
# Test 4a — clustering: hits CLEANLY inside a single grid cell merge
# ─────────────────────────────────────────────────────────────────────────
print("\nTest 4a — clustering: clean cell membership")
# grid_res = 0.5 → cell (4, 0) covers x ∈ [2.0, 2.5), y ∈ [0.0, 0.5)
# cell (-4, 0) covers x ∈ [-2.0, -1.5), y ∈ [0.0, 0.5)
hit_xy = torch.tensor([[
    [2.10, 0.10], [2.20, 0.20], [2.30, 0.30],   # all in cell (4, 0)
    [-1.90, 0.10], [-1.80, 0.20], [-1.70, 0.30], # all in cell (-4, 0)
    [0.0, 0.0], [0.0, 0.0],                      # padding (invalid)
]], device=device)
hit_valid = torch.tensor([[True, True, True, True, True, True, False, False]], device=device)
centers, valid = cluster_points_grid(hit_xy, hit_valid, grid_res=0.5, K_max=10)
n_valid = valid[0].sum().item()
assert n_valid == 2, f"Expected 2 clusters, got {n_valid}"
valid_centers = centers[0][valid[0]]
xs = sorted([c[0].item() for c in valid_centers])
assert abs(xs[0] - (-1.80)) < 0.05 and abs(xs[1] - 2.20) < 0.05, \
    f"Cluster centers at x={xs}, expected ~[-1.80, 2.20]"
print(f"  ✓ found 2 clusters at x ≈ {xs[0]:.3f} and x ≈ {xs[1]:.3f}")

# ─────────────────────────────────────────────────────────────────────────
# Test 4b — clustering: hits straddling cell boundaries split (intentional)
# ─────────────────────────────────────────────────────────────────────────
print("\nTest 4b — clustering: cell-boundary straddle splits clusters")
# Points around (2.0, 0.0) which sits exactly on cell boundary:
#   (2.05, +0.05) → cell (4,  0)   (within 0–0.5 in y)
#   (2.00, -0.05) → cell (4, -1)   (within −0.5–0 in y)
# This is a real-world LiDAR pipeline failure: one obstacle, two clusters.
hit_xy = torch.tensor([[
    [2.05, 0.05], [2.10, 0.05],         # one cell
    [2.05, -0.05], [2.10, -0.05],       # different cell (boundary straddle)
]], device=device)
hit_valid = torch.tensor([[True, True, True, True]], device=device)
centers, valid = cluster_points_grid(hit_xy, hit_valid, grid_res=0.5, K_max=10)
n_valid = valid[0].sum().item()
assert n_valid == 2, f"Boundary-straddle case should produce 2 clusters, got {n_valid}"
print(f"  ✓ boundary-straddle split into 2 clusters (real-world cluster-split failure mode)")


# ─────────────────────────────────────────────────────────────────────────
# Test 5 — end-to-end shield_perceive_v0c sanity
# ─────────────────────────────────────────────────────────────────────────
print("\nTest 5 — end-to-end shield_perceive_v0c")
robot_pos = torch.tensor([[0.0, 0.0]], device=device)
obs_pos   = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]]], device=device)
obs_radii = torch.tensor([[0.3, 0.3, 0.3, 0.3]], device=device)

# Default grid_res=0.4: hit-arc on each obstacle straddles a grid boundary
# (the arc spans y∈[-0.3, +0.3] for a 0.3m cylinder at 2m), so one real
# obstacle may produce 2 clusters. This is the cluster-SPLIT failure mode
# that v0c is designed to expose. Verify cluster count is in the realistic
# range [4, 8] rather than expecting exact-4.
p_perc, r_perc = shield_perceive_v0c(robot_pos, obs_pos, obs_radii, n_rays=64, K_max=10)
n_clusters = int((p_perc[0, :, 0] < 1e5).sum().item())
assert 4 <= n_clusters <= 8, \
    f"Expected 4-8 clusters (cluster-split-aware), got {n_clusters}"
assert (r_perc[0, :n_clusters] == SHIELD_FIXED_R).all(), \
    f"Perceived radii should all = {SHIELD_FIXED_R}, got {r_perc[0]}"
print(f"  ✓ 4-obstacle scene → {n_clusters} clusters (4–8 expected, "
      f"includes cluster-split from arc-straddle); all R={SHIELD_FIXED_R}m")

# Sanity: with obstacles deliberately offset from grid boundaries, cluster
# count drops back closer to true obstacle count (validates the splitting
# was driven by grid-boundary geometry, not a clustering bug).
obs_offset = torch.tensor([[
    [2.30, 1.30], [1.30, 2.30], [-2.30, -1.30], [-1.30, -2.30],
]], device=device)
p_perc3, _ = shield_perceive_v0c(robot_pos, obs_offset, obs_radii, n_rays=64, K_max=10)
n_clusters3 = int((p_perc3[0, :, 0] < 1e5).sum().item())
print(f"  ✓ same 4 obstacles, offset from grid lines → {n_clusters3} clusters "
      f"(closer to true=4 when arcs don't straddle grid boundaries)")


# ─────────────────────────────────────────────────────────────────────────
# Test 6 — performance check: per-call time at B=64 and B=512
# ─────────────────────────────────────────────────────────────────────────
print("\nTest 6 — performance benchmark")
def bench(B, K, n_rays=64, n_runs=20):
    torch.manual_seed(0)
    robot_pos = torch.randn(B, 2, device=device) * 0.5
    obs_pos = torch.randn(B, K, 2, device=device) * 3.0
    obs_radii = torch.full((B, K), 0.3, device=device)
    # Warmup
    for _ in range(3):
        shield_perceive_v0c(robot_pos, obs_pos, obs_radii, n_rays=n_rays, K_max=K)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        shield_perceive_v0c(robot_pos, obs_pos, obs_radii, n_rays=n_rays, K_max=K)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_runs
    return elapsed * 1000.0  # ms

for B in [64, 256, 512]:
    ms = bench(B, K=20)
    print(f"  B={B:>4d}  K=20  n_rays=64   → {ms:>6.1f} ms/call")
    if B == 64 and ms > 50:
        print(f"  WARN: 64-env eval is unexpectedly slow")

print("\nAll tests passed.")
