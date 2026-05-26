"""Smoke test for the per-obstacle position tracker (Wk3 v1 occlusion fix).

The Isaac Lab dependency makes the production code path hard to test
standalone, so this test replicates the tracker state machine math
directly and verifies the key behaviors:

  1. Visible obstacle → cache updates from this frame's hits
  2. Occluded obstacle (no rays hit) → cache HOLDS last-observed value
     for up to K_persist frames
  3. After K_persist frames of occlusion, cache expires (valid=False)
  4. Re-observation after partial occlusion refreshes the cache cleanly

This is the "back-row cylinder behind the front-row cylinder in the
corridor" case from the perception diagnostic.

Run on the lab box:
    python ~/Desktop/safety-go2/scripts/test_obstacle_tracker.py
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
SHIELD_R_MIN = _p.SHIELD_R_MIN
SHIELD_R_MAX = _p.SHIELD_R_MAX
SHIELD_R_SAFETY_MARGIN = _p.SHIELD_R_SAFETY_MARGIN
SHIELD_SENSOR_RANGE = _p.SHIELD_SENSOR_RANGE
SHIELD_N_RAYS_DEFAULT = _p.SHIELD_N_RAYS_DEFAULT


def step_tracker(
    robot_pos: torch.Tensor,   # (N, 2)
    obs_pos: torch.Tensor,     # (N, K, 2)
    obs_radii: torch.Tensor,   # (N, K)
    cache_pos: torch.Tensor,   # (N, K, 2) in-place updated
    cache_r: torch.Tensor,     # (N, K)
    cache_age: torch.Tensor,   # (N, K) long
    cache_valid: torch.Tensor, # (N, K) bool
    persist_steps: int,
    n_rays: int = SHIELD_N_RAYS_DEFAULT,
):
    """Replicates CbfGo2Env._step_obstacle_tracker. Mutates the cache_*
    tensors in place. Returns nothing (mirror production semantics)."""
    device = robot_pos.device
    N, K, _ = obs_pos.shape

    hit_xy, hit_valid, hit_obs_idx = synthetic_lidar_raycast(
        robot_pos, obs_pos, obs_radii,
        n_rays=n_rays, max_range=SHIELD_SENSOR_RANGE,
        return_obs_idx=True,
    )

    k_range = torch.arange(K, device=device)
    per_obs_mask = (
        hit_obs_idx.unsqueeze(-1) == k_range.view(1, 1, K)
    ) & hit_valid.unsqueeze(-1)
    n_hits_per_obs = per_obs_mask.sum(dim=1)
    visible = n_hits_per_obs > 0

    weighted_sum = torch.einsum(
        "nrc,nrk->nkc",
        hit_xy.to(torch.float32),
        per_obs_mask.to(torch.float32),
    )
    counts_f = n_hits_per_obs.clamp(min=1).to(torch.float32).unsqueeze(-1)
    centroids = weighted_sum / counts_f

    centroids_expanded = centroids.unsqueeze(1)
    hit_xy_expanded = hit_xy.unsqueeze(2)
    dists_all = (hit_xy_expanded - centroids_expanded).norm(dim=-1)
    dists_masked = torch.where(
        per_obs_mask, dists_all,
        torch.full_like(dists_all, float("-inf")),
    )
    max_dist_per_obs, _ = dists_masked.max(dim=1)
    max_dist_per_obs = max_dist_per_obs.clamp(min=0.0)
    r_est = (max_dist_per_obs + SHIELD_R_SAFETY_MARGIN).clamp(
        SHIELD_R_MIN, SHIELD_R_MAX,
    )

    vis = visible
    cache_pos.copy_(torch.where(vis.unsqueeze(-1), centroids, cache_pos))
    cache_r.copy_(torch.where(vis, r_est, cache_r))
    new_age = torch.where(vis, torch.zeros_like(cache_age), cache_age + 1)
    cache_age.copy_(new_age)
    not_expired = new_age <= persist_steps
    cache_valid.copy_(vis | (cache_valid & not_expired))

    return visible, n_hits_per_obs


def init_cache(N: int, K: int, persist_steps: int, device: str):
    cache_pos = torch.zeros((N, K, 2), device=device)
    cache_r = torch.full((N, K), SHIELD_R_MIN, device=device)
    cache_age = torch.full((N, K), persist_steps + 1, dtype=torch.long, device=device)
    cache_valid = torch.zeros((N, K), dtype=torch.bool, device=device)
    return cache_pos, cache_r, cache_age, cache_valid


def test_basic_visibility() -> bool:
    """Two obstacles, both visible. Both should land in cache with reasonable
    positions and radii."""
    print()
    print("Test 1: two visible obstacles → both in cache")
    device = "cpu"
    N, K = 1, 2
    persist = 10

    robot_pos = torch.tensor([[0.0, 0.0]], device=device)
    obs_pos = torch.tensor([[[2.0, +0.5], [2.0, -0.5]]], device=device)
    obs_radii = torch.tensor([[0.30, 0.30]], device=device)
    cache_pos, cache_r, cache_age, cache_valid = init_cache(N, K, persist, device)

    visible, n_hits = step_tracker(
        robot_pos, obs_pos, obs_radii,
        cache_pos, cache_r, cache_age, cache_valid, persist,
    )
    ok = True
    print(f"  visible mask = {visible.tolist()}, n_hits = {n_hits.tolist()}")
    print(f"  cache_pos = {cache_pos.tolist()}")
    print(f"  cache_r = {cache_r.tolist()}")
    print(f"  cache_age = {cache_age.tolist()}")
    print(f"  cache_valid = {cache_valid.tolist()}")
    if not bool(cache_valid.all().item()):
        print("  FAIL: not both slots valid")
        ok = False
    return ok


def test_persistence_after_occlusion() -> bool:
    """Step 1: two obstacles visible. Step 2+: rear obstacle gets a closer
    occluder placed in front of it. The rear obstacle's cache entry should
    HOLD its last-observed pos/r for persist_steps frames."""
    print()
    print("Test 2: rear obstacle occluded after step 1 → cache holds")
    device = "cpu"
    N, K = 1, 3
    persist = 5

    robot_pos = torch.tensor([[0.0, 0.0]], device=device)
    # Frame 1: obs 0 at (2, 0.5), obs 1 at (2, -0.5), obs 2 far away
    obs_pos_t1 = torch.tensor([[
        [2.0, +0.5],
        [2.0, -0.5],
        [50.0, 50.0],   # parked, won't be hit
    ]], device=device)
    # Frame 2+: obs 2 moves IN FRONT of obs 0 (between robot and obs 0),
    # which occludes obs 0. obs 1 still visible.
    obs_pos_occlude = torch.tensor([[
        [2.0, +0.5],
        [2.0, -0.5],
        [1.0, +0.5],    # right between robot and obs 0
    ]], device=device)
    obs_radii = torch.tensor([[0.30, 0.30, 0.30]], device=device)
    cache_pos, cache_r, cache_age, cache_valid = init_cache(N, K, persist, device)

    # Frame 1
    visible, n_hits = step_tracker(
        robot_pos, obs_pos_t1, obs_radii,
        cache_pos, cache_r, cache_age, cache_valid, persist,
    )
    print(f"  frame 1: n_hits = {n_hits.tolist()}, valid = {cache_valid.tolist()}")
    obs0_cached_pos_t1 = cache_pos[0, 0].clone()
    obs0_cached_age_t1 = cache_age[0, 0].item()
    if cache_age[0, 0].item() != 0 or not cache_valid[0, 0].item():
        print("  FAIL: obs 0 should be visible+age=0 after frame 1")
        return False

    # Frames 2..N: obs 0 occluded by obs 2. Cache should hold obs 0 for
    # persist=5 frames, then expire.
    for t in range(2, 10):
        visible, n_hits = step_tracker(
            robot_pos, obs_pos_occlude, obs_radii,
            cache_pos, cache_r, cache_age, cache_valid, persist,
        )
        obs0_age = cache_age[0, 0].item()
        obs0_valid = cache_valid[0, 0].item()
        obs0_n_hits = n_hits[0, 0].item()
        expected_valid = (t - 1) <= persist  # frames 2..6 hold, frame 7+ expire
        expected_age = t - 1                  # 1, 2, ..., 8
        marker = "OK" if (obs0_valid == expected_valid and obs0_age == expected_age) else "FAIL"
        print(f"  frame {t}: obs0 n_hits={obs0_n_hits} age={obs0_age} "
              f"valid={obs0_valid} (expected age={expected_age}, valid={expected_valid})  {marker}")
        if marker == "FAIL":
            return False
        # Also: while valid, the cached position should NOT have changed
        # from frame 1 (it's frozen).
        if obs0_valid:
            drift = (cache_pos[0, 0] - obs0_cached_pos_t1).norm().item()
            if drift > 1e-5:
                print(f"  FAIL: obs0 cached position drifted by {drift} while occluded")
                return False
    return True


def test_re_acquisition() -> bool:
    """Obstacle gets occluded, then unoccluded. The cache should snap back
    to the current observation when re-visible (age=0, fresh pos)."""
    print()
    print("Test 3: occluded then re-acquired → cache refreshes")
    device = "cpu"
    N, K = 1, 2
    persist = 5

    robot_pos = torch.tensor([[0.0, 0.0]], device=device)
    obs_pos_visible = torch.tensor([[
        [2.0, 0.0],
        [50.0, 50.0],
    ]], device=device)
    obs_pos_occluded = torch.tensor([[
        [2.0, 0.0],
        [1.0, 0.0],     # blocker right in front of obs 0
    ]], device=device)
    obs_radii = torch.tensor([[0.30, 0.30]], device=device)
    cache_pos, cache_r, cache_age, cache_valid = init_cache(N, K, persist, device)

    # Frame 1: visible
    step_tracker(robot_pos, obs_pos_visible, obs_radii,
                 cache_pos, cache_r, cache_age, cache_valid, persist)
    print(f"  frame 1: obs0 age={cache_age[0,0].item()} valid={cache_valid[0,0].item()}")

    # Frames 2-4: occluded (age 1, 2, 3 — within persist window)
    for t in range(2, 5):
        step_tracker(robot_pos, obs_pos_occluded, obs_radii,
                     cache_pos, cache_r, cache_age, cache_valid, persist)
    print(f"  after 3 occluded frames: obs0 age={cache_age[0,0].item()} valid={cache_valid[0,0].item()}")
    if cache_age[0, 0].item() != 3 or not cache_valid[0, 0].item():
        print("  FAIL: expected age=3, valid=True at frame 4")
        return False

    # Frame 5: re-acquired (move blocker out of the way)
    step_tracker(robot_pos, obs_pos_visible, obs_radii,
                 cache_pos, cache_r, cache_age, cache_valid, persist)
    print(f"  frame 5 (re-acquired): obs0 age={cache_age[0,0].item()} valid={cache_valid[0,0].item()}")
    if cache_age[0, 0].item() != 0 or not cache_valid[0, 0].item():
        print("  FAIL: expected age=0, valid=True on re-acquisition")
        return False
    return True


def test_corridor_scenario() -> bool:
    """The smoking-gun case from the perception diagnostic: corridor with
    front and back rows of cylinders. Front row visible, back row occluded.
    Without tracker, back row would be invisible to QP. With tracker, the
    first frame catches them all (no occluder ahead of any cylinder yet at
    the robot's initial position), then the robot moves forward and the
    back row gets occluded by the front row.

    For this test we simulate a robot starting BEHIND the corridor (so all
    cylinders visible) then teleporting to the corridor entrance (where
    the front row occludes the back). With persistence, the back row stays
    in cache for K_persist frames."""
    print()
    print("Test 4: corridor scenario — back row occluded after robot enters")
    device = "cpu"
    N, K = 1, 6
    persist = 10

    obs_pos = torch.tensor([[
        # +y rail
        [2.5, +0.55],
        [3.0, +0.55],
        [3.5, +0.55],
        # -y rail
        [2.5, -0.55],
        [3.0, -0.55],
        [3.5, -0.55],
    ]], device=device)
    obs_radii = torch.tensor([[0.30] * 6], device=device)

    # Phase 1: robot far behind, all cylinders visible from a wide vantage.
    robot_far = torch.tensor([[0.0, 0.0]], device=device)
    # Phase 2: robot has entered the corridor; back row occluded by front row.
    robot_entered = torch.tensor([[2.0, 0.0]], device=device)

    cache_pos, cache_r, cache_age, cache_valid = init_cache(N, K, persist, device)

    # Phase 1: visible (some, depending on ray density). The point is to
    # populate the cache for whatever rays do hit.
    for _ in range(3):
        visible, n_hits = step_tracker(
            robot_far, obs_pos, obs_radii,
            cache_pos, cache_r, cache_age, cache_valid, persist,
        )
    print(f"  phase 1 (robot at 0,0): n_hits = {n_hits.tolist()}")
    print(f"    valid mask = {cache_valid.tolist()}")
    phase1_seen = cache_valid.clone()

    # Phase 2: robot moves to corridor entrance. Back row at x=3.5 likely
    # occluded by front-row at x=2.5.
    visible, n_hits = step_tracker(
        robot_entered, obs_pos, obs_radii,
        cache_pos, cache_r, cache_age, cache_valid, persist,
    )
    print(f"  phase 2 (robot at 2,0): n_hits = {n_hits.tolist()}")
    print(f"    valid mask = {cache_valid.tolist()}")

    # Even if back row's n_hits == 0 at phase 2, valid should remain True
    # for any obstacle that was visible during phase 1 (within persist window).
    held = (cache_valid & phase1_seen).sum().item()
    print(f"  obstacles previously seen AND still in cache: {held}/6")
    if held < phase1_seen.sum().item():
        print("  FAIL: some previously-seen obstacles expired immediately")
        return False
    return True


def main() -> int:
    print("Per-obstacle position tracker smoke test")
    print(f"device: cpu, n_rays: {SHIELD_N_RAYS_DEFAULT}")
    all_ok = True
    all_ok = test_basic_visibility() and all_ok
    all_ok = test_persistence_after_occlusion() and all_ok
    all_ok = test_re_acquisition() and all_ok
    all_ok = test_corridor_scenario() and all_ok
    print()
    print("ALL PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
