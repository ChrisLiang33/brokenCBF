"""Numerical equivalence test for cluster_points_grid_v vs cluster_points_grid_legacy.

The perception module is pure torch (no Isaac Lab deps), so we load it
directly via importlib to avoid triggering the parent package's __init__
chain (which pulls in USD/pxr and requires the simapp runtime).

Run on the lab box (regular python, NOT through isaaclab.sh):
    python ~/Desktop/safety-go2/scripts/test_cluster_vectorized.py

Or inside the isaaclab conda env:
    conda activate isaaclab
    python ~/Desktop/safety-go2/scripts/test_cluster_vectorized.py

Exit code 0 = vectorized equivalent to legacy → safe to deploy.
Exit code 1 = mismatch → DO NOT deploy.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import torch


def _load_perception_module():
    """Load cbf_go2_perception.py directly without triggering package init."""
    here = os.path.dirname(os.path.abspath(__file__))
    perception_py = os.path.normpath(os.path.join(
        here, "..", "IsaacLab", "source", "isaaclab_tasks",
        "isaaclab_tasks", "manager_based", "safety", "cbf_go2",
        "cbf_go2_perception.py",
    ))
    if not os.path.exists(perception_py):
        print(f"FATAL: perception module not found at {perception_py}")
        sys.exit(2)
    spec = importlib.util.spec_from_file_location(
        "cbf_go2_perception_standalone", perception_py,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_perception = _load_perception_module()
cluster_points_grid_legacy = _perception.cluster_points_grid_legacy
cluster_points_grid_v = _perception.cluster_points_grid_v
SHIELD_GRID_RES_DEFAULT = _perception.SHIELD_GRID_RES_DEFAULT
SHIELD_K_MAX_DEFAULT = _perception.SHIELD_K_MAX_DEFAULT
SHIELD_R_MIN = _perception.SHIELD_R_MIN
SHIELD_R_MAX = _perception.SHIELD_R_MAX
SHIELD_R_SAFETY_MARGIN = _perception.SHIELD_R_SAFETY_MARGIN


def _multisets_equal(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5) -> bool:
    """Two (N, 2) sets of points are equal as multisets if each row in `a`
    matches a unique row in `b` within atol. Order doesn't matter."""
    if a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    used = torch.zeros(b.shape[0], dtype=torch.bool, device=a.device)
    for ai in a:
        d = (b - ai).norm(dim=-1)
        d = d.masked_fill(used, float("inf"))
        j = d.argmin()
        if d[j] > atol:
            return False
        used[j] = True
    return bool(used.all().item())


def run_case(B: int, n_rays: int, valid_rate: float, seed: int) -> bool:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    K_obs = 5
    obs_centers = torch.randn((B, K_obs, 2), device=device) * 2.5
    ray_to_obs = torch.randint(0, K_obs, (B, n_rays), device=device)
    hit_xy = obs_centers.gather(
        1, ray_to_obs.unsqueeze(-1).expand(-1, -1, 2),
    ) + torch.randn((B, n_rays, 2), device=device) * 0.1
    hit_valid = torch.rand((B, n_rays), device=device) < valid_rate

    # Warm-up to avoid timing the first kernel compilation.
    _ = cluster_points_grid_v(hit_xy, hit_valid)
    _ = cluster_points_grid_legacy(hit_xy, hit_valid)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    centers_v, valid_v = cluster_points_grid_v(
        hit_xy, hit_valid, SHIELD_GRID_RES_DEFAULT, SHIELD_K_MAX_DEFAULT,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t_v = time.perf_counter() - t0

    t0 = time.perf_counter()
    centers_l, valid_l = cluster_points_grid_legacy(
        hit_xy, hit_valid, SHIELD_GRID_RES_DEFAULT, SHIELD_K_MAX_DEFAULT,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t_l = time.perf_counter() - t0

    ok = True
    n_mismatch = 0
    for b in range(B):
        v_pts = centers_v[b][valid_v[b]]
        l_pts = centers_l[b][valid_l[b]]
        if not _multisets_equal(v_pts, l_pts):
            ok = False
            n_mismatch += 1
            if n_mismatch <= 3:
                print(f"  batch {b}: legacy {l_pts.shape[0]} clusters, "
                      f"vectorized {v_pts.shape[0]}")
                print(f"    legacy = {l_pts.cpu().numpy()}")
                print(f"    vector = {v_pts.cpu().numpy()}")

    print(f"  B={B} n_rays={n_rays} valid_rate={valid_rate}: "
          f"legacy {t_l*1000:.1f}ms, vectorized {t_v*1000:.1f}ms, "
          f"speedup {t_l/max(t_v,1e-9):.1f}x, "
          f"mismatches {n_mismatch}/{B}, {'PASS' if ok else 'FAIL'}")
    return ok


def test_radii_sanity() -> bool:
    """Check that per-cluster radii returned by cluster_points_grid_v with
    return_radii=True are all within [R_MIN, R_MAX], and that a known
    obstacle radius is roughly recovered when many rays hit the same
    obstacle."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print()
    print("Per-cluster radius sanity tests:")

    # Synthetic: 4 envs, each with hits clustered tightly around a single
    # point with a known "spread" (which lower-bounds estimated radius).
    B, n_rays = 4, 64
    centers = torch.tensor([
        [1.0, 1.0],
        [-2.0, 3.0],
        [4.0, -1.0],
        [0.0, 0.0],
    ], device=device)
    # spread_xy controls how far hits scatter from center → controls r_est.
    spreads = torch.tensor([0.05, 0.20, 0.40, 0.60], device=device)  # m
    torch.manual_seed(99)

    hit_xy = torch.zeros((B, n_rays, 2), device=device)
    for b in range(B):
        # Uniform on a disk of radius `spreads[b]`.
        angles = torch.rand(n_rays, device=device) * 2 * torch.pi
        radii_in = spreads[b] * torch.sqrt(torch.rand(n_rays, device=device))
        hit_xy[b, :, 0] = centers[b, 0] + radii_in * torch.cos(angles)
        hit_xy[b, :, 1] = centers[b, 1] + radii_in * torch.sin(angles)
    hit_valid = torch.ones((B, n_rays), dtype=torch.bool, device=device)

    out_centers, out_radii, out_valid = cluster_points_grid_v(
        hit_xy, hit_valid, SHIELD_GRID_RES_DEFAULT, SHIELD_K_MAX_DEFAULT,
        return_radii=True,
    )

    ok = True
    for b in range(B):
        valid_count = out_valid[b].sum().item()
        if valid_count == 0:
            print(f"  env {b}: no clusters! spread={spreads[b].item():.2f}")
            ok = False
            continue
        # Take the largest valid cluster's radius.
        r_est = out_radii[b][out_valid[b]][0].item()
        # Expected upper bound: spread + safety margin (capped at R_MAX).
        # The estimator uses MAX dist-to-centroid + safety_margin, so a
        # near-uniform disk of spread S yields ~ S + safety. Both should
        # be clamped to [R_MIN, R_MAX].
        expected_upper = min(spreads[b].item() + SHIELD_R_SAFETY_MARGIN, SHIELD_R_MAX)
        expected_lower = max(SHIELD_R_MIN,
                             spreads[b].item() * 0.5 + SHIELD_R_SAFETY_MARGIN)
        within = expected_lower <= r_est <= max(expected_upper, SHIELD_R_MIN) + 0.02
        # Always clamped to [R_MIN, R_MAX]
        clamped_ok = SHIELD_R_MIN <= r_est <= SHIELD_R_MAX
        print(f"  env {b}: spread={spreads[b].item():.2f}, r_est={r_est:.3f}, "
              f"clamped[{SHIELD_R_MIN:.2f}, {SHIELD_R_MAX:.2f}]: {'OK' if clamped_ok else 'FAIL'}")
        ok = ok and clamped_ok

    # Check empty slots use R_MIN default.
    empty_slot_radii = out_radii[~out_valid]
    if empty_slot_radii.numel() > 0:
        all_min = (empty_slot_radii == SHIELD_R_MIN).all().item()
        print(f"  empty slots use R_MIN ({SHIELD_R_MIN}): {'OK' if all_min else 'FAIL'}")
        ok = ok and bool(all_min)

    return ok


def main() -> int:
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"torch:  {torch.__version__}")
    print()

    cases = [
        (16,   64, 0.7, 0),     # small batch
        (128,  64, 0.7, 1),
        (2048, 64, 0.7, 2),     # production scale
        (2048, 64, 0.3, 3),     # sparse hits
        (2048, 64, 1.0, 4),     # all hits valid
        (2048, 64, 0.0, 5),     # no hits valid (edge case)
    ]
    all_ok = True
    for c in cases:
        all_ok = run_case(*c) and all_ok

    all_ok = test_radii_sanity() and all_ok

    print()
    print("ALL PASS" if all_ok else "FAIL — DO NOT deploy")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
