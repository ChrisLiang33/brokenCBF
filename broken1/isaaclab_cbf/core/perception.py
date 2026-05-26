"""Batched LiDAR perception (PyTorch).

Approach for Isaac Lab:
  - Use Isaac Lab's RayCaster sensor to get per-env range arrays (B, n_rays).
  - This module converts those ranges to (a) a top-down occupancy grid for
    the CNN feature extractor, and (b) a list of (center, radius) fits via
    1D clustering on contiguous rays + algebraic least-squares circle fit.
  - Both outputs are returned as fixed-shape tensors (with masks) so the
    downstream pipeline stays GPU-batched.

NOTE: variable-size cluster output is the usual GPU pain point. We resolve
it by capping at MAX_OBSTACLES per env and returning a validity mask.
"""
from __future__ import annotations

import torch


MAX_OBSTACLES = 8                # cap per env
MIN_CLUSTER_SIZE = 3
RANGE_JUMP_THRESHOLD = 0.35      # neighboring rays clustered if diff < this
MAX_FIT_RADIUS = 2.5
OCC_GRID_SIZE = 32               # H = W = 32 cells, robot-centered
OCC_GRID_EXTENT = 4.0            # meters; cell = 8 cm at 32 cells


# ---------------------------------------------------------------------------
# Analytical lidar (ray-circle intersection) against ground-truth obstacles
# ---------------------------------------------------------------------------
@torch.no_grad()
def analytical_lidar(
    robot_xy: torch.Tensor,           # (B, 2)
    robot_yaw: torch.Tensor,          # (B,)
    obs_centers: torch.Tensor,        # (B, N, 2)
    obs_radii: torch.Tensor,          # (B, N)
    obs_mask: torch.Tensor,           # (B, N)
    ray_dirs_body: torch.Tensor,      # (n_rays, 2)  body-frame unit vectors
    max_range: float = 6.0,
    noise_std: torch.Tensor | float = 0.0,
):
    """Closed-form 2D lidar — no USD raycasting needed.
    Returns ranges (B, n_rays) with optional per-env Gaussian range noise.
    """
    B = robot_xy.shape[0]
    n_rays = ray_dirs_body.shape[0]
    device = robot_xy.device

    # Rotate body-frame ray directions to world frame per env
    cy = torch.cos(robot_yaw).unsqueeze(-1)   # (B, 1)
    sy = torch.sin(robot_yaw).unsqueeze(-1)
    rx, ry = ray_dirs_body[:, 0], ray_dirs_body[:, 1]   # (n_rays,)
    rd_x = cy * rx.unsqueeze(0) - sy * ry.unsqueeze(0)   # (B, n_rays)
    rd_y = sy * rx.unsqueeze(0) + cy * ry.unsqueeze(0)

    ranges = torch.full((B, n_rays), max_range, device=device)
    inf = torch.full_like(ranges, float("inf"))
    for j in range(obs_centers.shape[1]):
        dx = (robot_xy[:, 0] - obs_centers[:, j, 0]).unsqueeze(-1)   # (B, 1)
        dy = (robot_xy[:, 1] - obs_centers[:, j, 1]).unsqueeze(-1)
        b = 2.0 * (dx * rd_x + dy * rd_y)                            # (B, n_rays)
        c_q = (dx * dx + dy * dy - obs_radii[:, j:j + 1] ** 2)       # (B, 1)
        disc = b * b - 4 * c_q                                       # (B, n_rays)
        valid = disc >= 0
        sqrt_d = torch.sqrt(torch.clamp_min(disc, 0))
        t1 = torch.where(valid, (-b - sqrt_d) / 2, inf)
        t2 = torch.where(valid, (-b + sqrt_d) / 2, inf)
        t1 = torch.where(t1 > 1e-6, t1, inf)
        t2 = torch.where(t2 > 1e-6, t2, inf)
        t = torch.minimum(t1, t2)
        # Mask invalid obstacles
        t = torch.where(obs_mask[:, j:j + 1] > 0, t, inf)
        ranges = torch.minimum(ranges, t)
    ranges = ranges.clamp_max(max_range)
    if isinstance(noise_std, torch.Tensor) or noise_std > 0:
        hit = ranges < max_range
        noise = torch.randn_like(ranges) * (
            noise_std if isinstance(noise_std, torch.Tensor) else float(noise_std)
        )
        ranges = torch.where(hit, ranges + noise, ranges)
    return ranges


# ---------------------------------------------------------------------------
# Occupancy grid for CNN input
# ---------------------------------------------------------------------------
@torch.no_grad()
def lidar_to_occgrid(
    ranges: torch.Tensor,            # (B, n_rays)  ray distances
    ray_dirs: torch.Tensor,          # (n_rays, 2)  unit directions in body frame
    max_range: float = 6.0,
    grid_size: int = OCC_GRID_SIZE,
    extent: float = OCC_GRID_EXTENT,
) -> torch.Tensor:
    """Return (B, 1, H, W) occupancy grid (1 = hit cell, 0 = free)."""
    B, n_rays = ranges.shape
    device = ranges.device

    # Body-frame hit points (B, n_rays, 2)
    hit_xy = ranges.unsqueeze(-1) * ray_dirs.unsqueeze(0)
    valid = ranges < max_range - 0.1                          # (B, n_rays)

    # Bin to grid cells
    cell_size = (2.0 * extent) / grid_size                    # meters per cell
    half = grid_size // 2
    idx = (hit_xy / cell_size + half).long()                  # (B, n_rays, 2)
    in_range = ((idx >= 0) & (idx < grid_size)).all(dim=-1) & valid

    grid = torch.zeros(B, grid_size, grid_size, device=device)
    # Use scatter; flatten indices
    flat_idx = idx[..., 0] * grid_size + idx[..., 1]          # (B, n_rays)
    flat_idx = flat_idx.clamp_(0, grid_size * grid_size - 1)
    grid_flat = grid.view(B, -1)
    grid_flat.scatter_(1, flat_idx, in_range.float())
    return grid.unsqueeze(1)                                  # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Cylinder fitting (per env)
# ---------------------------------------------------------------------------
def _fit_circle_torch(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Algebraic LS circle fit. pts: (M, 2) — assumes M >= 3.
    Returns center (2,), radius scalar tensor."""
    A = torch.cat([2 * pts, torch.ones_like(pts[:, :1])], dim=1)  # (M, 3)
    b = (pts ** 2).sum(dim=1)
    sol = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
    cx, cy, k = sol[0], sol[1], sol[2]
    r_sq = k + cx * cx + cy * cy
    return torch.stack([cx, cy]), torch.sqrt(r_sq.clamp_min(1e-9))


@torch.no_grad()
def fit_obstacles_per_env(
    ranges: torch.Tensor,            # (n_rays,)  for ONE env
    ray_dirs: torch.Tensor,          # (n_rays, 2)
    robot_xy: torch.Tensor,          # (2,)       robot world pos
    max_range: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-env clustering + circle fit. Returns:
        centers (MAX, 2), radii (MAX,), mask (MAX,)  — padded.
    Called in a vmap/Python loop over envs by perceive_batched().
    """
    device = ranges.device
    n_rays = ranges.shape[0]
    hit = ranges < max_range - 0.1                            # (n_rays,)
    centers = torch.zeros(MAX_OBSTACLES, 2, device=device)
    radii = torch.zeros(MAX_OBSTACLES, device=device)
    mask = torch.zeros(MAX_OBSTACLES, device=device)
    if hit.sum() < MIN_CLUSTER_SIZE:
        return centers, radii, mask

    # 1D clustering on ray index — range-continuity test
    clusters: list[list[int]] = []
    current: list[int] = []
    for i in range(n_rays):
        if not bool(hit[i]):
            if current:
                clusters.append(current); current = []
            continue
        if not current or abs(float(ranges[i] - ranges[current[-1]])) < RANGE_JUMP_THRESHOLD:
            current.append(i)
        else:
            clusters.append(current); current = [i]
    if current:
        clusters.append(current)
    # Wrap-around merge
    if (len(clusters) >= 2 and clusters[0][0] == 0 and clusters[-1][-1] == n_rays - 1
            and abs(float(ranges[0] - ranges[n_rays - 1])) < RANGE_JUMP_THRESHOLD):
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()

    # Convert each cluster's hits → world-frame points → circle fit
    world_pts = robot_xy + ranges.unsqueeze(-1) * ray_dirs    # (n_rays, 2)
    out_i = 0
    for idxs in clusters:
        if len(idxs) < MIN_CLUSTER_SIZE or out_i >= MAX_OBSTACLES:
            continue
        pts = world_pts[torch.tensor(idxs, device=device)]
        try:
            c, r = _fit_circle_torch(pts)
        except RuntimeError:
            continue
        if float(r) > MAX_FIT_RADIUS:
            continue
        centers[out_i] = c
        radii[out_i] = r
        mask[out_i] = 1.0
        out_i += 1
    return centers, radii, mask


@torch.no_grad()
def perceive_batched(
    ranges: torch.Tensor,            # (B, n_rays)
    ray_dirs: torch.Tensor,          # (n_rays, 2)  body-frame unit vectors
    robot_xy: torch.Tensor,          # (B, 2)
    yaw: torch.Tensor | None = None, # (B,) heading angle, if non-zero
    max_range: float = 6.0,
):
    """Batched fit. Returns:
        centers (B, MAX, 2), radii (B, MAX), mask (B, MAX)
    """
    B = ranges.shape[0]
    device = ranges.device
    centers = torch.zeros(B, MAX_OBSTACLES, 2, device=device)
    radii = torch.zeros(B, MAX_OBSTACLES, device=device)
    mask = torch.zeros(B, MAX_OBSTACLES, device=device)
    for i in range(B):
        rd = ray_dirs
        if yaw is not None:
            cy, sy = torch.cos(yaw[i]), torch.sin(yaw[i])
            rot = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])])
            rd = ray_dirs @ rot.T
        c, r, m = fit_obstacles_per_env(ranges[i], rd, robot_xy[i], max_range)
        centers[i] = c
        radii[i] = r
        mask[i] = m
    return centers, radii, mask


# ---------------------------------------------------------------------------
# Frame-to-frame velocity estimation
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_velocities(
    cur_centers: torch.Tensor,        # (B, MAX, 2)
    cur_mask:    torch.Tensor,        # (B, MAX)
    prev_centers: torch.Tensor,       # (B, MAX, 2)
    prev_mask:    torch.Tensor,       # (B, MAX)
    dt: float,
    match_thr: float = 0.5,
) -> torch.Tensor:
    """Match cur to prev by nearest-neighbor center distance, finite-diff for v.
    Returns velocities (B, MAX, 2). Zero for unmatched / first-frame entries.
    """
    B, M, _ = cur_centers.shape
    velocities = torch.zeros_like(cur_centers)
    # (B, M_cur, M_prev) pairwise distance
    diff = cur_centers.unsqueeze(2) - prev_centers.unsqueeze(1)
    dists = diff.norm(dim=-1)
    # Mask invalid prev entries with +inf
    dists = dists.masked_fill(prev_mask.unsqueeze(1) == 0, float("inf"))
    min_d, min_idx = dists.min(dim=-1)                        # (B, M_cur)
    matched = (min_d < match_thr) & (cur_mask > 0)

    batch_arange = torch.arange(B, device=cur_centers.device).unsqueeze(-1).expand(-1, M)
    matched_prev_xy = prev_centers[batch_arange, min_idx]
    v = (cur_centers - matched_prev_xy) / max(dt, 1e-6)
    velocities = torch.where(matched.unsqueeze(-1), v, torch.zeros_like(v))
    return velocities
