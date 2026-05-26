"""Batched smoothed-SDF CBF safety filter (PyTorch).

Direct port of the MVP's safety_filter (numpy) to batched torch, with all
operations vectorized over the batch dimension B (= num_envs in Isaac Lab).

Constraint:
    slack(u) = L_f h + A·u + α·(h − c) − φ·‖A‖² − a − b·‖u‖   ≥ 0

The b·‖u‖ term is handled by a fixed Picard iteration (5 passes) — pure
tensor math, no data-dependent branching, GPU-friendly across thousands of
parallel envs.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Smoothed SDF
# ---------------------------------------------------------------------------
def compute_h_smooth(
    x: torch.Tensor,                  # (B, 2)   robot xy
    obs_centers: torch.Tensor,        # (B, N, 2) obstacle centers
    obs_radii: torch.Tensor,          # (B, N)    obstacle radii
    obs_mask: torch.Tensor,           # (B, N)    1 = valid, 0 = padded
    lam: float = 1.0,
    gamma: float = 2.0,
):
    """Return (h, grad_h, idx_closest) where:
        h         (B,)     smoothed signed-distance value
        grad_h    (B, 2)   gradient of h w.r.t. robot xy
        idx_closest (B,)   index of the active (closest) obstacle per env
    """
    diff = x.unsqueeze(1) - obs_centers                       # (B, N, 2)
    dist = diff.norm(dim=-1).clamp_min(1e-9)                  # (B, N)
    sdf_per_obs = dist - obs_radii                            # (B, N)
    sdf_per_obs = sdf_per_obs.masked_fill(obs_mask == 0, float("inf"))

    sdf, idx = sdf_per_obs.min(dim=1)                         # (B,), (B,)
    # Gather closest obstacle's gradient direction (unit vector away from center)
    batch_arange = torch.arange(x.shape[0], device=x.device)
    diff_active = diff[batch_arange, idx]                     # (B, 2)
    dist_active = dist[batch_arange, idx].unsqueeze(-1)       # (B, 1)
    grad_sdf = diff_active / dist_active                      # (B, 2)

    exp_term = torch.exp(-gamma * sdf)
    h = lam * (1.0 - exp_term)
    grad_h = (lam * gamma * exp_term).unsqueeze(-1) * grad_sdf
    return h, grad_h, idx


# ---------------------------------------------------------------------------
# Safety filter
# ---------------------------------------------------------------------------
def safety_filter(
    x: torch.Tensor,                  # (B, 2)
    u_nom: torch.Tensor,              # (B, 2)
    obs_centers: torch.Tensor,        # (B, N, 2)
    obs_radii: torch.Tensor,          # (B, N)
    obs_mask: torch.Tensor,           # (B, N)
    alpha: torch.Tensor,              # (B,) — from policy
    phi:   torch.Tensor,              # (B,)
    a:     torch.Tensor,              # (B,)
    b:     torch.Tensor,              # (B,)
    c:     torch.Tensor,              # (B,)
    obs_velocities: torch.Tensor | None = None,   # (B, N, 2), L_f h feedforward
    lam: float = 1.0,
    gamma: float = 2.0,
    n_picard: int = 5,
):
    """Return u_safe (B, 2).

    All terms (alpha, phi, a, b, c) are per-env tensors — the RL policy
    produces them. Picard iteration handles the b·‖u‖ SOC constraint.
    """
    h, A, idx = compute_h_smooth(x, obs_centers, obs_radii, obs_mask, lam, gamma)

    A_sq = (A * A).sum(dim=-1)                                # (B,)

    # L_f h contribution from obstacle motion (frame-to-frame velocity estimate)
    L_f_h = torch.zeros_like(h)
    if obs_velocities is not None:
        batch_arange = torch.arange(x.shape[0], device=x.device)
        v_active = obs_velocities[batch_arange, idx]          # (B, 2)
        L_f_h = -(A * v_active).sum(dim=-1)                   # (B,)

    const = L_f_h + alpha * (h - c) - phi * A_sq - a          # (B,)
    A_dot_u_nom = (A * u_nom).sum(dim=-1)                     # (B,)

    # Picard iteration: u_{k+1} = project(u_nom | A·u ≥ −const + b·‖u_k‖)
    u = u_nom.clone()
    for _ in range(n_picard):
        u_norm = u.norm(dim=-1)                               # (B,)
        rhs = -const + b * u_norm
        gap = rhs - A_dot_u_nom                               # (B,)
        # only project when gap > 0; otherwise u_nom is feasible
        violation = (gap > 0).float().unsqueeze(-1)
        lam_coef = (gap / (A_sq + 1e-9)).unsqueeze(-1)
        u = u_nom + violation * lam_coef * A
    return u
