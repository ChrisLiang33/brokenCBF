"""CBF-QP safety filter for a single-integrator agent.

The filter operates on *perceived* obstacle positions, which may be offset
from the true positions by sensor noise. Collisions are detected in the
sim against the true geoms, so picking too aggressive an α with high noise
will cause crashes — that's the gradient the RL policy learns from.

CBF condition (ISSf):
    L_f h(x) + L_g h(x) · u  -  φ ||L_g h(x)||²  ≥  -α · h(x).

Single-integrator dynamics (`p_dot = u`): f = 0, g = I, so the QP reduces
to a single scalar constraint and is solved in closed form.

Barrier (smoothed SDF, eq. 19–20):
    sdf(p) = min_i  ||p - ρ̂_i|| - (R_robot + R_i)    (uses perceived ρ̂)
    h(p)   = λ (1 - exp(-γ · sdf(p)))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planner import Cylinder


@dataclass(frozen=True)
class CBFParams:
    robot_radius: float = 0.15
    alpha: float = 4.0           # default α used when no override is supplied
    lam: float = 1.0             # λ
    gamma: float = 0.5           # γ
    phi: float = 0.05            # ISSf relaxation


class CBFSafetyFilter:
    def __init__(self, obstacles: list[Cylinder], params: CBFParams | None = None) -> None:
        self.params = params or CBFParams()
        self._true_centers = np.array([[o.cx, o.cy] for o in obstacles], dtype=float)
        self._combined_R = np.array(
            [self.params.robot_radius + o.radius for o in obstacles])
        # What the filter *thinks* the obstacles are at. Defaults to truth.
        self._perceived_centers = self._true_centers.copy()

    def set_perceived_obstacles(self, centers: np.ndarray) -> None:
        """Override the perceived obstacle positions (e.g., true + sensor noise)."""
        c = np.asarray(centers, dtype=float)
        if c.shape != self._true_centers.shape:
            raise ValueError(
                f"perceived centers shape {c.shape} != true {self._true_centers.shape}")
        self._perceived_centers = c.copy()

    @property
    def perceived_centers(self) -> np.ndarray:
        return self._perceived_centers.copy()

    def h_and_grad(self, pos_xy: np.ndarray) -> tuple[float, np.ndarray]:
        diffs = pos_xy - self._perceived_centers
        dists = np.linalg.norm(diffs, axis=1)
        sdfs = dists - self._combined_R
        i = int(np.argmin(sdfs))
        sdf = float(sdfs[i])
        grad_sdf = np.zeros(2) if dists[i] < 1e-9 else diffs[i] / dists[i]
        e = float(np.exp(-self.params.gamma * sdf))
        h = self.params.lam * (1.0 - e)
        grad_h = self.params.lam * self.params.gamma * e * grad_sdf
        return h, grad_h

    def filter(self, pos_xy: np.ndarray, u_des: np.ndarray,
               alpha_override: float | None = None,
               phi_override: float | None = None) -> tuple[np.ndarray, dict]:
        h, Lgh = self.h_and_grad(pos_xy)
        a_eff = float(alpha_override) if alpha_override is not None else self.params.alpha
        phi_eff = float(phi_override) if phi_override is not None else self.params.phi
        denom = float(Lgh @ Lgh)
        rhs = -a_eff * h + phi_eff * denom
        lhs = float(Lgh @ u_des)
        sdf_min_perceived = float(np.min(
            np.linalg.norm(pos_xy - self._perceived_centers, axis=1) - self._combined_R))

        if lhs >= rhs or denom < 1e-12:
            return u_des.copy(), {"h": h, "sdf_min": sdf_min_perceived,
                                  "alpha": a_eff, "phi": phi_eff, "active": False}

        mu = (rhs - lhs) / denom
        u_safe = u_des + mu * Lgh
        return u_safe, {"h": h, "sdf_min": sdf_min_perceived, "alpha": a_eff,
                        "phi": phi_eff, "active": True, "mu": mu}
