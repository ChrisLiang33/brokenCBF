"""2D theoretical CBF sandbox.

Single-integrator dynamics: x_dot = u (so g(x) = I, L_g h = grad h).
Single cylinder obstacle.
CBF QP: min ||u - u_des||^2 s.t. L_g h . u + alpha (h - c) + phi ||L_g h||^2 + a >= 0

This module isolates the CBF math from the Isaac Lab RL stack. Used to
compute analytical optima for (alpha, phi, a, c) under specific disturbance
regimes and compare against trained-policy outputs.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class CBFParams:
    alpha: float = 2.0
    phi: float = 0.0
    a: float = 0.0
    c: float = 0.0


@dataclass
class Obstacle:
    pos: np.ndarray  # shape (2,)
    radius: float    # true radius


def barrier(x: np.ndarray, obs: Obstacle, perception_bias: float = 0.0):
    """h(x) = ||x - obs.pos|| - (obs.radius + perception_bias).

    perception_bias > 0 means perceived obstacle is LARGER than true.
    perception_bias < 0 means perceived obstacle is SMALLER than true.

    Returns (h, grad_h) where grad_h is shape (2,).
    """
    d = x - obs.pos
    dist = np.linalg.norm(d)
    h = dist - (obs.radius + perception_bias)
    if dist > 1e-9:
        grad_h = d / dist
    else:
        grad_h = np.zeros(2)
    return h, grad_h


def cbf_qp(
    u_des: np.ndarray,
    x: np.ndarray,
    obs: Obstacle,
    params: CBFParams,
    perception_bias: float = 0.0,
):
    """Closed-form CBF-QP solve, matching the Isaac Lab cbf_go2 sim formulation.

    Constraint (sim form):  L_g h . u  >=  rhs   where
        rhs = -alpha (h - c) + phi ||L_g h||^2 + a

    Equivalently: L_g h . u + alpha (h - c) >= phi ||L_g h||^2 + a.
    So alpha is positive on the LHS (higher alpha = constraint easier in
    safe region, harder near boundary — standard Nagumo). phi and a are
    on the RHS (higher phi/a = constraint tighter — robust CBF margins
    against actuation noise / bounded disturbance, Dean/Kolathaya).

    With single-integrator dynamics, g(x) = I, so L_g h = grad_h.

    KKT closed form:
        slack = L_g h . u_des - rhs
        if slack >= 0:           u_safe = u_des
        else:                    u_safe = u_des + lam * grad_h,
                                 lam = (rhs - L_g h . u_des) / ||grad_h||^2.

    Returns (u_safe, qp_active, info_dict).
    """
    h, grad_h = barrier(x, obs, perception_bias)
    Lgh_sq = np.dot(grad_h, grad_h)
    rhs = -params.alpha * (h - params.c) + params.phi * Lgh_sq + params.a
    slack = np.dot(grad_h, u_des) - rhs
    if slack >= 0 or Lgh_sq < 1e-12:
        return u_des.copy(), False, {"h": h, "slack": slack, "rhs": rhs}
    lam = (rhs - np.dot(grad_h, u_des)) / Lgh_sq
    u_safe = u_des + lam * grad_h
    return u_safe, True, {"h": h, "slack": slack, "rhs": rhs, "lambda": lam}


def simulate(
    x0: np.ndarray,
    goal: np.ndarray,
    obs: Obstacle,
    params: CBFParams,
    T: float = 10.0,
    dt: float = 0.05,
    u_max: float = 2.0,
    perception_bias: float = 0.0,
    actuation_noise_sigma: float = 0.0,
    actuation_noise_mode: str = "gaussian",
    push_fn: Optional[Callable[[int, float], np.ndarray]] = None,
    seed: int = 0,
    goal_radius: float = 0.1,
):
    """Roll out 2D dynamics with CBF safety filter.

    The policy is goal-tracking (u_des = unit vector toward goal, magnitude
    capped at u_max). CBF QP uses the perceived obstacle (radius +
    perception_bias). Actuation noise is added to u_safe BEFORE applying.
    Optional push_fn(i, dt) -> R^2 adds a velocity perturbation.

    Returns dict of arrays.
    """
    rng = np.random.RandomState(seed)
    N = int(T / dt)
    x = x0.copy()
    hist = {
        "t": [], "x": [], "u_des": [], "u_safe": [], "u_applied": [],
        "h_true": [], "h_perc": [], "deflection": [], "qp_active": [],
        "alpha_used": [], "phi_used": [], "a_used": [], "c_used": [],
    }
    reached = False
    for i in range(N):
        # u_des: goal-tracking with magnitude cap.
        dir_to_goal = goal - x
        norm = np.linalg.norm(dir_to_goal)
        if norm < 1e-9:
            u_des = np.zeros(2)
        else:
            u_des = (dir_to_goal / norm) * min(u_max, norm)

        # CBF QP (uses perceived obstacle).
        u_safe, qp_active, info = cbf_qp(u_des, x, obs, params, perception_bias)

        # Clamp u_safe to control authority.
        un = np.linalg.norm(u_safe)
        if un > u_max:
            u_safe = u_safe * (u_max / un)

        # Apply noise + push.
        u_applied = u_safe.copy()
        if actuation_noise_sigma > 0:
            if actuation_noise_mode == "gaussian":
                u_applied = u_applied + rng.normal(0, actuation_noise_sigma, size=2)
            elif actuation_noise_mode == "adversarial":
                # Worst-case noise: aligned with -grad_h (push state INTO obstacle).
                # Magnitude = 2 * sigma (matches ~95% Gaussian magnitude bound in 2D).
                _, grad_h_now = barrier(x, obs, perception_bias)
                gh_norm = np.linalg.norm(grad_h_now)
                if gh_norm > 1e-9:
                    eps = -2.0 * actuation_noise_sigma * (grad_h_now / gh_norm)
                else:
                    eps = np.zeros(2)
                u_applied = u_applied + eps
            else:
                raise ValueError(f"unknown actuation_noise_mode: {actuation_noise_mode}")
        if push_fn is not None:
            u_applied = u_applied + push_fn(i, dt)

        # Step dynamics.
        x_next = x + u_applied * dt

        # Log (state BEFORE the step, action this step).
        h_true, _ = barrier(x, obs, 0.0)
        h_perc, _ = barrier(x, obs, perception_bias)
        deflection = float(np.linalg.norm(u_safe - u_des))
        hist["t"].append(i * dt)
        hist["x"].append(x.copy())
        hist["u_des"].append(u_des.copy())
        hist["u_safe"].append(u_safe.copy())
        hist["u_applied"].append(u_applied.copy())
        hist["h_true"].append(h_true)
        hist["h_perc"].append(h_perc)
        hist["deflection"].append(deflection)
        hist["qp_active"].append(qp_active)
        hist["alpha_used"].append(params.alpha)
        hist["phi_used"].append(params.phi)
        hist["a_used"].append(params.a)
        hist["c_used"].append(params.c)

        x = x_next
        if np.linalg.norm(x - goal) < goal_radius:
            reached = True
            break

    out = {k: np.array(v) for k, v in hist.items()}
    out["reached_goal"] = reached
    out["n_steps"] = int(len(hist["t"]))
    out["min_h_true"] = float(out["h_true"].min()) if len(out["h_true"]) else 0.0
    out["collided"] = bool(out["min_h_true"] < 0.0)
    out["mean_deflection"] = float(out["deflection"].mean()) if len(out["deflection"]) else 0.0
    out["time_to_goal"] = float(out["t"][-1] + dt) if reached else float("inf")
    return out


def plot_trajectory(traj, obs, goal, ax=None, title="", show_qp_active=True):
    """Plot a 2D trajectory with obstacle and goal."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    theta = np.linspace(0, 2 * np.pi, 200)
    ox = obs.pos[0] + obs.radius * np.cos(theta)
    oy = obs.pos[1] + obs.radius * np.sin(theta)
    ax.fill(ox, oy, alpha=0.25, color="red")
    ax.plot(ox, oy, "r-", lw=1.5, label="obstacle (true)")

    xs = traj["x"]
    ax.plot(xs[:, 0], xs[:, 1], "b-", lw=1.2, alpha=0.9)
    ax.plot(xs[0, 0], xs[0, 1], "go", markersize=9, label="start")
    ax.plot(goal[0], goal[1], "g*", markersize=14, label="goal")

    if show_qp_active and len(traj["qp_active"]) > 0:
        qa = traj["qp_active"].astype(bool)
        if qa.any():
            ax.scatter(xs[qa, 0], xs[qa, 1], c="orange", s=8, alpha=0.7,
                       label="QP active")

    if traj.get("collided", False):
        cmask = traj["h_true"] < 0
        if cmask.any():
            ax.scatter(xs[cmask, 0], xs[cmask, 1], c="darkred", s=18,
                       marker="x", label="collision")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.25)
    return ax
