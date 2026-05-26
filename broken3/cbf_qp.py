"""Robustified CBF-QP safety filter -- multi-obstacle (command space).

The robot is a single integrator, so f_hat = 0 and g_hat = I. For each
obstacle i the distance barrier  h_i(x) = ||x - p_i|| - r_safe_i  has a
unit gradient, hence  L_g h_i = grad_h_i  and  ||L_g h_i||^2 = 1, and
Lucas's robustified constraint collapses to one linear inequality per
obstacle:

    grad_h_i . u  >=  phi - alpha * h_i           (a = b = c = 0)

All obstacle constraints are stacked into a single QP solved each step:

    minimize    ||u - u_nom||^2 + M * sum_i delta_i^2
    subject to  grad_h_i . u + delta_i >= rhs_i      for every obstacle i
                ||u||_2 <= u_max
                delta_i >= 0

A PER-CONSTRAINT slack delta_i keeps the QP always feasible without one
infeasible constraint relaxing the others -- which matters inside the
corridor, where two opposing constraints are active at once.

The QP is built once with cvxpy Parameters (DPP-compliant) so repeated
solves inside the RL loop reuse the compiled problem.
"""
import cvxpy as cp
import numpy as np


class CBFQPSolver:
    def __init__(self, u_max: float, slack_penalty: float, n_obstacles: int):
        self.u_max = u_max
        self.n = n_obstacles

        # parameters refreshed every solve
        self._grad_h = [cp.Parameter(2) for _ in range(n_obstacles)]
        self._rhs = [cp.Parameter() for _ in range(n_obstacles)]
        self._u_nom = cp.Parameter(2)

        # decision variables
        self._u = cp.Variable(2)
        self._delta = cp.Variable(n_obstacles, nonneg=True)

        objective = cp.Minimize(
            cp.sum_squares(self._u - self._u_nom)
            + slack_penalty * cp.sum_squares(self._delta)
        )
        constraints = [cp.norm(self._u, 2) <= u_max]
        for i in range(n_obstacles):
            constraints.append(
                self._grad_h[i] @ self._u + self._delta[i] >= self._rhs[i]
            )
        self._prob = cp.Problem(objective, constraints)

    def solve(self, x, obstacles, u_nom, phi, alpha, a=0.0, c=0.0):
        """Solve the safety QP.

        obstacles : list of (center_x, center_y, r_safe, v_x, v_y), one
                    per obstacle. (v_x, v_y) is the obstacle velocity in
                    units/sec -- zero for static obstacles.
        phi, alpha: the learned parameters (a, c default to 0 in the MVP).

        For a moving obstacle h_i(x,t) = ||x - p_i(t)|| - r_safe_i, the
        chain rule gives  d h_i/d t = grad_h_i . u - grad_h_i . v_i, so
        the robustified constraint picks up a + grad_h_i . v_i term:

            grad_h_i . u >= phi - alpha*(h_i - c) + a + grad_h_i . v_i

        Returns (u_safe, total_slack, h_min).
        """
        x = np.asarray(x, dtype=float)
        h_vals = []
        for i, obs in enumerate(obstacles):
            cx, cy, r_safe = obs[0], obs[1], obs[2]
            v = np.array([obs[3], obs[4]]) if len(obs) >= 5 else np.zeros(2)
            diff = x - np.array([cx, cy])
            dist = max(float(np.linalg.norm(diff)), 1e-6)
            h = dist - r_safe
            grad = diff / dist                         # unit -> ||L_g h||^2=1
            self._grad_h[i].value = grad
            # + grad.v is the moving-obstacle (d h/d t) compensation
            self._rhs[i].value = float(phi + a - alpha * (h - c)
                                       + grad @ v)
            h_vals.append(h)
        self._u_nom.value = np.asarray(u_nom, dtype=float)

        try:
            self._prob.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception:
            try:
                self._prob.solve(warm_start=True)
            except Exception:
                self._prob.solve()

        if self._u.value is None:                      # solver failed
            return np.asarray(u_nom, dtype=float), 0.0, min(h_vals)

        slack = (float(np.sum(self._delta.value))
                 if self._delta.value is not None else 0.0)
        return np.asarray(self._u.value, dtype=float), slack, min(h_vals)
