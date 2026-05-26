"""Robustified CBF-QP safety filter (command space, single integrator).

The robot is a single integrator, so f_hat = 0 and g_hat = I.  With the
distance barrier  h(x) = ||x - p_obs|| - r_safe, the gradient grad_h is a
unit vector, hence  L_g h = grad_h  and  ||L_g h||^2 = 1.

Lucas's robustified constraint

    dh/dx . f_hat + dh/dx . g_hat . u  - phi*||L_g h||^2 - a - b||u||
        + alpha*(h - c)  >= 0

therefore collapses, for this MVP, to a single linear inequality in u:

    grad_h . u  >=  phi - alpha * h           (a = b = c = 0)

The QP solved each step:

    minimize    ||u - u_nom||^2 + M * delta^2
    subject to  grad_h . u + delta >= rhs
                ||u||_2 <= u_max
                delta >= 0

The slack delta keeps the QP always feasible (important: an RL env that
can throw a solver error mid-rollout is a debugging nightmare).  A large
delta in `info` flags that the *nominal* request was un-satisfiable.

The QP is built once with cvxpy Parameters (DPP-compliant) so repeated
solves inside the RL loop reuse the compiled problem.
"""
import cvxpy as cp
import numpy as np


class CBFQPSolver:
    def __init__(self, u_max: float, slack_penalty: float):
        self.u_max = u_max

        # parameters refreshed every solve
        self._grad_h = cp.Parameter(2)
        self._u_nom = cp.Parameter(2)
        self._rhs = cp.Parameter()

        # decision variables
        self._u = cp.Variable(2)
        self._delta = cp.Variable(nonneg=True)

        objective = cp.Minimize(
            cp.sum_squares(self._u - self._u_nom)
            + slack_penalty * cp.square(self._delta)
        )
        constraints = [
            self._grad_h @ self._u + self._delta >= self._rhs,
            cp.norm(self._u, 2) <= u_max,
        ]
        self._prob = cp.Problem(objective, constraints)

    def solve(self, x, p_obs, r_safe, u_nom, phi, alpha, a=0.0, c=0.0):
        """Return (u_safe, slack, h).  `phi` and `alpha` are the learned
        parameters; a, c default to 0 in the MVP."""
        diff = np.asarray(x) - np.asarray(p_obs)
        dist = max(float(np.linalg.norm(diff)), 1e-6)
        h = dist - r_safe
        grad_h = diff / dist                    # unit vector  ->  ||L_g h||^2 = 1
        rhs = phi * 1.0 + a - alpha * (h - c)

        self._grad_h.value = grad_h
        self._u_nom.value = np.asarray(u_nom, dtype=float)
        self._rhs.value = float(rhs)

        try:
            self._prob.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception:
            try:
                self._prob.solve(warm_start=True)
            except Exception:
                self._prob.solve()

        if self._u.value is None:               # solver failed entirely
            return np.asarray(u_nom, dtype=float), 0.0, h

        slack = float(self._delta.value) if self._delta.value is not None else 0.0
        return np.asarray(self._u.value, dtype=float), slack, h
