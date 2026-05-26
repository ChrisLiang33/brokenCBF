"""Gymnasium environment for learning CBF parameters -- scatter layout
with one moving obstacle.

The RL policy does NOT drive the robot. Each step it emits parameters
(phi, alpha); the QP turns those into a safe command u_safe; the
single-integrator dynamics (with disturbance) advance the state. The QP
and the dynamics together ARE the environment.

  action  : Box(-1, 1, shape=(2,))   -> mapped to (phi, alpha)
  obs (7) : [h_min, h_second, dist_nearest, ||u_nom||, disturbance_est,
             last_slack, closing_rate]
            - h_min, h_second: smallest / second-smallest barrier value;
              both small => the robot is in the pinch.
            - last_slack: previous step's QP slack -> the filter is
              straining (a late correction is failing).
            - closing_rate: d h/d t of the nearest obstacle
              (= -grad_h . v_obs); negative => obstacle approaching.
            last_slack and closing_rate are the signals that give the
            CBF gain alpha a reason to adapt.
            (all zeros when cfg.state_conditioned is False -> Phase 1)
  reward  : w_progress * progress
            - w_intervention * ||u_safe - u_nom||
            - w_collision  if any h_true < 0   (terminates)
            + w_goal       if goal reached     (terminates)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cbf_qp import CBFQPSolver

_NO_SECOND = 10.0   # h_second placeholder when there is no 2nd obstacle
_OBS_DIM = 7


class CBFParamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.safe_radii = cfg.safe_radii()
        self.goal = np.asarray(cfg.goal, dtype=float)
        self.qp = CBFQPSolver(cfg.u_max, cfg.slack_penalty,
                              len(cfg.obstacles))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(_OBS_DIM,), dtype=np.float32
        )
        self._rng = np.random.default_rng()

    # ---- helpers -------------------------------------------------------
    def _sync_obstacles(self):
        """Refresh obstacle centers/velocities for the current step."""
        self._centers, self._vels = self.cfg.obstacle_state(self.t)

    def _map_action(self, a):
        """Normalized [-1, 1]^2  ->  (phi, alpha) in their config bounds."""
        a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
        lo_p, hi_p = self.cfg.phi_bounds
        lo_a, hi_a = self.cfg.alpha_bounds
        phi = lo_p + (a[0] + 1.0) / 2.0 * (hi_p - lo_p)
        alpha = lo_a + (a[1] + 1.0) / 2.0 * (hi_a - lo_a)
        return float(phi), float(alpha)

    def _nominal(self, x):
        """Go-to-goal P controller, clipped to the velocity limit."""
        u = self.cfg.kp * (self.goal - x)
        n = np.linalg.norm(u)
        if n > self.cfg.u_max:
            u = u / n * self.cfg.u_max
        return u

    def _barriers(self, x):
        """All barrier values (current obstacle positions), ascending."""
        return sorted(float(np.linalg.norm(x - c)) - r
                      for c, r in zip(self._centers, self.safe_radii))

    def _obstacle_list(self):
        """(cx, cy, r_safe, vx, vy) tuples for the QP."""
        return [(c[0], c[1], r, v[0], v[1])
                for c, r, v in zip(self._centers, self.safe_radii,
                                   self._vels)]

    def _obs(self, x, u_nom):
        if not self.cfg.state_conditioned:
            return np.zeros(_OBS_DIM, dtype=np.float32)
        dists = [float(np.linalg.norm(x - c)) for c in self._centers]
        hs = sorted((d - r, i)
                    for i, (d, r) in enumerate(zip(dists, self.safe_radii)))
        h_min, i_near = hs[0]
        h_second = hs[1][0] if len(hs) > 1 else _NO_SECOND
        # closing rate of the nearest obstacle: d h/d t = -grad_h . v_obs
        diff = x - self._centers[i_near]
        dist = max(float(np.linalg.norm(diff)), 1e-6)
        closing_rate = float(-(diff / dist) @ self._vels[i_near])
        return np.array(
            [h_min, h_second, min(dists), float(np.linalg.norm(u_nom)),
             self._dist_level, self._last_slack, closing_rate],
            dtype=np.float32,
        )

    def _resample_disturbance(self):
        theta = self._rng.uniform(0.0, 2.0 * np.pi)
        self._d = self._dist_level * np.array([np.cos(theta), np.sin(theta)])

    # ---- gym API -------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        self.x = np.asarray(self.cfg.start, dtype=float)
        self.t = 0
        self._last_slack = 0.0
        self._sync_obstacles()
        # explicit override (evaluation) wins; else sample from the
        # training range if configured; else use the fixed level
        if "disturbance" in options:
            self._dist_level = float(options["disturbance"])
        elif self.cfg.disturbance_range is not None:
            lo, hi = self.cfg.disturbance_range
            self._dist_level = float(self._rng.uniform(lo, hi))
        else:
            self._dist_level = float(self.cfg.disturbance)
        self._resample_disturbance()
        self.prev_dist_goal = float(np.linalg.norm(self.goal - self.x))
        return self._obs(self.x, self._nominal(self.x)), {}

    def step(self, action):
        cfg = self.cfg
        phi, alpha = self._map_action(action)

        # QP uses obstacle positions/velocities at the current step
        u_nom = self._nominal(self.x)
        u_safe, slack, _ = self.qp.solve(
            self.x, self._obstacle_list(), u_nom, phi, alpha
        )
        self._last_slack = slack

        # advance TRUE dynamics with disturbance (the QP never sees d)
        if self.t % cfg.disturbance_resample == 0:
            self._resample_disturbance()
        self.x = self.x + cfg.dt * (u_safe + self._d)
        self.t += 1
        self._sync_obstacles()                  # obstacle moves too

        h_true = self._barriers(self.x)[0]      # nearest, current positions
        dist_goal = float(np.linalg.norm(self.goal - self.x))
        intervention = float(np.linalg.norm(u_safe - u_nom))
        progress = self.prev_dist_goal - dist_goal
        self.prev_dist_goal = dist_goal

        reward = cfg.w_progress * progress - cfg.w_intervention * intervention
        terminated = False
        collided = h_true < 0.0
        reached = dist_goal < cfg.goal_tol
        if collided:
            reward -= cfg.w_collision
            terminated = True
        elif reached:
            reward += cfg.w_goal
            terminated = True
        truncated = self.t >= cfg.max_steps

        info = {
            "h": h_true,
            "intervention": intervention,
            "slack": slack,
            "collided": collided,
            "reached": reached,
            "phi": phi,
            "alpha": alpha,
            "x": self.x.copy(),
        }
        next_obs = self._obs(self.x, self._nominal(self.x))
        return next_obs, float(reward), terminated, truncated, info
