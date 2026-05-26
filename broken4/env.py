"""Gymnasium environment for learning CBF parameters.

The RL policy does NOT drive the robot.  Each step it emits parameters
(phi, alpha); the QP turns those into a safe command u_safe; the
single-integrator dynamics (with disturbance) advance the state.  The QP
and the dynamics together ARE the environment -- the policy acts through
the optimizer, so no gradients ever flow through the QP.

  action  : Box(-1, 1, shape=(2,))   -> mapped to (phi, alpha)
  obs     : [h, dist_to_obstacle, ||u_nom||, disturbance_estimate]
            (all zeros when cfg.state_conditioned is False -> Phase 1)
  reward  : w_progress * progress
            - w_intervention * ||u_safe - u_nom||
            - w_collision  if h_true < 0   (terminates)
            + w_goal       if goal reached (terminates)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cbf_qp import CBFQPSolver


class CBFParamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.p_obs = np.asarray(cfg.obstacle, dtype=float)
        self.goal = np.asarray(cfg.goal, dtype=float)
        self.r_safe = cfg.r_safe
        self.qp = CBFQPSolver(cfg.u_max, cfg.slack_penalty)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(4,), dtype=np.float32
        )
        self._rng = np.random.default_rng()

    # ---- helpers -------------------------------------------------------
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

    def _obs(self, x, u_nom):
        if not self.cfg.state_conditioned:
            return np.zeros(4, dtype=np.float32)
        diff = x - self.p_obs
        dist = float(np.linalg.norm(diff))
        h = dist - self.r_safe
        return np.array(
            [h, dist, float(np.linalg.norm(u_nom)), self._dist_level],
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
        # explicit override (evaluation) wins; else sample from the training
        # range if one is configured; else use the fixed level
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

        u_nom = self._nominal(self.x)
        u_safe, slack, _ = self.qp.solve(
            self.x, self.p_obs, self.r_safe, u_nom, phi, alpha
        )

        # advance TRUE dynamics with disturbance (the QP never sees d)
        if self.t % cfg.disturbance_resample == 0:
            self._resample_disturbance()
        self.x = self.x + cfg.dt * (u_safe + self._d)
        self.t += 1

        h_true = float(np.linalg.norm(self.x - self.p_obs)) - self.r_safe
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
