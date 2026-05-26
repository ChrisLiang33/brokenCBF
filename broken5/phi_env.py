"""Gymnasium env for adaptive-φ training.

Each episode samples a `kick_magnitude ∈ [0, KICK_MAX]`. When the agent
crosses the trigger line, a single position kick of that magnitude is
applied toward the nearest obstacle — unobservable to the filter. Low φ
→ crashes on big kicks. High φ → safe but slow / unable to reach.
Policy must read `kick_magnitude` and pick φ accordingly.

Observation (12-d):
    agent_xy          (2)
    agent_vel_xy      (2)
    rel_to_obs1       (2)
    rel_to_obs2       (2)
    rel_to_goal       (2)
    kick_magnitude    (1)
    has_kicked_flag   (1)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mvp import Sim
from planner import OBSTACLES

PHI_MIN = 0.0
PHI_MAX = 2.0

KICK_MAX = 0.20
GOAL = (1.4, 0.30)
TRIGGER_X = 0.6

CONTROL_DECIMATION = 10
MAX_CONTROL_STEPS = 150

TIME_PENALTY = 0.05
CRASH_PENALTY = 15.0
GOAL_BONUS = 12.0
PROGRESS_BONUS = 0.02


class AdaptivePhiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.sim = Sim(verbose=False)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._obs1 = np.array([OBSTACLES[0].cx, OBSTACLES[0].cy])
        self._obs2 = np.array([OBSTACLES[1].cx, OBSTACLES[1].cy])
        self._goal = np.array(GOAL)

        self._kick_mag = 0.0
        self._kicked = False
        self._step_count = 0

    @staticmethod
    def _action_to_phi(action: np.ndarray) -> float:
        a = float(np.clip(action[0], -1.0, 1.0))
        return PHI_MIN + (a + 1.0) * 0.5 * (PHI_MAX - PHI_MIN)

    def _obs(self) -> np.ndarray:
        pos = self.sim.data.qpos[:2].copy()
        vel = self.sim.data.qvel[:2].copy()
        rel_obs1 = pos - self._obs1
        rel_obs2 = pos - self._obs2
        rel_goal = self._goal - pos
        return np.concatenate([
            pos, vel, rel_obs1, rel_obs2, rel_goal,
            [self._kick_mag, float(self._kicked)],
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self._kick_mag = float(rng.uniform(0.0, KICK_MAX))
        self._kicked = False
        self._step_count = 0
        self.sim.reset(goal_xy=GOAL)
        return self._obs(), {"kick_mag": self._kick_mag}

    def set_scenario(self, kick_mag: float) -> None:
        self._kick_mag = float(kick_mag)
        self._kicked = False
        self._step_count = 0
        self.sim.reset(goal_xy=GOAL)

    def step(self, action):
        phi = self._action_to_phi(action)

        crashed = reached = False
        for _ in range(CONTROL_DECIMATION):
            if not self._kicked and self.sim.data.qpos[0] > TRIGGER_X:
                direction = self._obs1 - self.sim.data.qpos[:2]
                direction = direction / max(np.linalg.norm(direction), 1e-6)
                self.sim.pending_kick = direction * self._kick_mag
                self._kicked = True
            self.sim.step(phi_override=phi)
            if self.sim.terminated:
                if self.sim.termination_reason == "collision":
                    crashed = True
                elif self.sim.termination_reason == "goal_reached":
                    reached = True
                break

        self._step_count += 1
        terminated = self.sim.terminated
        truncated = (self._step_count >= MAX_CONTROL_STEPS) and not terminated

        pos = self.sim.data.qpos[:2]
        vel = self.sim.data.qvel[:2]
        to_goal = self._goal - pos
        dist = float(np.linalg.norm(to_goal))
        progress = float(vel @ to_goal) / max(dist, 1e-6)

        reward = -TIME_PENALTY + PROGRESS_BONUS * progress
        if crashed:
            reward -= CRASH_PENALTY
        if reached:
            reward += GOAL_BONUS

        info = {
            "phi": phi,
            "kick_mag": self._kick_mag,
            "kicked": self._kicked,
            "crashed": crashed,
            "reached": reached,
            "pos": pos.copy(),
        }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
