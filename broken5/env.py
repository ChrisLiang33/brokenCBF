"""Gymnasium env: randomize BOTH sensor noise σ AND goal position per episode.

Combines stage 1 (noise → drop α) and stage 2 (tight goal → raise α). Tests
whether a single policy learns a 2-axis decision: pick α from both how
trustworthy the sensors are AND how close to a wall the task forces you.

Observation (12-d):
    agent_xy                                (2)
    agent_vel_xy                            (2)
    rel_to_perceived_obs1                   (2)
    rel_to_perceived_obs2                   (2)
    rel_to_goal                             (2)
    sigma                                   (1)
    goal_to_nearest_perceived_obs           (1)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import mujoco

from mvp import Sim
from planner import OBSTACLES

ALPHA_MIN = 0.5
ALPHA_MAX = 4.0

SIGMA_MIN = 0.0
SIGMA_MAX = 0.12

CONTROL_DECIMATION = 10
MAX_CONTROL_STEPS = 130   # tight enough that α=α_min crawls fail to reach tight goals

GOAL_X_RANGE = (1.5, 3.5)
GOAL_Y_RANGE = (-0.6, 0.6)
GOAL_MIN_CLEAR = 0.08
GOAL_MAX_CLEAR = 0.80

TIME_PENALTY = 0.05
CRASH_PENALTY = 20.0
GOAL_BONUS = 15.0           # bigger payoff for reaching tight goals
PROGRESS_BONUS = 0.04       # stronger gradient toward "be fast when clear"


class AdaptiveCBFEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.sim = Sim(verbose=False)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._true_centers = np.array(
            [[OBSTACLES[0].cx, OBSTACLES[0].cy],
             [OBSTACLES[1].cx, OBSTACLES[1].cy]], dtype=float)
        self._obs_radii = np.array([OBSTACLES[0].radius, OBSTACLES[1].radius])

        self._perceived_mocap = [
            int(self.sim.model.body_mocapid[
                mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, name)])
            for name in ("perceived_obs1", "perceived_obs2")
        ]

        self._sigma = 0.0
        self._perceived = self._true_centers.copy()
        self._goal = np.array([3.0, 0.0])
        self._step_count = 0

    @staticmethod
    def _action_to_alpha(action: np.ndarray) -> float:
        a = float(np.clip(action[0], -1.0, 1.0))
        return ALPHA_MIN + (a + 1.0) * 0.5 * (ALPHA_MAX - ALPHA_MIN)

    def _goal_to_nearest_perceived(self, goal_xy: np.ndarray) -> float:
        d = np.linalg.norm(self._perceived - goal_xy, axis=1)
        return float(np.min(d - self._obs_radii))

    def _goal_to_nearest_true(self, goal_xy: np.ndarray) -> float:
        d = np.linalg.norm(self._true_centers - goal_xy, axis=1)
        return float(np.min(d - self._obs_radii))

    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        for _ in range(200):
            g = np.array([rng.uniform(*GOAL_X_RANGE), rng.uniform(*GOAL_Y_RANGE)])
            clear = self._goal_to_nearest_true(g)
            if GOAL_MIN_CLEAR <= clear <= GOAL_MAX_CLEAR:
                return g
        return np.array([3.0, 0.0])

    def _push_perceived_to_sim(self) -> None:
        self.sim.filter.set_perceived_obstacles(self._perceived)
        for i, mid in enumerate(self._perceived_mocap):
            self.sim.data.mocap_pos[mid] = [self._perceived[i, 0],
                                            self._perceived[i, 1], 0.01]

    def _obs(self) -> np.ndarray:
        pos = self.sim.data.qpos[:2].copy()
        vel = self.sim.data.qvel[:2].copy()
        rel_p1 = pos - self._perceived[0]
        rel_p2 = pos - self._perceived[1]
        rel_goal = self._goal - pos
        return np.concatenate([
            pos, vel, rel_p1, rel_p2, rel_goal,
            [self._sigma, self._goal_to_nearest_perceived(self._goal)],
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self._sigma = float(rng.uniform(SIGMA_MIN, SIGMA_MAX))
        noise = rng.normal(0.0, max(self._sigma, 1e-12), size=self._true_centers.shape)
        self._perceived = self._true_centers + noise
        self._goal = self._sample_goal(rng)
        self.sim.reset(goal_xy=tuple(self._goal))
        self._push_perceived_to_sim()
        self._step_count = 0
        return self._obs(), {"sigma": self._sigma, "goal": self._goal.copy()}

    def set_scenario(self, sigma: float | None = None,
                     goal: tuple[float, float] | None = None,
                     noise: np.ndarray | None = None) -> None:
        """Script σ and/or goal (used by eval)."""
        if sigma is not None:
            self._sigma = float(sigma)
            if noise is None:
                noise = self.np_random.normal(
                    0.0, max(self._sigma, 1e-12), size=self._true_centers.shape)
            self._perceived = self._true_centers + noise
        if goal is not None:
            self._goal = np.asarray(goal, dtype=float)
        self.sim.reset(goal_xy=tuple(self._goal))
        self._push_perceived_to_sim()
        self._step_count = 0

    def step(self, action):
        alpha = self._action_to_alpha(action)

        crashed = reached = False
        for _ in range(CONTROL_DECIMATION):
            self.sim.step(alpha_override=alpha)
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
            "alpha": alpha,
            "sigma": self._sigma,
            "goal": self._goal.copy(),
            "goal_clear_true": self._goal_to_nearest_true(self._goal),
            "crashed": crashed,
            "reached": reached,
            "pos": pos.copy(),
        }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
