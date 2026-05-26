"""Naive goal-reaching planner.

The planner only knows about the goal — it ignores obstacles entirely and
returns a straight-line path from start to goal. The agent will collide with
any cylinder that happens to be in the way; that's intentional. Adding a
safety layer (CBF, shield, etc.) on top is the next step.

The `OBSTACLES` list still mirrors what's in `scene_mvp.xml` so a future
safety filter has access to the obstacle geometry — the planner itself does
not use it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cylinder:
    """Top-down circular obstacle footprint."""
    cx: float
    cy: float
    radius: float

    def contains(self, x: float, y: float) -> bool:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius ** 2


# Mirrors scene_mvp.xml. Not consumed by the planner; available for a future
# safety layer that needs to know where obstacles are.
OBSTACLES: list[Cylinder] = [
    Cylinder(cx=1.0, cy=-0.15, radius=0.3),
    Cylinder(cx=2.0, cy=+0.15, radius=0.3),
]

AGENT_RADIUS = 0.10


def plan(start_xy: tuple[float, float],
         goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
    """Return a straight-line path from start to goal. Obstacles are ignored."""
    return [tuple(start_xy), tuple(goal_xy)]


class PurePursuit:
    """Pops waypoints as the agent reaches them; produces a world-frame velocity command."""

    def __init__(self, waypoints: list[tuple[float, float]],
                 reach_tolerance: float = 0.15,
                 speed: float = 0.6) -> None:
        self.waypoints = waypoints
        self.idx = 1 if len(waypoints) > 1 else 0
        self.reach_tolerance = reach_tolerance
        self.speed = speed

    def done(self) -> bool:
        return self.idx >= len(self.waypoints)

    def current_target(self) -> tuple[float, float]:
        return self.waypoints[min(self.idx, len(self.waypoints) - 1)]

    def command(self, pos_xy: np.ndarray) -> np.ndarray:
        while not self.done():
            tgt = np.asarray(self.waypoints[self.idx])
            if np.linalg.norm(tgt - pos_xy) < self.reach_tolerance:
                self.idx += 1
            else:
                break
        if self.done():
            return np.zeros(2)
        tgt = np.asarray(self.waypoints[self.idx])
        delta = tgt - pos_xy
        n = np.linalg.norm(delta)
        if n < 1e-6:
            return np.zeros(2)
        return delta / n * self.speed
