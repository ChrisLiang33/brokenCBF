"""Scene layouts ported from the 2D MVP.

In Isaac Lab these become RigidObject configurations — cylinders standing
upright (z-axis = world up). Each scene gives initial obstacle (x, y, radius)
in the ground plane and a goal (x, y).

The training script picks scenes per env (randomized or fixed depending on
the experiment).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scene:
    name: str
    obstacles: list[tuple[tuple[float, float], float]]   # ((cx, cy), r)
    goal: tuple[float, float]
    start_y_range: tuple[float, float] = (-0.4, 0.4)


SCENES: dict[str, Scene] = {
    "open":     Scene("open",
                      [((2.5, 0.0), 0.6)],
                      (5.0, 0.0)),

    "spath":    Scene("spath",
                      [((1.8,  0.5), 0.6), ((3.5, -0.5), 0.6)],
                      (5.0, 0.0)),

    "corridor": Scene("corridor",
                      [((2.5,  0.9), 0.775), ((2.5, -0.9), 0.775)],  # gap ~0.25 m
                      (5.0, 0.0)),

    "slalom":   Scene("slalom",
                      [((1.5,  0.45), 0.5),
                       ((3.0, -0.45), 0.5),
                       ((4.0,  0.45), 0.5)],
                      (5.5, 0.0)),

    # Held-out hard scenes (test only):
    "narrow":   Scene("narrow",
                      [((2.5,  0.9), 0.825), ((2.5, -0.9), 0.825)],  # gap ~0.15 m
                      (5.0, 0.0)),

    "gauntlet": Scene("gauntlet",
                      [((1.5,  0.45), 0.4),
                       ((2.5, -0.45), 0.4),
                       ((3.5,  0.45), 0.4),
                       ((4.5, -0.45), 0.4)],
                      (6.0, 0.0)),
}

TRAIN_SCENES = ["open", "spath"]   # corridor (0.25 m gap) is narrower than Go2 (~0.30 m); unsolvable
TEST_SCENES = list(SCENES.keys())
HELD_OUT = ["slalom", "narrow", "gauntlet"]
