"""Procedural terrain generators for the Phase 6 terrain pivot.

The α channel in our CBF parameterization theoretically responds to
*tracking error* (how poorly the locomotion executes the commanded
velocity). Flat ground gives ~zero tracking residual, so α has no
state-conditional optimum to learn -- exactly the problem we hit with
isotropic disturbance. Rough terrain forces the locomotion to track
poorly in proportion to roughness; that's the signal α can learn from.

Seven preset levels of escalating roughness:
    0 -- flat (control / baseline)
    1 -- light bumps        (~2.5 cm)
    2 -- medium bumps       (~5 cm)
    3 -- heavy bumps        (~7.5 cm)
    4 -- mild slopes + heavy bumps  (slopes up to 14 deg)
    5 -- steep slopes + heavy bumps (slopes up to 22 deg)
    6 -- pyramid stairs             (10 cm steps, 50 cm wide)

Note: the Go2 stock locomotion is robust up to ~7.5cm bumps -- alpha
gate runs 0..3 showed track_err essentially flat. To get α-signal we
NEED to push into the regime the loco wasn't trained on: stairs and
steeper slopes are the ones most likely to degrade tracking.

Used by:
  - `CBFAdaptiveGo2RoughEnvCfg(terrain_level=L)` in cbf_adaptive_env_cfg
  - `phase6_alpha_gate.py` which sweeps L and reports best-fixed α per level.
"""
from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg


def rough_terrain_generator(level: int) -> TerrainGeneratorCfg:
    """Build a homogeneous terrain generator for the given roughness level.

    Each tile is `size=(8, 8) m`. We use a moderate grid (8x8 = 64 tiles)
    so 256 parallel envs can be distributed across them (~4 envs/tile).
    All tiles within a generator share the SAME noise range -- this is
    the "homogeneous" version used for the α gate (one terrain level at
    a time). A heterogeneous version for training would mix sub_terrains.
    """
    if level < 0 or level > 6:
        raise ValueError(f"terrain level must be 0..6, got {level}")

    sub_terrains: dict[str, terrain_gen.SubTerrainBaseCfg] = {}
    if level <= 3:
        # levels 0..3: pure random bumps, height scales linearly
        noise_max = 0.025 * level         # 0, 2.5, 5, 7.5 cm
        sub_terrains["rough"] = terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.0, noise_max),
            noise_step=0.01,
            border_width=0.0,
        )
    elif level == 4:
        # mild slopes + heavy bumps
        sub_terrains = {
            "rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.5,
                noise_range=(0.0, 0.10),
                noise_step=0.01,
                border_width=0.0,
            ),
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.5,
                slope_range=(0.0, 0.25),    # up to ~14 deg
                platform_width=2.0,
                border_width=0.0,
            ),
        }
    elif level == 5:
        # steep slopes + heavy bumps
        sub_terrains = {
            "rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.4,
                noise_range=(0.0, 0.10),
                noise_step=0.01,
                border_width=0.0,
            ),
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.6,
                slope_range=(0.20, 0.40),   # 11 to 22 deg
                platform_width=2.0,
                border_width=0.0,
            ),
        }
    else:  # level 6: pyramid stairs
        sub_terrains = {
            "stairs": terrain_gen.HfPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.05, 0.10),
                step_width=0.30,
                platform_width=2.0,
                border_width=0.0,
            ),
        }

    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,             # flat border so robots can't fall off
        num_rows=8,
        num_cols=8,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains=sub_terrains,
    )
