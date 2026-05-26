"""Isaac Lab scene config: Go2 + ground + 8 cylinder obstacles + lidar.

Targets Isaac Lab 0.3.x API (mid/late 2024). If your version is older, the
imports may live under `omni.isaac.lab.*` instead of `isaaclab.*`.
"""
from __future__ import annotations

import math

# isaaclab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg, AssetBaseCfg, RigidObjectCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Go2 articulation config — Isaac Lab ships this pre-defined.
# If your install puts it elsewhere, replace this import.
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # noqa: F401


# How many cylinder slots to pre-spawn. Per-scene we'll move unused ones far
# below ground (hidden) — this keeps the scene tensor shape fixed across
# scenes so it batches cleanly across envs.
MAX_OBSTACLES = 8
GROUND_HIDE_Z = -10.0       # park unused cylinders here


def _cylinder_cfg(prim_path: str, init_xy=(50.0, 0.0), radius: float = 0.5,
                  height: float = 1.5) -> RigidObjectCfg:
    """One cylinder rigid object. We default to 'parked' far away; positions
    get overwritten on reset depending on the chosen scene."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,    # we'll set velocities directly for drift
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(init_xy[0], init_xy[1], GROUND_HIDE_Z + height / 2),
        ),
    )


@configclass
class Go2CbfSceneCfg(InteractiveSceneCfg):
    """Scene: ground + Go2 + N cylinders + lidar."""

    # Ground plane with explicit friction (Go2 locomotion needs grip)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # Lighting (Isaac Lab usually requires at least one light)
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.95)),
    )

    # Robot — use Go2's standard standing pose (calves bent within limits)
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*L_hip_joint":  0.1,
                ".*R_hip_joint": -0.1,
                "F[LR]_thigh_joint":  0.8,
                "R[LR]_thigh_joint":  1.0,
                ".*_calf_joint":     -1.5,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # NOTE: We DON'T use Isaac Lab's RayCaster because it only supports one
    # mesh prim. Instead the env computes lidar analytically (ray-circle
    # intersection) against the ground-truth obstacle positions, with optional
    # Gaussian range noise — same approach as the 2D MVP.

    # ---- Obstacles --------------------------------------------------------
    # Spawn MAX_OBSTACLES placeholder cylinders. Per-env, we'll teleport them
    # to scene-specific positions on reset (and park unused ones below ground).
    obs_0 = _cylinder_cfg("{ENV_REGEX_NS}/Obs_0")
    obs_1 = _cylinder_cfg("{ENV_REGEX_NS}/Obs_1")
    obs_2 = _cylinder_cfg("{ENV_REGEX_NS}/Obs_2")
    obs_3 = _cylinder_cfg("{ENV_REGEX_NS}/Obs_3")
    obs_4 = _cylinder_cfg("{ENV_REGEX_NS}/Obs_4")
    obs_5 = _cylinder_cfg("{ENV_REGEX_NS}/Obs_5")
    obs_6 = _cylinder_cfg("{ENV_REGEX_NS}/Obs_6")
    obs_7 = _cylinder_cfg("{ENV_REGEX_NS}/Obs_7")


def get_obs_attrs(scene_cfg: Go2CbfSceneCfg):
    """Convenience: return the obstacle config attribute names in order."""
    return [f"obs_{i}" for i in range(MAX_OBSTACLES)]
