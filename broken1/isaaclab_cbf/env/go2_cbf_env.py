"""Go2 + CBF env, inheriting from Isaac Lab's standard Go2 flat task.

The env config (Go2CbfFlatEnvCfg) inherits UnitreeGo2FlatEnvCfg so the
standard managers (observation, action, event, reward, command) all stay
correct for the frozen locomotion policy.

The env class (Go2CbfRLEnv) overrides step() so:
  - Outer action: 5-D log CBF params (alpha, phi, a, b, c) from the RL policy
  - Inside step:
      1. perceive obstacles (analytical lidar)
      2. estimate obstacle velocities (frame-to-frame matching)
      3. planner u_nom (P-controller toward goal, possibly adversarial)
      4. CBF filter -> u_safe
      5. inject u_safe into the standard env's velocity command buffer
      6. build standard loco obs via the std observation manager
      7. call frozen locomotion -> 12-D joint action
      8. process via std action manager + decimation + physics step
  - Outer obs = dict {proprio, occgrid, past_actions, priv_obs}
  - Outer reward = goal-progress + reach bonus - collision penalty
"""
from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

from core.safety_filter import safety_filter
from core.perception import (
    analytical_lidar,
    lidar_to_occgrid,
    MAX_OBSTACLES,
)
# perceive_batched and estimate_velocities are imported lazily in case we
# switch perception back on later for evaluation realism.
from core.scenes import SCENES, TRAIN_SCENES


# Constants
N_LIDAR_RAYS = 180
LIDAR_MAX_RANGE = 6.0
PAST_K = 4
GROUND_HIDE_Z = -10.0
# Robot's effective collision radius (Go2 is ~0.30 m wide, ~0.70 m long; use a
# conservative disk approximation). The CBF math assumes a POINT robot, so we
# inflate obstacle radii by this much before passing them to the safety filter.
# Without this, the CBF can declare "0.10 m clearance — safe" while the body
# is already 15 cm inside the obstacle. The 2D MVP did not have this issue
# because its simulated robot was literally a point.
ROBOT_RADIUS = 0.25

# Reward shape — rebalanced 2026-05-26 to break the rush-and-crash local min.
# Old: progress (+100 max) ≈ collision (-100) → indifference. New: reach pays
# ~150 net, crash pays ~10 net → strong gradient toward actually solving.
PROGRESS_K   = 20.0
TIME_PEN     = 0.05
COLLISION_P  = 50.0     # was 100 — still painful, not dominant
GOAL_BONUS   = 50.0     # was 20 — reaching the goal is now the dominant signal
TIMEOUT_P    = 30.0
FALL_PENALTY = 50.0     # std_term (base_contact) terminations — loco fell
REACH_RADIUS = 0.4
# Action-rate penalty on the 5 CBF params at *refresh ticks*. Decimation
# stops change within a hold block; this stops big jumps between blocks.
ACTION_RATE_K = 0.05
# Outer-loop decimation: CBF policy updates its params every CBF_DECIM physics
# steps instead of every step. At 50 Hz inner, decim=4 → 12.5 Hz outer control.
# Loco still runs at 50 Hz. The CBF math still runs at 50 Hz (using held params),
# so the controller responds to changing state — just the policy decision is slow.
CBF_DECIM    = 4


# ---------------------------------------------------------------------------
# Env config: standard Go2 flat env + 8 obstacle slots
# ---------------------------------------------------------------------------
def _cylinder_cfg(prim_path: str, init_xy=(50.0, 0.0)) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CylinderCfg(
            radius=0.5, height=1.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(init_xy[0], init_xy[1], GROUND_HIDE_Z + 0.75),
        ),
    )


def _goal_marker_cfg(prim_path: str) -> RigidObjectCfg:
    """A bright-green sphere we teleport to goal_xy per env each reset.
    Kinematic and no collision — pure visualization."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.2)),
            # no collision_props → robot walks through the marker
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 0.5)),
    )


@configclass
class Go2CbfFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """Standard Go2 flat env + 8 cylinder obstacle slots (parked at init)."""

    # Domain randomization ranges — every env samples uniformly in range
    # on every reset (no gating; no "off" envs).
    # DR ranges — narrowed so the loco policy stays in its training distribution.
    # Wider ranges caused 70% fall rate at steady state; these keep robots walkable.
    # ---- Perception / state-estimation
    rand_lidar_noise:  tuple[float, float] = (0.0, 0.05)
    rand_sigma_pose:   tuple[float, float] = (0.0, 0.03)
    rand_drift_std:    tuple[float, float] = (0.0, 0.15)
    rand_adv_prob:     tuple[float, float] = (0.0, 0.20)
    # ---- Actuation / tracking
    rand_tracking_err: tuple[float, float] = (0.0, 0.05)
    # ---- Physical / dynamics (applied via EventCfg in __post_init__)
    rand_mass_scale:     tuple[float, float] = (0.85, 1.15)
    rand_motor_strength: tuple[float, float] = (0.9, 1.1)
    rand_friction:       tuple[float, float] = (0.6, 1.2)
    # ---- Action constraint
    rand_v_max:          tuple[float, float] = (0.5, 1.0)

    def __post_init__(self):
        super().__post_init__()
        for j in range(MAX_OBSTACLES):
            setattr(self.scene, f"obs_{j}",
                    _cylinder_cfg(f"{{ENV_REGEX_NS}}/Obs_{j}",
                                  init_xy=(50.0 + 2.0 * j, 0.0)))
        # Visible goal marker (bright-green sphere) — teleported per env on reset.
        self.scene.goal_marker = _goal_marker_cfg("{ENV_REGEX_NS}/Goal")
        # Physics DR via Isaac Lab's stock event terms (correct indexing, no
        # manual PhysX writes). All fire on reset, so per-episode variation.
        self.events.dr_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": self.rand_mass_scale,
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.dr_friction = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "static_friction_range":  self.rand_friction,
                "dynamic_friction_range": self.rand_friction,
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
        self.events.dr_motor = EventTermCfg(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": self.rand_motor_strength,
                "damping_distribution_params":   self.rand_motor_strength,
                "operation": "scale",
                "distribution": "uniform",
            },
        )


# ---------------------------------------------------------------------------
# Env class: outer action is 5-D CBF params, step() runs the full pipeline
# ---------------------------------------------------------------------------
class Go2CbfRLEnv(ManagerBasedRLEnv):
    cfg: Go2CbfFlatEnvCfg

    def __init__(self, cfg: Go2CbfFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        device = self.sim.device
        B = self.num_envs

        # Frozen locomotion policy — set externally before reset()
        self.locomotion = None
        # If True, zero the priv slice of the outer obs (used for nopriv ablation)
        self.priv_masked = False

        # Body-frame ray directions
        angles = torch.linspace(0, 2 * math.pi, N_LIDAR_RAYS + 1, device=device)[:-1]
        self.ray_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        # Per-env state
        self.goal_xy = torch.zeros(B, 2, device=device)
        self.true_obs_centers = torch.zeros(B, MAX_OBSTACLES, 2, device=device)
        self.true_obs_radii   = torch.zeros(B, MAX_OBSTACLES,    device=device)
        self.true_obs_mask    = torch.zeros(B, MAX_OBSTACLES,    device=device)
        self.drift_vel        = torch.zeros(B, MAX_OBSTACLES, 2, device=device)
        self.past_actions     = torch.zeros(B, PAST_K, 5, device=device)
        self.prev_obs_centers = torch.zeros(B, MAX_OBSTACLES, 2, device=device)
        self.prev_obs_mask    = torch.zeros(B, MAX_OBSTACLES,    device=device)
        self.prev_dist        = torch.zeros(B, device=device)

        # Per-rollout privileged scalars (every env always sampled, no gating)
        # ---- Perception / state-estimation
        self.sigma_e      = torch.zeros(B, device=device)   # lidar range noise
        self.sigma_pose   = torch.zeros(B, device=device)   # robot pose estimation noise
        self.drift_e      = torch.zeros(B, device=device)   # obstacle drift speed
        self.adv_e        = torch.zeros(B, device=device)   # adversarial planner prob
        # ---- Actuation / tracking
        self.tracking_err = torch.zeros(B, device=device)   # loco cmd noise std
        # Outer-loop decimation: per-env tick counter (0..CBF_DECIM-1) plus
        # the held log-params that the CBF math actually uses.
        self._cbf_tick         = torch.zeros(B, dtype=torch.long, device=device)
        self._held_log_params  = torch.zeros(B, 5, device=device)
        # EMA on the velocity command sent to loco. Separate from policy
        # smoothing — handles discontinuities that the CBF math itself produces
        # when an active constraint switches on (e.g., entering an obstacle's
        # safe-set boundary). The loco was trained on slow commands and falls
        # when it gets a step jump in vel_command. Real robots always have this.
        self._prev_cmd_xy      = torch.zeros(B, 2, device=device)
        # ---- Action-time DR (applied each step in step()).
        # Physical-dynamics DR (mass / friction / motor / com) now happens via
        # stock EventCfg terms in Go2CbfFlatEnvCfg.__post_init__ — no manual
        # PhysX writes. We don't expose those realized values in priv for now;
        # the encoder gets a smaller priv vector (perception + actuation only).
        self.dr_v_max = torch.ones(B, device=device)

    # =====================================================================
    # Reset
    # =====================================================================
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        n = env_ids.numel()
        device = self.device

        # Sample all DR uniformly across the cfg ranges — no gating, every env
        # gets a randomized world.
        def _uni(lo_hi, shape):
            lo, hi = lo_hi
            return lo + (hi - lo) * torch.rand(shape, device=device)
        self.sigma_e[env_ids]      = _uni(self.cfg.rand_lidar_noise,  (n,))
        self.sigma_pose[env_ids]   = _uni(self.cfg.rand_sigma_pose,   (n,))
        self.drift_e[env_ids]      = _uni(self.cfg.rand_drift_std,    (n,))
        self.adv_e[env_ids]        = _uni(self.cfg.rand_adv_prob,     (n,))
        self.tracking_err[env_ids] = _uni(self.cfg.rand_tracking_err, (n,))
        self.dr_v_max[env_ids]     = _uni(self.cfg.rand_v_max,        (n,))
        self.drift_vel[env_ids] = self.drift_e[env_ids][:, None, None] \
                                  * torch.randn(n, MAX_OBSTACLES, 2, device=device)
        # Physics DR (mass / friction / motor) is now applied automatically by
        # the EventCfg event terms registered in Go2CbfFlatEnvCfg.__post_init__.
        # No manual PhysX writes here.

        # Random scene choice
        scene_idx = torch.randint(0, len(TRAIN_SCENES), (n,), device=device)
        self._set_scene_obstacles(env_ids, scene_idx)

        # Reset short-term buffers
        self.past_actions[env_ids] = 0.0
        self.prev_obs_centers[env_ids] = 0.0
        self.prev_obs_mask[env_ids] = 0.0
        self._cbf_tick[env_ids] = 0
        self._held_log_params[env_ids] = 0.0
        self._prev_cmd_xy[env_ids] = 0.0

        robot_xy = (self.scene["robot"].data.root_pos_w[:, :2]
                    - self.scene.env_origins[:, :2])
        self.prev_dist[env_ids] = (robot_xy[env_ids] - self.goal_xy[env_ids]).norm(dim=-1)

    def _set_scene_obstacles(self, env_ids, scene_idx):
        """Teleport cylinders to scene-specific positions; park unused below ground."""
        device = self.device
        n = env_ids.numel()
        new_pos   = torch.zeros(n, MAX_OBSTACLES, 3, device=device)
        new_radii = torch.zeros(n, MAX_OBSTACLES,    device=device)
        new_mask  = torch.zeros(n, MAX_OBSTACLES,    device=device)
        goals     = torch.zeros(n, 2, device=device)
        new_pos[..., 2] = GROUND_HIDE_Z

        for si, sname in enumerate(TRAIN_SCENES):
            m = scene_idx == si
            if not m.any():
                continue
            sc = SCENES[sname]
            for j, ((cx, cy), r) in enumerate(sc.obstacles):
                new_pos[m, j, 0] = cx
                new_pos[m, j, 1] = cy
                new_pos[m, j, 2] = 0.75
                new_radii[m, j] = r
                new_mask[m, j]  = 1.0
            goals[m] = torch.tensor(sc.goal, device=device)

        self.goal_xy[env_ids] = goals
        self.true_obs_centers[env_ids] = new_pos[..., :2]
        self.true_obs_radii[env_ids]   = new_radii
        self.true_obs_mask[env_ids]    = new_mask

        # Teleport the visible goal marker to per-env goal position.
        if "goal_marker" in self.scene.keys():
            goal_world = torch.cat([
                goals + self.scene.env_origins[env_ids, :2],
                torch.full((n, 1), 0.5, device=device),                   # z=0.5 m
            ], dim=-1)
            gquat = torch.zeros(n, 4, device=device); gquat[:, 0] = 1.0
            self.scene["goal_marker"].write_root_pose_to_sim(
                torch.cat([goal_world, gquat], dim=-1), env_ids=env_ids,
            )

        # Push to sim
        for j in range(MAX_OBSTACLES):
            obj = self.scene[f"obs_{j}"]
            world_pos = new_pos[:, j, :] + torch.cat([
                self.scene.env_origins[env_ids, :2],
                torch.zeros(n, 1, device=device),
            ], dim=-1)
            quat = torch.zeros(n, 4, device=device); quat[:, 0] = 1.0
            obj.write_root_pose_to_sim(torch.cat([world_pos, quat], dim=-1), env_ids=env_ids)
            obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

    # =====================================================================
    # Reset returns our outer obs (not the standard locomotion obs)
    # =====================================================================
    def reset(self, seed=None, options=None):
        # Standard reset → physics reset + observation manager compute
        _ = super().reset(seed=seed, options=options)
        device = self.device
        B = self.num_envs
        robot_xy_true = (self.scene["robot"].data.root_pos_w[:, :2]
                         - self.scene.env_origins[:, :2])
        quat = self.scene["robot"].data.root_quat_w
        siny_cosp = 2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
        cosy_cosp = 1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
        yaw_true = torch.atan2(siny_cosp, cosy_cosp)
        ranges = analytical_lidar(
            robot_xy_true, yaw_true,
            self.true_obs_centers, self.true_obs_radii, self.true_obs_mask,
            self.ray_dirs, LIDAR_MAX_RANGE, noise_std=self.sigma_e.unsqueeze(-1),
        )
        # Pose seen by the policy is the noisy estimate
        s = self.sigma_pose.unsqueeze(-1)
        robot_xy_est = robot_xy_true + s * torch.randn(B, 2, device=device)
        yaw_est      = yaw_true      + self.sigma_pose * torch.randn(B, device=device)
        dummy_params = torch.zeros(B, 5, device=device)
        outer_obs = self._build_outer_obs(robot_xy_est, yaw_est, ranges, dummy_params)
        return outer_obs, {}

    # =====================================================================
    # Step: outer action -> CBF -> loco -> physics
    # =====================================================================
    def step(self, raw_cbf_log_params: torch.Tensor):
        """raw_cbf_log_params: (B, 5) — the RL policy's raw output.

        Outer-loop decimation: the CBF policy's params are HELD for CBF_DECIM
        physics steps before being refreshed. PPO still gets a transition per
        physics step (rsl_rl expects that), but the effective control rate of
        the params is 1/CBF_DECIM. CBF math + loco still run at full rate.
        """
        assert self.locomotion is not None, "env.locomotion must be set before step()"
        device = self.device
        B = self.num_envs
        # Clamp log params to a sane range so exp() can't blow up.
        # exp(-3, 3) ≈ (0.05, 20) — covers any reasonable α/φ/a/b/c.
        raw_cbf_log_params = raw_cbf_log_params.clamp(-3.0, 3.0)

        # Decimation: only refresh held params for envs whose counter hits 0.
        self._cbf_tick = (self._cbf_tick + 1) % CBF_DECIM
        update_mask_1d = (self._cbf_tick == 0)                          # (B,)
        update_mask = update_mask_1d.unsqueeze(-1)                      # (B, 1)
        # Action-rate penalty (only meaningful at refresh ticks; held steps
        # have zero rate by construction).
        delta_sq = ((raw_cbf_log_params - self._held_log_params) ** 2).sum(dim=-1)
        action_rate_sq = torch.where(update_mask_1d, delta_sq, torch.zeros_like(delta_sq))
        self._held_log_params = torch.where(
            update_mask, raw_cbf_log_params, self._held_log_params,
        ).detach()
        cbf_log_params = self._held_log_params

        # 1. Robot state — TRUE pose from sim
        robot_xy_true = (self.scene["robot"].data.root_pos_w[:, :2]
                         - self.scene.env_origins[:, :2])
        quat = self.scene["robot"].data.root_quat_w
        siny_cosp = 2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
        cosy_cosp = 1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
        yaw_true = torch.atan2(siny_cosp, cosy_cosp)

        # 1b. Estimated pose — TRUE + per-env state-estimation noise. Used by
        # everything the robot has to *reason* with (planner, CBF, goal_body).
        # Lidar still ray-casts from TRUE pose (sensor-side noise is sigma_e).
        s = self.sigma_pose.unsqueeze(-1)                       # (B, 1)
        robot_xy = robot_xy_true + s * torch.randn(B, 2, device=device)
        yaw      = yaw_true      + self.sigma_pose * torch.randn(B, device=device)

        # 2. Perception — fast path: skip clustering/circle-fit (Python loop is
        #    too slow at scale), feed CBF the ground-truth obstacles directly.
        #    The lidar OCCUPANCY GRID is still computed (fast GPU op) and
        #    flows through the obs → later a CNN can consume it.
        ranges = analytical_lidar(
            robot_xy_true, yaw_true,
            self.true_obs_centers, self.true_obs_radii, self.true_obs_mask,
            self.ray_dirs, LIDAR_MAX_RANGE, noise_std=self.sigma_e.unsqueeze(-1),
        )
        cur_centers = self.true_obs_centers
        # Inflate obstacle radii so the CBF (which treats the robot as a point)
        # actually keeps the BODY clear of obstacles. See ROBOT_RADIUS docstring.
        cur_radii   = self.true_obs_radii + ROBOT_RADIUS
        cur_mask    = self.true_obs_mask
        # Ground-truth drift velocity for the L_f h feedforward
        v_obs = self.drift_vel

        # 3. Planner u_nom in body frame, possibly adversarial.
        #    Includes a tangent-bias: when an obstacle is between us and the
        #    goal, u_nom is rotated to point AROUND the obstacle (toward
        #    whichever side is closer to the goal direction). This prevents
        #    the CBF from being asked to "go through an obstacle" — it can
        #    just follow u_nom and stay safe.
        to_goal = self.goal_xy - robot_xy
        u_world = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp_min(1e-3)

        diffs = self.true_obs_centers - robot_xy.unsqueeze(1)        # (B, N, 2)
        ob_dists = diffs.norm(dim=-1)                                # (B, N)
        ob_dists = ob_dists.masked_fill(self.true_obs_mask == 0, float("inf"))
        min_d, min_idx = ob_dists.min(dim=-1)                        # (B,)
        batch_idx = torch.arange(B, device=device)
        near_obs  = self.true_obs_centers[batch_idx, min_idx]        # (B, 2)
        near_r    = self.true_obs_radii[batch_idx, min_idx]          # (B,)

        to_obs = near_obs - robot_xy
        to_obs_d = to_obs.norm(dim=-1).clamp_min(1e-3)
        to_obs_dir = to_obs / to_obs_d.unsqueeze(-1)
        surface_d = to_obs_d - near_r - ROBOT_RADIUS                 # > 0 outside
        align_g = (to_obs_dir * u_world).sum(dim=-1)                 # is obs in front?

        # Two tangents (CW and CCW perpendicular to to_obs_dir)
        tan_cw  = torch.stack([ to_obs_dir[:, 1], -to_obs_dir[:, 0]], dim=-1)
        tan_ccw = torch.stack([-to_obs_dir[:, 1],  to_obs_dir[:, 0]], dim=-1)
        a_cw  = (tan_cw  * u_world).sum(dim=-1)
        a_ccw = (tan_ccw * u_world).sum(dim=-1)
        tan_best = torch.where((a_cw > a_ccw).unsqueeze(-1), tan_cw, tan_ccw)

        # Blend: route only when obstacle is in front AND we're close to it.
        ENGAGE = 1.5                                                  # meters of surface_d
        weight_route = ((align_g > 0.3) & (surface_d < ENGAGE)).float()
        blend = weight_route * (1.0 - (surface_d / ENGAGE).clamp(0.0, 1.0))   # (B,)
        u_world = (1.0 - blend.unsqueeze(-1)) * u_world \
                  + blend.unsqueeze(-1) * tan_best
        u_world = u_world / u_world.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        # Adversarial: aim at perceived obstacle with prob adv_e
        adv_dice = torch.rand(B, device=device)
        adv_mask = adv_dice < self.adv_e
        if adv_mask.any():
            diff = self.prev_obs_centers - robot_xy.unsqueeze(1)
            dist_to = diff.norm(dim=-1).masked_fill(self.prev_obs_mask == 0, float("inf"))
            idx = dist_to.argmin(dim=1)
            batch = torch.arange(B, device=device)
            adv_target = self.prev_obs_centers[batch, idx]
            adv_u = (adv_target - robot_xy)
            adv_u = adv_u / adv_u.norm(dim=-1, keepdim=True).clamp_min(1e-3)
            u_world = torch.where(adv_mask.unsqueeze(-1), adv_u, u_world)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        u_nom = torch.stack([
             cy * u_world[:, 0] + sy * u_world[:, 1],
            -sy * u_world[:, 0] + cy * u_world[:, 1],
        ], dim=-1)

        # 4. CBF filter
        params = cbf_log_params.exp()
        u_safe = safety_filter(
            robot_xy, u_nom,
            cur_centers, cur_radii, cur_mask,
            alpha=params[:, 0], phi=params[:, 1],
            a=params[:, 2], b=params[:, 3], c=params[:, 4],
            obs_velocities=v_obs,
        )

        # 5. Inject velocity command — turn AND go simultaneously.
        #
        # The CBF outputs body-frame (vx, vy). Naive (vx, vy, 0) crab-walks
        # at full speed and the loco falls. Pure turn-then-go stalls the
        # robot. The fix: forward speed scales with alignment via cos(heading)
        # so the robot keeps moving while turning, and lateral velocity is
        # capped tight so it never high-speed strafes.
        #
        #   speed   = ||u_safe||
        #   heading = atan2(u_safe.y, u_safe.x)
        #   omega_z = K_yaw · heading            (clipped)
        #   v_fwd   = speed · max(cos(h), 0)     (clipped to v_fwd_max)
        #   v_lat   = u_safe.y                   (clipped tight, e.g. ±0.25)
        #
        # Loco-cmd clip limits match the standard Go2 training distribution
        # (lin_vel_x ±1, lin_vel_y ±1, ang_vel_z ±1). Lateral cap is 0.25 m/s
        # because Go2 strafes much worse than it walks.
        track_noise = self.tracking_err.unsqueeze(-1) \
                      * torch.randn(B, 2, device=device)
        v_max = self.dr_v_max.unsqueeze(-1)                     # (B, 1)
        u_clamped = torch.clamp(u_safe + track_noise, -v_max, v_max)

        speed   = u_clamped.norm(dim=-1)                        # (B,)
        heading = torch.atan2(u_clamped[:, 1], u_clamped[:, 0]) # (B,) ∈ [-π, π]

        # Knobs we sweep via env attributes (set from test script / train)
        K_YAW     = getattr(self, "K_YAW",     2.0)
        OMEGA_MAX = getattr(self, "OMEGA_MAX", 1.0)
        V_FWD_MAX = getattr(self, "V_FWD_MAX", 1.0)
        V_LAT_MAX = getattr(self, "V_LAT_MAX", 0.25)

        # Critical: scale ω_z by speed. When CBF holds back (||u_safe|| ≈ 0
        # near an obstacle boundary), we MUST NOT spin in place — that's the
        # bug where robot pins at the boundary rotating forever. With this,
        # ω_z → 0 as the CBF clamps the motion.
        omega_z = (K_YAW * heading * speed.clamp_max(1.0)) \
                       .clamp(-OMEGA_MAX, OMEGA_MAX)
        # Forward velocity: take the body-frame x-component directly (this is
        # exactly speed·cos(heading)), clipped to loco's training range.
        v_fwd   = u_clamped[:, 0].clamp(-V_FWD_MAX, V_FWD_MAX)
        # Lateral: pass through tight-clipped y component so the robot creeps
        # tangentially around obstacles instead of stopping at the boundary.
        v_lat   = u_clamped[:, 1].clamp(-V_LAT_MAX, V_LAT_MAX)

        cmd_3 = torch.zeros(B, 3, device=device)
        cmd_3[:, 0] = v_fwd
        cmd_3[:, 1] = v_lat
        cmd_3[:, 2] = omega_z
        self.command_manager._terms["base_velocity"].vel_command_b[:] = cmd_3

        # 6. Build standard locomotion obs via the standard manager
        loco_obs = self.observation_manager.compute()["policy"]

        # 7. Run frozen locomotion
        with torch.no_grad():
            joint_action = self.locomotion(loco_obs)

        # 8. Standard step — handles decimation, episode_length_buf++,
        #    auto-reset of done envs, etc. We let it compute (and discard) the
        #    standard locomotion reward / termination; we override below.
        _, _, std_term, std_trunc, info = super().step(joint_action)

        # 9. Outer obs (post-step state) — TRUE pose for lidar/reward/collision,
        #    NOISY pose for what the policy sees (goal_body).
        new_xy_true = (self.scene["robot"].data.root_pos_w[:, :2]
                       - self.scene.env_origins[:, :2])
        new_quat = self.scene["robot"].data.root_quat_w
        siny_cosp = 2.0 * (new_quat[:, 0] * new_quat[:, 3] + new_quat[:, 1] * new_quat[:, 2])
        cosy_cosp = 1.0 - 2.0 * (new_quat[:, 2] ** 2 + new_quat[:, 3] ** 2)
        new_yaw_true = torch.atan2(siny_cosp, cosy_cosp)
        # Fresh estimation noise for the post-step "reading"
        s = self.sigma_pose.unsqueeze(-1)
        new_xy_est  = new_xy_true  + s * torch.randn(B, 2, device=device)
        new_yaw_est = new_yaw_true + self.sigma_pose * torch.randn(B, device=device)
        new_ranges = analytical_lidar(
            new_xy_true, new_yaw_true,
            self.true_obs_centers, self.true_obs_radii, self.true_obs_mask,
            self.ray_dirs, LIDAR_MAX_RANGE, noise_std=self.sigma_e.unsqueeze(-1),
        )
        outer_obs = self._build_outer_obs(new_xy_est, new_yaw_est, new_ranges, params)

        # 10. Custom reward (goal progress + collision + reach) — uses TRUE pose
        outer_reward = self._compute_outer_reward(new_xy_true)
        new_xy = new_xy_true                                    # alias for the block below

        # Push action to history
        self.past_actions = torch.cat([
            self.past_actions[:, 1:], cbf_log_params.detach().unsqueeze(1),
        ], dim=1)

        # Use std terminations (base_contact = robot fell, time_out = max steps)
        # plus our own goal-reached/collision-with-obstacle
        cur_dist = (new_xy - self.goal_xy).norm(dim=-1)
        reached = cur_dist < REACH_RADIUS
        # Distance to true obstacles for our collision check
        diff_obs = self.true_obs_centers - new_xy.unsqueeze(1)
        d_obs = diff_obs.norm(dim=-1)
        # Body-aware collision: fires when robot's effective disk touches obstacle.
        # Slightly less margin than the CBF inflation so CBF activates before the
        # collision termination — gives the safety filter room to actually save us.
        collided_obs = ((d_obs < self.true_obs_radii + ROBOT_RADIUS - 0.05)
                        & (self.true_obs_mask > 0)).any(dim=-1)
        # Bonus/penalty wraps
        outer_reward = torch.where(reached,      outer_reward + GOAL_BONUS,  outer_reward)
        outer_reward = torch.where(collided_obs, outer_reward - COLLISION_P, outer_reward)
        outer_reward = torch.where(std_trunc & ~reached & ~collided_obs,
                                   outer_reward - TIMEOUT_P, outer_reward)
        # Fall penalty: std_term here is the loco's base_contact termination.
        outer_reward = torch.where(std_term & ~reached & ~collided_obs,
                                   outer_reward - FALL_PENALTY, outer_reward)
        # Action-rate penalty — Lipschitz pressure on the refresh ticks.
        outer_reward = outer_reward - ACTION_RATE_K * action_rate_sq

        terminated = std_term | reached | collided_obs
        truncated  = std_trunc
        # Manually reset envs that hit OUR custom terminations (reached /
        # collided_obs) — super().step() only auto-resets on std_term/std_trunc,
        # so without this our envs stay at the goal forever.
        custom_done = (reached | collided_obs) & ~std_term & ~std_trunc
        if custom_done.any():
            self._reset_idx(custom_done.nonzero(as_tuple=False).squeeze(-1))
        # Termination breakdown for diagnostics — rsl_rl logs anything under
        # info["log"][...] or info["episode"][...] to TensorBoard / printout.
        # Use info["log"] which the rsl_rl wrapper picks up as scalar episode
        # metrics. Fractions over the batch of envs this step.
        info.setdefault("log", {})
        info["log"]["term/reached"]      = reached.float().mean()
        info["log"]["term/collided_obs"] = collided_obs.float().mean()
        info["log"]["term/fell"]         = std_term.float().mean()
        info["log"]["term/timeout"]      = std_trunc.float().mean()
        return outer_obs, outer_reward, terminated, truncated, info

    # =====================================================================
    # Outer obs and reward
    # =====================================================================
    # Fixed obs layout for rsl_rl flat-tensor compatibility.
    PROPRIO_DIM = 35       # 3+3+3+2+12+12
    OCC_DIM     = 32 * 32  # 1024
    PAST_DIM    = PAST_K * 5
    # Privileged: env/dynamics ONLY. Every dim here is a LIVE perturbation —
    # no dead inputs. Obstacle info is perceivable from lidar so it never lives
    # here (would defeat sim-to-real — see cbf-architecture memory).
    # Physics DR (mass / friction / motor / com) now goes through EventCfg event
    # terms, randomizing per reset but NOT exposed in priv (we'd need to read
    # back realized PhysX values — TODO). Encoder sees perception + actuation DR.
    # Layout: [v_max(1) | sigma(1) | sigma_pose(1) | drift(1) | adv(1) | track_err(1)] = 6
    PRIV_DIM    = 6
    FLAT_OBS_DIM = PROPRIO_DIM + OCC_DIM + PAST_DIM + PRIV_DIM    # 1085

    def _build_outer_obs(self, xy, yaw, ranges, params):
        del params
        data = self.scene["robot"].data

        # xy / yaw passed in are the *estimated* pose (true + sigma_pose noise),
        # so goal_body inherits the pose-estimation error automatically.
        goal_local = self.goal_xy - xy
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        goal_body = torch.stack([
             cy * goal_local[:, 0] + sy * goal_local[:, 1],
            -sy * goal_local[:, 0] + cy * goal_local[:, 1],
        ], dim=-1)
        proprio = torch.cat([
            data.root_lin_vel_b,
            data.root_ang_vel_b,
            data.projected_gravity_b,
            goal_body,
            data.joint_pos - data.default_joint_pos,
            data.joint_vel,
        ], dim=-1)                                              # (B, 35)

        occ = lidar_to_occgrid(ranges, self.ray_dirs).flatten(start_dim=1)   # (B, 1024)
        past = self.past_actions.flatten(start_dim=1)                        # (B, 20)
        priv = torch.cat([
            self.dr_v_max.unsqueeze(-1),              # 1  (per-env speed cap, m/s)
            self.sigma_e.unsqueeze(-1),               # 1  (lidar range noise)
            self.sigma_pose.unsqueeze(-1),            # 1  (robot pose estimation noise)
            self.drift_e.unsqueeze(-1),               # 1  (obstacle drift speed)
            self.adv_e.unsqueeze(-1),                 # 1  (adversarial planner prob)
            self.tracking_err.unsqueeze(-1),          # 1  (loco cmd noise std)
        ], dim=-1)                                              # (B, 6)
        if self.priv_masked:                                    # nopriv ablation
            priv = torch.zeros_like(priv)

        # Flat layout:  [proprio(35) | occgrid_flat(1024) | past(20) | priv(6)]
        #                offsets:     0           35              1059       1079    1085
        # When we add the CNN actor, the policy network will slice out the
        # occgrid chunk (35:1059) and reshape to (B, 1, 32, 32) for conv.
        flat = torch.cat([proprio, occ, past, priv], dim=-1)    # (B, 1085)
        return {"policy": flat}

    def _compute_outer_reward(self, xy):
        """Progress + time penalty. Goal/collision/timeout wrappers are
        applied in step() after we know the std terminations."""
        cur_dist = (xy - self.goal_xy).norm(dim=-1)
        progress = (self.prev_dist - cur_dist) * PROGRESS_K
        self.prev_dist = cur_dist
        return progress - TIME_PEN
