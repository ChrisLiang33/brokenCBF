"""CBF parameter action term -- batched over Isaac Lab envs.

The outer policy emits a per-env 2-dim action mapped to (φ, α). This
term then, FOR EACH ENV IN PARALLEL:

1. Reads base state (position, yaw, lin_vel_b, ang_vel_b, gravity_b,
   joint_pos, joint_vel) directly from the scene's robot articulation.
2. Computes nominal P-controller velocity in world frame toward a fixed
   goal, rotates to base frame.
3. Computes barrier `h = ||p_base - p_obs|| - r_safe` and the base-frame
   gradient `R(yaw)ᵀ · grad_h_world`.
4. Closed-form CBF projection -> u_safe in base frame.
5. Builds the locomotion's expected 48-dim observation per env (with
   u_safe in the velocity_commands slot, previous joint action in the
   actions slot).
6. Forwards the frozen locomotion actor -> raw 12-dim joint action.
7. Scales to actual joint targets: target = default_joint_pos + scale *
   raw_action  (matching the stock Go2 JointPositionActionCfg scale).
8. Samples a per-env disturbance direction (resampled every
   `disturbance_resample` steps) and writes external base force.

The term caches per-step quantities (u_nom, u_safe, h_realized,
dist_to_goal, intervention, prev_dist_to_goal) so reward terms can read
them via `env.action_manager._terms["cbf_param"]`.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Forward-declaration token. The class_type default on the cfg is set
# below, AFTER the action term class itself is defined.
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _closed_form_cbf_batched(
    grad_h_base: torch.Tensor,    # (N, 2) unit
    rhs: torch.Tensor,            # (N,)
    u_nom_b: torch.Tensor,        # (N, 2)
    v_max,                         # float OR (N,) tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (v_safe (N, 2), slack (N,)).  v_max can be a scalar OR a
    per-env tensor (N,) -- the per-env form is used by `v_max_range` DR.
    """
    deficit = rhs - (grad_h_base * u_nom_b).sum(dim=-1)         # (N,)
    safe = deficit <= 0.0
    v = torch.where(
        safe.unsqueeze(-1),
        u_nom_b,
        u_nom_b + deficit.unsqueeze(-1) * grad_h_base,
    )
    n = torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-9)
    if torch.is_tensor(v_max) and v_max.dim() > 0:
        v_max_b = v_max.unsqueeze(-1)                            # (N, 1)
        over = n.squeeze(-1) > v_max
        v = torch.where(over.unsqueeze(-1), v * (v_max_b / n), v)
    else:
        over = (n.squeeze(-1) > v_max)
        v = torch.where(over.unsqueeze(-1), v * (v_max / n), v)
    slack = (rhs - (grad_h_base * v).sum(dim=-1)).clamp(min=0.0)
    return v, slack


# ---------------------------------------------------------------------------
# Action term
# ---------------------------------------------------------------------------
class CBFParamActionTerm(ActionTerm):
    cfg: "CBFParamActionTermCfg"

    def __init__(self, cfg: "CBFParamActionTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._env = env
        device = env.device
        N = env.num_envs

        # robot articulation
        self._robot = env.scene[cfg.asset_name]
        self._n_joints = self._robot.data.joint_pos.shape[1]

        if cfg.locomotion_policy_obj is None:
            raise ValueError(
                "CBFParamActionTermCfg.locomotion_policy_obj is None. "
                "Set it in your training script to the loaded locomotion nn.Module."
            )
        self._loco = cfg.locomotion_policy_obj

        # scenario per env: goal and obstacles are specified in env-LOCAL
        # coordinates; add per-env world origins so each env's scenario is
        # translated to its own spawn location.
        env_origins_xy = env.scene.env_origins[:, :2].to(device)             # (N, 2)
        goal_local = torch.tensor(cfg.goal_xy, device=device, dtype=torch.float32)
        self._goal_xy = env_origins_xy + goal_local.unsqueeze(0)             # (N, 2)

        # Resolve obstacles list. If the user didn't set cfg.obstacles, fall
        # back to the single-obstacle (obstacle_xy, obstacle_radius) tuple
        # for backward compatibility with Phase 0/0.5/0.6/1/2 cfgs.
        obstacles_local = cfg.obstacles if cfg.obstacles is not None else [
            (float(cfg.obstacle_xy[0]), float(cfg.obstacle_xy[1]),
             float(cfg.obstacle_radius)),
        ]
        K = len(obstacles_local)
        centers_local = torch.tensor(
            [(o[0], o[1]) for o in obstacles_local], device=device, dtype=torch.float32
        )                                                                    # (K, 2)
        radii_local = torch.tensor(
            [float(o[2]) for o in obstacles_local], device=device, dtype=torch.float32
        )                                                                    # (K,)
        # per-env world centers: (N, K, 2)
        self._obs_centers_w = (env_origins_xy.unsqueeze(1) + centers_local.unsqueeze(0))
        # nominal (no-jitter) positions, kept for re-jittering at reset
        self._obs_centers_nominal_w = self._obs_centers_w.clone()
        if cfg.obstacle_pos_jitter_range is not None:
            self._obs_jitter_lo = float(cfg.obstacle_pos_jitter_range[0])
            self._obs_jitter_hi = float(cfg.obstacle_pos_jitter_range[1])
        else:
            self._obs_jitter_lo = 0.0
            self._obs_jitter_hi = 0.0

        # Random topology config (v8-light). When True, per-reset we
        # re-sample obstacle xy positions via rejection sampling in the
        # configured corridor instead of jittering nominal positions.
        # `obstacles` cfg field is still used for INITIAL placement +
        # radii (K obstacles, each with its r_safe); spawn corridor and
        # exclusion zones come from random_topology_* fields.
        self._random_topology = bool(getattr(cfg, "random_topology", False))
        if self._random_topology:
            self._rt_x_lo = float(cfg.random_topology_x_range[0])
            self._rt_x_hi = float(cfg.random_topology_x_range[1])
            self._rt_y_lo = float(cfg.random_topology_y_range[0])
            self._rt_y_hi = float(cfg.random_topology_y_range[1])
            self._rt_start_excl = float(cfg.random_topology_start_exclusion_r)
            self._rt_goal_excl  = float(cfg.random_topology_goal_exclusion_r)
            self._rt_min_sep    = float(cfg.random_topology_min_separation)
            self._rt_max_attempts = int(cfg.random_topology_max_attempts)
            # cache env-local goal for exclusion check (env_origins are
            # added back when sampling positions per env)
            self._env_origins_xy = env_origins_xy.clone()
            self._goal_local = goal_local.clone()
        # per-obstacle "safe radius" = obstacle physical radius + robot radius
        self._r_safe = (radii_local + float(cfg.robot_radius)).unsqueeze(0)  # (1, K)
        self._n_obstacles = K
        self._h_lambda = float(cfg.h_smooth_lambda)
        self._h_gamma = float(cfg.h_smooth_gamma)
        # Observation masking flags (ablation). Cached for the mdp obs
        # functions to read via _cbf_term(env).
        self._obs_mask_priv = bool(getattr(cfg, "obs_mask_priv", False))
        self._obs_mask_proprio = bool(getattr(cfg, "obs_mask_proprio", False))

        # Proprio history buffer (rolled per step). Allocated only if
        # `proprio_history_length` > 0; otherwise stays None (the
        # default RMA teachers don't use it).
        self._proprio_history_length = int(getattr(cfg, "proprio_history_length", 0))
        # PROPRIO_DIM is fixed at 45 (mdp.deployable_obs layout).
        self._proprio_history = (
            torch.zeros((N, self._proprio_history_length, 45), device=device)
            if self._proprio_history_length > 0 else None
        )
        # B.4 SIM2REAL (SHIELD-aligned, Yang et al. 2025): if True, the
        # CBF uses obstacle positions perturbed by per-step Gaussian
        # noise to simulate Livox Mid-360 + clustering + cylinder-fit
        # detection accuracy. Default False keeps existing privileged
        # behavior. `perception_noise_std` controls the noise std (m)
        # added to each obstacle's xy each step.
        self._use_lidar_sdf = bool(getattr(cfg, "use_lidar_sdf", False))
        self._perception_noise_std = float(
            getattr(cfg, "perception_noise_std", 0.05)
        )
        self._perception_dropout_prob = float(
            getattr(cfg, "perception_dropout_prob", 0.02)
        )
        self._lidar_max_range = float(
            getattr(cfg, "lidar_max_range", 20.0)
        )
        self._kp = float(cfg.kp)
        # v_max: scalar default, becomes per-env tensor when v_max_range is set
        # (the α-channel signal validated in phase6_vmax_gate -- the policy
        # learns to lower α when v_max is high, because tracking degrades).
        if cfg.v_max_range is not None:
            self._v_max_lo = float(cfg.v_max_range[0])
            self._v_max_hi = float(cfg.v_max_range[1])
        else:
            self._v_max_lo = self._v_max_hi = float(cfg.v_max)
        self._v_max = torch.full((N,), self._v_max_lo, device=device)
        self._phi_lo, self._phi_hi = cfg.phi_bounds
        self._alpha_lo, self._alpha_hi = cfg.alpha_bounds
        self._loco_action_scale = float(cfg.locomotion_action_scale)

        # disturbance magnitude: scalar -> broadcast to per-env tensor;
        # range -> sampled fresh per env on every reset (Phase 2+ DR).
        if cfg.disturbance_force_range is not None:
            lo, hi = float(cfg.disturbance_force_range[0]), float(cfg.disturbance_force_range[1])
        else:
            lo = hi = float(cfg.disturbance_force)
        self._disturbance_force_lo = lo
        self._disturbance_force_hi = hi
        self._disturbance_force = torch.full((N,), lo, device=device)
        self._disturbance_resample = int(cfg.disturbance_resample)

        # RMA additional privileged factors. Each is a per-env scalar
        # sampled on reset; the actual physics-level effect is applied
        # below (motor_strength in process_actions; friction + mass via
        # PhysX view writes here). For phases that don't DR these, set
        # the range tuple to a singleton so all envs share one value.
        self._friction_lo, self._friction_hi = (
            cfg.friction_range if cfg.friction_range is not None else (0.6, 0.6)
        )
        self._base_mass_lo, self._base_mass_hi = (
            cfg.base_mass_range if cfg.base_mass_range is not None else (0.0, 0.0)
        )
        self._motor_strength_lo, self._motor_strength_hi = (
            cfg.motor_strength_range if cfg.motor_strength_range is not None else (1.0, 1.0)
        )
        self._friction_coef = torch.full((N,), self._friction_lo, device=device)
        # base_mass_delta is an ADDITIVE delta to the default mass (matches
        # Isaac Lab's `add_base_mass` convention so values stay
        # interpretable in priv_obs).
        self._base_mass_delta = torch.full((N,), 0.0, device=device)
        self._motor_strength = torch.full((N,), 1.0, device=device)
        # NEW (B.1) extra priv channels for the unified teacher:
        # - actuation_noise_std: per-env Gaussian std added to processed
        #   joint targets each step. Theoretically maps to phi (input
        #   uncertainty hedge).
        # - com_offset: per-env 1D forward/back shift of the base rigid-
        #   body's center of mass (meters). Theoretically maps to alpha
        #   (tracking residual; off-CoM body has worse tracking).
        self._actuation_noise_std_lo, self._actuation_noise_std_hi = (
            cfg.actuation_noise_std_range
            if getattr(cfg, "actuation_noise_std_range", None) is not None
            else (0.0, 0.0)
        )
        self._actuation_noise_std = torch.full((N,), self._actuation_noise_std_lo,
                                                device=device)
        self._com_offset_lo, self._com_offset_hi = (
            cfg.com_offset_range
            if getattr(cfg, "com_offset_range", None) is not None
            else (0.0, 0.0)
        )
        self._com_offset = torch.full((N,), 0.0, device=device)
        # one-shot warning flags for set_material_properties / set_masses
        # / set_coms in case the PhysX view APIs aren't available -- print
        # only once rather than per-reset.
        self._phys_apply_warned_friction = False
        self._phys_apply_warned_mass = False
        self._phys_apply_warned_com = False

        # per-env mutable state
        self._raw_actions = torch.zeros((N, 2), device=device)
        # Lipschitz rate-limit state for hardware-safety smoothing
        # (B.5). Bounds the per-step change in the normalized [-1, 1]
        # action so the decoded (phi, alpha) are time-Lipschitz with
        # constant L = action_max_step / dt.
        self._action_max_step = float(getattr(cfg, "action_max_step", 0.0))
        self._action_prev = torch.zeros((N, 2), device=device)
        self._processed_actions = torch.zeros((N, self._n_joints), device=device)
        self._prev_loco_action = torch.zeros((N, self._n_joints), device=device)
        self._dist_theta = torch.zeros((N,), device=device)
        self._dist_step_count = torch.zeros((N,), dtype=torch.long, device=device)

        # ---- Clumsy-human u_nom planner state (Phase 10 / V2) ----
        # Off by default (mode="straight") -> the legacy P-controller toward
        # goal. Mode="clumsy_human" adds OU lateral noise, OU speed wobble,
        # and occasional adversarial swerves toward the nearest obstacle.
        # The user rationale: a straight-line u_nom is unrealistic for
        # deployment (teleop / partial autonomy), so we feed a noisier,
        # sometimes adversarial nominal at BOTH train and eval.
        #
        # CLUMSINESS PRESETS (user spec: training is harder than eval):
        #   "child" (training, 5-year-old): more wobble, slower, more
        #     mistakes -> harder distribution
        #   "teen"  (eval, 10-year-old):    less wobble, faster, fewer
        #     mistakes -> closer to deployment, still imperfect
        #   "default": back-compat values (no preset).
        # The preset overrides the OU sigma / swerve_prob / speed_mean
        # knobs below. Individual knobs on cfg still win if explicitly set
        # to a non-default value, but in practice you should pick a preset.
        self._unom_mode = str(getattr(cfg, "unom_mode", "straight"))
        # Preset is source of truth. Explicit knobs (unom_lat_sigma etc.)
        # ONLY apply when preset="default" -- otherwise cfg's class-level
        # defaults (0.30 / 0.18 / 0.75 / ...) would silently override the
        # preset because getattr can't distinguish "user-set" from
        # "class-default". The preset itself is what V2 env cfgs set
        # (child for train, teen for eval).
        _preset = str(getattr(cfg, "unom_clumsiness", "default"))
        if _preset == "child":
            self._unom_lat_sigma = 0.40
            self._unom_speed_sigma = 0.25
            self._unom_speed_mean = 0.65
            self._unom_swerve_prob = 0.008
            self._unom_swerve_steps = 30
        elif _preset == "teen":
            self._unom_lat_sigma = 0.20
            self._unom_speed_sigma = 0.12
            self._unom_speed_mean = 0.85
            self._unom_swerve_prob = 0.003
            self._unom_swerve_steps = 20
        else:
            self._unom_lat_sigma = float(getattr(cfg, "unom_lat_sigma", 0.30))
            self._unom_speed_sigma = float(getattr(cfg, "unom_speed_sigma", 0.18))
            self._unom_speed_mean = float(getattr(cfg, "unom_speed_mean", 0.75))
            self._unom_swerve_prob = float(getattr(cfg, "unom_swerve_prob", 0.005))
            self._unom_swerve_steps = int(getattr(cfg, "unom_swerve_steps", 25))
        # universal knobs (preset-independent)
        self._unom_lat_decay = float(getattr(cfg, "unom_lat_decay", 0.95))
        self._unom_speed_decay = float(getattr(cfg, "unom_speed_decay", 0.97))
        self._unom_speed_min = float(getattr(cfg, "unom_speed_min", 0.35))
        # u_nom rate-limit (Lipschitz in time on the WORLD-frame u_nom_w).
        # 0.0 -> off. > 0.0 hard-clips ||Δu_nom_w|| per step. Smooths the
        # noisy planner so the CBF sees a continuous reference (no spikes
        # from swerve onset / OU sample-to-sample jumps), independent of
        # the (phi, alpha) action_max_step which limits the policy output.
        self._unom_max_step = float(getattr(cfg, "unom_max_step", 0.0))
        self._unom_lat_noise = torch.zeros((N,), device=device)
        self._unom_speed_mult = torch.full((N,), self._unom_speed_mean, device=device)
        self._unom_swerve_counter = torch.zeros((N,), dtype=torch.long, device=device)
        self._unom_swerve_dir = torch.zeros((N, 2), device=device)
        self._unom_prev_b = torch.zeros((N, 2), device=device)
        # stuck-flag threshold (steps). The sticky `episode_stuck_any`
        # latches once an env accumulates this many "slow-and-not-at-goal"
        # steps in a single episode. V2 raises this to 250 (5 sec at 50Hz)
        # so brief slowdowns don't tag the env as stuck before it has a
        # chance to wiggle out. NOT a termination -- env still runs until
        # time_out / collision / fall / goal.
        self._stuck_threshold_steps = int(getattr(cfg, "stuck_threshold_steps", 100))

        # cached for reward + termination terms
        self.last_phi = torch.zeros((N,), device=device)
        self.last_alpha = torch.zeros((N,), device=device)
        self.last_u_nom = torch.zeros((N, 2), device=device)
        self.last_u_safe = torch.zeros((N, 2), device=device)
        self.last_intervention = torch.zeros((N,), device=device)
        self.last_h_pre = torch.zeros((N,), device=device)
        self.last_h_realized = torch.zeros((N,), device=device)
        self.last_h_smooth_realized = torch.zeros((N,), device=device)
        self.last_dist_to_goal = torch.zeros((N,), device=device)
        self.prev_dist_to_goal = torch.zeros((N,), device=device)
        self.last_slack = torch.zeros((N,), device=device)
        # action-smoothness diagnostic: |Δphi|/phi_width + |Δalpha|/alpha_width
        # per step (normalized so the units are comparable across phi and
        # alpha bound widths). Cached every process_actions; reward terms
        # and diagnostics read it.
        self.last_action_jitter = torch.zeros((N,), device=device)
        # lidar cache + previous-frame snapshot (computed in
        # update_post_physics). Lazy-initialized on first compute
        # (we don't know n_rays here).
        self.last_lidar = None                    # (N, R) or None
        self.last_lidar_prev = None               # (N, R) or None

        # sticky per-CELL (not per-episode) flags: set inside termination
        # functions, never reset by action_term.reset (which runs on env
        # auto-reset and would otherwise clobber the True we just set).
        # Cleared manually by the grid-search / eval code at cell boundaries.
        self.episode_reach_any = torch.zeros((N,), dtype=torch.bool, device=device)
        self.episode_collide_any = torch.zeros((N,), dtype=torch.bool, device=device)
        # fall = base low / robot tipped (locomotion failure, NOT a CBF
        # safety failure; tracked separately so reports don't lump it
        # into collisions).
        self.episode_fall_any = torch.zeros((N,), dtype=torch.bool, device=device)
        # stuck = walking-but-not-progressing. Counts slow-and-not-at-goal
        # steps; the sticky flag fires once an env has accumulated > 100
        # such steps (~2 sec). Helps tell "policy did nothing useful" apart
        # from "policy collided" / "policy reached".
        self.episode_stuck_steps = torch.zeros((N,), dtype=torch.long, device=device)
        self.episode_stuck_any = torch.zeros((N,), dtype=torch.bool, device=device)

        self._device = device

    # ------------------------ helpers -------------------------------
    def _compute_sdf_smooth(self, base_xy: torch.Tensor):
        """PRIVILEGED multi-obstacle SDF + smoothed barrier (paper eqs 19, 20).

        Uses the action term's cached `_obs_centers_w` (ground-truth
        obstacle positions). This is the "teacher cheats" version --
        accurate but doesn't translate to a real robot.

        sdf(x)        = min_i (||x - p_i|| - R_i)
        h_smooth(x)   = lambda * (1 - exp(-gamma * sdf))
        grad h_smooth = lambda * gamma * exp(-gamma * sdf) * grad(sdf)
                      = g_eff * grad_sdf       (grad_sdf is unit)

        Returns (sdf, h_smooth, grad_sdf_world, g_eff) -- all per env.
        """
        # diff: (N, K, 2)
        diff = base_xy.unsqueeze(1) - self._obs_centers_w
        dist = torch.linalg.norm(diff, dim=-1).clamp(min=1e-6)               # (N, K)
        h_per = dist - self._r_safe                                          # (N, K)
        sdf, idx = h_per.min(dim=-1)                                         # (N,), (N,)
        # gradient of sdf is the unit vector from active obstacle to robot
        gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)        # (N, 1, 2)
        active_diff = torch.gather(diff, 1, gather_idx).squeeze(1)           # (N, 2)
        active_dist = torch.linalg.norm(active_diff, dim=-1, keepdim=True).clamp(min=1e-6)
        grad_sdf = active_diff / active_dist                                 # (N, 2), unit
        # smoothed value + gradient magnitude
        exp_term = torch.exp(-self._h_gamma * sdf)                           # (N,)
        h_smooth = self._h_lambda * (1.0 - exp_term)
        g_eff = self._h_lambda * self._h_gamma * exp_term                    # (N,)
        return sdf, h_smooth, grad_sdf, g_eff

    def _compute_sdf_smooth_perception(self, base_xy: torch.Tensor):
        """SHIELD-aligned perception SDF (Yang et al. 2025, eqs 19-20).

        Same math as `_compute_sdf_smooth` BUT obstacle list is
        corrupted to match what a real Mid-360 + clustering pipeline
        produces:
          1. Position noise: each obstacle's xy perturbed by Gaussian
             noise (~5cm std). Simulates cluster-centroid accuracy.
          2. Range cutoff: obstacles farther than lidar_max_range are
             dropped from the SDF (perception literally doesn't see
             them).
          3. Dropout: each visible obstacle is randomly dropped with
             probability `perception_dropout_prob` per step. Simulates
             brief clustering failures / occlusion.

        When NO obstacles survive masking (all out of range AND/OR
        dropped), sdf falls back to lidar_max_range -> h ≈ λ, gradient
        magnitude g_eff ≈ 0, so the CBF naturally goes silent (no
        constraint binds). This matches the "if nothing detected,
        assume clear" behaviour of a real safety filter.

        Returns (sdf, h_smooth, grad_sdf_WORLD, g_eff).
        """
        # 1) position noise
        if self._perception_noise_std > 0.0:
            noise = torch.randn_like(self._obs_centers_w) * self._perception_noise_std
            noisy_centers = self._obs_centers_w + noise
        else:
            noisy_centers = self._obs_centers_w

        diff = base_xy.unsqueeze(1) - noisy_centers                          # (N, K, 2)
        dist = torch.linalg.norm(diff, dim=-1).clamp(min=1e-6)               # (N, K)

        # 2) range cutoff (lidar can't see past lidar_max_range)
        in_range = dist < self._lidar_max_range                              # (N, K) bool

        # 3) random per-step dropout per (env, obstacle)
        if self._perception_dropout_prob > 0.0:
            keep_random = torch.rand_like(dist) >= self._perception_dropout_prob
            visible = in_range & keep_random
        else:
            visible = in_range

        any_visible = visible.any(dim=-1)                                    # (N,)

        # set masked entries to a HUGE per-h so they never win the min.
        # Use a finite large value (not inf) for numerical stability.
        # _lidar_max_range is a scalar; _r_safe is (1, K) tensor.
        h_per = dist - self._r_safe
        h_per_masked = torch.where(
            visible, h_per,
            torch.full_like(h_per, self._lidar_max_range)
        )
        sdf, idx = h_per_masked.min(dim=-1)                                  # (N,)

        # When no obstacles are visible for an env, clamp sdf to "lidar
        # max range" (minus the tightest safety radius) so h saturates
        # near λ and the CBF naturally goes silent (g_eff ≈ 0 in
        # process_actions's rhs_unit step). Exact value doesn't matter
        # past saturation; use a scalar so dtype/shape are clean.
        fallback_sdf = self._lidar_max_range - float(self._r_safe.min().item())
        sdf = torch.where(
            any_visible, sdf,
            torch.full_like(sdf, fallback_sdf)
        )

        # gradient -- gather the active diff regardless. For no-visible
        # envs the grad is arbitrary but g_eff is ~0 so it doesn't bind.
        gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
        active_diff = torch.gather(diff, 1, gather_idx).squeeze(1)
        active_dist = torch.linalg.norm(active_diff, dim=-1, keepdim=True).clamp(min=1e-6)
        grad_sdf = active_diff / active_dist

        exp_term = torch.exp(-self._h_gamma * sdf)
        h_smooth = self._h_lambda * (1.0 - exp_term)
        g_eff = self._h_lambda * self._h_gamma * exp_term
        return sdf, h_smooth, grad_sdf, g_eff

    # ------------------------ ActionTerm API ------------------------
    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # initialize prev_dist_to_goal from the current base position post-reset
        base_xy = self._robot.data.root_pos_w[env_ids, :2]
        goal_xy = self._goal_xy[env_ids] if not isinstance(env_ids, slice) \
            else self._goal_xy
        dist = torch.linalg.norm(goal_xy - base_xy, dim=-1)
        self.prev_dist_to_goal[env_ids] = dist
        self.last_dist_to_goal[env_ids] = dist
        # zero per-env disturbance state and previous loco action
        self._dist_step_count[env_ids] = 0
        self._dist_theta[env_ids] = torch.rand(
            (self._dist_step_count[env_ids].shape[0],), device=self._device,
        ) * (2.0 * math.pi)
        if isinstance(env_ids, slice):
            self._prev_loco_action.zero_()
            self.last_phi.zero_()
            self.last_alpha.zero_()
            self.last_action_jitter.zero_()
            self._action_prev.zero_()
            self.episode_stuck_steps.zero_()
            self._unom_lat_noise.zero_()
            self._unom_speed_mult.fill_(self._unom_speed_mean)
            self._unom_swerve_counter.zero_()
            self._unom_swerve_dir.zero_()
            self._unom_prev_b.zero_()
            if self._proprio_history is not None:
                self._proprio_history.zero_()
        else:
            self._prev_loco_action[env_ids] = 0.0
            self.last_phi[env_ids] = 0.0
            self.last_alpha[env_ids] = 0.0
            self.last_action_jitter[env_ids] = 0.0
            self._action_prev[env_ids] = 0.0
            self.episode_stuck_steps[env_ids] = 0
            self._unom_lat_noise[env_ids] = 0.0
            self._unom_speed_mult[env_ids] = self._unom_speed_mean
            self._unom_swerve_counter[env_ids] = 0
            self._unom_swerve_dir[env_ids] = 0.0
            self._unom_prev_b[env_ids] = 0.0
            if self._proprio_history is not None:
                self._proprio_history[env_ids] = 0.0
        # resample per-env disturbance magnitude from the configured range
        # so each new episode has a fresh draw of the OOD signal
        n_sel = self._disturbance_force[env_ids].shape[0] \
            if not isinstance(env_ids, slice) else self._env.num_envs

        def _sample(lo, hi):
            if hi > lo:
                return torch.rand(n_sel, device=self._device) * (hi - lo) + lo
            return torch.full((n_sel,), lo, device=self._device)

        self._disturbance_force[env_ids] = _sample(
            self._disturbance_force_lo, self._disturbance_force_hi)
        self._friction_coef[env_ids] = _sample(
            self._friction_lo, self._friction_hi)
        self._base_mass_delta[env_ids] = _sample(
            self._base_mass_lo, self._base_mass_hi)
        self._motor_strength[env_ids] = _sample(
            self._motor_strength_lo, self._motor_strength_hi)
        # v_max per-episode DR -- validated α-channel (phase6_vmax_gate).
        self._v_max[env_ids] = _sample(self._v_max_lo, self._v_max_hi)
        # B.1 extra DR channels.
        self._actuation_noise_std[env_ids] = _sample(
            self._actuation_noise_std_lo, self._actuation_noise_std_hi)
        self._com_offset[env_ids] = _sample(
            self._com_offset_lo, self._com_offset_hi)

        # Push friction / mass / COM offset through to PhysX. Guard is
        # `>=` (not `>`) so eval-time pinning to a single nominal value
        # ALSO triggers the write -- otherwise stale physics from the
        # previous cell silently persists. For legacy phases without DR
        # the per-env tensor is the same constant every reset, so the
        # extra write is a no-op in effect.
        if self._friction_hi >= self._friction_lo:
            self._apply_friction(env_ids)
        if self._base_mass_hi >= self._base_mass_lo:
            self._apply_mass(env_ids)
        if self._com_offset_hi >= self._com_offset_lo:
            self._apply_com_offset(env_ids)

        # per-episode obstacle position jitter. Re-sample relative to the
        # NOMINAL (init-time) positions so jitter doesn't accumulate.
        # If random_topology is on, that path takes over instead (below).
        if self._random_topology:
            self._sample_random_topology(env_ids)
        elif self._obs_jitter_hi > self._obs_jitter_lo:
            lo, hi = self._obs_jitter_lo, self._obs_jitter_hi
            if isinstance(env_ids, slice):
                shape = self._obs_centers_w.shape   # (N, K, 2)
                jitter = torch.rand(shape, device=self._device) * (hi - lo) + lo
                self._obs_centers_w = self._obs_centers_nominal_w + jitter
            else:
                n_sel = env_ids.shape[0]
                shape = (n_sel, self._n_obstacles, 2)
                jitter = torch.rand(shape, device=self._device) * (hi - lo) + lo
                self._obs_centers_w[env_ids] = (
                    self._obs_centers_nominal_w[env_ids] + jitter
                )

    def _sample_random_topology(self, env_ids):
        """Rejection-sample K obstacle positions per env in the configured
        corridor. Reject if:
          - any obstacle is within `start_exclusion_r` of (0, 0)
          - any obstacle is within `goal_exclusion_r` of goal_xy (env-local)
          - any pair of obstacles is within `min_separation` of each other

        Vectorized across envs: each attempt samples a fresh full layout
        per env; failed envs roll again. Envs that fail after
        `max_attempts` retries fall back to their nominal slalom layout
        (rare in practice for reasonable corridor sizes).
        """
        K = self._n_obstacles
        if isinstance(env_ids, slice):
            N = self._env.num_envs
            ids_t = torch.arange(N, device=self._device)
        else:
            N = env_ids.shape[0]
            ids_t = env_ids
        dev = self._device
        x_lo, x_hi = self._rt_x_lo, self._rt_x_hi
        y_lo, y_hi = self._rt_y_lo, self._rt_y_hi
        start_sq = self._rt_start_excl ** 2
        goal_sq  = self._rt_goal_excl  ** 2
        sep_sq   = self._rt_min_sep    ** 2

        goal_local = self._goal_local                              # (2,)
        out = self._obs_centers_nominal_w[ids_t].clone()           # (N, K, 2) world
        pending = torch.ones(N, dtype=torch.bool, device=dev)

        for _ in range(self._rt_max_attempts):
            n_pending = int(pending.sum().item())
            if n_pending == 0:
                break
            cand = torch.empty(n_pending, K, 2, device=dev)
            cand[..., 0] = torch.rand(n_pending, K, device=dev) * (x_hi - x_lo) + x_lo
            cand[..., 1] = torch.rand(n_pending, K, device=dev) * (y_hi - y_lo) + y_lo

            # start exclusion (env-local origin is (0, 0))
            d2_start = (cand ** 2).sum(dim=-1)                     # (n_pending, K)
            ok_start = (d2_start > start_sq).all(dim=-1)           # (n_pending,)

            # goal exclusion
            d_goal = cand - goal_local.unsqueeze(0).unsqueeze(0)   # (n_pending, K, 2)
            d2_goal = (d_goal ** 2).sum(dim=-1)
            ok_goal = (d2_goal > goal_sq).all(dim=-1)

            # pairwise separation: for K=3, only 3 pairs; brute-force is fine
            if K > 1:
                # (n_pending, K, K) pairwise squared distance
                diff = cand.unsqueeze(2) - cand.unsqueeze(1)
                d2_pair = (diff ** 2).sum(dim=-1)
                # mask out diagonal (self vs self == 0) before min
                eye = torch.eye(K, dtype=torch.bool, device=dev)
                d2_pair_masked = d2_pair.masked_fill(eye, float("inf"))
                ok_sep = (d2_pair_masked > sep_sq).all(dim=-1).all(dim=-1)
            else:
                ok_sep = torch.ones(n_pending, dtype=torch.bool, device=dev)

            valid = ok_start & ok_goal & ok_sep                    # (n_pending,)
            # map pending->absolute env idx, write the valid envs out
            pend_idx = torch.where(pending)[0]                     # (n_pending,)
            winners = pend_idx[valid]                              # absolute idx in this batch
            if winners.numel() > 0:
                # winners is row idx into `out`; cand[valid] is the layouts
                cand_local = cand[valid]                           # (n_winners, K, 2)
                # convert env-local to world by adding env origin xy
                env_origins_win = self._env_origins_xy[ids_t[winners]]
                out[winners] = env_origins_win.unsqueeze(1) + cand_local
                pending[winners] = False

        # envs still pending after max_attempts: leave at nominal layout
        # (already initialized in `out` from _obs_centers_nominal_w)
        if isinstance(env_ids, slice):
            self._obs_centers_w[ids_t] = out
        else:
            self._obs_centers_w[env_ids] = out

    def _apply_friction(self, env_ids):
        try:
            view = self._robot.root_physx_view
            materials = view.get_material_properties().clone()    # (N, M, 3)
            md = materials.device                                  # PhysX views are CPU
            if isinstance(env_ids, slice):
                friction_md = self._friction_coef.to(md).unsqueeze(-1)
                materials[:, :, 0] = friction_md
                materials[:, :, 1] = friction_md
                idx = torch.arange(self._env.num_envs, device=md,
                                   dtype=torch.int32)
            else:
                ids_md = env_ids.to(md)
                friction_md = self._friction_coef[env_ids].to(md).unsqueeze(-1)
                materials[ids_md, :, 0] = friction_md
                materials[ids_md, :, 1] = friction_md
                idx = ids_md.to(torch.int32)
            view.set_material_properties(materials, idx)
        except Exception as e:
            if not self._phys_apply_warned_friction:
                print(f"[cbf_action_term] WARN couldn't apply friction "
                      f"to PhysX: {type(e).__name__}: {e}. Channel will "
                      f"be a 'fake' priv signal; gate should reject it.")
                self._phys_apply_warned_friction = True

    def _apply_mass(self, env_ids):
        try:
            view = self._robot.root_physx_view
            masses = view.get_masses().clone()                    # (N, B)
            md = masses.device
            default_mass = self._robot.data.default_mass.to(md)
            if isinstance(env_ids, slice):
                masses[:, 0] = (default_mass[:, 0]
                                + self._base_mass_delta.to(md))
                idx = torch.arange(self._env.num_envs, device=md,
                                   dtype=torch.int32)
            else:
                ids_md = env_ids.to(md)
                masses[ids_md, 0] = (
                    default_mass[ids_md, 0]
                    + self._base_mass_delta[env_ids].to(md)
                )
                idx = ids_md.to(torch.int32)
            view.set_masses(masses, idx)
        except Exception as e:
            if not self._phys_apply_warned_mass:
                print(f"[cbf_action_term] WARN couldn't apply mass "
                      f"to PhysX: {type(e).__name__}: {e}. Channel will "
                      f"be a 'fake' priv signal; gate should reject it.")
                self._phys_apply_warned_mass = True

    def _apply_com_offset(self, env_ids):
        """Shift the base body's center of mass along its body-frame x
        axis by per-env `_com_offset` (meters). Other 5 bodies untouched.
        Theoretically: shifting CoG forward/back makes the body lean,
        which the frozen locomotion can't perfectly compensate for, so
        velocity tracking residual grows -- the α-axis signal we want
        to gate.
        """
        try:
            view = self._robot.root_physx_view
            coms = view.get_coms().clone()                        # (N, B, 7) pose
            md = coms.device
            if isinstance(env_ids, slice):
                coms[:, 0, 0] = self._com_offset.to(md)           # body 0 = base, x
                idx = torch.arange(self._env.num_envs, device=md,
                                   dtype=torch.int32)
            else:
                ids_md = env_ids.to(md)
                coms[ids_md, 0, 0] = self._com_offset[env_ids].to(md)
                idx = ids_md.to(torch.int32)
            view.set_coms(coms, idx)
        except Exception as e:
            if not self._phys_apply_warned_com:
                print(f"[cbf_action_term] WARN couldn't apply COM offset "
                      f"to PhysX: {type(e).__name__}: {e}. Channel will "
                      f"be a 'fake' priv signal; gate should reject it.")
                self._phys_apply_warned_com = True

    def process_actions(self, actions: torch.Tensor) -> None:
        """Outer (phi, alpha) -> joint targets, stored in _processed_actions.
        Also applies the external disturbance force."""
        N = self._env.num_envs
        device = self._device

        a = torch.clamp(actions, -1.0, 1.0)
        # HARDWARE-SAFETY Lipschitz rate-limit: per-step change in the
        # normalized [-1, 1] action is hard-clipped to action_max_step.
        # This makes (phi, alpha) Lipschitz-continuous in time with
        # constant L = action_max_step / dt -- a strict bound, not a
        # soft filter. Prevents high-frequency u_safe that would
        # damage real-robot actuators. The policy trains WITH the
        # limit active so it learns to issue continuous commands.
        # action_max_step <= 0 disables (backward compat).
        if self._action_max_step > 0.0:
            delta = a - self._action_prev
            delta = torch.clamp(delta,
                                -self._action_max_step,
                                +self._action_max_step)
            a = self._action_prev + delta
            self._action_prev = a.detach().clone()
        self._raw_actions = a

        # map to (phi, alpha)
        phi = self._phi_lo + (a[:, 0] + 1.0) * 0.5 * (self._phi_hi - self._phi_lo)
        alpha = self._alpha_lo + (a[:, 1] + 1.0) * 0.5 * (self._alpha_hi - self._alpha_lo)

        # base state
        base_xy = self._robot.data.root_pos_w[:, :2]
        yaw = _yaw_from_quat_wxyz(self._robot.data.root_quat_w)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        # multi-obstacle SDF + smoothed barrier. Two paths share the
        # same closed-form CBF projection downstream:
        # - privileged: SDF from exact obstacle positions.
        # - perception (SHIELD-aligned): SDF from noisy positions
        #   (~5cm std per axis), simulating lidar + clustering accuracy.
        # Both return grad in WORLD frame; we rotate to body here.
        if self._use_lidar_sdf:
            sdf, h_smooth, grad_sdf_w, g_eff = self._compute_sdf_smooth_perception(base_xy)
        else:
            sdf, h_smooth, grad_sdf_w, g_eff = self._compute_sdf_smooth(base_xy)
        grad_sdf_b = torch.stack([
             cy * grad_sdf_w[:, 0] + sy * grad_sdf_w[:, 1],
            -sy * grad_sdf_w[:, 0] + cy * grad_sdf_w[:, 1],
        ], dim=-1)

        # nominal velocity in WORLD frame.
        # mode="straight": legacy P-controller toward goal (used by every
        # phase prior to V2). mode="clumsy_human": simulates a clumsy
        # teleop / partial-autonomy human:
        #   1) lateral OU noise perpendicular to goal direction (wobble)
        #   2) speed wobble (OU around 0.75 * v_max, clipped to >= 0.35)
        #   3) occasional adversarial swerve toward the nearest obstacle
        #      (rare, persists ~0.5s) to exercise the CBF under adversarial
        #      reference inputs the policy actually has to filter.
        # Real deployment is closer to clumsy than to perfect; training on
        # clumsy means the policy sees the same u_nom distribution at eval.
        err_w = self._goal_xy - base_xy
        v_max_b = self._v_max.unsqueeze(-1)                                # (N, 1)
        if self._unom_mode == "clumsy_human":
            n_err = torch.linalg.norm(err_w, dim=-1, keepdim=True).clamp(min=1e-9)
            goal_dir = err_w / n_err                                       # (N, 2) unit
            perp = torch.stack([-goal_dir[:, 1], goal_dir[:, 0]], dim=-1)  # (N, 2)
            # OU lateral noise (m/s along perp direction)
            self._unom_lat_noise = (
                self._unom_lat_decay * self._unom_lat_noise
                + (1.0 - self._unom_lat_decay) ** 0.5
                  * self._unom_lat_sigma
                  * torch.randn((N,), device=device)
            )
            # OU speed multiplier around mean
            self._unom_speed_mult = (
                self._unom_speed_decay * self._unom_speed_mult
                + (1.0 - self._unom_speed_decay) * self._unom_speed_mean
                + (1.0 - self._unom_speed_decay) ** 0.5
                  * self._unom_speed_sigma
                  * torch.randn((N,), device=device)
            ).clamp(min=self._unom_speed_min, max=1.0)
            # base goal-attractor command, modulated by speed mult
            speed_b = (self._unom_speed_mult * self._v_max).unsqueeze(-1)  # (N, 1)
            u_nom_w = goal_dir * speed_b + perp * self._unom_lat_noise.unsqueeze(-1)
            # adversarial swerve toggle: if not currently swerving, sample
            # a new swerve with low probability; commit to it for
            # `unom_swerve_steps` ticks toward the nearest obstacle.
            not_swerving = self._unom_swerve_counter == 0
            trigger = (torch.rand((N,), device=device) < self._unom_swerve_prob) & not_swerving
            if trigger.any():
                # nearest obstacle direction in WORLD frame
                diff_obs = self._obs_centers_w - base_xy.unsqueeze(1)      # (N, K, 2)
                d2 = (diff_obs * diff_obs).sum(dim=-1)                     # (N, K)
                nearest = d2.argmin(dim=-1)                                # (N,)
                gather_idx = nearest.view(N, 1, 1).expand(-1, 1, 2)
                near_diff = torch.gather(diff_obs, 1, gather_idx).squeeze(1)
                near_norm = torch.linalg.norm(near_diff, dim=-1, keepdim=True).clamp(min=1e-6)
                near_dir = near_diff / near_norm                           # (N, 2)
                # write swerve dir + counter only for triggered envs
                self._unom_swerve_dir = torch.where(
                    trigger.unsqueeze(-1), near_dir, self._unom_swerve_dir)
                self._unom_swerve_counter = torch.where(
                    trigger, torch.full_like(self._unom_swerve_counter,
                                              self._unom_swerve_steps),
                    self._unom_swerve_counter)
            swerving = self._unom_swerve_counter > 0
            if swerving.any():
                # override u_nom toward the nearest obstacle at full v_max
                # for those envs
                u_swerve = self._unom_swerve_dir * v_max_b
                u_nom_w = torch.where(swerving.unsqueeze(-1), u_swerve, u_nom_w)
                self._unom_swerve_counter = (self._unom_swerve_counter - 1).clamp(min=0)
            # final v_max clip
            n_nom = torch.linalg.norm(u_nom_w, dim=-1, keepdim=True).clamp(min=1e-9)
            u_nom_w = torch.where(n_nom > v_max_b, u_nom_w * (v_max_b / n_nom), u_nom_w)
        else:
            u_nom_w = self._kp * err_w
            n_nom = torch.linalg.norm(u_nom_w, dim=-1, keepdim=True).clamp(min=1e-9)
            u_nom_w = torch.where(n_nom > v_max_b, u_nom_w * (v_max_b / n_nom), u_nom_w)

        u_nom_b = torch.stack([
             cy * u_nom_w[:, 0] + sy * u_nom_w[:, 1],
            -sy * u_nom_w[:, 0] + cy * u_nom_w[:, 1],
        ], dim=-1)

        # Lipschitz rate-limit on u_nom_b (BODY frame -- the CBF's actual
        # input; the constraint grad_sdf_b · u_nom_b sees body-frame).
        # 0.0 = disabled. > 0.0 hard-clips ||Δu_nom_b|| per step. With
        # unom_max_step=0.15 at v_max=2.0 the reference can develop a
        # full direction change in ~25 steps (0.5 s) -- enough to allow
        # adversarial swerves while killing any sub-step jitter the CBF
        # would otherwise have to filter. NB: applied AFTER the world->
        # body rotation so yaw rotation contributions are also bounded.
        if self._unom_max_step > 0.0:
            delta_unom = u_nom_b - self._unom_prev_b
            n_delta = torch.linalg.norm(delta_unom, dim=-1, keepdim=True).clamp(min=1e-9)
            over = n_delta.squeeze(-1) > self._unom_max_step
            delta_unom = torch.where(
                over.unsqueeze(-1),
                delta_unom * (self._unom_max_step / n_delta),
                delta_unom,
            )
            u_nom_b = self._unom_prev_b + delta_unom
        self._unom_prev_b = u_nom_b.detach().clone()

        # CBF constraint: grad h_smooth . v >= phi - alpha * h_smooth
        # grad h_smooth = g_eff * grad_sdf (non-unit). Fold into rhs so the
        # closed-form solver still uses the unit-gradient direction.
        rhs_smooth = phi - alpha * h_smooth                                  # (N,)
        rhs_unit = rhs_smooth / g_eff.clamp(min=1e-9)                        # (N,)
        u_safe_b, slack = _closed_form_cbf_batched(
            grad_sdf_b, rhs_unit, u_nom_b, self._v_max,
        )
        h_pre = sdf  # keep legacy name for cached state ("h before step")

        # build locomotion's 48-dim obs
        base_lin_b = self._robot.data.root_lin_vel_b               # (N, 3)
        base_ang_b = self._robot.data.root_ang_vel_b               # (N, 3)
        gravity_b = self._robot.data.projected_gravity_b           # (N, 3)
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        vel_cmd_for_loco = torch.cat(
            [u_safe_b, torch.zeros((N, 1), device=device)], dim=-1,  # (vx, vy, wz=0)
        )

        loco_obs = torch.cat([
            base_lin_b,
            base_ang_b,
            gravity_b,
            vel_cmd_for_loco,
            joint_pos_rel,
            joint_vel,
            self._prev_loco_action,
        ], dim=-1)   # (N, 48)

        # frozen locomotion policy.
        # NB: use no_grad, NOT inference_mode -- inference_mode marks the
        # output tensor, and that mark propagates into env state via the
        # joint_targets we write next, eventually breaking env.reset().
        with torch.no_grad():
            joint_raw = self._loco(loco_obs)                       # (N, 12)
        # match stock JointPositionActionCfg: target = default + scale * raw
        # RMA motor_strength: scales the magnitude of joint-action deviations
        # from default (1.0 = nominal). Per-env scalar -> broadcast to (N, 1).
        joint_targets = (self._robot.data.default_joint_pos
                         + self._loco_action_scale
                         * self._motor_strength.unsqueeze(-1)
                         * joint_raw)
        # B.1 actuation noise: per-env Gaussian noise added to joint
        # targets each step. Per-env std = self._actuation_noise_std.
        # Maps to phi (input uncertainty).
        if (self._actuation_noise_std > 0.0).any():
            noise = torch.randn_like(joint_targets) \
                    * self._actuation_noise_std.unsqueeze(-1)
            joint_targets = joint_targets + noise
        self._processed_actions = joint_targets
        self._prev_loco_action = joint_raw.detach()

        # external disturbance force (applied to base before physics tick).
        # `set_external_force_and_torque` writes a PERSISTENT force --
        # what you write stays until you overwrite it. So we ALWAYS write
        # (zeros when no disturbance) to avoid stale forces leaking from
        # a previous DR setting into a later eval cell with disturbance=0.
        self._dist_step_count += 1
        resample = (self._dist_step_count % self._disturbance_resample) == 0
        if resample.any():
            new_theta = torch.rand((N,), device=device) * (2.0 * math.pi)
            self._dist_theta = torch.where(resample, new_theta, self._dist_theta)
        forces = torch.zeros((N, 1, 3), device=device)
        forces[:, 0, 0] = self._disturbance_force * torch.cos(self._dist_theta)
        forces[:, 0, 1] = self._disturbance_force * torch.sin(self._dist_theta)
        torques = torch.zeros_like(forces)
        self._robot.set_external_force_and_torque(forces, torques, body_ids=[0])

        # cache for reward terms. Compute step-to-step jitter (normalized
        # by bound width) BEFORE overwriting last_phi/last_alpha.
        phi_width = self._phi_hi - self._phi_lo
        alpha_width = self._alpha_hi - self._alpha_lo
        self.last_action_jitter = (
            (phi - self.last_phi).abs() / max(phi_width, 1e-9)
            + (alpha - self.last_alpha).abs() / max(alpha_width, 1e-9)
        )
        self.last_phi = phi
        self.last_alpha = alpha
        self.last_u_nom = u_nom_b
        self.last_u_safe = u_safe_b
        self.last_intervention = torch.linalg.norm(u_safe_b - u_nom_b, dim=-1)
        self.last_h_pre = h_pre
        self.last_slack = slack

    def apply_actions(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions)

    # ------------------------ post-step convenience ------------------------
    def update_post_physics(self) -> None:
        """Called after physics step; refreshes h_realized + dist_to_goal +
        progress buffers. Reward terms then read these.

        `last_h_realized` = SDF (signed distance, min over obstacles) --
        we keep this as the metric for collision and the obs-side
        geometric stand-in. `last_h_smooth_realized` is also available
        for any consumer that wants the smoothed value.
        """
        base_xy = self._robot.data.root_pos_w[:, :2]
        sdf, h_smooth, _, _ = self._compute_sdf_smooth(base_xy)
        self.last_h_realized = sdf
        self.last_h_smooth_realized = h_smooth
        self.last_dist_to_goal = torch.linalg.norm(self._goal_xy - base_xy, dim=-1)


# ---------------------------------------------------------------------------
# Cfg (defined AFTER the term class so class_type can be a proper default;
# configclass freezes field defaults at class-creation time, so a late
# attribute assignment wouldn't take.)
# ---------------------------------------------------------------------------
@configclass
class CBFParamActionTermCfg(ActionTermCfg):
    """Cfg for `CBFParamActionTerm`.

    `locomotion_policy_obj` must be set externally (typically in the training
    script after the policy is loaded from a checkpoint); it cannot be a
    configclass default because it's a torch.nn.Module.
    """
    class_type: type = CBFParamActionTerm
    asset_name: str = "robot"

    locomotion_policy_obj: Any = None     # nn.Module, set in train script
    locomotion_action_scale: float = 0.25

    phi_bounds: tuple[float, float] = (0.0, 1.0)
    alpha_bounds: tuple[float, float] = (0.2, 4.0)

    goal_xy: tuple[float, float] = (6.0, 0.0)
    # Multi-obstacle support: list of (x, y, radius). Each obstacle is
    # specified in env-LOCAL coordinates; env_origins are added per env.
    # Default = single obstacle matching the original Phase 1/2 scenario.
    obstacles: list[tuple[float, float, float]] = None   # set in __post_init__
    # Robot's safety radius added to each obstacle's physical radius to form
    # `r_safe` in the CBF. 0.35 m covers the Go2's body half-diagonal
    # (footprint ~0.65×0.31 m -> half-diagonal ~0.36) up to a small margin.
    # Previously 0.30 (inscribed-circle) which under-covered diagonal
    # approaches.  Slalom passage is still feasible: robot deflects to
    # y=±0.35 to pass obstacles at y=±0.5 (well within the 1.0 m corridor).
    robot_radius: float = 0.35
    # Smoothed barrier h_smooth(sdf) = lambda * (1 - exp(-gamma * sdf)).
    # See `_compute_sdf_smooth`. lambda controls asymptotic h magnitude;
    # gamma controls smoothness/saturation rate.
    h_smooth_lambda: float = 1.0
    h_smooth_gamma: float = 2.0
    # B.4 SIM2REAL (SHIELD-aligned, Yang et al. 2025): switch the CBF
    # to use obstacle positions perturbed by per-step Gaussian noise,
    # simulating Livox Mid-360 + Euclidean clustering + cylinder-fit
    # detection accuracy. Default False keeps privileged behavior.
    use_lidar_sdf: bool = False
    # std (meters) of per-axis Gaussian noise added to each obstacle's
    # xy each step. 5cm is realistic for Mid-360 + clustering.
    perception_noise_std: float = 0.05
    # per-step probability that each visible obstacle is dropped from
    # the SDF list. Simulates clustering failures / occlusion. 2% per
    # step ~= 18% chance of being dropped for at least one step in any
    # 10-step window. Realistic for crowded real-world scenes.
    perception_dropout_prob: float = 0.02
    # max range of the simulated lidar (m). Obstacles farther than this
    # are not in the SDF (perception literally doesn't see them).
    # 20m matches Livox Mid-360 horizontal coverage.
    lidar_max_range: float = 20.0
    # ----- Mid-360 lidar fidelity knobs (consumed by mdp._compute_lidar) -----
    # Per-ray Gaussian range noise (m), FIXED in range. Mid-360 datasheet:
    # ±2 cm @ 10 m. Replaces the previous range-scaling model that was
    # overly pessimistic at distance.
    lidar_noise_std: float = 0.02
    # Per-step Gaussian jitter on each ray's angle (radians). Approximates
    # the Mid-360's non-repetitive scan pattern: different bearings each
    # tick rather than 72 fixed angles. 0.0087 rad ~ 0.5 deg.
    lidar_angle_jitter_std: float = 0.0087
    # Range-weighted Bernoulli dropout on each ray:
    #     p_drop = base + slope * (range / max_range)
    # so far returns drop more often than near returns (mirrors real
    # incidence-angle / low-reflectivity failures). Dropped rays are set
    # to `lidar_max_range`.
    lidar_dropout_base_prob: float = 0.005
    lidar_dropout_range_slope: float = 0.03
    # B.5 HARDWARE-SAFETY Lipschitz rate-limit on the policy action.
    # Per-step change in the normalized [-1, 1] action is hard-clipped
    # to this value. Makes (phi, alpha) time-Lipschitz with constant
    # L = action_max_step / dt. With dt = 0.02s (50Hz):
    #   0.04 -> L = 2/s   (50 steps to traverse full [-1, 1] range)
    #   0.08 -> L = 4/s   (25 steps)
    #   0.20 -> L = 10/s  (10 steps, fast)
    #   0.0  -> disabled  (default, backward compat)
    # The policy trains WITH the rate-limit active so it learns to
    # issue continuous commands.
    action_max_step: float = 0.0
    # back-compat single-obstacle CLI knobs (consumed in __post_init__ if
    # `obstacles` is left unset)
    obstacle_xy: tuple[float, float] = (2.5, 0.3)
    obstacle_radius: float = 0.9

    kp: float = 1.0
    v_max: float = 1.3

    # ---- Clumsy-human u_nom planner (Phase 10 / V2) ----
    # "straight" (default) -> legacy P-controller toward goal (every phase
    # prior to V2). "clumsy_human" -> OU lateral noise + speed wobble +
    # occasional adversarial swerve toward nearest obstacle. Active at
    # BOTH train and eval so the policy actually sees this u_nom
    # distribution. See the process_actions branch for the math.
    unom_mode: str = "straight"
    # Preset selector: "child" (5yo: bigger wobble, slower, more mistakes
    # -> harder train distribution), "teen" (10yo: less wobble, faster,
    # fewer mistakes -> closer to deployment, still imperfect), or
    # "default" (back-compat values). The preset overrides the OU sigma /
    # swerve_prob / speed_mean knobs below at action-term init time;
    # individual knobs still win if explicitly set.
    unom_clumsiness: str = "default"
    unom_lat_decay: float = 0.95     # OU on lateral wobble (per-step decay)
    unom_lat_sigma: float = 0.30     # m/s, std of lateral wobble at steady state
    unom_speed_decay: float = 0.97   # OU on speed multiplier
    unom_speed_sigma: float = 0.18   # std of speed-mult wobble
    unom_speed_mean: float = 0.75    # OU mean of speed multiplier (vs v_max)
    unom_speed_min: float = 0.35     # hard floor on speed mult (no full stop)
    unom_swerve_prob: float = 0.005  # per-step trigger probability
    unom_swerve_steps: int = 25      # persistence of one swerve (~0.5s @ 50Hz)
    # Hard cap on ||Δu_nom_w|| per step (m/s). 0.0 disables. > 0.0 keeps
    # the CBF reference Lipschitz in time so swerve onsets / OU jumps
    # can't spike u_nom -- the CBF sees a smooth nominal.
    unom_max_step: float = 0.0
    # Stuck-flag latency (steps). Sticky `episode_stuck_any` fires only
    # after this many CONTINUOUS slow-and-not-at-goal steps within one
    # episode. V2 raises this to 250 (5 s @ 50 Hz) so brief slowdowns
    # don't false-tag the env as stuck before it can wiggle out.
    stuck_threshold_steps: int = 100

    v_max_range: tuple[float, float] | None = None
    """If set, sample v_max per env on reset. The phase6_vmax_gate showed
    this is the cleanest α channel: best α shifts 4.0 -> 2.5 across
    v_max in [1.0, 2.0] (92% of bound width) because higher commanded
    speed -> harder to brake -> need more-conservative α. Policy can
    infer commanded v_max from observed base_lin_vel_b (proprio)."""

    obstacle_pos_jitter_range: tuple[float, float] | None = None
    """If set, per-episode uniform offset applied to each obstacle's xy
    position (same range used for x and y, independent per obstacle).
    Use this to force the policy to actually read lidar -- with fixed
    obstacle positions, the robot can infer "I'm near the obstacle"
    from proprio + dist_to_goal alone (verified empirically in
    phase6_lidar_attention.py). Randomizing position makes lidar the
    only signal that carries obstacle location."""

    # ---- random topology (v8-light: random per-episode obstacle spawn) ----
    # When True, IGNORE `obstacles` + `obstacle_pos_jitter_range` and instead
    # spawn `random_topology_K` obstacles per env per reset, sampled
    # uniformly in `random_topology_{x,y}_range` (env-local coords), with
    # rejection sampling for non-overlap and start/goal exclusion zones.
    # The original RMA paper trained with random obstacle layouts; the v8
    # plan calls for this as the "topology" axis of generalization. Static
    # in the sense that obstacles don't MOVE within an episode (no drift).
    # ---- proprio history buffer (for history-MLP teacher) ----
    # When > 0, the action term maintains a rolling buffer of
    # `proprio_history_length` past `deployable_obs` snapshots per env.
    # Exposed to the policy via `mdp.proprio_history_obs`. Used by the
    # RMAHistory teacher to learn whether PPO can extract priv-equivalent
    # info from proprio dynamics end-to-end (vs the explicit RMA
    # student-distillation stage).
    proprio_history_length: int = 0

    # ---- observation-level masking (ablation experiments) ----
    # When True, the corresponding slice of teacher_obs is forcibly
    # zeroed BEFORE being concatenated into the actor's obs. Used to
    # test "what does the policy learn if it can't see X?" without
    # writing parallel obs functions or model variants. The slots stay
    # in the obs layout (model shape unchanged) -- only the values are
    # masked.
    #
    # Combinations of interest:
    #   obs_mask_priv=True, obs_mask_proprio=False    -> lidar + proprio
    #   obs_mask_priv=False, obs_mask_proprio=True    -> lidar + priv z
    obs_mask_priv: bool = False
    obs_mask_proprio: bool = False

    random_topology: bool = False
    random_topology_K: int = 3
    random_topology_x_range: tuple[float, float] = (1.5, 6.0)
    random_topology_y_range: tuple[float, float] = (-1.5, 1.5)
    random_topology_start_exclusion_r: float = 0.8   # m around (0, 0)
    random_topology_goal_exclusion_r: float = 0.8    # m around goal_xy
    random_topology_min_separation: float = 1.4      # m between obstacle centers
    random_topology_max_attempts: int = 30           # rejection-sampling retries

    disturbance_force: float = 30.0
    disturbance_force_range: tuple[float, float] | None = None
    # RMA additional privileged factors (per-episode DR). Leave as None to
    # keep them pinned at nominal (no DR). Setting a range activates DR
    # AND the physics-apply path in `reset` (friction/mass via PhysX
    # view writes; motor_strength as a multiplier in process_actions).
    friction_range: tuple[float, float] | None = None
    base_mass_range: tuple[float, float] | None = None
    """Additive delta to the default base mass per episode (kg)."""
    motor_strength_range: tuple[float, float] | None = None
    """Multiplier on the joint-action deviation from default."""
    # ---- B.1 extra DR channels (unified-teacher candidates) ----
    actuation_noise_std_range: tuple[float, float] | None = None
    """If set, sample a per-episode Gaussian std (rad) and add N(0, std)
    noise to processed joint targets EVERY step. Theoretically a φ-axis
    channel (input uncertainty). Typical range (0.0, 0.05) rad."""
    com_offset_range: tuple[float, float] | None = None
    """If set, sample a per-episode forward/backward (x-axis) offset (m)
    of the base body's center of mass. Shifts CoG without changing mass,
    making the body lean and tracking residual rise. Theoretically an
    α-axis channel. Typical range (-0.05, 0.05) m."""
    """If set, override the scalar `disturbance_force` with a uniform
    range; each env's disturbance magnitude is sampled fresh on every
    reset. Use this for Phase 2 / 3 (DR-style adaptation training).
    Leave as None for Phase 1 (fixed magnitude across all envs / episodes).
    """
    disturbance_resample: int = 50
