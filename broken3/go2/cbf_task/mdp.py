"""Custom MDP function library for the CBF-adaptive Go2 task.

Reward, termination, and observation term implementations. These are
standalone callables in the Isaac Lab manager-based env pattern; they
receive `env` (the ManagerBasedRLEnv) and return per-env tensors.

Most of the heavy lifting (CBF QP, h, intervention, etc.) is done inside
`CBFParamActionTerm.process_actions` and cached as attributes. These MDP
functions just call `_cbf_term(env).update_post_physics()` once per step
and then read the cached buffers.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .cbf_action_term import _yaw_from_quat_wxyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cbf_term(env: "ManagerBasedRLEnv"):
    return env.action_manager._terms["cbf_param"]


# ---------------------------------------------------------------------------
# A small once-per-step post-physics refresh trampoline.
#
# Isaac Lab's reward/termination managers call each term function once per
# step. The first one we route through this helper triggers the CBF term's
# `update_post_physics()` (which reads the post-step robot pose to compute
# h_realized / dist_to_goal). All subsequent terms in the same step then
# read the cached buffers cheaply.
# ---------------------------------------------------------------------------
def _ensure_post_physics(env: "ManagerBasedRLEnv") -> None:
    cbf = _cbf_term(env)
    # We use the env's common_step_counter as a "have we updated this step?" marker.
    step = int(env.common_step_counter)
    if getattr(cbf, "_post_physics_done_at_step", -1) != step:
        cbf.update_post_physics()
        # also bump the stuck-tracking counters once per step (cheap)
        robot = cbf._robot
        lin_speed = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
        slow_now = (lin_speed < 0.15) & (cbf.last_dist_to_goal > 0.4)
        cbf.episode_stuck_steps += slow_now.long()
        # V2 raises this threshold to 250 (5 s @ 50 Hz) so brief stalls
        # don't tag the env as stuck before it can wiggle out.
        cbf.episode_stuck_any |= (
            cbf.episode_stuck_steps > int(getattr(cbf, "_stuck_threshold_steps", 100))
        )
        # cache lidar at t and t-1 so the CNN sees raw consecutive frames
        # rather than a pre-computed delta (CNN can derive delta itself
        # via a learned 1x2 conv across the channel dim if useful).
        cfg = cbf.cfg
        new_lidar = _compute_lidar(
            env, n_rays=72,
            max_range=float(getattr(cfg, "lidar_max_range", 20.0)),
            noise_std=float(getattr(cfg, "lidar_noise_std", 0.02)),
            angle_jitter_std=float(getattr(cfg, "lidar_angle_jitter_std", 0.0087)),
            dropout_base_prob=float(getattr(cfg, "lidar_dropout_base_prob", 0.005)),
            dropout_range_slope=float(getattr(cfg, "lidar_dropout_range_slope", 0.03)),
        )
        if cbf.last_lidar is None or cbf.last_lidar.shape != new_lidar.shape:
            cbf.last_lidar = new_lidar.clone()
        cbf.last_lidar_prev = cbf.last_lidar
        cbf.last_lidar = new_lidar
        cbf._post_physics_done_at_step = step


# ---------------------------------------------------------------------------
# Reward terms (return per-env reward tensors; weights applied by cfg)
# ---------------------------------------------------------------------------
def progress_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-step reduction in distance to goal: prev_dist - current_dist.
    Updates `prev_dist_to_goal` after computing the delta so the next step's
    reward is well-defined.
    """
    _ensure_post_physics(env)
    cbf = _cbf_term(env)
    delta = cbf.prev_dist_to_goal - cbf.last_dist_to_goal
    cbf.prev_dist_to_goal = cbf.last_dist_to_goal.clone()
    return delta


def intervention_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """||u_safe - u_nom||. Positive value; sign applied by cfg weight."""
    return _cbf_term(env).last_intervention


def action_smoothness_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """|Δphi|/phi_width + |Δalpha|/alpha_width per step, computed by the
    action term in `process_actions`. Positive value; cfg applies the
    sign weight. Adds back-pressure so the policy keeps (phi, alpha)
    continuous step-to-step -- the locomotion controller below can't
    track jittery commands.
    """
    return _cbf_term(env).last_action_jitter


def collision_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """One-shot 1.0 in envs where h_realized < 0. Cfg applies the weight."""
    _ensure_post_physics(env)
    return (_cbf_term(env).last_h_realized < 0.0).float()


def goal_reached_bonus(env: "ManagerBasedRLEnv", goal_tol: float = 0.4) -> torch.Tensor:
    """One-shot 1.0 in envs that have reached the goal."""
    _ensure_post_physics(env)
    return (_cbf_term(env).last_dist_to_goal < goal_tol).float()


def fall_penalty(env: "ManagerBasedRLEnv",
                  height_thr: float = 0.15,
                  grav_z_thr: float = -0.3) -> torch.Tensor:
    """One-shot 1.0 per env on the step the robot first falls. Same
    detector as `fall_termination` (base low + tilted). Paired with a
    large negative weight (e.g. -100) so the policy is strongly
    incentivized not to fall, separately from the collision penalty.
    """
    cbf = _cbf_term(env)
    robot = cbf._robot
    base_z = robot.data.root_pos_w[:, 2]
    grav_b_z = robot.data.projected_gravity_b[:, 2]
    return ((base_z < height_thr) & (grav_b_z > grav_z_thr)).float()


def stuck_penalty(env: "ManagerBasedRLEnv",
                   speed_thresh: float = 0.15,
                   goal_tol: float = 0.4) -> torch.Tensor:
    """Per-step penalty when the robot is slow AND not yet at the goal.

    Default weight is 0.0 in CBFRewardsCfg (wired but inactive). Turn on
    if the teacher converges to a 'freeze under disturbance' optimum
    (observed in earlier teacher runs): without this term, standing
    still earns 0 progress reward, 0 collision penalty, 0 intervention
    cost -- a stable degenerate fixed point. A small negative weight
    (e.g. -0.05) tilts the basin toward forward motion.

    Reuses the same `slow_now` mask that drives `episode_stuck_*`
    tracking in `_ensure_post_physics`, so the reward and the eval
    diagnostic agree on what "stuck" means.
    """
    _ensure_post_physics(env)
    cbf = _cbf_term(env)
    lin_speed = torch.linalg.norm(cbf._robot.data.root_lin_vel_b[:, :2], dim=-1)
    slow_now = (lin_speed < speed_thresh) & (cbf.last_dist_to_goal > goal_tol)
    return slow_now.float()


# ---------------------------------------------------------------------------
# Termination terms (return per-env bool tensors)
# ---------------------------------------------------------------------------
def collision_termination(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _ensure_post_physics(env)
    cbf = _cbf_term(env)
    fired = cbf.last_h_realized < 0.0
    # latch into the sticky per-cell flag BEFORE the auto-reset runs and
    # otherwise clobbers our chance to observe this from outside env.step
    cbf.episode_collide_any |= fired
    return fired


def goal_reached_termination(env: "ManagerBasedRLEnv",
                             goal_tol: float = 0.4) -> torch.Tensor:
    _ensure_post_physics(env)
    cbf = _cbf_term(env)
    fired = cbf.last_dist_to_goal < goal_tol
    cbf.episode_reach_any |= fired
    return fired


def base_contact_termination(env: "ManagerBasedRLEnv",
                              threshold: float = 1.0) -> torch.Tensor:
    """Legacy contact-sensor variant. Kept for reference, NOT WIRED IN
    -- the SceneEntityCfg.body_ids didn't resolve cleanly inside the
    wrapper and it fired on every-step foot contact. Use `fall_termination`
    (base height + orientation check) instead.
    """
    from isaaclab.envs.mdp.terminations import illegal_contact
    from isaaclab.managers import SceneEntityCfg
    fired = illegal_contact(
        env,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names="base"),
        threshold=threshold,
    )
    cbf = _cbf_term(env)
    cbf.episode_fall_any |= fired
    return fired


def fall_termination(env: "ManagerBasedRLEnv",
                      height_thr: float = 0.15,
                      grav_z_thr: float = -0.3) -> torch.Tensor:
    """Robot fell -- require BOTH base on the ground AND clearly tilted.
       - base z < height_thr (0.15m, well below Go2's 0.33m standing)
       - gravity_b[2] > grav_z_thr (>70 deg tilt from vertical)
    AND-conjunction (not OR) so transient spawn states and quick turns
    don't false-trigger. First eval attempt had OR with looser thresholds
    and fired at spawn (gravity_b[2] was 0-initialized -> > -0.5 was
    always True), tagging every env as fallen before physics stabilized.
    """
    cbf = _cbf_term(env)
    robot = cbf._robot
    base_z = robot.data.root_pos_w[:, 2]
    grav_b_z = robot.data.projected_gravity_b[:, 2]
    fired = (base_z < height_thr) & (grav_b_z > grav_z_thr)
    cbf.episode_fall_any |= fired
    return fired


def stuck_termination(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """OPT-IN termination on the sticky 'stuck' flag (>100 slow-not-at-goal
    steps, ~2 sec at 50Hz). Reads the same `episode_stuck_any` flag that
    `_ensure_post_physics` maintains -- DOES NOT increment any counters
    here (the old `stuck_check` function did, and was a duplication
    footgun: wiring it as a DoneTerm would have double-counted with
    `_ensure_post_physics`).

    To use: add `stuck = DoneTerm(func=mdp.stuck_termination)` in a
    `CBFTerminationsCfg` subclass post-init, and pair with a small
    `stuck_penalty` reward weight so the policy is pressured away from
    the freeze-under-disturbance optimum without being able to escape
    the cost by waiting for time_out.
    """
    return _cbf_term(env).episode_stuck_any


# ---------------------------------------------------------------------------
# Observation terms
# ---------------------------------------------------------------------------
def zeroed_obs(env: "ManagerBasedRLEnv", obs_dim: int = 7) -> torch.Tensor:
    """Phase 1: zeroed observation forces the outer policy to be
    state-INdependent. Returns (num_envs, obs_dim) of zeros."""
    return torch.zeros((env.num_envs, obs_dim), device=env.device)


def geometric_stand_in(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """SWAP-OUT MARKER: this term is a proof-of-concept stand-in for
    lidar-derived geometric awareness. It exposes the analytic barrier
    value `h_realized` (distance from obstacle minus safe radius) as a
    single scalar per env.

    Why: Phase 1.5 proved proprio carries the disturbance signal, but the
    policy is still failing on safety. Hypothesis: it lacks geometric
    awareness of where the obstacle IS. A real Mid-360 lidar grid + CNN
    would learn this implicitly; for POC we cheat with the scalar.

    To replace with lidar later: drop this obs term, add a lidar obs
    term that returns the raycast grid + CNN encoder in the policy.
    """
    _ensure_post_physics(env)
    cbf = _cbf_term(env)
    return cbf.last_h_realized.unsqueeze(-1)   # (N, 1)


def _compute_lidar(env: "ManagerBasedRLEnv",
                    n_rays: int = 72,
                    max_range: float = 20.0,
                    noise_std: float = 0.02,
                    angle_jitter_std: float = 0.0087,
                    dropout_base_prob: float = 0.005,
                    dropout_range_slope: float = 0.03) -> torch.Tensor:
    """Analytic 2D lidar ring (ray-cylinder intersection) tuned to match
    a real Livox Mid-360 after ring-extraction.

    Mid-360 sim2real model:
      * `noise_std` (m): per-ray Gaussian range noise, FIXED in range.
        Mid-360 datasheet quotes ±2cm @ 10m; the scaling-with-range
        noise the previous version used was overly pessimistic at
        distance and didn't match the spec.
      * `angle_jitter_std` (rad): per-STEP Gaussian jitter on every ray
        angle. Approximates the Mid-360's non-repetitive scan pattern
        (the lidar samples slightly different bearings each tick instead
        of the same 72 fixed angles). 0.0087 rad ~ 0.5deg, which is
        about the angular grid spacing of 360 / 72 = 5deg divided by 10.
      * `dropout_base_prob` + `dropout_range_slope`: a ray is dropped
        with probability `base + slope * (range / max_range)`. Closer
        rays drop less, far rays drop more (mirrors the reality that
        far returns fail more often due to incidence angle, beam
        divergence, and low-reflectivity targets). Dropped rays are set
        to `max_range` (i.e. "no obstacle in this direction"), which is
        what a real point cloud post-clustering would produce.
      * `max_range`: hard ceiling. Mid-360 spec is 40m but Isaac warp
        starts to lose accuracy past ~20m so we cap conservatively.

    Computed analytically because Isaac Lab's warp raycaster can't
    target per-env-replicated obstacles.

    Returns (N, R).
    """
    cbf = _cbf_term(env)
    device = env.device

    base_xy = cbf._robot.data.root_pos_w[:, :2]                  # (N, 2)
    yaw = _yaw_from_quat_wxyz(cbf._robot.data.root_quat_w)       # (N,)

    relative_angles = torch.linspace(
        -math.pi, math.pi, n_rays + 1, device=device,
    )[:-1]                                                       # (R,)
    # PER-STEP, PER-ENV angle jitter -- different bearings each tick
    # and each env, mirroring the non-repetitive Mid-360 scan
    if angle_jitter_std > 0.0:
        N = base_xy.shape[0]
        jitter = torch.randn((N, n_rays), device=device) * angle_jitter_std
        world_angles = yaw.unsqueeze(-1) + relative_angles.unsqueeze(0) + jitter
    else:
        world_angles = yaw.unsqueeze(-1) + relative_angles.unsqueeze(0)  # (N, R)
    dirs = torch.stack([torch.cos(world_angles),
                        torch.sin(world_angles)], dim=-1)        # (N, R, 2)

    centers = cbf._obs_centers_w                                 # (N, K, 2)
    obs_radii = (cbf._r_safe.squeeze(0)
                 - float(cbf.cfg.robot_radius)).clamp(min=0.0)   # (K,)

    base_b = base_xy.unsqueeze(1).unsqueeze(2)                   # (N, 1, 1, 2)
    centers_b = centers.unsqueeze(1)                             # (N, 1, K, 2)
    dirs_b = dirs.unsqueeze(2)                                   # (N, R, 1, 2)
    diff = base_b - centers_b                                    # (N, 1, K, 2)
    b = (dirs_b * diff).sum(dim=-1)                              # (N, R, K)
    c = (diff * diff).sum(dim=-1) \
        - obs_radii.unsqueeze(0).unsqueeze(0) ** 2               # (N, 1, K)
    disc = b * b - c                                             # (N, R, K)
    sqrt_d = torch.sqrt(disc.clamp(min=0.0))
    t_near = -b - sqrt_d
    t_far = -b + sqrt_d
    t_hit = torch.where(t_near > 0.0, t_near, t_far)             # (N, R, K)
    valid = (disc >= 0.0) & (t_hit > 0.0)
    t_hit = torch.where(valid, t_hit,
                        torch.full_like(t_hit, max_range))
    ranges = t_hit.min(dim=-1).values.clamp(max=max_range)       # (N, R)

    if noise_std > 0.0:
        # fixed per-ray Gaussian noise (Mid-360 datasheet)
        ranges = ranges + torch.randn_like(ranges) * noise_std

    if dropout_base_prob > 0.0 or dropout_range_slope > 0.0:
        # range-weighted Bernoulli dropout: far rays fail more often
        p_drop = (dropout_base_prob
                  + dropout_range_slope * (ranges / max_range)).clamp(0.0, 1.0)
        dropped = torch.rand_like(ranges) < p_drop
        ranges = torch.where(dropped, torch.full_like(ranges, max_range), ranges)

    return ranges.clamp(min=0.0, max=max_range)


def lidar_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Cached lidar (72 rays). Computed once per step in
    `_ensure_post_physics`; this function just returns the cache so
    obs/reward/diagnostic terms see a consistent snapshot.
    """
    _ensure_post_physics(env)
    cbf = _cbf_term(env)
    return cbf.last_lidar


def lidar_prev_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Previous-step lidar ranges (per ray). Paired with current ranges
    via the lidar CNN's 2-channel input, so the CNN sees raw (t-1, t)
    frames and learns the temporal relationship itself (instead of us
    pre-computing a delta).
    """
    _ensure_post_physics(env)
    cbf = _cbf_term(env)
    return cbf.last_lidar_prev


def state_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Small state vector the RMA actor consumes alongside z + lidar:
    [dist_to_goal, base_lin_vel_b(3)]. Heading is intentionally omitted
    -- the CBF + locomotion handle steering; the outer policy just needs
    'how urgent' (dist_to_goal) and 'how fast' (lin_vel) to decide its
    risk posture. (N, 4).
    """
    cbf = _cbf_term(env)
    dist = cbf.last_dist_to_goal.unsqueeze(-1)             # (N, 1)
    lin_vel = cbf._robot.data.root_lin_vel_b               # (N, 3)
    return torch.cat([dist, lin_vel], dim=-1)              # (N, 4)


def prev_action_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Previous (phi, alpha) the policy emitted, raw values (not the
    normalized [-1, 1] policy output). (N, 2).
    """
    cbf = _cbf_term(env)
    return torch.stack([cbf.last_phi, cbf.last_alpha], dim=-1)


def teacher_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """RMA-teacher single flat observation (SHIELD layout):
       [priv(7), deployable(45), prev_action(2),
        lidar_prev(72), lidar(72)] = (N, 198).
    """
    priv = priv_obs(env)                                   # (N, 7)
    proprio = deployable_obs(env)                          # (N, 45)
    pa = prev_action_obs(env)                              # (N, 2)
    rays_prev = lidar_prev_obs(env)                        # (N, 72)
    rays = lidar_obs(env)                                  # (N, 72)
    return torch.cat([priv, proprio, pa, rays_prev, rays], dim=-1)  # (N, 198)


def proprio_history_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Rolling buffer of the last K `deployable_obs` snapshots, where
    K = cfg.proprio_history_length on the action term. Returns (N, K, 45)
    NOT flattened -- the policy's _ProprioCNN reshapes for its time-axis
    convolution.

    Buffer is rolled by THIS function (idempotent within a step via the
    `_post_physics_done_at_step` marker like the lidar cache). Reset
    zeroing of per-env slices is handled by the action term's reset().
    """
    _ensure_post_physics(env)
    cbf = _cbf_term(env)
    if cbf._proprio_history is None:
        raise RuntimeError(
            "proprio_history_obs called but action term was not configured "
            "with proprio_history_length > 0"
        )
    # one-time roll per env step. The marker check piggybacks on the
    # lidar caching path: if last_lidar was just updated this step, the
    # proprio history hasn't been rolled yet for this step.
    step = int(env.common_step_counter)
    if getattr(cbf, "_proprio_history_done_at_step", -1) != step:
        new_proprio = deployable_obs(env)                  # (N, 45)
        cbf._proprio_history = torch.roll(cbf._proprio_history, shifts=-1, dims=1)
        cbf._proprio_history[:, -1, :] = new_proprio
        cbf._proprio_history_done_at_step = step
    return cbf._proprio_history                            # (N, K, 45)


def teacher_obs_history(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """RMAHistory teacher obs: proprio history + prev_action + lidar.
       [proprio_history(K*45), prev_action(2),
        lidar_prev(72), lidar(72)] = (N, K*45 + 146)

    NO priv slot at all. The hypothesis is that PPO can extract priv-
    equivalent info from proprio dynamics end-to-end, removing the
    need for a separate student-distillation stage.
    """
    hist = proprio_history_obs(env)                        # (N, K, 45)
    N, K, D = hist.shape
    hist_flat = hist.reshape(N, K * D)                     # (N, K*45)
    pa = prev_action_obs(env)                              # (N, 2)
    rays_prev = lidar_prev_obs(env)                        # (N, 72)
    rays = lidar_obs(env)                                  # (N, 72)
    return torch.cat([hist_flat, pa, rays_prev, rays], dim=-1)


def teacher_obs_rma_classic(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """RMA-classic teacher obs (4 priv channels):
       [priv(4), deployable(45), prev_action(2),
        lidar_prev(72), lidar(72)] = (N, 195).

    Same layout as SHIELD's teacher_obs but with the RMA-paper-canonical
    4-channel priv (disturbance, friction, mass, motor) instead of the
    v7 7-channel priv. Pairs with `RMAClassicMLPModel` and the
    `Isaac-CBF-Adaptive-Go2-RMAStatic-v0` task.

    Honors `obs_mask_priv` and `obs_mask_proprio` on the action term
    cfg (ablation flags): when set, the corresponding slice is zeroed
    BEFORE concat. Slot stays in the layout so the model shape is
    unchanged; only the values are masked.
    """
    cbf = _cbf_term(env)
    priv = priv_obs_rma_classic(env)                       # (N, 4)
    proprio = deployable_obs(env)                          # (N, 45)
    pa = prev_action_obs(env)                              # (N, 2)
    rays_prev = lidar_prev_obs(env)                        # (N, 72)
    rays = lidar_obs(env)                                  # (N, 72)
    if getattr(cbf, "_obs_mask_priv", False):
        priv = torch.zeros_like(priv)
    if getattr(cbf, "_obs_mask_proprio", False):
        proprio = torch.zeros_like(proprio)
    return torch.cat([priv, proprio, pa, rays_prev, rays], dim=-1)  # (N, 195)


def priv_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """RMA-teacher privileged observation -- environment factors the
    student must learn to infer from proprio history.

    Layout (N, 7):
       [:, 0]  disturbance_force        (N) -- per-episode DR (φ/α)
       [:, 1]  friction_coef            -- per-episode DR (PhysX material) (φ)
       [:, 2]  base_mass_delta          -- per-episode DR (PhysX mass) (α)
       [:, 3]  motor_strength           -- per-episode DR (action scale) (φ)
       [:, 4]  actuation_noise_std      -- per-episode DR (joint noise std) (φ) [B.1]
       [:, 5]  com_offset               -- per-episode DR (PhysX CoM x offset) (α) [B.1]
       [:, 6]  v_max                    -- per-episode DR (kinematic urgency) (α) [B.6]
                                          THE validated α channel: 92% bound span
                                          per phase6_vmax_gate. Higher v_max = more
                                          momentum = need smaller α (start braking
                                          earlier). Recoverable from base_lin_vel_b
                                          (saturates at v_max under goal command),
                                          but explicit slot makes inference unneeded.

    Channels that don't have DR enabled (range set to None on the action
    term) will be constant per env; the priv-attention diagnostic will
    reveal whether each carries learnable signal in the policy output.
    """
    cbf = _cbf_term(env)
    return torch.stack([
        cbf._disturbance_force,
        cbf._friction_coef,
        cbf._base_mass_delta,
        cbf._motor_strength,
        cbf._actuation_noise_std,
        cbf._com_offset,
        cbf._v_max,
    ], dim=-1)   # (N, 7)


def priv_obs_rma_classic(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """RMA-paper-canonical privileged observation (4 channels).

    Drops the experimental v7 additions (v_max, actuation_noise,
    com_offset) -- the priv_attention diagnostic on v7.1 showed the
    trained policy ignored all 3 (see [[v7_priv_channel_dead]]). Going
    back to the 4 channels the original RMA paper used (or near-
    analogues for our CBF-param task) to test whether a smaller, cleaner
    priv space gives a better-distilled z latent.

    Layout (N, 4):
       [:, 0]  disturbance_force        per-episode DR (external force)
       [:, 1]  friction_coef            per-episode DR (PhysX material)
       [:, 2]  base_mass_delta          per-episode DR (PhysX mass)
       [:, 3]  motor_strength           per-episode DR (action scale)
    """
    cbf = _cbf_term(env)
    return torch.stack([
        cbf._disturbance_force,
        cbf._friction_coef,
        cbf._base_mass_delta,
        cbf._motor_strength,
    ], dim=-1)   # (N, 4)


def deployable_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Outer-policy proprio observation. 45-dim, deliberately *excludes*
    the velocity_commands slot.

    Rationale (added 2026-05-24): the outer (φ, α) policy doesn't need
    to know what the goal is. Adaptation of CBF parameters depends on
    the world (priv z), current physical state (base/joint vel),
    and obstacles (lidar) -- NOT the commanded velocity. The CBF uses
    u_nom internally; the policy just sets the filter's knobs.

    Side benefit: removing u_nom kills the "goal-proxy" confound that
    plagued earlier teachers (where ||u_nom|| correlated with
    dist_to_goal -> policy used goal_dist instead of lidar to infer
    obstacles). With u_nom out of the obs, lidar is the only way the
    policy can know about obstacles.

    Earlier history:
      - 2026-05-23: vel_cmd was u_safe (post-CBF) -- this leaked
        obstacle info directly (policy could read CBF deflection)
      - 2026-05-24a: switched vel_cmd to u_nom (pre-CBF) -- removed
        the CBF-deflection leak but still leaked goal_dist via ||u_nom||
      - 2026-05-24b (this version): removed vel_cmd entirely

    Layout (45):
      [0:3]    base_lin_vel_b
      [3:6]    base_ang_vel_b
      [6:9]    projected_gravity_b
      [9:21]   joint_pos_rel
      [21:33]  joint_vel
      [33:45]  prev_loco_action (raw, de-scaled)
    """
    cbf = _cbf_term(env)
    robot = env.scene["robot"]

    base_lin_b = robot.data.root_lin_vel_b                  # (N, 3)
    base_ang_b = robot.data.root_ang_vel_b                  # (N, 3)
    gravity_b = robot.data.projected_gravity_b              # (N, 3)
    joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos     # (N, 12)
    joint_vel = robot.data.joint_vel                                         # (N, 12)
    last_jt = (cbf._processed_actions - robot.data.default_joint_pos) \
              / max(cbf._loco_action_scale, 1e-9)                            # (N, 12)
    return torch.cat([
        base_lin_b, base_ang_b, gravity_b,
        joint_pos_rel, joint_vel, last_jt,
    ], dim=-1)   # (N, 45)
