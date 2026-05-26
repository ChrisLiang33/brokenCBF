"""CBFAdaptiveGo2EnvCfg -- the manager-based env cfg for the CBF-adaptive
Go2 task.

Inherits scene + physics from `UnitreeGo2FlatEnvCfg` and then, INSIDE
`__post_init__()` (after the parent chain has run), replaces every MDP
manager (actions / rewards / terminations / observations) with our own.

Doing the replacement post-init avoids fighting the stock Go2 cfg's
`__post_init__`, which assumes the inherited `actions.joint_pos.scale`
exists, etc. We let the parent set itself up however it likes, then we
swap the MDP wholesale.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.mdp.terminations import time_out as time_out_fn
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

from . import mdp
from .cbf_action_term import CBFParamActionTermCfg


# ---------------------------------------------------------------------------
# Visualization helpers. Obstacles and goal exist as pure math in the CBF
# (analytic ray-cylinder lidar) -- there are no USD prims by default, so
# the play UI shows the robot walking through "invisible" obstacles. These
# helpers add cosmetic-only visual prims (no physics, no collision) at the
# nominal positions so the UI is interpretable. Per-episode position jitter
# applied by the action term is NOT reflected in the visuals -- they stay
# at the nominal config-time positions, which is close enough for "what is
# the policy doing" visualization.
# ---------------------------------------------------------------------------
def _make_obstacle_visual(prim_path: str, x: float, y: float, radius: float,
                          height: float = 1.0,
                          color: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, height / 2.0)),
    )


def _make_goal_visual(prim_path: str, x: float, y: float,
                      radius: float = 0.2,
                      color: tuple[float, float, float] = (0.1, 0.8, 0.1)) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.SphereCfg(
            radius=radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, 0.3)),
    )


# ---------------------------------------------------------------------------
# MDP manager cfgs (constructed inside __post_init__, not as class-level
# defaults, so the parent's __post_init__ chain doesn't trip on
# `self.actions.joint_pos.scale = ...`)
# ---------------------------------------------------------------------------
@configclass
class CBFActionsCfg:
    cbf_param: CBFParamActionTermCfg = CBFParamActionTermCfg()


@configclass
class CBFRewardsCfg:
    # Reward weights have been touched twice already; each iteration was
    # because PPO found a "collide cheaply, env auto-resets, try again"
    # local optimum. Phase 1 fixed it by dropping intervention 10×.
    # Phase 2 with per-episode disturbance DR re-opened that basin
    # (62-100% collision rates at higher disturbances). Bumping collision
    # 5× to -500 makes the per-collision penalty dominate the
    # per-episode-summed intervention (~50 with weight=0.1 at int=500),
    # so the safe-but-slightly-more-intervention basin clearly wins.
    progress = RewTerm(func=mdp.progress_reward, weight=1.0)
    # CURRICULUM step 2: anneal intervention back up from 0 to give the
    # policy a cost gradient again (without it the policy wanders -- 77%
    # reach at d=0 vs 100% for the trivial fixed baseline). -0.05 is half
    # the Phase 1 value so it tie-breaks between safe policies without
    # dominating the collision penalty.
    intervention = RewTerm(func=mdp.intervention_penalty, weight=-0.05)
    collision = RewTerm(func=mdp.collision_penalty, weight=-1000.0)
    goal_reached = RewTerm(func=mdp.goal_reached_bonus, weight=50.0,
                           params={"goal_tol": 0.4})
    # Stuck penalty: wired here at weight=0 (no behavior change). Flip to
    # e.g. -0.05 in an env-cfg post_init to discourage the 'freeze under
    # disturbance' degenerate optimum, where the policy avoids collision
    # by standing still and earns 0 from every other term. See
    # mdp.stuck_penalty for the per-step formula.
    stuck = RewTerm(func=mdp.stuck_penalty, weight=0.0,
                    params={"speed_thresh": 0.15, "goal_tol": 0.4})


@configclass
class CBFTerminationsCfg:
    time_out = DoneTerm(func=time_out_fn, time_out=True)
    collision = DoneTerm(func=mdp.collision_termination)
    goal_reached = DoneTerm(func=mdp.goal_reached_termination,
                            params={"goal_tol": 0.4})
    # fall detection via base height + orientation (replaces the failed
    # base_contact_termination attempt). Sets sticky `episode_fall_any`
    # on the action term so eval can distinguish "stuck or wandering"
    # from "actually fell over".
    fall = DoneTerm(func=mdp.fall_termination)


@configclass
class CBFObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        zeros = ObsTerm(func=mdp.zeroed_obs, params={"obs_dim": 7})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Env cfg
# ---------------------------------------------------------------------------
@configclass
class CBFAdaptiveGo2EnvCfg(UnitreeGo2FlatEnvCfg):
    """CBF-adaptive Go2 task. Outer policy emits (φ, α) every step."""

    episode_length_s: float = 25.0

    def __post_init__(self):
        # 1) run the stock Go2 init chain first -- builds scene, terrain,
        #    sim cfg, the (now-irrelevant-to-us) stock actions/rewards/etc.
        super().__post_init__()

        # 2) wholesale-replace the MDP managers
        self.actions = CBFActionsCfg()
        self.rewards = CBFRewardsCfg()
        self.terminations = CBFTerminationsCfg()
        self.observations = CBFObservationsCfg()

        # 3) pin friction at training-distribution default (the channel
        #    we sweep separately in Phase 0.6; held fixed during Phase 1)
        try:
            pm = self.events.physics_material
            pm.params["static_friction_range"] = (0.6, 0.6)
            pm.params["dynamic_friction_range"] = (0.6, 0.6)
        except AttributeError:
            pass

        # 4) spawn near origin facing the goal; minimal jitter so all envs
        #    have ~the same starting condition
        try:
            rb = self.events.reset_base
            rb.params["pose_range"] = {
                "x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.05, 0.05),
            }
            rb.params["velocity_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            }
        except AttributeError:
            pass


@configclass
class CBFAdaptiveGo2EnvCfg_PLAY(CBFAdaptiveGo2EnvCfg):
    """Single-env play variant."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5


# ---------------------------------------------------------------------------
# Phase 2 variant: deployable observation + per-episode disturbance DR.
# ---------------------------------------------------------------------------
@configclass
class CBFPhase2ObsCfg:
    """Phase 2 observation: 20-step history of the 48-dim deployable obs
    PLUS a single-scalar geometric stand-in for lidar-derived obstacle
    awareness. See `mdp.geometric_stand_in` for the swap-out plan.
    Total obs dim = 20*48 + 1 = 961.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        deployable = ObsTerm(func=mdp.deployable_obs, history_length=20)
        # SWAP-OUT MARKER: stand-in for lidar. Replace with lidar+CNN later.
        geometric = ObsTerm(func=mdp.geometric_stand_in)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CBFAdaptiveGo2Phase2EnvCfg(CBFAdaptiveGo2EnvCfg):
    """Phase 2: state-conditional policy.

    Differences from Phase 1:
    - Observation: 20-step windowed deployable obs (proprio + CBF-filtered
      cmd + last loco action), 48 * 20 = 960-dim flattened.
    - Disturbance: per-episode random magnitude in [0, 45] N (the range
      Phase 0.6 / 1.5 already covered). Gives the policy a real OOD
      distribution to adapt over.
    """
    def __post_init__(self):
        super().__post_init__()
        self.observations = CBFPhase2ObsCfg()
        # randomize disturbance per episode -- the OOD signal the policy
        # must adapt to
        self.actions.cbf_param.disturbance_force_range = (0.0, 45.0)


# ---------------------------------------------------------------------------
# RMA variant: privileged-info observation group for the teacher policy.
#
# Step 1 of the RMA build -- surface the existing per-env disturbance
# magnitude as a separate observation group named "priv". This is what the
# teacher's privileged encoder mu will map to z. The student's adaptation
# module phi(history) will be trained later to predict the same z from
# proprio. (Confusingly, RMA calls these "Phase 1 / Phase 2" too, distinct
# from our project's Phase 1-4.)
#
# `policy` group is left as the Phase 2 deployable window for now -- the
# actual swap to (state + prev_action + lidar_feature + z) happens once
# we land the custom actor-critic with branched encoders.
# ---------------------------------------------------------------------------
@configclass
class CBFRMAObsCfg:
    """Single flat obs group of size 43 -- `mdp.teacher_obs` concatenates
    [disturbance(1), state(4), prev_action(2), lidar(36)]. The actor-
    critic (RMAActorCritic) splits internally and routes the disturbance
    slice through z_enc; no other path lets priv info reach the action.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        obs = ObsTerm(func=mdp.teacher_obs)   # (N, 43)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CBFRMAClassicObsCfg:
    """RMA-classic obs cfg: 4 priv channels (RMA-paper canonical) instead
    of v7's 7. Wraps `mdp.teacher_obs_rma_classic` (195-dim)."""

    @configclass
    class PolicyCfg(ObsGroup):
        obs = ObsTerm(func=mdp.teacher_obs_rma_classic)  # (N, 195)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CBFRMAHistoryObsCfg:
    """RMAHistory obs cfg: no priv, proprio as 50-step history.
    Wraps `mdp.teacher_obs_history` (2396-dim for K=50)."""

    @configclass
    class PolicyCfg(ObsGroup):
        obs = ObsTerm(func=mdp.teacher_obs_history)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CBFAdaptiveGo2RMAEnvCfg(CBFAdaptiveGo2Phase2EnvCfg):
    """RMA-style two-stage training env. Adds the privileged observation
    group. The "lidar" the policy will see is computed analytically
    (ray-cylinder intersection against the action term's cached obstacle
    positions) -- see `mdp.lidar_obs`. We considered Isaac Lab's warp
    raycaster but its `mesh_prim_paths` doesn't accept the `env_.*` regex
    for per-env-replicated obstacles, and exact paths would only see one
    env's obstacle. Analytic lidar sidesteps that and is multi-obstacle
    friendly out of the box.
    """
    def __post_init__(self):
        super().__post_init__()
        self.observations = CBFRMAObsCfg()

        # turn on RMA privileged-factor DR. Ranges chosen to roughly
        # match the RMA paper (Kumar 2021), proportional to typical
        # quadruped operating ranges:
        # - friction: substantial spread around the locked-down 0.6
        # - base mass: +/- a couple of kg around default base mass
        # - motor strength: +/- 20% of nominal
        at = self.actions.cbf_param
        at.friction_range = (0.3, 1.0)
        at.base_mass_range = (-2.0, 2.0)
        at.motor_strength_range = (0.8, 1.2)

        # action-smoothness penalty. Reduced from -0.5 -> -0.2 since the
        # heavier weight pushed the policy into a too-constant solution
        # (jitter dropped to ~0.04 but safety regressed at high d). -0.2
        # still curbs the action_std=0.92 jitter ceiling without locking
        # action variation completely.
        self.rewards.action_smoothness = RewTerm(
            func=mdp.action_smoothness_penalty, weight=-0.2,
        )
        # EXPERIMENTAL: intervention=0 for "confirm modulation" retrain.
        # Phase 6 fixed-param sweep (2026-05-24) showed signal exists --
        # optimal (phi, alpha) moves with obstacle distance at d=0/15/30.
        # But every teacher we've trained pegs at (max, max). Hypothesis:
        # even at intervention=-0.3, the policy can't escape the (max,
        # max) basin because collision (-1000) dominates the early-train
        # gradient. Set intervention=0 to remove ALL pressure toward
        # (max, max), retrain, check if the policy MODULATES at all.
        # If yes -> anneal intervention back up. If still pegs -> aux
        # obstacle-prediction loss on lidar_feat.
        # REVERT to weight=-0.3 if the experiment doesn't pan out --
        # other people train against this cfg (see CLAUDE memory
        # feedback_shared_repo_reverts).
        self.rewards.intervention = RewTerm(
            func=mdp.intervention_penalty, weight=0.0,
        )


# ---------------------------------------------------------------------------
# Phase 6 sibling: per-episode random obstacle position. The lidar
# attention diagnostic showed our learned policy didn't actually use the
# lidar slice -- it inferred obstacle proximity from proprio (dist_to_goal +
# vel_cmd + prev_action) since the obstacle position was fixed across
# every env and every episode. Randomizing position decorrelates obstacle
# location from proprio, forcing the policy to read lidar to localize it.
# ---------------------------------------------------------------------------
@configclass
class CBFAdaptiveGo2RandObsEnvCfg(CBFAdaptiveGo2RMAEnvCfg):
    """RMA env with per-episode random obstacle position jitter. Same
    priv/lidar/reward shape as RMA, just the obstacle's xy is uniformly
    re-sampled per env per reset within `obstacle_pos_jitter_range`.
    """
    def __post_init__(self):
        super().__post_init__()
        # +/- 1.5m box around nominal (2.5, 0.3). Combined with
        # symmetric jitter range, the obstacle spans roughly x in
        # [1.0, 4.0] and y in [-1.2, +1.8] -- enough lateral spread
        # that "directly between robot and goal" is no longer a
        # reliable assumption.
        self.actions.cbf_param.obstacle_pos_jitter_range = (-1.5, 1.5)


# ---------------------------------------------------------------------------
# Phase 6 variant: RMA + slalom obstacles. Combines the RMA env's DR +
# branched-encoder + lidar obs with Phase3's 3-obstacle slalom layout.
# Motivation: the fixed-(phi,alpha) sweep (2026-05-24) showed the
# adaptation signal on single-obstacle RandObs is real but small (~1.3%
# vs 1.0% progress gaps across bins). Slalom geometry provides much
# stronger per-segment differentiation (left-pinch / right-pinch / left-
# pinch sequence) so the optimal (phi, alpha) should genuinely move per
# step, not just per episode. Combined with intervention=0 (set on the
# RMA parent), gives the cleanest possible "does the policy modulate"
# test.
# ---------------------------------------------------------------------------
@configclass
class CBFAdaptiveGo2SlalomEnvCfg(CBFAdaptiveGo2RMAEnvCfg):
    """RMA env with Phase3's 3-obstacle slalom and small per-episode
    obstacle jitter (to keep the obstacle-localization leak plugged --
    fixed positions let the policy infer location from proprio alone).
    """
    def __post_init__(self):
        super().__post_init__()
        at = self.actions.cbf_param
        # 3-obstacle slalom; tighter radius (0.5m) than the default 0.9m
        # since obstacles are closer together. Goal at (7, 0) gives room
        # to traverse all three.
        at.obstacles = [
            (2.0,  0.5, 0.5),
            (4.0, -0.5, 0.5),
            (5.5,  0.5, 0.5),
        ]
        at.goal_xy = (7.0, 0.0)
        # small per-episode jitter on each obstacle's xy. Keeps the
        # leak-plug from RandObs (proprio can no longer perfectly
        # localize obstacles) without destroying the slalom geometry.
        at.obstacle_pos_jitter_range = (-0.3, 0.3)
        # cosmetic visual markers (no physics, no collision) so the UI
        # shows where the obstacles + goal are.
        for i, (x, y, r) in enumerate(at.obstacles):
            setattr(self.scene, f"obstacle_visual_{i}",
                    _make_obstacle_visual(f"{{ENV_REGEX_NS}}/ObstacleVis{i}", x, y, r))
        self.scene.goal_visual = _make_goal_visual(
            "{ENV_REGEX_NS}/GoalVis", at.goal_xy[0], at.goal_xy[1])


# ---------------------------------------------------------------------------
# B.3 UNIFIED teacher env. Slalom geometry + all priv-channel DR active
# (both z-driving channels and lidar-driving setup). Goal: train a single
# teacher that uses BOTH z and lidar for adaptation (the slalom teacher
# uses z; the decorr teacher uses lidar; neither does both).
#
# Disturbance capped at 30N -- at d=45 the frozen locomotion fails
# regardless of CBF policy (43% fall rate observed), so we exclude it
# from the training distribution.
# ---------------------------------------------------------------------------
@configclass
class CBFAdaptiveGo2UnifiedEnvCfg(CBFAdaptiveGo2SlalomEnvCfg):
    """Unified teacher env: slalom geometry + intervention=0 (inherited)
    + all validated/candidate DR channels active simultaneously."""
    def __post_init__(self):
        super().__post_init__()
        at = self.actions.cbf_param
        # validated channels (kept from slalom + RMA parents, made
        # explicit here for clarity)
        at.friction_range = (0.3, 1.0)
        at.base_mass_range = (-3.0, 3.0)
        at.motor_strength_range = (0.7, 1.3)
        # disturbance kept but capped at 30N (45N collapses locomotion)
        at.disturbance_force_range = (0.0, 30.0)
        # B.1 new channels
        at.actuation_noise_std_range = (0.0, 0.05)
        at.com_offset_range = (-0.05, 0.05)
        # B.6 v_max DR -- THE validated α channel (92% bound span per
        # phase6_vmax_gate). Higher v_max -> more momentum -> need
        # smaller α (start braking earlier). Now exposed in priv_obs
        # slot 6.
        at.v_max_range = (1.0, 2.0)
        # slightly bigger obstacle jitter than vanilla Slalom (0.3) to
        # give the lidar more meaningful per-episode variation
        at.obstacle_pos_jitter_range = (-0.5, 0.5)


# ---------------------------------------------------------------------------
# B.4 SIM2REAL teacher env (SHIELD-aligned, Yang et al. 2025). Same as
# Unified BUT the CBF uses obstacle positions corrupted by per-step
# Gaussian noise (~5cm) to simulate the accuracy of a real Livox
# Mid-360 + Euclidean clustering + cylinder-fit pipeline. The CBF math
# is identical to privileged -- only the obstacle position estimates
# are noisy.
#
# Trade-offs vs Unified (privileged-SDF teacher):
#   - More realistic: CBF can be wrong, collisions possible
#   - Forces the policy to be robust to perception noise (drives real
#     lidar use, not just goal-proxy)
#   - Slightly harder to train (noise compounds)
# ---------------------------------------------------------------------------
@configclass
class CBFAdaptiveGo2UnifiedLidarSDFEnvCfg(CBFAdaptiveGo2UnifiedEnvCfg):
    """Unified teacher + perception-SDF (SHIELD-aligned). 5cm position
    noise. Lipschitz rate-limit on (phi, alpha) for hardware safety --
    real-robot actuators can't track step changes."""
    def __post_init__(self):
        super().__post_init__()
        at = self.actions.cbf_param
        at.use_lidar_sdf = True
        at.perception_noise_std = 0.05   # meters
        # B.5 hardware-safety: bound per-step change in normalized
        # action to 0.05 -> L_a = 2.5/s on the [-1, 1] action (dt=0.02s).
        # Decoded (phi, alpha) inherit Lipschitz via the linear-decode
        # slopes: dphi/da = 0.5*(phi_hi-phi_lo) = 0.5*1.0 = 0.5,
        # dalpha/da = 0.5*(alpha_hi-alpha_lo) = 0.5*3.8 = 1.9. So:
        #   L_phi   = L_a * 0.5 = 1.25 phi-units/sec
        #   L_alpha = L_a * 1.9 = 4.75 alpha-units/sec
        # Strict bound, not a soft filter -- CBF input is guaranteed
        # continuous in time.
        at.action_max_step = 0.05


@configclass
class CBFAdaptiveGo2RMAStaticEnvCfg(CBFAdaptiveGo2UnifiedLidarSDFEnvCfg):
    """RMA-classic teacher env: 4 priv channels + random obstacle
    topology (static within episode) + everything else from SHIELD
    (Lipschitz, Mid-360 lidar fidelity, perception SDF, stuck penalty
    NOT active by default -- v7.1's stuck=-0.05 belongs to a different
    experiment).

    Differences vs SHIELD parent:
      - obs cfg: 4-channel priv (CBFRMAClassicObsCfg)
      - DR: drop v_max_range, actuation_noise_range, com_offset_range
            (the 3 channels v7.1 ignored per [[v7_priv_channel_dead]])
      - obstacles: random_topology=True with K=3 in corridor
        x in [1.5, 6.0], y in [-1.5, 1.5]
      - intervention=0 inherited from RMAEnvCfg
      - perception SDF + Lipschitz inherited from SHIELD parent

    Pairs with: RMAClassicMLPModel + CBFAdaptiveGo2RMAStaticRunnerCfg.
    """
    def __post_init__(self):
        super().__post_init__()
        # 4-channel obs cfg replaces SHIELD's 7-channel
        self.observations = CBFRMAClassicObsCfg()
        at = self.actions.cbf_param
        # drop the 3 v7 experimental DR channels; keep the 4 RMA-canonical
        at.v_max_range = None
        at.actuation_noise_std_range = None
        at.com_offset_range = None
        # random topology (replaces the slalom + jitter inherited from Slalom)
        at.random_topology = True
        at.random_topology_K = 3
        at.random_topology_x_range = (1.5, 6.0)
        at.random_topology_y_range = (-1.5, 1.5)
        at.random_topology_start_exclusion_r = 0.8
        at.random_topology_goal_exclusion_r = 0.8
        at.random_topology_min_separation = 1.4
        at.random_topology_max_attempts = 30
        # The inherited Slalom layout's `at.obstacles = [...]` stays as
        # the FALLBACK for envs that fail rejection sampling. K obstacles
        # with radius matching Slalom (0.5 m) so the action term's
        # `_n_obstacles` and `_r_safe` shapes are unchanged at init.


@configclass
class CBFAdaptiveGo2RMAStaticNoPrivEnvCfg(CBFAdaptiveGo2RMAStaticEnvCfg):
    """RMA-Static ablation: train with priv slot ZEROED (lidar + proprio
    only). Tests: can the policy learn safety + reach with NO privileged
    info? If yes, the RMA premise was load-bearing in name only and we
    can drop the priv pipeline. If no, priv was necessary (just not
    used the way we thought)."""
    def __post_init__(self):
        super().__post_init__()
        self.actions.cbf_param.obs_mask_priv = True


@configclass
class CBFAdaptiveGo2RMAHistoryEnvCfg(CBFAdaptiveGo2RMAStaticEnvCfg):
    """RMAHistory teacher env: no priv obs, proprio fed as 50-step
    history through a 1D CNN. Same physics + DR + random topology as
    RMAStatic; only the obs cfg changes."""
    def __post_init__(self):
        super().__post_init__()
        self.observations = CBFRMAHistoryObsCfg()
        self.actions.cbf_param.proprio_history_length = 50
        # priv masking is irrelevant -- teacher_obs_history doesn't
        # include priv in the obs at all. Leave the masking flags as-is.


@configclass
class CBFAdaptiveGo2RMAStaticNoProprioEnvCfg(CBFAdaptiveGo2RMAStaticEnvCfg):
    """RMA-Static ablation: train with proprio slot ZEROED (lidar + priv
    z only). Tests: if proprio is unavailable, does PPO start using
    priv via z? If z's R^2 jumps high here, priv WAS informative but
    proprio was a stronger signal that crowded it out. If z still dead
    here, priv is genuinely uninformative for the policy."""
    def __post_init__(self):
        super().__post_init__()
        self.actions.cbf_param.obs_mask_proprio = True


@configclass
class CBFAdaptiveGo2UnifiedLidarSDFStuckEnvCfg(CBFAdaptiveGo2UnifiedLidarSDFEnvCfg):
    """SHIELD env + active stuck penalty (v7.1 experimental variant).

    Motivation: v7 cross-scene eval (E1/E3/E4) showed the teacher gives up
    too much reach -- "max-hedge + bail" pattern. Reach on E1 SINGLE was
    40%, E3 DENSE 4%, E4 NARROW 12%. Hypothesis: the policy is exploiting
    the lack of a stuck-cost gradient -- standing still earns 0 from every
    reward term, which is a stable local optimum at the cost of timeout.

    Fix: flip the stuck_penalty weight (wired at 0 in CBFRewardsCfg) to
    -0.05. Per-step penalty when slow (<0.15 m/s) AND not at goal. Should
    tilt the basin toward forward motion without making the policy
    suicidal (collision is still -1000).

    Revert path: if v7.1 underperforms v7 on safety (>10% delta on
    worst_coll), drop this env and the registered task entry. The base
    SHIELD env is untouched.
    """
    def __post_init__(self):
        super().__post_init__()
        # Only need to flip the weight; the term is already wired in
        # CBFRewardsCfg with the right params (speed_thresh=0.15, goal_tol=0.4).
        self.rewards.stuck.weight = -0.05


# ---------------------------------------------------------------------------
# B.1 GATE envs. Each is Slalom geometry + intervention=0 (inherited
# from RMA parent) but with ONLY ONE DR channel active at a time, so a
# fixed-(phi, alpha) sweep can tell whether optimal (phi, alpha) shifts
# with that channel. Only channels that pass the gate (>10% bound span)
# get added to the unified teacher's priv obs.
# ---------------------------------------------------------------------------
@configclass
class CBFAdaptiveGo2ActNoiseGateEnvCfg(CBFAdaptiveGo2SlalomEnvCfg):
    """Slalom + only `actuation_noise_std` DR active. Other DR off so
    the gate sweep isolates this channel's effect."""
    def __post_init__(self):
        super().__post_init__()
        at = self.actions.cbf_param
        # Disable all other DR
        at.friction_range = (0.6, 0.6)
        at.base_mass_range = (0.0, 0.0)
        at.motor_strength_range = (1.0, 1.0)
        at.disturbance_force_range = (0.0, 0.0)
        at.com_offset_range = (0.0, 0.0)
        # Only this channel is active
        at.actuation_noise_std_range = (0.0, 0.08)


@configclass
class CBFAdaptiveGo2ComOffsetGateEnvCfg(CBFAdaptiveGo2SlalomEnvCfg):
    """Slalom + only `com_offset` DR active."""
    def __post_init__(self):
        super().__post_init__()
        at = self.actions.cbf_param
        at.friction_range = (0.6, 0.6)
        at.base_mass_range = (0.0, 0.0)
        at.motor_strength_range = (1.0, 1.0)
        at.disturbance_force_range = (0.0, 0.0)
        at.actuation_noise_std_range = (0.0, 0.0)
        # Only this channel is active. +/- 5cm in body x.
        at.com_offset_range = (-0.05, 0.05)


# ---------------------------------------------------------------------------
# Phase 6 variant: DECORRELATION TEST env. Single obstacle randomly
# placed in a wide 2D region that includes positions PAST the goal and
# OFF to the side of the straight start->goal line, so dist_to_goal can
# no longer proxy obstacle distance (the structural correlation that
# slalom has). Used only for *evaluation* of a slalom-trained teacher
# to test whether its alpha modulation is driven by lidar or by the
# goal-distance proxy. NOT a training env.
# ---------------------------------------------------------------------------
@configclass
class CBFAdaptiveGo2DecorrEnvCfg(CBFAdaptiveGo2RMAEnvCfg):
    """RMA env, single obstacle uniformly jittered across a wide 2D
    region (~ +/-4m x, +/-3m y around (4, 0)). With goal fixed at (7, 0),
    obstacles can fall before, beside, or beyond the goal, decorrelating
    obstacle distance from goal distance.
    """
    def __post_init__(self):
        super().__post_init__()
        at = self.actions.cbf_param
        # one obstacle, nominal at the midpoint of start->goal line
        at.obstacles = [(4.0, 0.0, 0.5)]
        at.goal_xy = (7.0, 0.0)
        # WIDE jitter -- +/-4m in x and y. Both axes use the same range
        # in the current cfg (single scalar range, applied independently
        # to x and y per obstacle).
        at.obstacle_pos_jitter_range = (-4.0, 4.0)
        # cosmetic visual markers. Note: position jitter is wide here, so
        # the visual marker (at nominal position) will often NOT be where
        # the actual obstacle is during a given episode. The marker just
        # shows the *nominal* obstacle location.
        for i, (x, y, r) in enumerate(at.obstacles):
            setattr(self.scene, f"obstacle_visual_{i}",
                    _make_obstacle_visual(f"{{ENV_REGEX_NS}}/ObstacleVis{i}", x, y, r))
        self.scene.goal_visual = _make_goal_visual(
            "{ENV_REGEX_NS}/GoalVis", at.goal_xy[0], at.goal_xy[1])


# ---------------------------------------------------------------------------
# Phase 6 variant: v_max DR + motor_strength DR. Two validated
# adaptation channels paired to their theoretical CBF role:
#   alpha <- v_max  (kinematic urgency, validated phase6_vmax_gate at 92% bound)
#   phi   <- motor_strength  (control effectiveness, validated per-channel sweep)
# All other DR off -- isolated, clean training signal.
# ---------------------------------------------------------------------------
@configclass
class CBFAdaptiveGo2VmaxEnvCfg(CBFAdaptiveGo2RMAEnvCfg):
    """Adaptive CBF teacher env with two validated channels:
       v_max in [0.5, 2.0] (alpha signal), motor_strength in [0.8, 1.2]
       (phi signal). Disturbance, friction, mass DR turned off.
    """
    def __post_init__(self):
        super().__post_init__()
        at = self.actions.cbf_param
        # alpha channel: per-episode v_max DR (validated 92% span)
        at.v_max_range = (0.5, 2.0)
        # phi channel: keep motor_strength DR (validated 26% on prev teacher)
        at.motor_strength_range = (0.8, 1.2)
        # turn off the dead channels so they don't dilute training signal
        at.disturbance_force_range = (0.0, 0.0)
        at.friction_range = (0.6, 0.6)
        at.base_mass_range = (0.0, 0.0)


# ---------------------------------------------------------------------------
# Phase 6 variant: rough terrain. Replaces flat ground with a procedural
# heightfield generator so locomotion tracking error scales with roughness.
# This is the α signal the project has been missing: optimal α should
# DECREASE on rough terrain (lower class-K = more conservative recovery
# when the body is lagging the command).
#
# Use this env for the strong α gate: instantiate at increasing
# terrain_level and verify optimal fixed α actually moves with level.
# DR ranges are slimmed (no disturbance, no friction DR by default) so
# the gate isolates terrain as the only varying channel.
# ---------------------------------------------------------------------------
@configclass
class CBFAdaptiveGo2RoughEnvCfg(CBFAdaptiveGo2RMAEnvCfg):
    """RMA env on procedural rough terrain. Configure roughness via
    `terrain_level` (0=flat, 4=heavy bumps + slopes). For the α gate we
    pin disturbance/friction/mass/motor at nominal and vary terrain only.
    """
    terrain_level: int = 0

    def __post_init__(self):
        super().__post_init__()
        # disable all the other DR channels so the gate isolates terrain
        at = self.actions.cbf_param
        at.disturbance_force_range = (0.0, 0.0)
        at.friction_range = (0.6, 0.6)
        at.base_mass_range = (0.0, 0.0)
        at.motor_strength_range = (1.0, 1.0)
        # also drop the smoothness penalty during gate -- we want raw
        # optimum per cell, not a smoothness-coupled compromise
        if hasattr(self.rewards, "action_smoothness"):
            self.rewards.action_smoothness = None

        # swap flat ground for procedural rough terrain
        from isaaclab.terrains import TerrainImporterCfg
        import isaaclab.sim as sim_utils
        from .terrain_helpers import rough_terrain_generator

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=rough_terrain_generator(self.terrain_level),
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )


# ---------------------------------------------------------------------------
# Phase 3 variant: multi-obstacle slalom.
#
# Phase 2's single-static-obstacle + isotropic-disturbance scenario does
# NOT reward state-conditional adaptation -- the optimal fixed (φ, α)
# wins everywhere. To extract real adaptation signal, we need a scenario
# where the optimal (φ, α) genuinely varies with state the policy can
# perceive. A multi-obstacle slalom provides that: different segments of
# the path (open stretch vs. near-miss vs. pinch) have different optimal
# margins, and the policy can read which segment it's in from the
# obstacle-distance sensor (current h or eventually lidar).
# ---------------------------------------------------------------------------
@configclass
class CBFAdaptiveGo2Phase3EnvCfg(CBFAdaptiveGo2Phase2EnvCfg):
    """Phase 3: multi-obstacle slalom + smoothed multi-obstacle SDF CBF.

    Uses the generalized SDF h(x) = min_i (||x - p_i|| - R_i) with
    smoothing h_smooth = lambda * (1 - exp(-gamma * h)) to keep the
    constraint differentiable at obstacle-switch points.
    """
    def __post_init__(self):
        super().__post_init__()
        at = self.actions.cbf_param
        # 3-obstacle slalom along the start->goal path; each pair forces a
        # different lateral deflection, so the optimal (φ, α) on one
        # segment is not optimal on another.
        at.obstacles = [
            (2.0,  0.5, 0.5),   # near-miss right
            (4.0, -0.5, 0.5),   # near-miss left
            (5.5,  0.5, 0.5),   # near-miss right (second time)
        ]
        # goal moved farther to give the policy room to traverse all three
        at.goal_xy = (7.0, 0.0)
        # SDF smoothing knobs (paper defaults: lambda=1, gamma=2)
        at.h_smooth_lambda = 1.0
        at.h_smooth_gamma = 2.0


# ---------------------------------------------------------------------------
# Held-out EVAL SCENES E0-E4. All inherit the deployment-target SHIELD
# env (full DR + Mid-360 lidar + Lipschitz rate-limit) and only override
# the obstacle layout + goal + jitter range. Used by
# phase6_eval_scenes.py to cross-test a teacher checkpoint and the
# train-tuned baselines on geometries the policy has NOT trained on.
#
# Visual cleanup: each subclass first nulls out the parent slalom's
# obstacle_visual_{0..N} prims, then re-adds visuals for its own
# obstacles. Headless eval doesn't care about visuals but play mode
# would show stale slalom cylinders otherwise.
# ---------------------------------------------------------------------------
def _reset_obstacle_visuals(scene, max_n: int = 10) -> None:
    """Drop any obstacle_visual_{i} attrs the parent chain added."""
    for i in range(max_n):
        attr = f"obstacle_visual_{i}"
        if hasattr(scene, attr):
            setattr(scene, attr, None)


def _apply_eval_scene(self, obstacles: list[tuple[float, float, float]],
                       goal_xy: tuple[float, float],
                       jitter: tuple[float, float] | None) -> None:
    """Set obstacles + goal + jitter on this cfg and refresh visuals.
    Keeps every other knob (DR, lidar fidelity, Lipschitz) inherited from
    the SHIELD parent untouched."""
    at = self.actions.cbf_param
    at.obstacles = list(obstacles)
    at.goal_xy = tuple(goal_xy)
    at.obstacle_pos_jitter_range = jitter
    _reset_obstacle_visuals(self.scene)
    for i, (x, y, r) in enumerate(obstacles):
        setattr(self.scene, f"obstacle_visual_{i}",
                _make_obstacle_visual(f"{{ENV_REGEX_NS}}/ObstacleVis{i}", x, y, r))
    self.scene.goal_visual = _make_goal_visual(
        "{ENV_REGEX_NS}/GoalVis", goal_xy[0], goal_xy[1])


# NOTE: E0 (empty scene) was dropped 2026-05-24. The CBF action term's
# tensor init (cbf_action_term.py:117) crashes on `len(obstacles)==0`
# because `torch.tensor([(o[0], o[1]) for o in []])` returns shape (0,)
# instead of (0, 2), and the SDF/lidar paths assume K>=1. Sanity-floor
# value didn't justify the multi-line action-term refactor; B-trivial
# (phi=0, alpha=2.5) in the comparison serves as the no-tuning baseline.


@configclass
class CBFAdaptiveGo2EvalE1EnvCfg(CBFAdaptiveGo2UnifiedLidarSDFEnvCfg):
    """E1 SINGLE -- one cylinder centered on the start->goal axis. Forces
    a single swerve; tests basic CBF avoidance isolated from multi-
    obstacle SDF effects (the parent SDF is min over obstacles, which
    reduces to a single term here)."""
    def __post_init__(self):
        super().__post_init__()
        _apply_eval_scene(self,
            obstacles=[(4.0, 0.0, 0.7)],
            goal_xy=(7.0, 0.0),
            jitter=(-0.5, 0.5))


@configclass
class CBFAdaptiveGo2EvalE2EnvCfg(CBFAdaptiveGo2UnifiedLidarSDFEnvCfg):
    """E2 SLALOM -- the training geometry. In-distribution regression
    check: teacher should match its training-time numbers here. If E2
    underperforms training metrics, look for an eval pipeline bug
    BEFORE concluding anything about generalization."""
    def __post_init__(self):
        super().__post_init__()
        _apply_eval_scene(self,
            obstacles=[(2.0, 0.5, 0.5), (4.0, -0.5, 0.5), (5.5, 0.5, 0.5)],
            goal_xy=(7.0, 0.0),
            jitter=(-0.5, 0.5))


@configclass
class CBFAdaptiveGo2EvalE3EnvCfg(CBFAdaptiveGo2UnifiedLidarSDFEnvCfg):
    """E3 DENSE FIELD -- 5 obstacles staggered across a 3m x 2.4m corridor
    between start and goal. Tests multi-obstacle SDF blending and forces
    the policy to commit to a lateral lane vs zig-zag. Hardest scene by
    expected collision rate."""
    def __post_init__(self):
        super().__post_init__()
        _apply_eval_scene(self,
            obstacles=[
                (2.5,  0.8, 0.5),
                (2.5, -0.8, 0.5),
                (4.0,  0.0, 0.5),
                (5.5,  0.8, 0.5),
                (5.5, -0.8, 0.5),
            ],
            goal_xy=(7.0, 0.0),
            jitter=(-0.5, 0.5))


@configclass
class CBFAdaptiveGo2EvalE4EnvCfg(CBFAdaptiveGo2UnifiedLidarSDFEnvCfg):
    """E4 NARROW GAP -- two cylinders forming a ~1.2m corridor at x=3.5.
    Robot is 0.32m wide so the corridor is ~4x robot width. Tests tight-
    tolerance navigation and phi-hedging under proximity (alpha alone
    can't safely thread the gap if the SDF gradient is noisy)."""
    def __post_init__(self):
        super().__post_init__()
        _apply_eval_scene(self,
            obstacles=[(3.5, -0.6, 0.5), (3.5, 0.6, 0.5)],
            goal_xy=(7.0, 0.0),
            jitter=(-0.5, 0.5))


# ---------------------------------------------------------------------------
# NoPriv variants of E1-E4 for evaluating the lidar+proprio teacher.
# Same geometry as their 7-priv counterparts, but inherit from
# CBFAdaptiveGo2RMAStaticNoPrivEnvCfg so obs_dim=195 (matches the NoPriv
# teacher) and random_topology is OFF (replaced by the fixed eval scene
# layout). Used by phase6_eval_scenes.py with --scene_prefix EvalNoPriv.
# ---------------------------------------------------------------------------
def _apply_eval_scene_static(self, obstacles, goal_xy, jitter):
    """Same as _apply_eval_scene but ALSO turns off random_topology so the
    fixed obstacles win over per-reset random sampling."""
    self.actions.cbf_param.random_topology = False
    _apply_eval_scene(self, obstacles=obstacles, goal_xy=goal_xy, jitter=jitter)


@configclass
class CBFAdaptiveGo2EvalNoPrivE1EnvCfg(CBFAdaptiveGo2RMAStaticNoPrivEnvCfg):
    """NoPriv E1 SINGLE -- one cylinder. 195-dim obs (priv masked)."""
    def __post_init__(self):
        super().__post_init__()
        _apply_eval_scene_static(self,
            obstacles=[(4.0, 0.0, 0.7)],
            goal_xy=(7.0, 0.0),
            jitter=(-0.5, 0.5))


@configclass
class CBFAdaptiveGo2EvalNoPrivE2EnvCfg(CBFAdaptiveGo2RMAStaticNoPrivEnvCfg):
    """NoPriv E2 SLALOM -- training-geometry regression check."""
    def __post_init__(self):
        super().__post_init__()
        _apply_eval_scene_static(self,
            obstacles=[(2.0, 0.5, 0.5), (4.0, -0.5, 0.5), (5.5, 0.5, 0.5)],
            goal_xy=(7.0, 0.0),
            jitter=(-0.5, 0.5))


@configclass
class CBFAdaptiveGo2EvalNoPrivE3EnvCfg(CBFAdaptiveGo2RMAStaticNoPrivEnvCfg):
    """NoPriv E3 DENSE FIELD -- 5 obstacles."""
    def __post_init__(self):
        super().__post_init__()
        _apply_eval_scene_static(self,
            obstacles=[
                (2.5,  0.8, 0.5),
                (2.5, -0.8, 0.5),
                (4.0,  0.0, 0.5),
                (5.5,  0.8, 0.5),
                (5.5, -0.8, 0.5),
            ],
            goal_xy=(7.0, 0.0),
            jitter=(-0.5, 0.5))


@configclass
class CBFAdaptiveGo2EvalNoPrivE4EnvCfg(CBFAdaptiveGo2RMAStaticNoPrivEnvCfg):
    """NoPriv E4 NARROW GAP."""
    def __post_init__(self):
        super().__post_init__()
        _apply_eval_scene_static(self,
            obstacles=[(3.5, -0.6, 0.5), (3.5, 0.6, 0.5)],
            goal_xy=(7.0, 0.0),
            jitter=(-0.5, 0.5))


# ===========================================================================
# Phase 10 / V2 — UNIFIED COMPARISON EXPERIMENT
# ---------------------------------------------------------------------------
# All 5 architectures train on the SAME training distribution so the
# delta between architectures is the only variable. Distribution:
#   - K=3 random obstacles per episode (random_topology, fixed r=0.5)
#   - 7 DR channels active simultaneously: friction, motor_strength,
#     v_max, base_mass, com_offset, actuation_noise, disturbance
#   - Clumsy-human u_nom (OU lateral + speed wobble + adversarial swerve)
#   - Rewards: progress(+1), goal(+100), fall(-100), stuck(-0.05),
#     collision(-1000), action_smoothness(-0.2), intervention(0 OR -0.05)
#   - SHIELD-aligned perception SDF + Mid-360 lidar fidelity + Lipschitz
#     rate-limit (inherited from UnifiedLidarSDF parent)
# ===========================================================================


@configclass
class CBFV2RewardsCfg:
    """Phase 10 / V2 reward shape (per user spec):
      - progress(+1): per-step Δdist_to_goal
      - goal(+100): one-shot on reach
      - collision(-1000): one-shot
      - fall(-100): one-shot (NEW; distinct from collision)
      - stuck(-0.05): per-step when slow + not at goal
      - action_smoothness(-0.2): step-to-step (φ, α) jitter
      - intervention(0 or -0.05): swept across the two variants
    """
    progress = RewTerm(func=mdp.progress_reward, weight=1.0)
    goal_reached = RewTerm(func=mdp.goal_reached_bonus, weight=100.0,
                           params={"goal_tol": 0.4})
    collision = RewTerm(func=mdp.collision_penalty, weight=-1000.0)
    fall = RewTerm(func=mdp.fall_penalty, weight=-100.0)
    stuck = RewTerm(func=mdp.stuck_penalty, weight=-0.05,
                    params={"speed_thresh": 0.15, "goal_tol": 0.4})
    action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=-0.2)
    intervention = RewTerm(func=mdp.intervention_penalty, weight=0.0)


@configclass
class CBFAdaptiveGo2UnifiedV2EnvCfg(CBFAdaptiveGo2UnifiedLidarSDFEnvCfg):
    """Phase 10 / V2 training distribution -- Full 7-priv obs layout.
    Base class for the NoPriv / NoProprio masking variants (same obs
    shape; masking flips a flag on the action term). RMA-classic and
    History have their own env cfgs because their obs layouts differ
    structurally (priv_dim=4 and proprio-history respectively).

    Training distribution:
      - K=3 random obstacles per episode (random_topology=True)
      - 7 DR channels active: friction, motor_strength, v_max, mass,
        com_offset, actuation_noise, disturbance (all per-episode)
      - Clumsy-human u_nom (OU wobble + adversarial swerve)
      - V2 rewards: goal=+100, fall=-100, stuck=-0.05, intervention=0
        (override to -0.05 via env_cfg.rewards.intervention.weight to
        train the alternate variant)
      - SHIELD perception SDF + Mid-360 lidar + Lipschitz inherited
    """
    def __post_init__(self):
        super().__post_init__()
        # V2 reward shape (overwrites all parent reward terms in one shot)
        self.rewards = CBFV2RewardsCfg()
        at = self.actions.cbf_param
        # all 7 DR channels active simultaneously
        at.friction_range = (0.3, 1.0)
        at.motor_strength_range = (0.7, 1.3)
        at.v_max_range = (1.0, 2.0)
        at.base_mass_range = (-3.0, 3.0)
        at.com_offset_range = (-0.05, 0.05)
        at.actuation_noise_std_range = (0.0, 0.05)
        at.disturbance_force_range = (0.0, 30.0)
        # random K=3 topology per episode (fixed r=0.5)
        at.random_topology = True
        at.random_topology_K = 3
        at.random_topology_x_range = (1.5, 6.0)
        at.random_topology_y_range = (-1.5, 1.5)
        at.random_topology_start_exclusion_r = 0.8
        at.random_topology_goal_exclusion_r = 0.8
        at.random_topology_min_separation = 1.4
        at.random_topology_max_attempts = 30
        # the inherited Slalom layout becomes the FALLBACK if rejection
        # sampling fails (rare in practice for the configured corridor)
        # clumsy-human u_nom (active in BOTH train and eval).
        # TRAINING uses the "child" preset -- bigger wobble, more swerves,
        # slower base speed -> harder distribution. Eval scenes override
        # to "teen" so the policy is tested against the (still imperfect)
        # operator the deployment system is meant to assist.
        at.unom_mode = "clumsy_human"
        at.unom_clumsiness = "child"
        # rate-limit u_nom_w so the CBF reference is smooth (no jitter
        # from OU sample-to-sample jumps or instantaneous swerve onsets).
        # 0.15 m/s/step at dt=0.02s -> 7.5 m/s² accel cap -> swerves can
        # still develop in ~0.5 s but no per-step spikes propagate.
        at.unom_max_step = 0.15
        # give the policy 5 s to wiggle out before tagging it stuck
        at.stuck_threshold_steps = 250
        # masking flags off by default; subclasses flip them
        at.obs_mask_priv = False
        at.obs_mask_proprio = False


@configclass
class CBFAdaptiveGo2UnifiedV2NoPrivEnvCfg(CBFAdaptiveGo2UnifiedV2EnvCfg):
    """V2 + priv channels zeroed in obs (still 7-priv shape; values
    forced to zero by the obs functions). Tests whether the policy
    needs priv at all on the V2 distribution."""
    def __post_init__(self):
        super().__post_init__()
        self.actions.cbf_param.obs_mask_priv = True


@configclass
class CBFAdaptiveGo2UnifiedV2NoProprioEnvCfg(CBFAdaptiveGo2UnifiedV2EnvCfg):
    """V2 + proprio channels zeroed in obs. Tests whether priv z is
    informative when proprio is unavailable."""
    def __post_init__(self):
        super().__post_init__()
        self.actions.cbf_param.obs_mask_proprio = True


@configclass
class CBFAdaptiveGo2UnifiedV2RMAClassicEnvCfg(CBFAdaptiveGo2UnifiedV2EnvCfg):
    """V2 distribution + 4-priv RMA-classic obs (195-dim).

    The 4-priv obs only surfaces (disturbance, friction, mass, motor).
    The other 3 channels (v_max, com_offset, actuation_noise) are still
    physically sampled per episode -- the policy just doesn't see them
    explicitly and must infer / be robust.
    """
    def __post_init__(self):
        super().__post_init__()
        self.observations = CBFRMAClassicObsCfg()


@configclass
class CBFAdaptiveGo2UnifiedV2HistoryEnvCfg(CBFAdaptiveGo2UnifiedV2EnvCfg):
    """V2 distribution + 50-step proprio history obs (no priv slot)."""
    def __post_init__(self):
        super().__post_init__()
        self.observations = CBFRMAHistoryObsCfg()
        self.actions.cbf_param.proprio_history_length = 50


# ---------------------------------------------------------------------------
# V2 eval scenes -- four held-out geometries (GAP / SLALOM / WALL / FIELD).
# Each scene is built for three obs layouts:
#   - Full (7-priv, 198-dim) -- shared by Full / NoPriv / NoProprio policies
#     (masking flipped at eval time via the action-term cfg flags)
#   - RMAClassic (4-priv, 195-dim)
#   - History (50-step proprio history, no priv)
#
# `random_topology` is OFF in every eval scene -- fixed obstacle layout
# defines the scene. The clumsy-human u_nom stays active so eval matches
# training distribution.
# ---------------------------------------------------------------------------
SCENES_V2 = {
    # E1 GAP: two cylinders form a narrow corridor (~2.0 m center-to-
    # center, with r_safe=0.85m each so clearance ~0.3 m). Forces a
    # commit-and-thread maneuver -- single hedge can't avoid both.
    "E1Gap": [(3.5, -1.0, 0.5), (3.5, 1.0, 0.5)],
    # E2 SLALOM: in-distribution geometry (training has random K=3 in a
    # similar corridor; this is a fixed weave). Regression check.
    "E2Slalom": [(2.0, 0.7, 0.5), (4.0, -0.7, 0.5), (5.5, 0.7, 0.5)],
    # E3 WALL: 4 obstacles forming a wall at x=2.5 with a center gap of
    # ~2.0 m (clearance ~0.3 m), plus 1 obstacle after the wall to force
    # post-wall avoidance. K=5.
    "E3Wall": [
        (2.5, 1.8, 0.5), (2.5, 1.0, 0.5),
        (2.5, -1.0, 0.5), (2.5, -1.8, 0.5),
        (4.5, 0.0, 0.5),
    ],
    # E4 FIELD: 7 scattered obstacles, smaller radius (0.4) to allow
    # density. r_safe = 0.4 + 0.35 = 0.75 m per obstacle.
    "E4Field": [
        (2.0, 0.8, 0.4), (2.5, -0.7, 0.4), (3.5, 0.0, 0.4),
        (4.0, -1.0, 0.4), (4.5, 1.0, 0.4),
        (5.0, -0.4, 0.4), (5.5, 0.6, 0.4),
    ],
}
GOAL_V2 = (7.0, 0.0)


def _apply_v2_eval_scene(self, scene_key: str) -> None:
    """Pin the obstacle layout + goal for a V2 eval scene; disable
    random_topology so the fixed scene wins. Also downgrade clumsiness
    to "teen" so eval matches a more capable (still imperfect) operator
    than the 5-year-old used at training time."""
    obstacles = SCENES_V2[scene_key]
    at = self.actions.cbf_param
    at.random_topology = False
    at.obstacles = list(obstacles)
    at.goal_xy = tuple(GOAL_V2)
    # small per-episode jitter (~0.3 m) keeps the obstacle-localization
    # leak plugged without changing the scene topology
    at.obstacle_pos_jitter_range = (-0.3, 0.3)
    # eval = 10-year-old; train was 5-year-old. Same OU/swerve mechanism,
    # smaller noise + fewer mistakes.
    at.unom_clumsiness = "teen"
    _reset_obstacle_visuals(self.scene)
    for i, (x, y, r) in enumerate(obstacles):
        setattr(self.scene, f"obstacle_visual_{i}",
                _make_obstacle_visual(f"{{ENV_REGEX_NS}}/ObstacleVis{i}", x, y, r))
    self.scene.goal_visual = _make_goal_visual(
        "{ENV_REGEX_NS}/GoalVis", GOAL_V2[0], GOAL_V2[1])


# Concrete eval cfgs (12 total = 4 scenes × 3 obs layouts).
# Each scene exists in three obs layouts so eval matches the training
# obs of each architecture. NoPriv / NoProprio reuse the Full-layout
# eval cfg and flip masking at eval time.
@configclass
class CBFV2EvalE1GapEnvCfg(CBFAdaptiveGo2UnifiedV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E1Gap")


@configclass
class CBFV2EvalE2SlalomEnvCfg(CBFAdaptiveGo2UnifiedV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E2Slalom")


@configclass
class CBFV2EvalE3WallEnvCfg(CBFAdaptiveGo2UnifiedV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E3Wall")


@configclass
class CBFV2EvalE4FieldEnvCfg(CBFAdaptiveGo2UnifiedV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E4Field")


# RMAClassic eval cfgs (4-priv, 195-dim obs)
@configclass
class CBFV2EvalE1GapRMAClassicEnvCfg(CBFAdaptiveGo2UnifiedV2RMAClassicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E1Gap")


@configclass
class CBFV2EvalE2SlalomRMAClassicEnvCfg(CBFAdaptiveGo2UnifiedV2RMAClassicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E2Slalom")


@configclass
class CBFV2EvalE3WallRMAClassicEnvCfg(CBFAdaptiveGo2UnifiedV2RMAClassicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E3Wall")


@configclass
class CBFV2EvalE4FieldRMAClassicEnvCfg(CBFAdaptiveGo2UnifiedV2RMAClassicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E4Field")


# History eval cfgs (proprio-history obs)
@configclass
class CBFV2EvalE1GapHistoryEnvCfg(CBFAdaptiveGo2UnifiedV2HistoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E1Gap")


@configclass
class CBFV2EvalE2SlalomHistoryEnvCfg(CBFAdaptiveGo2UnifiedV2HistoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E2Slalom")


@configclass
class CBFV2EvalE3WallHistoryEnvCfg(CBFAdaptiveGo2UnifiedV2HistoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E3Wall")


@configclass
class CBFV2EvalE4FieldHistoryEnvCfg(CBFAdaptiveGo2UnifiedV2HistoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v2_eval_scene(self, "E4Field")
