"""RSL-RL PPO config for the CBF-Adaptive Go2 task.

Outer policy is 2-action (φ, α) on a zeroed 7-dim obs (Phase 1). A tiny
MLP is appropriate; we use 32x32 since there's nothing to condition on.

For Phase 2 (real obs), bump net_arch and capacity in a separate cfg.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CBFAdaptiveGo2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 300
    save_interval: int = 50
    experiment_name: str = "cbf_adaptive_go2"
    empirical_normalization: bool = False
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CBFAdaptiveGo2RMARunnerCfg(RslRlOnPolicyRunnerCfg):
    """RMA teacher PPO cfg, rsl-rl 5.0.1 NEW-FORMAT.

    We MUST NOT use the deprecated `policy` field here -- when the
    deprecation handler translates `policy.class_name` -> `actor.class_name`
    it silently drops our custom class_name and defaults to "MLPModel".
    The branched encoder + lidar CNN we wrote then never get used.

    Instead, set `actor` + `critic` (and `obs_groups`) directly. The
    qualified `class_name="module.path:ClassName"` form goes through
    rsl_rl's `resolve_callable` importlib path, which finds our class
    via `cbf_task.agents.rma_actor_critic`. Auto-import of that module
    via the training script is enough.
    """
    num_steps_per_env: int = 24
    max_iterations: int = 1000
    save_interval: int = 100
    experiment_name: str = "cbf_adaptive_go2_rma"
    # Required by the base class (MISSING by default). The new format
    # uses obs_normalization on each model cfg instead, but the field
    # must still be set to avoid the configclass validation error.
    empirical_normalization: bool = False

    # New-format actor / critic / obs_groups (rsl-rl >= 4.0.0).
    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="cbf_task.agents.rma_actor_critic:RMAMLPModel",
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        # NEW distribution config (rsl-rl >= 5.0.0). init_std=0.3 to
        # avoid the action_std=0.92 jitter ceiling we hit at init=1.0.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.3,
            std_type="scalar",
        ),
        # legacy fields are MISSING by default and the configclass would
        # error on access. Set them harmlessly (distribution_cfg above
        # takes precedence in rsl-rl >= 5.0.0).
        stochastic=True,
        init_noise_std=0.3,
        noise_std_type="scalar",
    )
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="cbf_task.agents.rma_actor_critic:RMAMLPModel",
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,     # critic is deterministic
        stochastic=False,
        init_noise_std=0.0,
    )
    obs_groups: dict = {"actor": ["policy"], "critic": ["policy"]}

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # entropy_coef 0.002 -> 0.0. With 1024 envs and the unified
        # env's sparse reward, the entropy bonus was driving action_std
        # to ~4.6 (way above the [-1, 1] action range), collapsing all
        # priv-driven adaptation -- policy became "low-alpha + random
        # actions". Zero bonus forces PPO to rely on its own intrinsic
        # exploration via init_std=0.3 + adaptive LR schedule.
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
@configclass
class CBFAdaptiveGo2RMAStaticRunnerCfg(CBFAdaptiveGo2RMARunnerCfg):
    """RMA-classic teacher PPO cfg: same structure as
    CBFAdaptiveGo2RMARunnerCfg but with the model class swapped to
    RMAClassicMLPModel (priv_dim=4 baked in). All other hyperparams
    inherited (entropy_coef=0, init_std=0.3, etc).

    Pair with `Isaac-CBF-Adaptive-Go2-RMAStatic-v0` task. The model
    class string uses the qualified module:Class form so rsl_rl 5.0.1's
    resolve_callable path finds it without bare-name fallback.
    """
    experiment_name: str = "cbf_adaptive_go2_rma_static"
    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="cbf_task.agents.rma_classic_actor_critic:RMAClassicMLPModel",
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.3, std_type="scalar",
        ),
        stochastic=True,
        init_noise_std=0.3,
        noise_std_type="scalar",
    )
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="cbf_task.agents.rma_classic_actor_critic:RMAClassicMLPModel",
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
        stochastic=False,
        init_noise_std=0.0,
    )


@configclass
class CBFAdaptiveGo2RMAHistoryRunnerCfg(CBFAdaptiveGo2RMARunnerCfg):
    """RMAHistory teacher PPO cfg: uses RMAHistoryMLPModel (proprio CNN
    over 50-step history, no priv path). All other hyperparams inherited
    from CBFAdaptiveGo2RMARunnerCfg."""
    experiment_name: str = "cbf_adaptive_go2_rma_history"
    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="cbf_task.agents.rma_history_actor_critic:RMAHistoryMLPModel",
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.3, std_type="scalar",
        ),
        stochastic=True,
        init_noise_std=0.3,
        noise_std_type="scalar",
    )
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="cbf_task.agents.rma_history_actor_critic:RMAHistoryMLPModel",
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
        stochastic=False,
        init_noise_std=0.0,
    )


@configclass
class CBFAdaptiveGo2Phase2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 2 PPO config -- larger MLP for the 960-dim windowed obs,
    more iterations to converge on a state-conditional policy."""
    num_steps_per_env: int = 24
    max_iterations: int = 1000
    save_interval: int = 100
    experiment_name: str = "cbf_adaptive_go2_phase2"
    empirical_normalization: bool = True   # 960-dim obs -- normalize
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
