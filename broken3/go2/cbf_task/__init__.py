"""CBF-Adaptive Go2 task package.

Registers `Isaac-CBF-Adaptive-Go2-v0` and the Play variant. The env cfg
inherits the scene/robot/physics from the stock Go2 flat task and
overrides the MDP managers (actions, rewards, terminations,
observations) to install the CBF-parameter action term and our custom
reward shape.

Usage from a training script:

    import gymnasium as gym
    import cbf_task  # noqa: F401  (registers the task)

    env_cfg = ...
    env_cfg.actions.cbf_param.locomotion_policy_obj = loaded_actor   # nn.Module
    env = gym.make("Isaac-CBF-Adaptive-Go2-v0", cfg=env_cfg)
"""
import gymnasium as gym

from . import agents
from .cbf_adaptive_env_cfg import (
    CBFAdaptiveGo2ActNoiseGateEnvCfg,
    CBFAdaptiveGo2ComOffsetGateEnvCfg,
    CBFAdaptiveGo2DecorrEnvCfg,
    CBFAdaptiveGo2EnvCfg,
    CBFAdaptiveGo2EnvCfg_PLAY,
    CBFAdaptiveGo2Phase2EnvCfg,
    CBFAdaptiveGo2Phase3EnvCfg,
    CBFAdaptiveGo2RMAEnvCfg,
    CBFAdaptiveGo2RandObsEnvCfg,
    CBFAdaptiveGo2RoughEnvCfg,
    CBFAdaptiveGo2SlalomEnvCfg,
    CBFAdaptiveGo2UnifiedEnvCfg,
    CBFAdaptiveGo2UnifiedLidarSDFEnvCfg,
    CBFAdaptiveGo2UnifiedV2EnvCfg,
    CBFAdaptiveGo2UnifiedV2NoPrivEnvCfg,
    CBFAdaptiveGo2UnifiedV2NoProprioEnvCfg,
    CBFAdaptiveGo2UnifiedV2RMAClassicEnvCfg,
    CBFAdaptiveGo2UnifiedV2HistoryEnvCfg,
    CBFAdaptiveGo2VmaxEnvCfg,
)


gym.register(
    id="Isaac-CBF-Adaptive-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-Phase2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2Phase2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2Phase2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-Phase3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2Phase3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2Phase2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-RMA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2RMAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-RandObs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2RandObsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-Vmax-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2VmaxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-Slalom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2SlalomEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-Decorr-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2DecorrEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-ActNoiseGate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2ActNoiseGateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-ComOffsetGate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2ComOffsetGateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-Unified-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2UnifiedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
    },
)

gym.register(
    id="Isaac-CBF-Adaptive-Go2-UnifiedLidarSDF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2UnifiedLidarSDFEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
    },
)

# v7.1 experimental variant: SHIELD env + stuck_penalty active at -0.05.
# Built to test whether activating the wired-at-0 stuck term recovers
# the reach loss seen in v7 cross-scene eval (E1=40%, E3=4%, E4=12%).
# Revert path: drop this entry + the env subclass if v7.1 underperforms.
gym.register(
    id="Isaac-CBF-Adaptive-Go2-UnifiedLidarSDFStuck-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2UnifiedLidarSDFStuckEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
    },
)

# RMA-classic experiment: 4 priv channels (RMA paper canonical) + random
# obstacle topology (static within episode). Built after v7.1's priv_attention
# diagnostic showed only disturbance was used; this strips the v7 experimental
# DR additions and adds topology randomization to test the "exact RMA + lidar"
# setup. Different model class (RMAClassicMLPModel, priv_dim=4 hardcoded) so
# does NOT interfere with the SHIELD pipeline.
gym.register(
    id="Isaac-CBF-Adaptive-Go2-RMAStatic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2RMAStaticEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMAStaticRunnerCfg",
    },
)

# RMA-Static ablations: train with one of {priv, proprio} masked to zero.
# Same model architecture, same DR, same scene; only obs masking differs.
# Answer the question: "if the policy can't see X, what does it learn?"
gym.register(
    id="Isaac-CBF-Adaptive-Go2-RMAStaticNoPriv-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2RMAStaticNoPrivEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMAStaticRunnerCfg",
    },
)
gym.register(
    id="Isaac-CBF-Adaptive-Go2-RMAStaticNoProprio-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2RMAStaticNoProprioEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMAStaticRunnerCfg",
    },
)

# RMAHistory teacher: no priv obs, 50-step proprio history through 1D
# CNN. Tests whether PPO can extract priv-equivalent info from proprio
# dynamics end-to-end (no student-distillation stage needed).
gym.register(
    id="Isaac-CBF-Adaptive-Go2-RMAHistory-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cbf_adaptive_env_cfg:CBFAdaptiveGo2RMAHistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMAHistoryRunnerCfg",
    },
)

# NoPriv eval scenes (E1-E4). Same geometries as Eval{E1..E4} but with
# the 4-priv RMA-classic obs layout (195-dim) + priv masked, matching the
# NoPriv teacher's training distribution. Used with
# phase6_eval_scenes.py --scene_prefix EvalNoPriv.
for _eval_id, _eval_cls in [
    ("E1", "CBFAdaptiveGo2EvalNoPrivE1EnvCfg"),
    ("E2", "CBFAdaptiveGo2EvalNoPrivE2EnvCfg"),
    ("E3", "CBFAdaptiveGo2EvalNoPrivE3EnvCfg"),
    ("E4", "CBFAdaptiveGo2EvalNoPrivE4EnvCfg"),
]:
    gym.register(
        id=f"Isaac-CBF-Adaptive-Go2-EvalNoPriv{_eval_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                f"{__name__}.cbf_adaptive_env_cfg:{_eval_cls}",
            "rsl_rl_cfg_entry_point":
                f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMAStaticRunnerCfg",
        },
    )

# Held-out eval scenes E1-E4. Geometries the teacher has NOT trained on.
# Used by phase6_eval_scenes.py for cross-scene comparison vs baselines.
# All inherit the SHIELD env (full DR + lidar fidelity + Lipschitz).
# E0 (empty) was dropped -- see comment in cbf_adaptive_env_cfg.py.
for _eval_id, _eval_cls in [
    ("E1", "CBFAdaptiveGo2EvalE1EnvCfg"),
    ("E2", "CBFAdaptiveGo2EvalE2EnvCfg"),
    ("E3", "CBFAdaptiveGo2EvalE3EnvCfg"),
    ("E4", "CBFAdaptiveGo2EvalE4EnvCfg"),
]:
    gym.register(
        id=f"Isaac-CBF-Adaptive-Go2-Eval{_eval_id}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                f"{__name__}.cbf_adaptive_env_cfg:{_eval_cls}",
            "rsl_rl_cfg_entry_point":
                f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
        },
    )


# ===========================================================================
# Phase 10 / V2 — Unified comparison experiment (5 architectures on the
# SAME training distribution; 4 eval scenes; per-scene DR sweep).
# ---------------------------------------------------------------------------
# Training tasks (5). Each takes an env_cfg the train script overrides with
# the intervention cost (0 or -0.05) to produce the 10 trainings.
#   - V2Full         : 7-priv obs (198-dim) + RMAMLPModel
#   - V2NoPriv       : 7-priv obs, priv slot masked to zero + RMAMLPModel
#   - V2NoProprio    : 7-priv obs, proprio slot masked to zero + RMAMLPModel
#   - V2RMAClassic   : 4-priv obs (195-dim) + RMAClassicMLPModel
#   - V2History      : proprio history obs (no priv) + RMAHistoryMLPModel
# ===========================================================================
_V2_TRAIN_TASKS = [
    ("V2Full",       "CBFAdaptiveGo2UnifiedV2EnvCfg",            "CBFAdaptiveGo2RMARunnerCfg"),
    ("V2NoPriv",     "CBFAdaptiveGo2UnifiedV2NoPrivEnvCfg",      "CBFAdaptiveGo2RMARunnerCfg"),
    ("V2NoProprio",  "CBFAdaptiveGo2UnifiedV2NoProprioEnvCfg",   "CBFAdaptiveGo2RMARunnerCfg"),
    ("V2RMAClassic", "CBFAdaptiveGo2UnifiedV2RMAClassicEnvCfg",  "CBFAdaptiveGo2RMAStaticRunnerCfg"),
    ("V2History",    "CBFAdaptiveGo2UnifiedV2HistoryEnvCfg",     "CBFAdaptiveGo2RMAHistoryRunnerCfg"),
]
for _arch, _env_cls, _runner_cls in _V2_TRAIN_TASKS:
    gym.register(
        id=f"Isaac-CBF-Adaptive-Go2-{_arch}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                f"{__name__}.cbf_adaptive_env_cfg:{_env_cls}",
            "rsl_rl_cfg_entry_point":
                f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner_cls}",
        },
    )

# Eval scene tasks (4 scenes × 3 obs layouts = 12 task IDs).
# The Full layout is shared by Full / NoPriv / NoProprio policies -- the
# eval script flips obs_mask_priv / obs_mask_proprio on the action term
# at eval time (no separate task IDs needed).
_V2_EVAL_SCENES = ["E1Gap", "E2Slalom", "E3Wall", "E4Field"]
for _scene in _V2_EVAL_SCENES:
    # Full layout (7-priv)
    gym.register(
        id=f"Isaac-CBF-Adaptive-Go2-V2Eval-{_scene}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                f"{__name__}.cbf_adaptive_env_cfg:CBFV2Eval{_scene}EnvCfg",
            "rsl_rl_cfg_entry_point":
                f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMARunnerCfg",
        },
    )
    # RMA-classic (4-priv)
    gym.register(
        id=f"Isaac-CBF-Adaptive-Go2-V2Eval-{_scene}-RMAClassic-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                f"{__name__}.cbf_adaptive_env_cfg:CBFV2Eval{_scene}RMAClassicEnvCfg",
            "rsl_rl_cfg_entry_point":
                f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMAStaticRunnerCfg",
        },
    )
    # History
    gym.register(
        id=f"Isaac-CBF-Adaptive-Go2-V2Eval-{_scene}-History-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                f"{__name__}.cbf_adaptive_env_cfg:CBFV2Eval{_scene}HistoryEnvCfg",
            "rsl_rl_cfg_entry_point":
                f"{agents.__name__}.rsl_rl_ppo_cfg:CBFAdaptiveGo2RMAHistoryRunnerCfg",
        },
    )
