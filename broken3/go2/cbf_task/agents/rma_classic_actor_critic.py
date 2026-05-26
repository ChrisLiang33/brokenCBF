"""RMA-classic variant: 4 priv channels (RMA-paper canonical) instead
of v7's 7.

Drops v7's experimental v_max/com_offset/actuation_noise additions. The
priv_attention diagnostic on v7.1 showed the trained policy ignored all
3 ([[v7_priv_channel_dead]] memory). Going back to the 4 channels that
match the original RMA paper's priv vector for our CBF-param task to
test whether a smaller, cleaner priv space yields a z latent the
student can actually distill.

Reuses the parametric `_BranchedMLP` and `_LidarCNN` from
rma_actor_critic; only difference is `priv_dim=4` baked into the model
constructor. Obs layout becomes:

    [priv(4) | proprio(45) | prev_act(2) | lidar_prev(72) | lidar(72)] = 195

Paired with:
  - `priv_obs_rma_classic` in mdp.py for the env's obs term
  - `Isaac-CBF-Adaptive-Go2-RMAStatic-v0` task with random obstacle topology
  - `CBFAdaptiveGo2RMAStaticRunnerCfg` PPO runner that references this class
"""
from __future__ import annotations

import rsl_rl.models

from .rma_actor_critic import RMAMLPModel, PROPRIO_DIM, LIDAR_DIM


PRIV_DIM = 4
EXPECTED_OBS_DIM = PRIV_DIM + PROPRIO_DIM + 2 + 2 * LIDAR_DIM  # 195
PRIV_SLICE = slice(0, PRIV_DIM)
PROPRIO_SLICE = slice(PRIV_DIM, PRIV_DIM + PROPRIO_DIM)
PREV_ACT_SLICE = slice(PRIV_DIM + PROPRIO_DIM, PRIV_DIM + PROPRIO_DIM + 2)
LIDAR_PREV_SLICE = slice(
    PRIV_DIM + PROPRIO_DIM + 2,
    PRIV_DIM + PROPRIO_DIM + 2 + LIDAR_DIM,
)
LIDAR_SLICE = slice(
    PRIV_DIM + PROPRIO_DIM + 2 + LIDAR_DIM,
    EXPECTED_OBS_DIM,
)


class RMAClassicMLPModel(RMAMLPModel):
    """RMAMLPModel with priv_dim=4 hardcoded.

    Subclass so the runner cfg can resolve `class_name=
    "cbf_task.agents.rma_classic_actor_critic:RMAClassicMLPModel"`
    without needing to pass priv_dim through the (typed) MLPModelCfg.
    """

    def __init__(self, *args, **kwargs):
        # rsl_rl passes (obs, obs_groups, obs_set, output_dim) positionally
        # then **cfg["actor"] as kwargs. Accept both.
        kwargs.setdefault("priv_dim", PRIV_DIM)
        super().__init__(*args, **kwargs)


# Mirror the parent's auto-registration pattern so the runner cfg can
# also resolve the bare-name form if needed.
rsl_rl.models.RMAClassicMLPModel = RMAClassicMLPModel
