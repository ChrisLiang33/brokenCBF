"""RMA-History teacher: no priv obs at all, proprio fed as a 50-step
history through a 1D CNN over the time axis (RMA paper's adaptation-
module pattern, but baked into the teacher directly so no separate
student-distillation stage is needed).

Hypothesis: PPO can implicitly extract priv-equivalent info from
proprio dynamics if given enough history, removing the RMA two-stage
pipeline. Result will tell us whether end-to-end "blind" training
suffices for adaptive CBF on this task.

Obs layout (no priv):
    [proprio_history(K*45), prev_act(2), lidar_prev(72), lidar(72)]
    where K = proprio_history_length (default 50)
    Total = K*45 + 146

Architecture:
    proprio_history (N, K*45)
        reshape -> (N, 45, K)               # channels = features, length = time
        _ProprioCNN -> (N, 64)              # 1D conv over time axis
    lidar (N, 144)
        _LidarCNN (existing) -> (N, 16)     # 1D conv over azimuth
    main MLP input: concat(proprio_feat, lidar_feat, prev_act) = 64+16+2 = 82
        -> hidden -> output (2 for actor, 1 for critic)

Pairs with:
    - `mdp.teacher_obs_history` / `mdp.proprio_history_obs`
    - cbf cfg `proprio_history_length` > 0 (default 50 in RMAHistory env)
    - `CBFAdaptiveGo2RMAHistoryEnvCfg` + `Isaac-CBF-Adaptive-Go2-RMAHistory-v0`
"""
from __future__ import annotations

import torch
import torch.nn as nn

import rsl_rl.models
from rsl_rl.models.mlp_model import MLPModel

from .rma_actor_critic import _ACT_MAP, _LidarCNN, _mlp, PROPRIO_DIM, LIDAR_DIM


# Default history window
DEFAULT_HISTORY_LEN = 50
PROPRIO_FEAT_DIM = 64
LIDAR_FEAT_DIM_DEFAULT = 16
PREV_ACT_DIM = 2


class _ProprioCNN(nn.Module):
    """1D CNN over the proprio HISTORY axis. Input shape (N, K*45),
    reshaped to (N, 45, K) where 45 channels = proprio features and K
    is the time dimension. Convolutions slide along time, weight-shared
    across timesteps. Mirrors the RMA paper's adaptation module.

    Architecture (for K=50, D=45):
        (N, 45, 50)
          -> Conv1d(45, 32, k=5) -> ELU                # local temporal patterns
          -> Conv1d(32, 32, k=5) -> ELU
          -> AvgPool1d(2)                              # 50 -> 25
          -> Conv1d(32, 16, k=3) -> ELU
          -> AvgPool1d(2)                              # 25 -> 12
          -> Flatten -> Linear(16*12=192, 64) -> ELU -> Linear(64, 64)
    """

    def __init__(self, history_len: int, proprio_dim: int = PROPRIO_DIM,
                 out_dim: int = PROPRIO_FEAT_DIM,
                 activation_cls: type[nn.Module] = nn.ELU):
        super().__init__()
        self.history_len = history_len
        self.proprio_dim = proprio_dim
        self.conv = nn.Sequential(
            nn.Conv1d(proprio_dim, 32, kernel_size=5, padding=2),
            activation_cls(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            activation_cls(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            activation_cls(),
            nn.AvgPool1d(2),
        )
        flat_dim = 16 * (history_len // 4)               # two pools by 2
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 64),
            activation_cls(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat: (N, K*D). Reshape so channels=D, time=K.
        N = x_flat.shape[0]
        x = x_flat.view(N, self.history_len, self.proprio_dim)
        x = x.transpose(1, 2).contiguous()                # (N, D, K)
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        return self.head(h)


class _HistoryBranchedMLP(nn.Module):
    """Branched MLP for the RMAHistory teacher: proprio_CNN + lidar_CNN
    + main MLP. NO priv path.

    Obs layout (flat):
        [0 : K*45)              proprio history
        [K*45 : K*45+2)         prev_act
        [K*45+2 : K*45+74)      lidar_prev
        [K*45+74 : K*45+146)    lidar
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: list[int] | tuple[int, ...],
                 activation: str = "elu",
                 history_len: int = DEFAULT_HISTORY_LEN,
                 proprio_feat_dim: int = PROPRIO_FEAT_DIM,
                 lidar_feat_dim: int = LIDAR_FEAT_DIM_DEFAULT):
        super().__init__()
        expected = history_len * PROPRIO_DIM + PREV_ACT_DIM + 2 * LIDAR_DIM
        if input_dim != expected:
            raise ValueError(
                f"RMAHistory branched MLP expects input_dim={expected} "
                f"(history_len={history_len}), got {input_dim}."
            )
        act_cls = _ACT_MAP.get(activation.lower(), nn.ELU)
        self._history_len = history_len
        ph_end = history_len * PROPRIO_DIM
        self._ph_slice  = slice(0, ph_end)
        self._pact_slice = slice(ph_end, ph_end + PREV_ACT_DIM)
        lp_start = ph_end + PREV_ACT_DIM
        self._lp_slice  = slice(lp_start, lp_start + LIDAR_DIM)
        self._ld_slice  = slice(lp_start + LIDAR_DIM, lp_start + 2 * LIDAR_DIM)

        self.proprio_enc = _ProprioCNN(
            history_len=history_len, proprio_dim=PROPRIO_DIM,
            out_dim=proprio_feat_dim, activation_cls=act_cls,
        )
        self.lidar_enc = _LidarCNN(
            n_rays=LIDAR_DIM, out_dim=lidar_feat_dim, activation_cls=act_cls,
        )
        main_in = proprio_feat_dim + lidar_feat_dim + PREV_ACT_DIM
        self.main = _mlp(main_in, list(hidden_dims), output_dim, act_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proprio_hist = x[..., self._ph_slice]
        prev_act     = x[..., self._pact_slice]
        lidar_prev   = x[..., self._lp_slice]
        lidar        = x[..., self._ld_slice]
        proprio_feat = self.proprio_enc(proprio_hist)
        lidar_feat   = self.lidar_enc(torch.cat([lidar_prev, lidar], dim=-1))
        return self.main(torch.cat([proprio_feat, lidar_feat, prev_act], dim=-1))


class RMAHistoryMLPModel(MLPModel):
    """rsl_rl 5.0.1 MLPModel with a history-based branched encoder.
    No priv slot in the obs (it's not in teacher_obs_history)."""

    def __init__(self,
                 obs,                                       # TensorDict
                 obs_groups: dict,
                 obs_set: str,
                 output_dim: int,
                 hidden_dims=(256, 256, 256),
                 activation: str = "elu",
                 obs_normalization: bool = False,
                 distribution_cfg: dict | None = None,
                 history_len: int = DEFAULT_HISTORY_LEN,
                 proprio_feat_dim: int = PROPRIO_FEAT_DIM,
                 lidar_feat_dim: int = LIDAR_FEAT_DIM_DEFAULT,
                 **kwargs):
        super().__init__(
            obs=obs, obs_groups=obs_groups, obs_set=obs_set,
            output_dim=output_dim, hidden_dims=hidden_dims,
            activation=activation, obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg, **kwargs,
        )
        mlp_output_dim = (self.distribution.input_dim
                          if self.distribution is not None
                          else output_dim)
        self.mlp = _HistoryBranchedMLP(
            input_dim=self.obs_dim,
            output_dim=mlp_output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            history_len=history_len,
            proprio_feat_dim=proprio_feat_dim,
            lidar_feat_dim=lidar_feat_dim,
        )
        if self.distribution is not None:
            try:
                self.distribution.init_mlp_weights(self.mlp)
            except Exception:
                pass


rsl_rl.models.RMAHistoryMLPModel = RMAHistoryMLPModel
