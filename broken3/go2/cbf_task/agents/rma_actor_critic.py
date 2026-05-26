"""RMA-style branched-encoder model for the CBF-adaptive Go2 task,
written against rsl_rl 5.0.1's `MLPModel` interface.

Observation contract (set by `mdp.teacher_obs`):
    [0:7]      priv              (disturbance, friction, mass_delta,
                                  motor_strength, actuation_noise_std,
                                  com_offset, v_max; teacher only)
                                  [B.1: 4->6, B.6: +v_max -> 7]
    [7:52]     proprio           (deployable_obs: base lin/ang vel,
                                  gravity_b, joint_pos_rel, joint_vel,
                                  prev_loco_action; 45-dim. NO vel_cmd
                                  -- removed 2026-05-24b to kill the
                                  goal-proxy leak via ||u_nom||.)
    [52:54]    prev_action       (outer policy's prev phi, alpha)
    [54:126]   lidar_prev        (72-ray ring at t-1, noisy ranges (m))
    [126:198]  lidar             (72-ray ring at t,  noisy ranges (m))

By network construction the disturbance slice can only reach the
output through `z_enc`. For the student phase, slice [0:1] is replaced
with phi(history); everything else is unchanged.

In rsl_rl 5.0.1 the actor and critic are *separate* `MLPModel` instances.
We subclass it and replace `self.mlp` (which the parent built as a
plain `MLP`) with a branched module that splits the obs, encodes the
priv and lidar slices via small sub-MLPs, then runs a main MLP on the
concatenation. The parent handles obs normalization, distribution
wrapping (Gaussian by default), JIT/ONNX export, etc.

To use:
    1. Import this module before the runner is constructed -- monkey-
       patches `RMAMLPModel` into `rsl_rl.models` so `class_name=
       "RMAMLPModel"` in the runner cfg can resolve it.
    2. Set `policy.class_name = "RMAMLPModel"` in the runner cfg.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import rsl_rl.models
from rsl_rl.models.mlp_model import MLPModel


# Slice layout for the 198-dim teacher obs (B.6: added v_max -> 7 priv).
PRIV_SLICE = slice(0, 7)
PROPRIO_SLICE = slice(7, 52)
PREV_ACT_SLICE = slice(52, 54)
LIDAR_PREV_SLICE = slice(54, 126)
LIDAR_SLICE = slice(126, 198)
EXPECTED_OBS_DIM = 198
PRIV_DIM = 7
PROPRIO_DIM = 45
LIDAR_DIM = 72

_ACT_MAP: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def _mlp(in_dim: int, hidden_dims: list[int] | tuple[int, ...],
         out_dim: int, activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(last, h), activation()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class _LidarCNN(nn.Module):
    """1D CNN over the lidar ring. Input is the concatenation of two
    consecutive lidar frames at t-1 and t (shape (N, 2*R)); we reshape
    to (N, 2, R) so each ray index has a (prev, curr) 2-channel input.
    The CNN learns the temporal feature itself (a 1x2 conv across the
    channel dim is effectively a learned delta).

    Circular padding (`padding_mode='circular'`) handles the fact that
    ray 71 is adjacent to ray 0 around the back of the robot -- without
    it, the conv would treat them as on opposite ends of a sequence.

    Architecture:
        (N, 2, 72) -> Conv1d(2, 16, k=5, circ) -> ELU
                  -> Conv1d(16, 32, k=5, circ) -> ELU
                  -> MaxPool1d(2)                       # 72 -> 36
                  -> Conv1d(32, 16, k=3, circ) -> ELU
                  -> MaxPool1d(2)                       # 36 -> 18
                  -> Flatten + Linear(288, out_dim)
    """

    def __init__(self, n_rays: int = 72, out_dim: int = 16,
                 activation_cls: type[nn.Module] = nn.ELU):
        super().__init__()
        self.n_rays = n_rays
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2, padding_mode="circular"),
            activation_cls(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2, padding_mode="circular"),
            activation_cls(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1, padding_mode="circular"),
            activation_cls(),
            nn.MaxPool1d(2),
        )
        # after two pool-by-2: n_rays // 4 = 18 for n_rays=72
        flat_dim = 16 * (n_rays // 4)
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 64),
            activation_cls(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 2*R) with first R = lidar at t-1, next R = lidar at t
        N = x.shape[0]
        x2 = x.view(N, 2, self.n_rays)
        h = self.conv(x2)
        h = h.flatten(start_dim=1)
        return self.head(h)


class _BranchedMLP(nn.Module):
    """Drop-in replacement for rsl_rl's flat `MLP` that splits the obs,
    encodes the priv and lidar slices, then runs a main MLP on the
    concatenation. Same forward signature (single 1D input, single 1D
    output) so it slots into `MLPModel.mlp`.

    Layout-parametric: pass `priv_dim` to switch between the 7-channel
    SHIELD layout (default, matches module constants) and the 4-channel
    RMA-classic layout (priv_dim=4, EXPECTED_OBS_DIM=195).
    PROPRIO_DIM and LIDAR_DIM are fixed.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: list[int] | tuple[int, ...],
                 activation: str = "elu",
                 z_dim: int = 8,
                 lidar_feat_dim: int = 16,
                 priv_dim: int = PRIV_DIM):
        super().__init__()
        # Compute slices from priv_dim. Layout is always:
        # [priv | proprio(45) | prev_act(2) | lidar_prev(72) | lidar(72)]
        expected_obs_dim = priv_dim + PROPRIO_DIM + 2 + LIDAR_DIM + LIDAR_DIM
        if input_dim != expected_obs_dim:
            raise ValueError(
                f"RMA branched MLP expects input_dim={expected_obs_dim} "
                f"(priv_dim={priv_dim}), got {input_dim}. Check "
                f"mdp.teacher_obs / obs cfg."
            )
        self._priv_dim = priv_dim
        self._priv_slice  = slice(0, priv_dim)
        self._propr_slice = slice(priv_dim, priv_dim + PROPRIO_DIM)
        self._pact_slice  = slice(priv_dim + PROPRIO_DIM,
                                   priv_dim + PROPRIO_DIM + 2)
        lp_start = priv_dim + PROPRIO_DIM + 2
        self._lp_slice    = slice(lp_start, lp_start + LIDAR_DIM)
        self._ld_slice    = slice(lp_start + LIDAR_DIM,
                                   lp_start + 2 * LIDAR_DIM)

        act_cls = _ACT_MAP.get(activation.lower(), nn.ELU)
        self.z_enc = _mlp(priv_dim, [16], z_dim, act_cls)
        # lidar_enc: 1D CNN with circular padding over the 72-ray ring,
        # 2 input channels (raw lidar at t-1 and t).
        self.lidar_enc = _LidarCNN(
            n_rays=LIDAR_DIM, out_dim=lidar_feat_dim, activation_cls=act_cls,
        )
        # main input = z + lidar_feat + proprio (45) + prev_act (2)
        main_in = z_dim + lidar_feat_dim + PROPRIO_DIM + 2
        self.main = _mlp(main_in, list(hidden_dims), output_dim, act_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        priv = x[..., self._priv_slice]
        proprio = x[..., self._propr_slice]
        prev_act = x[..., self._pact_slice]
        lidar_prev = x[..., self._lp_slice]
        lidar = x[..., self._ld_slice]
        z = self.z_enc(priv)
        # channel0=t-1, channel1=t; CNN learns delta-like features itself
        lidar_feat = self.lidar_enc(torch.cat([lidar_prev, lidar], dim=-1))
        return self.main(torch.cat([z, lidar_feat, proprio, prev_act], dim=-1))


class RMAMLPModel(MLPModel):
    """rsl_rl 5.0.1-compatible MLP model with branched encoders.

    Inherits all of `MLPModel`'s machinery (obs concat, normalization,
    distribution wrapping, export, etc.) and just swaps the inner MLP.
    """

    def __init__(self,
                 obs,                                       # TensorDict
                 obs_groups: dict,
                 obs_set: str,
                 output_dim: int,
                 hidden_dims=(256, 256, 256),
                 activation: str = "elu",
                 obs_normalization: bool = False,
                 distribution_cfg: dict | None = None,
                 # RMA-specific knobs
                 z_dim: int = 8,
                 lidar_feat_dim: int = 16,
                 priv_dim: int = PRIV_DIM,
                 **kwargs):
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            **kwargs,
        )
        # Replace the flat MLP with our branched one. The parent already
        # built self.mlp with the right output dimension (matches
        # `distribution.input_dim` if a distribution was configured,
        # otherwise `output_dim`).
        mlp_output_dim = (self.distribution.input_dim
                          if self.distribution is not None
                          else output_dim)
        self.mlp = _BranchedMLP(
            input_dim=self.obs_dim,
            output_dim=mlp_output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            z_dim=z_dim,
            lidar_feat_dim=lidar_feat_dim,
            priv_dim=priv_dim,
        )
        # Re-init distribution-specific weights on the new MLP if the
        # distribution needs special init (e.g., small last-layer std).
        if self.distribution is not None:
            try:
                self.distribution.init_mlp_weights(self.mlp)
            except Exception:
                # graceful fallback: pytorch defaults are usually fine
                pass


# Make `class_name="RMAMLPModel"` resolvable via rsl_rl.models lookup
# without requiring callers to monkey-patch themselves.
rsl_rl.models.RMAMLPModel = RMAMLPModel
