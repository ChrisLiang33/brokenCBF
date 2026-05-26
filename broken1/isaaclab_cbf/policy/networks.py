"""rsl_rl-compatible ActorCritic for the CBF param policy.

The env returns flat tensor obs of shape (B, FLAT_OBS_DIM=1085) wrapped in
{"policy": ...}. This module parses that flat tensor back into:
    proprio        (B, 35)
    occgrid        (B, 1, 32, 32)
    past_actions   (B, 20)
    priv_obs       (B, 6)   — perception + actuation DR only (v_max, sigma_e,
                              sigma_pose, drift, adv, tracking_err).
                              Physics DR (mass/friction/motor) runs via
                              EventCfg but isn't exposed here yet.

and feeds them through the same architecture we sketched in the 2D MVP:
    occgrid → CNN → feat
    priv_obs → encoder → z       (only if use_priv=True)
    concat(z, proprio, past_actions, feat) → MLP → 5-D action mean

Two variants:
    use_priv=True  ↔ "fullsetup"   (sees privileged obs)
    use_priv=False ↔ "nopriv"      (priv channel zeroed out before the encoder)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------
class PrivEncoder(nn.Module):
    def __init__(self, priv_dim: int, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(priv_dim, 128), nn.ELU(),
            nn.Linear(128, 64),       nn.ELU(),
            nn.Linear(64, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, priv: torch.Tensor) -> torch.Tensor:
        return self.net(priv)


class LidarCNN(nn.Module):
    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
        )
        self.fc = nn.Linear(32 * 4 * 4, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, occ: torch.Tensor) -> torch.Tensor:
        return F.elu(self.fc(self.conv(occ).flatten(start_dim=1)))


# ---------------------------------------------------------------------------
# rsl_rl-compatible ActorCritic
# ---------------------------------------------------------------------------
class CBFActorCritic(nn.Module):
    """Compatible with rsl_rl 4.x ActorCritic interface.
    Single flat-obs input; symmetric actor and critic obs.
    """
    # Flat obs slicing (must match Go2CbfRLEnv's _build_outer_obs)
    PROPRIO_DIM = 35
    OCC_HW      = (32, 32)
    OCC_DIM     = 1024
    PAST_DIM    = 20
    PRIV_DIM    = 6         # v_max, sigma_e, sigma_pose, drift_e, adv_e, tracking_err
    FLAT_DIM    = PROPRIO_DIM + OCC_DIM + PAST_DIM + PRIV_DIM    # 1085

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int = 5,
        init_noise_std: float = 0.5,
        use_priv: bool = True,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super().__init__()
        assert num_actions == 5, "CBF policy outputs 5 params (alpha, phi, a, b, c)"
        self.use_priv = use_priv

        self.priv_encoder = PrivEncoder(self.PRIV_DIM, latent_dim) if use_priv else None
        self.lidar_cnn    = LidarCNN(feat_dim=64)

        z_dim = latent_dim if use_priv else 0
        fused = z_dim + self.PROPRIO_DIM + self.PAST_DIM + self.lidar_cnn.feat_dim
        self.actor_mlp = nn.Sequential(
            nn.Linear(fused, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 5),
        )
        # Sensible action-mean init: log of (alpha=3, phi=0.1, a=b=c=0.05)
        init_bias = torch.tensor([1.099, -2.302, -2.996, -2.996, -2.996])
        with torch.no_grad():
            self.actor_mlp[-1].bias.copy_(init_bias)

        # Critic always sees priv (asymmetric is encouraged in rsl_rl)
        critic_in = self.PROPRIO_DIM + self.PAST_DIM + self.PRIV_DIM + self.lidar_cnn.feat_dim
        self.critic_mlp = nn.Sequential(
            nn.Linear(critic_in, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Action distribution
        self.std = nn.Parameter(init_noise_std * torch.ones(5))
        self.distribution = None

    # ---- Obs parsing -------------------------------------------------------
    def _parse(self, flat: torch.Tensor):
        i = 0
        proprio      = flat[:, i:i + self.PROPRIO_DIM]; i += self.PROPRIO_DIM
        occ          = flat[:, i:i + self.OCC_DIM].reshape(-1, 1, *self.OCC_HW); i += self.OCC_DIM
        past_actions = flat[:, i:i + self.PAST_DIM]; i += self.PAST_DIM
        priv_obs     = flat[:, i:i + self.PRIV_DIM]
        return proprio, occ, past_actions, priv_obs

    def _actor_features(self, flat: torch.Tensor):
        proprio, occ, past_actions, priv_obs = self._parse(flat)
        feats = []
        if self.use_priv:
            feats.append(self.priv_encoder(priv_obs))
        feats.extend([proprio, past_actions, self.lidar_cnn(occ)])
        return torch.cat(feats, dim=-1)

    def _critic_features(self, flat: torch.Tensor):
        proprio, occ, past_actions, priv_obs = self._parse(flat)
        return torch.cat([proprio, past_actions, priv_obs, self.lidar_cnn(occ)], dim=-1)

    # ---- rsl_rl ActorCritic API -------------------------------------------
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError("Use update_distribution, act, evaluate")

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor_mlp(self._actor_features(observations))
        std = self.std.expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor_mlp(self._actor_features(observations))

    def evaluate(self, critic_observations, **kwargs):
        return self.critic_mlp(self._critic_features(critic_observations))


def log_params_to_cbf(log_params: torch.Tensor) -> dict:
    """Convenience: split (B, 5) into the 5-term dict used by safety_filter()."""
    p = log_params.exp()
    return {"alpha": p[:, 0], "phi": p[:, 1], "a": p[:, 2], "b": p[:, 3], "c": p[:, 4]}
