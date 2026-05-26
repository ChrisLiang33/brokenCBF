"""Wrapper for the pre-trained Go2 velocity-tracking locomotion policy.

Isaac Lab's `Isaac-Velocity-Flat-Unitree-Go2-v0` task trains a policy that
maps proprio + velocity command → joint targets. We load that checkpoint,
freeze it, and expose a simple callable:

    joint_targets = loco(proprio_tensor, velocity_cmd_xy)

The velocity command here is (vx_body, vy_body, yaw_rate). For our CBF
output (vx, vy), we set yaw_rate = 0 (or steer toward goal heading; either
works for an MVP).

USAGE:
    loco = FrozenGo2Locomotion("./go2_policy.pt", device="cuda")
    env.locomotion = loco
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrozenGo2Locomotion:
    """Holds a TorchScript or pickled state-dict policy in eval mode."""

    def __init__(self, ckpt_path: str, device: str = "cuda",
                 obs_layout: dict | None = None):
        # Try TorchScript first (most portable), then state-dict
        try:
            self.policy = torch.jit.load(ckpt_path, map_location=device)
        except Exception:
            # Assume it's a state_dict — user constructs the network shell.
            # rsl_rl saves with a wrapper; consult their save format.
            raise RuntimeError(
                f"Could not torch.jit.load {ckpt_path}. "
                "If using rsl_rl native checkpoints, export to TorchScript "
                "first (rsl_rl provides `OnPolicyRunner.export()`). "
                "Or write a small loader for your specific checkpoint format."
            )
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.device = device

        # Default obs layout assumes the policy was trained on the standard
        # Isaac Lab Go2 velocity task with this concatenation order. Override
        # via obs_layout if your training was different.
        self.obs_layout = obs_layout or {
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "velocity_cmd": 3,
            "joint_pos": 12,
            "joint_vel": 12,
            "actions": 12,
        }
        self.obs_dim = sum(self.obs_layout.values())   # = 48 for default

    @torch.no_grad()
    def __call__(self, proprio: torch.Tensor, u_safe: torch.Tensor,
                 last_action: torch.Tensor | None = None) -> torch.Tensor:
        """Build the policy input from the env's proprio block + velocity cmd.

        proprio is expected to be (B, P_dim) with the same channel layout as
        obs_layout EXCEPT 'velocity_cmd' — we override that with u_safe.

        u_safe: (B, 2) — planar velocity in body frame from the CBF filter.
        last_action: (B, 12) previous joint targets; some loco policies use
                     this. If None, the env should track and pass it.
        """
        B = u_safe.shape[0]
        # Build velocity command: (vx, vy, yaw_rate). For MVP yaw_rate = 0.
        cmd = torch.zeros(B, 3, device=u_safe.device)
        cmd[:, 0] = u_safe[:, 0]
        cmd[:, 1] = u_safe[:, 1]
        # cmd[:, 2] = 0.0

        # Compose the policy input. The exact slicing here is the part most
        # likely to need adjustment to match the trained policy.
        obs_pieces = []
        idx = 0
        for key, dim in self.obs_layout.items():
            if key == "velocity_cmd":
                obs_pieces.append(cmd)
            elif key == "actions" and last_action is not None:
                obs_pieces.append(last_action)
            else:
                obs_pieces.append(proprio[:, idx:idx + dim])
                idx += dim
        obs = torch.cat(obs_pieces, dim=-1)
        joint_targets = self.policy(obs)
        return joint_targets
