"""Deploy-side wrapper around the trained CBF parameter policy.

Loads a PyTorch checkpoint from the cbf_rl_mvp / go2 RL training side and
exposes a single ``infer()`` entrypoint that returns physical CBF
parameters (alpha, phi) to be consumed by cbf_filter_node.

Action space (matches sim — see env.py / go2/):
    raw_action[0] -> alpha  in [ALPHA_MIN, ALPHA_MAX]
    raw_action[1] -> phi    in [PHI_MIN,   PHI_MAX]

a, b, c are pinned to 0 in the MVP and are NOT emitted here. Re-add a
slot if the policy is later extended to output them.

Status — V0 SCAFFOLD:
    The infer() method below is a stub that returns hand-tuned safe
    defaults. This lets the full ROS pipeline (lidar -> grid -> inference
    -> filter -> walking_bridge) come up end-to-end and be validated
    before any learned policy is wired in.

    To swap in the real policy:
      1. Implement _load_checkpoint() — load the SB3/rsl_rl checkpoint
         produced by the training stack, build the encoder + MLP modules,
         restore state_dict.
      2. Implement _build_obs() — assemble the observation vector in the
         exact order the trained policy expects. Cross-check against
         go2/<latest_phase>/env.py or the MVP's obs builder.
      3. Implement _forward() — torch.no_grad() inference, tanh-decode
         raw outputs into physical (alpha, phi).

    The watchdog / fail-safe layer in cbf_inference_node.py is policy-
    agnostic and does not need to change when you swap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


# Physical parameter ranges. Must match the action decoder in the training
# env (see go2/.../env.py and cbf_rl_mvp/cbf_qp.py). Update here if the
# training side changes its decode bounds.
ALPHA_MIN, ALPHA_MAX = 0.5, 3.0
PHI_MIN,   PHI_MAX   = 0.0, 5.0

# Conservative-but-functional hand-tuned defaults. Used by the stub
# infer() and as fallback values inside cbf_inference_node.
SAFE_ALPHA = 2.0
SAFE_PHI   = 0.5


class CbfDeployModel:
    """V0 stub. Returns hand-tuned (alpha, phi) until a real checkpoint
    is wired in. See module docstring for the swap-in procedure."""

    def __init__(self, checkpoint: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint
        self.is_stub = True

        if checkpoint:
            ckpt_path = Path(checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
            # TODO(deploy): replace with real loading once the trained
            # policy format is locked.
            #   ckpt = torch.load(ckpt_path, map_location=self.device)
            #   self._build_policy(ckpt)
            #   self.policy.eval()
            #   self.is_stub = False
            print(f"[deploy] checkpoint provided ({ckpt_path}) but loader "
                  f"not yet implemented; running in SAFE-DEFAULTS stub mode.")
        else:
            print("[deploy] no checkpoint provided; running in SAFE-DEFAULTS stub mode.")

    def reset_history(self, proprio: np.ndarray | None = None) -> None:
        """Reset any rolling-history buffer the policy carries (no-op for
        the stub). Called by cbf_inference_node when an input gap > the
        history_reset window is detected."""
        return

    @torch.no_grad()
    def infer(self, proprio: np.ndarray, grid: np.ndarray) -> dict:
        """Single-step inference.

        Args:
            proprio: 1D proprio observation vector (shape depends on the
                trained policy; see _build_obs in cbf_inference_node).
            grid: 2-channel ego-centric LiDAR occupancy grid, shape
                (2, 64, 64).

        Returns:
            {"alpha": float, "phi": float, "raw": np.ndarray | None}
            where ``raw`` is the pre-decode policy output (None in stub
            mode).
        """
        if self.is_stub:
            return {"alpha": SAFE_ALPHA, "phi": SAFE_PHI, "raw": None}

        # TODO(deploy): real forward pass.
        #   x_proprio = torch.from_numpy(proprio).float().to(self.device)
        #   x_grid    = torch.from_numpy(grid).float().to(self.device)
        #   raw = self.policy(x_proprio, x_grid).squeeze(0)
        #   alpha = _decode(raw[0], ALPHA_MIN, ALPHA_MAX)
        #   phi   = _decode(raw[1], PHI_MIN,   PHI_MAX)
        #   return {"alpha": alpha, "phi": phi, "raw": raw.cpu().numpy()}
        raise NotImplementedError("real inference path not yet wired")


def _decode(raw: torch.Tensor, lo: float, hi: float) -> float:
    """tanh + scale into [lo, hi]. Matches the env's action decoder."""
    return float(lo + (torch.tanh(raw) + 1.0) * 0.5 * (hi - lo))
