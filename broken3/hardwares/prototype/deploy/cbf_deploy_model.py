"""Standalone CBF deploy inference model.

Loads V13.1 teacher checkpoint + V13.1 student adapter, exposes a single
`infer()` entrypoint. NO IsaacLab dependency — pure PyTorch + numpy.
Designed to run on the Go2's Jetson Orin via ROS 2.

Architecture mirrors V13's two-stream teacher, with the priv encoder
replaced by the student adapter:

    priv_observable_19 (sensor)  → proprio_normalize  ─┐
    history of (proprio, action) → student_adapter → ẑ_env ─┼─→ π_teacher → (α, φ, a, b, c)
    grid_8192          (LiDAR)   → grid_encoder ─────────┘

Usage:
    model = CbfDeployModel("teacher.pt", "student.pt")
    while True:
        action = model.infer(proprio_19, grid_2x64x64)  # returns (α, φ, a, b, c)
"""
from __future__ import annotations

import torch
from torch import nn
import numpy as np


# Layout constants — must match cbf_go2_teacher_rma.py + cbf_go2_env_cfg.py.
# Width constants (z_priv_dim, action_dim) are DERIVED from the checkpoint
# at load time — V13.1 uses z_priv_dim=16 (overridden from the 8 default).
_PRIV_HIDDEN_DIM = 14
_PRIV_OBSERVABLE_DIM = 19
_GRID_CHANNELS = 2
_GRID_H = 64
_GRID_W = 64
_GRID_FLAT = _GRID_CHANNELS * _GRID_H * _GRID_W  # 8192
_Z_GRID_DIM = 64

# Action decoding (same as cbf_go2_env._cbf_filter).
ALPHA_MIN, ALPHA_MAX = 0.5, 3.0
PHI_MIN, PHI_MAX = 0.0, 5.0
A_MIN, A_MAX = 0.0, 0.5
B_MIN, B_MAX = 0.0, 1.0
C_MIN, C_MAX = -0.10, 0.0


class _PrivRunningMeanStd(nn.Module):
    """Per-feature running mean/std normalizer. Stats are loaded from
    the saved teacher checkpoint and frozen at deploy."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + self.eps)


class _PrivEncoder(nn.Module):
    """Deploy version. NOT USED at inference — student adapter replaces
    it. Kept here so the saved teacher state_dict loads cleanly."""

    def __init__(self, input_dim: int, z_priv_dim: int) -> None:
        super().__init__()
        self.normalizer = _PrivRunningMeanStd(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ELU(),
            nn.Linear(64, 64),         nn.ELU(),
            nn.Linear(64, z_priv_dim), nn.ELU(),
        )
        self.z_priv_dim = z_priv_dim
        self.input_dim = input_dim

    def forward(self, priv: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalizer(priv))


class _GridEncoder(nn.Module):
    """LiDAR occupancy grid → z_grid (64-D). Matches cbf_go2_teacher_rma."""

    def __init__(self, z_grid_dim: int = _Z_GRID_DIM) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(_GRID_CHANNELS, 16, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),             nn.ELU(),
        )
        conv_h = _GRID_H // 4
        conv_w = _GRID_W // 4
        conv_flat = 32 * conv_h * conv_w
        self.proj = nn.Sequential(nn.Linear(conv_flat, z_grid_dim), nn.ELU())
        self.z_grid_dim = z_grid_dim

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        feat = self.conv(grid).flatten(1)
        return self.proj(feat)


class _PiTeacherMLP(nn.Module):
    """π_teacher head. Takes z_env ⊕ z_proprio ⊕ z_grid → action mean.
    rsl_rl GaussianDistribution with state_dependent_std=False stores std
    as a separate Parameter, so the MLP outputs action_dim directly (5)."""

    def __init__(self, input_dim: int, hidden: tuple[int, int] = (128, 128),
                 output_dim: int = 5) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ELU(),
            nn.Linear(h1, h2),         nn.ELU(),
            nn.Linear(h2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _StudentAdapter(nn.Module):
    """Mirror of StudentAdaptationModule from cbf_go2_student.py.
    Standalone to avoid the IsaacLab import path."""

    def __init__(self, proprio_dim: int, action_dim: int, z_env_dim: int,
                 history_len: int) -> None:
        super().__init__()
        in_channels = proprio_dim + action_dim
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=4), nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),          nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),          nn.ELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, history_len)
            conv_flat = self.conv(dummy).flatten(1).shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(conv_flat, 128), nn.ELU(),
            nn.Linear(128, 64),         nn.ELU(),
            nn.Linear(64, z_env_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (B, T, F)
        x = history.permute(0, 2, 1)
        z = self.conv(x).flatten(1)
        return self.mlp(z)


# ──────────────────────────────────────────────────────────────────────
# Deploy model — wires it all together.
# ──────────────────────────────────────────────────────────────────────

class CbfDeployModel:
    """V13.1 two-stream deploy inference.

    Loads teacher pi_teacher + grid_encoder + proprio_normalizer from the
    rsl_rl checkpoint. Loads student adapter from its own checkpoint.
    Maintains a single-rollout history buffer (one env at deploy).
    """

    def __init__(self, teacher_checkpoint: str, student_checkpoint: str,
                 device: str = "cpu"):
        self.device = torch.device(device)

        # Peek at the teacher checkpoint to derive widths (z_priv_dim,
        # action_dim, pi_teacher hidden) before constructing modules.
        ckpt = torch.load(teacher_checkpoint, map_location="cpu")
        # rsl_rl save format: top-level 'actor_state_dict' / 'critic_state_dict'.
        sd_root = ckpt.get("actor_state_dict",
                  ckpt.get("model_state_dict", ckpt))
        # priv_encoder last layer: mlp.0.net.4.weight  shape (z_priv_dim, 64)
        z_priv_w = sd_root.get("mlp.0.net.4.weight")
        if z_priv_w is None:
            raise RuntimeError(
                f"Couldn't find 'mlp.0.net.4.weight' in teacher checkpoint "
                f"{teacher_checkpoint}. Top-level keys: {list(sd_root.keys())[:8]}"
            )
        z_priv_dim = z_priv_w.shape[0]                  # 16 for V13.1
        # pi_teacher is a flat nn.Sequential — keys mlp.2.{0,2,4}.weight.
        pi_last_w = sd_root["mlp.2.4.weight"]
        action_dim = pi_last_w.shape[0]                 # 5
        h1 = sd_root["mlp.2.0.weight"].shape[0]
        h2 = sd_root["mlp.2.2.weight"].shape[0]
        z_grid_dim = sd_root["mlp.1.proj.0.weight"].shape[0]
        pi_input_dim = sd_root["mlp.2.0.weight"].shape[1]
        # Sanity: should equal z_priv_dim + 19 + z_grid_dim.
        expected_pi_in = z_priv_dim + _PRIV_OBSERVABLE_DIM + z_grid_dim
        if pi_input_dim != expected_pi_in:
            print(f"[deploy] WARNING: pi_input_dim={pi_input_dim} "
                  f"(expected {expected_pi_in} = z_priv {z_priv_dim} + "
                  f"proprio {_PRIV_OBSERVABLE_DIM} + z_grid {z_grid_dim})")

        self.z_priv_dim = z_priv_dim
        self.z_grid_dim = z_grid_dim
        self.action_dim = action_dim
        print(f"[deploy] derived dims: z_priv={z_priv_dim}, z_grid={z_grid_dim}, "
              f"pi_in={pi_input_dim}, action={action_dim}, hidden=({h1},{h2})")

        # Build modules with the right widths.
        self.priv_encoder = _PrivEncoder(input_dim=_PRIV_HIDDEN_DIM,
                                          z_priv_dim=z_priv_dim).to(self.device)
        self.grid_encoder = _GridEncoder(z_grid_dim=z_grid_dim).to(self.device)
        self.proprio_normalizer = _PrivRunningMeanStd(_PRIV_OBSERVABLE_DIM).to(self.device)
        self.pi_teacher = _PiTeacherMLP(input_dim=pi_input_dim,
                                         hidden=(h1, h2),
                                         output_dim=action_dim).to(self.device)

        self._load_teacher(teacher_checkpoint, sd_root)

        # Load student adapter (config saved in its checkpoint).
        sckpt = torch.load(student_checkpoint, map_location=self.device)
        cfg = sckpt["config"]
        self.student = _StudentAdapter(
            proprio_dim=cfg["proprio_dim"], action_dim=cfg["action_dim"],
            z_env_dim=cfg["z_env_dim"], history_len=cfg["history_len"],
        ).to(self.device)
        self.student.load_state_dict(sckpt["state_dict"])
        self.history_len = cfg["history_len"]
        print(f"[deploy] student loaded. R²_test_best={sckpt.get('best_test_r2', '?')}")

        # Eval mode (no dropout, no batchnorm updates).
        self.priv_encoder.eval()
        self.grid_encoder.eval()
        self.proprio_normalizer.eval()
        self.pi_teacher.eval()
        self.student.eval()

        # Single-env history buffer (deploy = 1 robot).
        self._history = torch.zeros(
            1, self.history_len, _PRIV_OBSERVABLE_DIM + self.action_dim,
            device=self.device,
        )
        self._prev_action = torch.zeros(1, self.action_dim, device=self.device)
        self._initialized = False

    def _load_teacher(self, path: str, sd: dict | None = None) -> None:
        """Load teacher's pi_teacher + grid_encoder + proprio_normalizer weights
        from an rsl_rl checkpoint. State-dict keys live under 'model_state_dict'
        with nested paths like 'mlp.0.normalizer.mean', 'mlp.1.proj.0.weight'."""
        if sd is None:
            ckpt = torch.load(path, map_location=self.device)
            # rsl_rl save format: dict with 'model_state_dict' (actor) etc.
            if "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            elif "actor_state_dict" in ckpt:
                sd = ckpt["actor_state_dict"]
            elif "actor" in ckpt and isinstance(ckpt["actor"], dict):
                sd = ckpt["actor"]
            else:
                sd = ckpt

        def strip_prefix(sd, prefix):
            return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

        # Extract sub-state-dicts via prefix matching.
        priv_sd = strip_prefix(sd, "mlp.0.")
        grid_sd = strip_prefix(sd, "mlp.1.")
        pi_sd = strip_prefix(sd, "mlp.2.")
        proprio_sd = strip_prefix(sd, "mlp.proprio_normalizer.")

        if not pi_sd:
            raise RuntimeError(
                f"Could not extract pi_teacher weights from {path}. "
                f"Top-level keys: {list(sd.keys())[:5]}"
            )

        self.priv_encoder.load_state_dict(priv_sd, strict=False)
        self.grid_encoder.load_state_dict(grid_sd, strict=False)
        # pi_teacher saved state is flat {0.weight, 0.bias, 2.weight, ...},
        # our nn.Module wraps these under self.net.{...}. Try both remappings.
        try:
            self.pi_teacher.net.load_state_dict(pi_sd, strict=True)
        except Exception:
            remapped = {f"net.{k}": v for k, v in pi_sd.items()}
            self.pi_teacher.load_state_dict(remapped, strict=False)
        self.proprio_normalizer.load_state_dict(proprio_sd, strict=False)

        print(f"[deploy] teacher loaded from {path}")
        print(f"  priv_encoder keys: {len(priv_sd)}  grid_encoder keys: {len(grid_sd)}  "
              f"pi_teacher keys: {len(pi_sd)}  proprio_normalizer keys: {len(proprio_sd)}")

    def reset_history(self, proprio_19: np.ndarray | torch.Tensor | None = None) -> None:
        """Clear history buffer. Called on episode start / robot stand-up.
        If `proprio_19` provided, fill history with that state (avoids the
        50-step warmup with zeros)."""
        if proprio_19 is None:
            self._history.zero_()
        else:
            if isinstance(proprio_19, np.ndarray):
                proprio_19 = torch.from_numpy(proprio_19).float()
            proprio_19 = proprio_19.to(self.device).view(1, -1)
            zero_a = torch.zeros(1, self.action_dim, device=self.device)
            step = torch.cat([proprio_19, zero_a], dim=-1)  # (1, F)
            self._history = step.unsqueeze(1).expand(-1, self.history_len, -1).clone()
        self._prev_action.zero_()
        self._initialized = True

    @torch.no_grad()
    def infer(self, proprio_19: np.ndarray | torch.Tensor,
              grid_2x64x64: np.ndarray | torch.Tensor) -> dict:
        """Single-step inference.
        proprio_19: (19,) — base_height(1) + tracking_err(15) + base_ang_vel(3)
        grid_2x64x64: (2, 64, 64) — 2-channel ego-centric LiDAR occupancy
        Returns dict with raw_action (5,) and decoded physical params.
        """
        if not self._initialized:
            self.reset_history(proprio_19)

        if isinstance(proprio_19, np.ndarray):
            proprio_19 = torch.from_numpy(proprio_19).float()
        if isinstance(grid_2x64x64, np.ndarray):
            grid_2x64x64 = torch.from_numpy(grid_2x64x64).float()

        proprio = proprio_19.to(self.device).view(1, _PRIV_OBSERVABLE_DIM)
        grid = grid_2x64x64.to(self.device).view(1, _GRID_CHANNELS, _GRID_H, _GRID_W)

        # Push (proprio_t, prev_action) onto history.
        new_step = torch.cat([proprio, self._prev_action], dim=-1)
        self._history = torch.cat(
            [self._history[:, 1:], new_step.unsqueeze(1)], dim=1
        )

        # Forward.
        z_env = self.student(self._history)                # (1, 8)
        z_proprio = self.proprio_normalizer(proprio)        # (1, 19)
        z_grid = self.grid_encoder(grid)                    # (1, 64)
        joint = torch.cat([z_env, z_proprio, z_grid], dim=-1)
        raw_action = self.pi_teacher(joint).squeeze(0)     # (5,)
        self._prev_action = raw_action.unsqueeze(0).detach()

        # Decode raw action → physical params (same tanh+scale as env).
        def decode(r, lo, hi):
            return lo + (torch.tanh(r) + 1.0) * 0.5 * (hi - lo)

        alpha = decode(raw_action[0], ALPHA_MIN, ALPHA_MAX).item()
        phi = decode(raw_action[1], PHI_MIN, PHI_MAX).item()
        a = decode(raw_action[2], A_MIN, A_MAX).item()
        b = decode(raw_action[3], B_MIN, B_MAX).item()
        c = decode(raw_action[4], C_MIN, C_MAX).item()

        return {
            "raw": raw_action.cpu().numpy(),
            "alpha": alpha, "phi": phi, "a": a, "b": b, "c": c,
        }
