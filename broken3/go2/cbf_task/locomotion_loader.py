"""Load the frozen Go2 velocity-tracking locomotion policy from an rsl_rl
checkpoint, WITHOUT needing rsl_rl's OnPolicyRunner (and therefore without
needing a live env).

The stock rsl_rl checkpoint stores the actor under `actor_state_dict` with
keys nested as `mlp.0.weight`, `mlp.0.bias`, ..., `mlp.6.weight`,
`mlp.6.bias`. Our stock Go2 actor is a 4-layer MLP with ELU activations:
    Linear(48, 128) -> ELU -> Linear(128, 128) -> ELU -> Linear(128, 128)
    -> ELU -> Linear(128, 12)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def load_locomotion_actor(checkpoint_path: str, device: str) -> nn.Module:
    """Returns an eval-mode MLP that maps 48-dim obs -> 12-dim joint action."""
    actor = nn.Sequential(
        nn.Linear(48, 128), nn.ELU(),
        nn.Linear(128, 128), nn.ELU(),
        nn.Linear(128, 128), nn.ELU(),
        nn.Linear(128, 12),
    )
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "actor_state_dict" not in loaded:
        raise KeyError(
            "Expected 'actor_state_dict' in checkpoint. "
            f"Top-level keys present: {list(loaded.keys())}"
        )
    actor_sd = loaded["actor_state_dict"]
    # Strip the `mlp.` prefix so keys align with our nn.Sequential indices.
    mlp_sd = {k[len("mlp."):]: v for k, v in actor_sd.items() if k.startswith("mlp.")}
    if not mlp_sd:
        raise KeyError(
            "No 'mlp.*' keys found in actor_state_dict. "
            f"Keys present: {list(actor_sd.keys())}"
        )
    actor.load_state_dict(mlp_sd, strict=True)
    actor = actor.to(device).eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    return actor
