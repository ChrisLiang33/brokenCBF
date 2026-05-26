"""
Step 2d.1: verify the trained locomotion policy can be loaded and run.

Loads the exported TorchScript policy and runs a dummy forward pass to
confirm it accepts 48D observations and outputs 12D joint targets.

Run on the lab desktop with the isaaclab conda env activated.
"""

import torch

CHECKPOINT = (
    "/home/chrisliang/Desktop/safety-go2/IsaacLab/logs/rsl_rl/"
    "unitree_go2_flat/2026-04-15_19-24-45/exported/policy.pt"
)

# Load TorchScript policy. This is a frozen, compiled model — no rsl_rl
# dependency needed, just pure torch.
policy = torch.jit.load(CHECKPOINT)
policy.eval()
print(f"Loaded policy from {CHECKPOINT}")
print(f"Policy type: {type(policy).__name__}")

# Move to GPU to match what Isaac Lab uses at runtime
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy = policy.to(device)
print(f"Moved to device: {device}")

# Dummy observation — 48D per the Go2 locomotion env setup:
#   3 base_lin_vel + 3 base_ang_vel + 3 projected_gravity +
#   3 velocity_commands + 12 joint_pos + 12 joint_vel + 12 last_actions
dummy_obs = torch.zeros((1, 48), device=device)

# Forward pass (no gradients, we're just using the policy)
with torch.no_grad():
    action = policy(dummy_obs)

print(f"\nDummy obs shape:  {tuple(dummy_obs.shape)}")
print(f"Action shape:     {tuple(action.shape)}")
print(f"Action dtype:     {action.dtype}")
print(f"Action min/max:   {action.min().item():.3f} / {action.max().item():.3f}")

# Expected: action shape (1, 12), dtype float32 (cuda).
assert action.shape == (1, 12), f"Expected (1, 12), got {action.shape}"
print("\nStep 2d.1: trained locomotion policy loads and runs inference.")
