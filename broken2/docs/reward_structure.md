# Reward Structure — Layer 2

Reference doc for the CBF teacher's reward stack.
Last updated: 2026-05-12 (Layer 2 v2 training in flight)

## All terms (current, in order of action manager)

| # | Term | Weight | Formula | Density | Purpose |
|---|------|--------|---------|---------|---------|
| 1 | `collision` | **-100** | 1.0 on collision event, 0 otherwise | Terminal (~1×/ep) | Episode-level: did the robot hit an obstacle |
| 2 | `base_contact_penalty` | **-500** | 1.0 if base/torso contacts ground or obstacle | Terminal (~1×/ep) | Fall detection. Larger than `collision` because falls are catastrophic and harder to recover from on hardware |
| 3 | `infeasibility` | **-10** | 1.0 if CBF-QP returned infeasible last step | Dense-rare | Penalizes the policy for setting (α, φ, a, c) such that no safe `u` exists. Prevents the QP from being driven into degenerate regions |
| 4 | `proximity` | **-0.5** | `exp(-min_surface_dist / 0.5)` | Dense (when near obstacle) | Soft "fear field" — geometric distance penalty that decays with `exp`. Effective range ~1.5m. Indirect gradient on CBF params via trajectory |
| 5 | `cbf_lhs_margin` | **-0.1** | `softplus(-slack)` where `slack = L_g h · u_des + α·h + φ·‖L_g h‖² + a` | Dense (when slack small) | The load-bearing dense reward. Direct gradient on α, φ, a via slack math. Specifically designed to fix the zero-gradient problem when the QP is idle |
| 6 | `u_safe_deviation` | **-0.1** | `‖u_safe - u_des‖₂` | Dense (when QP active) | Penalizes the magnitude of the CBF's deflection from the nominal velocity. Soft pressure to "let u_des through" when safe |
| 7 | `action_rate` | **-0.005** | `Σ(a_t - a_{t-1})²` over 5 CBF params | Dense | Smoothness: penalizes step-to-step jerkiness in the (α, φ, a, b, c) outputs. Important for stability on hardware |
| 8 | `stuck` | **-1.0** | 1.0 if base speed < 0.15 m/s | Dense (when slow) | Performance pressure: discourages over-conservative behavior that paralyzes the robot |
| 9 | `velocity_along_cmd` | **0.0 (OFF)** | `v · u_des / ‖u_des‖` (positive when moving toward goal) | DEAD | Currently disabled. Was the explicit "forward progress" reward — turned off due to instability in earlier iterations. Code retained for potential re-enable |

Term definitions live in [IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_rewards.py](../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_rewards.py).

Weights are configured in [IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py](../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py).

## Grouped by role

```
Safety (push CBF params HIGH):
  collision (-100)        : terminal, sparse, "did I hit?"
  base_contact (-500)     : terminal, sparse, "did I fall?"
  infeasibility (-10)     : dense-rare, "did I pick bad params?"
  proximity (-0.5)        : dense, geometric, "am I close to an obstacle?"
  cbf_lhs_margin (-0.1)   : dense, *direct gradient*, "at QP boundary?"

Filter behavior (regularization):
  u_safe_deviation (-0.1) : dense, "how much did CBF deflect from u_des?"
  action_rate (-0.005)    : dense, "are param outputs smooth?"

Performance (push CBF params LOW / get the robot moving):
  stuck (-1.0)            : dense, "is robot slow?"
  velocity_along_cmd (0.0): DEAD, was forward-progress reward
```

## Key things to know

### 1. `cbf_lhs_margin` is load-bearing

This term is what broke the v3.0e/f saturation. Without it, the policy gets zero gradient on α/φ when the QP is idle (most steps). The mechanism: it provides a direct analytical gradient on α, φ, a via the slack equation:

```
slack = L_g h · u_des + α·h + φ·‖L_g h‖² + a
∂R/∂α = -0.1 · sigmoid(-slack) · h
∂R/∂φ = -0.1 · sigmoid(-slack) · ‖L_g h‖²
∂R/∂a = -0.1 · sigmoid(-slack) · 1
```

All non-zero whenever the policy is near the constraint boundary, regardless of QP activation. This is the fix for the "zero-gradient problem" common to differentiable-control + RL setups.

### 2. Two terminal penalties dominate cumulative reward when they fire

`collision` (-100) and `base_contact` (-500) are the env-class signal carriers. Low-friction episodes have more falls → more terminal penalty → policy could in principle learn to adapt. But this signal is sparse: at most ~1-3 events per 800-step episode.

The dense rewards (proximity, lhs_margin) provide ~80× more update signal but are pure functions of geometry — friction doesn't appear in their formulas.

### 3. Velocity-along-cmd is currently OFF

The most natural "performance pull" reward is disabled. Without it, the only performance pressure on the policy is `stuck` (binary threshold) and `u_safe_deviation` (small). This contributes to the safety-skewed equilibrium.

### 4. The geometric-bias problem

Five of the dense terms (proximity, cbf_lhs_margin, infeasibility, u_safe_deviation, partly action_rate) are pure functions of robot-obstacle geometry. They don't depend on env-class properties (friction, mass, COM offset).

Consequence: dense gradient on CBF params is geometry-conditioned. α correlates strongly with slack/h/‖L_g h‖² (Pearson > 0.6) and not with friction/mass/COM (Pearson < 0.07). This is the Layer 1 ceiling we hit.

### 5. The FOV gating effect

The dense rewards (proximity, lhs_margin) are also effectively FOV-gated:
- `proximity` decays as `exp(-d/0.5)` — essentially zero past 1.5m
- `lhs_margin` is silent in open space because we clamp `h = 100` when no obstacle is within 3.2m FOV (priv_fov mode in v3.2+)

Net: when no obstacle is in FOV (~10% of training steps), there's no dense gradient on the CBF params. The policy's behavior in open space is shaped only by `action_rate` smoothness and the (currently weak) terminal signal.

## Layer 2's question, in this context

Layer 2 releases φ and pairs it with a new DR axis (`actuation_noise_sigma`, σ_max=0.20). Same reward stack. Hypothesis: φ has a physical pairing with actuation noise (both relate to L_g h estimate uncertainty) that α lacked, so the env-class adaptation might emerge for φ where it didn't for α.

The headline test: `Pearson(φ, actuation_noise_sigma) > 0.20`.

## Queued ablations / refinements

If Layer 2 v2 passes, these are nice-to-haves. If it fails, they become the active investigation queue.

| # | Change | Rationale | Risk |
|---|--------|-----------|------|
| A1 | Reduce `cbf_lhs_margin` weight: -0.1 → -0.02 | Shift dense:sparse signal ratio without removing dense entirely | Possible saturation regression |
| A2 | Drop `cbf_lhs_margin` AND `proximity` entirely | Test whether removing geometric dense signal lets env-class signal surface | High — v3.0e/f-style saturation |
| A3 | Add `r_efficiency = -w·a - w·c` for Layer 3/4 | Creates "tug of war" equilibrium with LHS reward → forces margins to track actual disturbance level (env-class adaptive) | Low, well-motivated by external advisor |
| A4 | Re-enable `velocity_along_cmd` with reweighting | Re-introduce explicit performance pull | Possible re-introduction of instability |
| A5 | Increase `action_rate`: -0.005 → -0.02 | Make smoothness cost meaningful enough to enforce baseline-park behavior in open space | Mild, well-bounded |
| A6 | Drop `ALPHA_MIN` floor (1.0 → 0.0) | Let PPO discover natural baseline through smoothness + obstacle dynamics | Requires A5 to work |

## Status of each parameter

| Slot | Released in | DR axis paired | Current state |
|------|------------|----------------|---------------|
| α | Layer 1 (v3.0+) | None (intentionally) | Geometry-learnable, NOT env-class-learnable |
| φ | Layer 2 (in test) | `actuation_noise_sigma` ∈ [0, 0.2] | Layer 2 v2 in training, headline pending |
| a | Layer 3 (planned) | `cbf_obs_pos_noise_sigma` (perception bias) | Frozen at 0 |
| b | Never (intentionally) | None — would make constraint SOCP | Frozen at 0, action slot ignored |
| c | Layer 4 (planned) | `cbf_obs_radius_error` (radius mismatch) | Frozen at 0 |
