# Professor Update — 2026-05-11

## Headline

Layer 1 (α-only adaptive CBF) reached its ceiling. α adapts cleanly to local CBF geometry but has no gradient path to environment class. Moving to Layer 2: release φ + actuation-noise DR, where the physical lever and the randomization axis line up.

## Setup recap

- Two-level: teacher PPO outputs CBF params, frozen locomotion controller below, hard CBF half-space projection between them.
- RMA-style split encoder (Kumar et al.): privileged DR features → z_priv (16-dim MLP), local occupancy grid → z_grid (CNN). cbf_state routed to value head only, not policy input.
- Released params currently: α only. Hand-coded defaults for φ, a, b, c.
- Headline metric: combined fall + stuck rate. Win = released-param adaptive beats best-fixed-α by ≥3pp on ≥1 task.
- Adaptation diagnostic: Pearson(α, DR feature) > 0.20 within distribution.

## Diagnostic chain (Layer 1)

v3.0a-d failed with monolithic CNN (encoder collapse — grid path starved dyn path for gradient). v3.0e-f tried pretrain+freeze + analytical aux loss, but α saturated at 5.0 with std=0. v3.1 onward used RMA split.

| Run | Key change | α μ/σ | max \|Pearson(α, DR)\| | HeavyCOM Δcombined |
|---|---|---|---|---|
| v3.1 | RMA split + AUX=0.005 + drop v_along_cmd + no cbf_state in policy | 4.0 / 1.5 | 0.008 | tie |
| v3.2 | + 45% adversarial planner + grazing 28° + speed 1.2 + priv_fov FOV-gated CBF + AUX=0 | 3.05 / 1.96 | <0.10 | +3.01pp |
| v3.2.1 | + z_priv 8→16 (capacity test) | 3.27 / 1.91 | 0.063 | -0.15pp |

## What v3.2.1 settled

- **Encoder capacity is not the bottleneck.** Doubling z_priv left DR correlations essentially unchanged.
- **v3.2's HeavyCOM +3.01pp did not replicate.** Re-train collapsed it to noise. Not load-bearing.
- **α correlates strongly with CBF state across all runs**: slack r=0.84, h r=0.71, ‖L_g h‖² r=-0.66.
- **DR correlations stay at |r| < 0.07** regardless of architecture, encoder capacity, or training distribution.

α is geometry-reactive: it ramps up as the CBF QP approaches activation and ramps down in open space. It does not distinguish episodes by friction, mass, COM offset, or applied disturbance magnitude.

## Why α has no env-class gradient path

The dense reward term `cbf_lhs_margin -0.1` penalizes negative slack (= L_g h · u_des + α·h), and α·h is the policy's direct lever on it. That's the proximate cause of the geometry coupling — and it's the same term that broke v3.0f's saturation, so it's the signal carrier we want to keep, not the obstacle.

Deeper structural reason: α scales the h-recovery rate. The DR axes (friction, mass, COM) act on the robot through slip dynamics or inertia, all routed through the frozen locomotion controller. There's no control path from α to those dynamics. Even with perfect credit assignment, the friction-optimal α isn't meaningfully different from the high-friction-optimal α, because "am I about to hit" dominates "how slippery is the floor" in the QP's loss landscape.

Read: this is a parameter-vs-DR-axis mismatch, not a training failure. No amount of encoder tuning or reward shaping fixes it. The fix is to pair each released parameter with a DR axis whose physics it actually touches.

## Layer 2 plan (launching today)

Release φ alongside α. CBF QP rhs:

```
rhs = -α(h - c) + φ · ‖L_g h‖² + a
```

φ multiplies ‖L_g h‖² — this is Kolathaya's ISSf input-disturbance margin term. It has a clean physical match to **actuation noise**: both are about L_g h estimate uncertainty.

- Add `actuation_noise_sigma` to teacher priv obs (priv_dim 15 → 16).
- Widen actuation noise DR: σ_max 0.10 → 0.20.
- Hold Layer 1 setup constant otherwise (RMA arch, adversarial planner, AUX=0, priv_fov, lhs reward).
- New OOD eval task: high-noise (σ_max=0.30) to test φ adaptation transfer.

Layer 2 PASS criterion: Pearson(φ, actuation_noise_sigma) > 0.20, combined-metric win ≥3pp on ≥1 task (in-dist or high-noise), Layer 1 geometry behavior on α preserved.

If Layer 2 also fails: Layer 3 (release `a` + perception-noise DR), Layer 4 (release `c` + radius-error DR). If all fail, we revisit whether the per-step-adaptive-param framing is viable at all and pivot toward per-episode adaptation only.

## Open questions

1. **Is the "α has no lever on slip/inertia" argument sound?** My read is that under hard CBF projection with a frozen locomotion controller below, α's gradient path to friction/mass DR axes is structurally absent — not just hard to find. If you'd push back, would value the pointer.

2. **φ ↔ actuation noise pairing** — picked this because it's the cleanest L_g h uncertainty proxy. Alternatives I considered: motor strength DR, contact stiffness DR. Reasonable, or is there a better one?

3. **What claim Layer 2 results would best support** — would rather align with you on framing before writing than after.

## Status

- v3.2.1 done. Layer 2 fully drafted locally, syncing to lab today (~12h pipeline).
- Hardware deploy on Go2 still on track for late May.
- All diagnostic tools (linear probe, α-corr, Z health) running per-version, artifacts archived per run.
