# Wk3 Scenario Design Doc — Bidirectional Robust-Param Stress

Authored: 2026-05-14, after v20/v21 Goodhart failures revealed the "Fat Robot" exploit.

## 0. Methodology principle

**For each released robust CBF parameter, the training distribution must contain scenarios where BOTH extremes of the parameter's value are penalized.** A parameter only learns dynamic adaptation if PPO is shown that both "too small" and "too large" carry cost. If only one direction is penalized, the policy collapses to the unpenalized extreme — what we just empirically watched happen with φ in v20/v21 (Fat Robot: locked φ at 3.83 / 3.99 because nothing in the open-spawn training distribution punished a large safety bubble).

Each parameter's theoretical role gives us the LOW-is-wrong scenario (the uncertainty it defends against). The HIGH-is-wrong scenario must be designed by hand — we have to create environmental conditions where excessive defense costs the policy reward.

## 1. The full mapping (α, φ, a, c)

| Param | Theoretical defense | LOW is wrong when… | HIGH is wrong when… |
|---|---|---|---|
| **α** (class-K rate) | Dynamics uncertainty | High mass/friction/COM/force → can't stop in time → fall | Calm dynamics → over-decelerates → loses velocity reward |
| **φ** (ISSf input buffer) | Actuation noise σ | High σ → margin insufficient → noisy collision | Tight space → bubble doesn't fit → stuck (v22 corridor) |
| **a** (additive slack) | Bounded ḣ disturbance from perception/tracker | High v_obs noise on fast obstacles → margin too tight → fall | Clean tracker output → constant additive margin → always slow / over-conservative |
| **c** (h-shift) | Systematic h-bias (LiDAR range error) | Persistent LiDAR range bias → perceived h overstates safety → fall | Unbiased perception → c shifts boundary inward unnecessarily → stuck near obstacles |

α and φ already covered as of v22 (corridor scenes just added the φ-HIGH penalty for the first time). a and c need explicit design before release.

## 2. Per-parameter Wk3 design

### 2.1 `a` (additive slack)

**Role:** absorbs zero-mean bounded ḣ uncertainty — anything α/φ can't pin down. Most relevant Wk3 source: tracker output noise on `v_obs` (the velocity estimate fed to the L_f h drift term).

**LOW-is-wrong scenario — "Noisy Tracker, Fast Obstacles":**
- 30% of episodes
- High obstacle velocity (`max_speed_range = (0.6, 1.0)`)
- High zero-mean noise on tracker `v_obs` output (σ ≈ 0.2 m/s per axis)
- Occasional dropouts (15% of obstacles get v_obs = 0 for a window each episode)
- Without `a > 0`, the L_f h drift term will be wrong by O(0.2 m/s × ‖L_g h‖), constraint margin insufficient when approaching fast-drifting obstacles → fall

**HIGH-is-wrong scenario — "Clean Tracker, Slow Obstacles":**
- 30% of episodes
- Low obstacle velocity (`max_speed_range = (0.0, 0.2)`)
- Low/zero tracker noise (σ ≈ 0.02 m/s)
- No dropouts
- If `a` is locked high, the constraint LHS has a permanent positive offset → QP forces u_safe deflection even when not needed → robot constantly biased away from u_des → velocity tracking drops → slow / stuck near goal

**Remaining 40%:** random mix (current behavior) — moderate tracker noise, moderate obstacle speed.

**Implementation:** extend the existing `cbf_obstacle_velocities` machinery in `cbf_go2_env.py`. Add `cbf_v_obs_tracker_noise_per_episode` buffer (per-env, per-obstacle, sampled at reset). When the CBF queries v_obs for L_f h, it reads the noisy version. Pure tensor ops, fits the existing fake-tracker hack pattern.

### 2.2 `c` (h-shift)

**Role:** compensates for systematic bias in h itself — LiDAR range measurement error that biases the perceived distance to obstacles.

**LOW-is-wrong scenario — "Biased LiDAR":**
- 30% of episodes
- Per-episode persistent positive bias on LiDAR range measurement (σ = 0.10 m, bias drawn once at reset)
- This means the QP-side `h` is biased HIGH (perceived farther than true) → without `c > 0`, constraint kicks in too late → fall

**HIGH-is-wrong scenario — "Honest LiDAR":**
- 30% of episodes
- Zero bias on LiDAR range
- If `c` is locked high, the effective boundary is shifted inward → the QP intervenes when h is still large → robot can't approach obstacles even when safe → can't reach goals that require close approach → stuck or timed out

**Remaining 40%:** mixed bias levels.

**Implementation:** add `cbf_obstacle_radius_bias_per_episode` and `cbf_obstacle_position_bias_per_episode` buffers. Already partially exists per `obstacle_radius_perception_error_max=0.10` and `obstacle_position_noise_sigma_max=0.05`. For v23+, **vary these biases per scene type** instead of using the same range for every episode. Some episodes get zero bias, some get ±10 cm — that's the bidirectional stress.

## 3. Order of operations (revised Wk3 plan)

Prerequisite chain — each step depends on the previous:

| # | Step | What it enables |
|---|---|---|
| 1 | Switch h computation from ground-truth obstacle positions to LiDAR-derived estimates | Foundation for everything Wk3 |
| 2 | Add classical tracker (or fake-tracker hack) for v_obs estimation, with knobs for noise/latency/dropout | Foundation for `a` scenario design |
| 3 | Add per-episode scene-type sampling for LiDAR/tracker noise levels (not just per-axis DR) | Enables bidirectional `a` and `c` stress |
| 4 | FOV-gate h to 3.2m radius (so policy and h see the same world) | Required for student distillation parity |
| 5 | Bump obstacle `max_speed_range` to `(0.0, 1.0)` for the "Fast Obstacles" sub-scenario | Makes L_f h meaningful, gives `a` real work |
| 6 | Release `a` and `c` from frozen=0 | Now the params have signal to learn against |
| 7 | Baseline parity: B0/B1/B2 must use SAME noisy LiDAR-derived h + tracker v_obs as BR. **Critical** — otherwise baselines have an unfair perfect-information advantage and ablation is meaningless | Table 2 foundation |

Each step is ~1–3 days of implementation + validation. Total Wk3: ~5–7 days.

## 4. Training distribution composition (final Wk3+)

Combining v22's φ-stress + the new a/c stress, the training scene-type mix becomes:

| Scene type | Proportion | What it stresses |
|---|---|---|
| Random (default open) | ~40% | General competence; mild stress on all params |
| Corridor (tight gap) | 20% | φ-HIGH wrong |
| Dense pack (weave) | 10% | φ-HIGH wrong (different form) |
| Noisy Tracker + Fast Obstacles | 15% | a-LOW wrong |
| Clean Tracker + Slow Obstacles | 5% | a-HIGH wrong (mild — also covered by random) |
| Biased LiDAR | 7% | c-LOW wrong |
| Honest LiDAR + Goal-near-obstacle | 3% | c-HIGH wrong (force robot to approach close) |

Each scene type's reward signal naturally penalizes the wrong parameter value. Pre-condition: the encoder must encode the relevant DR axis (v19's normalization fix handles this).

## 5. Diagnostic gates per parameter

For Wk3 release, before declaring a parameter "learned its theoretical role":

- **α:** existing diagnostics (Pearson(α, mass), Pearson(α, |force|)) — should hold from v19+
- **φ:** Pearson(φ, σ) > 0.20 AND Pearson(φ, grid_change) < −0.20 (BOTH couplings, not just one)
- **a:** Pearson(a, v_obs_tracker_noise_mag) > 0.20 AND avg a-value bounded (NOT pegged near max)
- **c:** Pearson(c, lidar_range_bias) > 0.20 AND avg c-value bounded

If any param fails the "bounded" check (i.e., is pegged at extreme), it means the HIGH-is-wrong scenario isn't strong enough — increase the proportion or sharpen the penalty.

## 6. Baseline parity (Wk3 Step 7)

This is the most easily-missed requirement and where the paper claim could break:

Currently B0/B1/B2 use the SAME `_encode_dim` pathway as BR but read h from ground-truth obstacle positions (with the same 5 cm bias the QP sees for BR). At Wk3, **BR will use LiDAR-derived h + tracker v_obs while baselines still use ground-truth** — unfair comparison.

Fix: in `eval_baseline.py`, when v ≥ 23, B0/B1/B2 must:
- Read h from the LiDAR-derived occupancy grid (not ground-truth positions)
- Read v_obs from the tracker (not ground-truth velocities)
- Their `a` and `c` remain pinned at 0 (their architectural limitation)

This is where the paper's central claim lands: under realistic perception, BR's adaptive `a` and `c` should outperform fixed CBF baselines that can't compensate for perception noise.

## 7. Risks and unknowns

1. **The encoder might not see persistent bias** — `c` requires the encoder to distinguish biased vs unbiased episodes from priv obs. Need to ensure the bias level is exposed in priv obs (not just used internally by the QP).
2. **PPO may not discover dynamic a/c in 1500 iters** — the signals for these params are subtler than α/φ. May need 2500+ iters or curriculum staging.
3. **Locomotion controller robustness** — fast obstacles + dynamic a/c could create new u_safe behaviors the locomotion hasn't seen. May need locomotion re-tuning if falls spike.
4. **Tracker hack realism gap** — synthetic latency/dropout may not match real Kalman behavior on hardware. Acceptable for sim → student transfer if we document the gap.

## 8. Open questions

- Do we want the "scene type" to be exposed to the policy as a feature? (Probably no — policy should infer from priv obs)
- Should v23 introduce dense-pack alongside corridor, or wait for v22 result? (Lean: wait — v22 might be enough for φ.)
- Is releasing a and c simultaneously safe, or do we need a single-param phase first? (Per "try simple first" — try simultaneous release. Curriculum only if it fails.)

## 9. Provenance

This design emerged from the v20/v21 failure analysis:
- **v20:** added u_safe_rate=−0.05 (jerk tax). Policy locked φ at 3.83 (zero derivative satisfies the tax). Fall_rate 0.526.
- **v21:** dropped u_safe_rate, split action_rate so φ-change is nearly free. Policy STILL locked φ, this time at 3.99 (no incentive to drop without magnitude penalty). Fall_rate 0.443, combined 0.572 (worst on record).
- **Realization:** the issue isn't reward shape, it's training distribution. Three reward-tweak attempts in a row failed because the open spawn area never punished a large safety bubble. The robot could always detour around obstacles.
- **v22 fix:** corridor scenes (30% of envs). Two parallel cylinder rows form a ~0.6m gap. Robot must drop φ to fit through. **First time a HIGH-φ-is-wrong scenario exists in our training distribution.**
- This doc generalizes the principle to all four robust params for Wk3 release.
