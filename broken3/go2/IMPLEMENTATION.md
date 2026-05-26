****# CBF-Adaptive Go2 — Implementation Doc

Snapshot as of 2026-05-24, right before the SHIELD-aligned teacher training run.
Re-check whenever a fundamental change is made.

---

## 1. Research goal

Train a PPO policy that **adaptively parameterizes** a robustified Control Barrier Function (CBF) for a Unitree Go2 quadruped. Outer policy emits 2-dim `(φ, α)` per step; frozen locomotion controller tracks the safe velocity that the CBF QP produces. Target deployment: real Go2 + Livox Mid-360 lidar, CoRL 2026.

Two-stage RMA architecture:
- **Teacher** (current focus) — sees privileged DR factors + lidar.
- **Student** (future) — predicts privileged factors from proprio history; deployed on hardware.

---

## 2. Observation: 198-dim `teacher_obs`

Built in `cbf_task/mdp.py:teacher_obs`. Layout:

| Slice | Dim | Contents | Source |
|---|---|---|---|
| `[0:7]` | 7 | **priv** (z input) | `priv_obs` |
| `[7:52]` | 45 | **proprio** (current physical state) | `deployable_obs` |
| `[52:54]` | 2 | **prev_action** (last (φ, α), post-Lipschitz) | `prev_action_obs` |
| `[54:126]` | 72 | **lidar_prev** (t-1 frame) | `lidar_prev_obs` |
| `[126:198]` | 72 | **lidar** (t frame) | `lidar_obs` |
| **Total** | **198** | | |

### 2.1 Priv channels (7) — what world am I in

| Idx | Channel | Range | Theoretical axis | Notes |
|---|---|---|---|---|
| 0 | disturbance_force | 0 – 30 N | α / φ | — |
| 1 | friction_coef | 0.3 – 1.0 | φ | — |
| 2 | base_mass_delta | -3 – +3 kg | α | — |
| 3 | motor_strength | 0.7 – 1.3 | φ | — |
| 4 | actuation_noise_std | 0 – 0.05 rad | φ | — |
| 5 | com_offset | -0.05 – +0.05 m | α | — |
| 6 | **v_max** | 1.0 – 2.0 m/s | **α** | **Validated 92% bound span (phase6_vmax_gate). Recoverable from base_lin_vel_b in proprio (saturates at v_max under goal cmd), but explicit slot removes inference burden.** |

Only the teacher sees these. Student will predict from proprio history.

### 2.2 Proprio (45) — how am I moving

- `base_lin_vel_b` (3), `base_ang_vel_b` (3), `projected_gravity_b` (3)
- `joint_pos_rel` (12), `joint_vel` (12), `prev_loco_action` (12)
- **NO `velocity_commands`** — removed 2026-05-24b to kill the goal-proxy leak via ‖u_nom‖.

### 2.3 Lidar (72 + 72) — what's around me

- 72 analytic ray-cylinder distances in body frame (sim of Mid-360 horizontal ring after vertical-cone projection).
- Two consecutive frames (t-1 and t) for temporal features.
- Mid-360-tuned noise model — see [§6.2 Lidar fidelity](#62-lidar-fidelity-mid-360-sim).
- Computed in `mdp._compute_lidar`.

---

## 3. Architecture: `RMAMLPModel` + `_BranchedMLP`

In `cbf_task/agents/rma_actor_critic.py`. Subclasses rsl_rl 5.0.1's `MLPModel`.

```
input 198-d
   ├─ [0:7]   priv  → z_enc (Linear 7→16, ELU, 16→8)  → z (8)
   ├─ [7:52]  proprio → passthrough (45)
   ├─ [52:54] prev_action → passthrough (2)
   └─ [54:198] lidar_prev + lidar (144) → lidar_enc CNN → lidar_feat (16)
              │
              └─ _LidarCNN: 1D convs with circular padding over 72-ray ring
                  Conv1d(2→16, k=5, circular) → ELU
                  Conv1d(16→32, k=5, circular) → ELU
                  MaxPool1d(2)
                  Conv1d(32→16, k=3, circular) → ELU
                  MaxPool1d(2)
                  Linear(288→64) → ELU → Linear(64→16)

concat → main MLP (75 → 128 → 64 → 2)
       → actor output (φ, α)
```

Critic shares architecture; outputs scalar value instead of 2-dim.

---

## 4. Action: (φ, α) with Lipschitz rate-limit

In `cbf_action_term.py:process_actions`:

1. **Clamp** raw policy action to [-1, 1]
2. **Lipschitz rate-limit**: `a = a_prev + clamp(a - a_prev, ±0.05)`
   - Hard bound on per-step normalized action change.
   - With dt=0.02s, gives L_a = 2.5/s on normalized action.
   - Decoded (φ, α) inherit Lipschitz via the linear-decode slopes:
     dφ/da = 0.5·(φ_hi − φ_lo) = 0.5; dα/da = 0.5·(α_hi − α_lo) = 1.9.
     So **L_φ = 1.25/s**, **L_α = 4.75/s**.
   - Configured via `action_max_step = 0.05` in SHIELD env.
3. **Decode** to:
   - φ ∈ [0, 1] — input-uncertainty hedge
   - α ∈ [0.2, 4.0] — class-K gain

Policy trains WITH the rate-limit active so it learns to issue continuous commands. The CBF (and downstream actuators) sees only smoothed (φ, α).

---

## 5. CBF safety filter

In `cbf_action_term.py:process_actions`.

### 5.1 Nominal control

`u_nom = kp · (goal_xy - robot_xy)` (P-controller toward goal)
- Capped at per-env `v_max`.
- Rotated to body frame.
- Default kp=1.0, v_max=1.3 m/s.

### 5.1b Safety radius

`r_safe[i] = obstacle_radius[i] + robot_radius` per-obstacle. `robot_radius = 0.35 m` (covers Go2 body half-diagonal ≈ 0.36 m up to a small margin; bumped from 0.30 inscribed-circle 2026-05-24). With SHIELD's `obstacle_radius=0.5`, `r_safe = 0.85 m`. The CBF QP enforces `dist_center_to_center ≥ r_safe`, i.e. `h = dist - r_safe ≥ 0`.

### 5.2 Signed distance function — two modes

Both share math: `sdf = min_i ||x - ρ̂_i|| - R_i`, smoothed as `h_smooth = λ(1 - exp(-γ·sdf))` (SHIELD eqs 19-20).

| Mode | `ρ̂_i` source | Used in |
|---|---|---|
| **Privileged** (`use_lidar_sdf=False`) | exact `_obs_centers_w` | Phase 5 / RMA / Unified envs |
| **Perception** (`use_lidar_sdf=True`) | noisy + dropout-prone | **SHIELD env** (deployment-target) |

### 5.3 Perception SDF (SHIELD-aligned, `_compute_sdf_smooth_perception`)

Simulates Livox Mid-360 + Euclidean clustering + cylinder-fit pipeline:

| Source of error | Default | What it models |
|---|---|---|
| Position noise | Gaussian std 0.05 m per axis per step | Cluster centroid accuracy |
| Range cutoff | 20 m | Mid-360 horizontal coverage |
| Random dropout | 2% per obstacle per step | Brief clustering failures / occlusion |

When NO obstacles survive masking: sdf saturates at `lidar_max_range - r_safe.min()` → h ≈ λ → g_eff ≈ 0 → CBF naturally goes silent.

### 5.4 Closed-form QP

`_closed_form_cbf_batched` projects `u_nom` along `∇h_smooth` direction to satisfy `∇h_smooth · u_safe ≥ φ - α·h_smooth`. Result clamped at `v_max`.

### 5.5 Smoothing params (SHIELD defaults)

- `h_smooth_lambda = 1.0`
- `h_smooth_gamma = 2.0`

---

## 6. Environments (registered tasks)

In `cbf_task/__init__.py`. Hierarchy:

```
UnitreeGo2FlatEnvCfg (Isaac Lab stock)
  └─ CBFAdaptiveGo2EnvCfg                       # base CBF setup
       └─ CBFAdaptiveGo2Phase2EnvCfg            # adds disturbance DR
            └─ CBFAdaptiveGo2RMAEnvCfg          # adds branched encoder + lidar + 4-chan priv
                 ├─ CBFAdaptiveGo2RandObsEnvCfg # single-obstacle ±1.5m jitter
                 ├─ CBFAdaptiveGo2VmaxEnvCfg    # v_max DR (validated α-channel)
                 ├─ CBFAdaptiveGo2RoughEnvCfg   # rough terrain
                 ├─ CBFAdaptiveGo2DecorrEnvCfg  # wide-jitter test env
                 ├─ CBFAdaptiveGo2ActNoiseGateEnvCfg, CBFAdaptiveGo2ComOffsetGateEnvCfg  # gate sweeps
                 └─ CBFAdaptiveGo2SlalomEnvCfg  # 3-obstacle slalom
                      └─ CBFAdaptiveGo2UnifiedEnvCfg  # slalom + all 6 DR channels
                           └─ CBFAdaptiveGo2UnifiedLidarSDFEnvCfg  # SHIELD (deployment-target)
```

### 6.1 The deployment-target env: `Isaac-CBF-Adaptive-Go2-UnifiedLidarSDF-v0`

Built from `CBFAdaptiveGo2UnifiedLidarSDFEnvCfg`:
- 3-obstacle slalom at (2.0, 0.5), (4.0, -0.5), (5.5, 0.5), r=0.5m
- Goal at (7.0, 0.0)
- Per-episode obstacle jitter: ±0.5 m
- All 7 priv DR channels active (including `v_max_range = (1.0, 2.0)`)
- `use_lidar_sdf=True` (perception SDF)
- `perception_noise_std=0.05`, `perception_dropout_prob=0.02`, `lidar_max_range=20.0`
- `action_max_step=0.05` (Lipschitz)
- Visual markers spawned (red cylinders + green sphere) for play UI

### 6.2 Lidar fidelity (Mid-360 sim)

The sim lidar (`mdp._compute_lidar`) is a 2D analytic ray-cylinder ring tuned to match a real **Livox Mid-360** after vertical-cone projection and ring extraction. We use the analytic form (not Isaac's warp raycaster) because warp can't target per-env-replicated obstacles, and the cylindrical-obstacle assumption is exact for our scene.

Channels (all configurable on `CBFActionTermCfg`):

| Knob | Default | Real-world correspondence |
|---|---|---|
| `n_rays` | 72 (5° spacing) | Mid-360 horizontal angular density after ring extraction |
| `lidar_max_range` | 20 m | Mid-360 spec is 40 m; cap conservatively (warp loses accuracy >20 m) |
| `lidar_noise_std` | 0.02 m | Datasheet: ±2 cm @ 10 m. **Fixed** — replaced the previous `0.03·(1+0.05·R)` model that was overly pessimistic at distance |
| `lidar_angle_jitter_std` | 0.0087 rad (0.5°) | Approximates the Mid-360's **non-repetitive flower scan** — each tick samples slightly different bearings rather than 72 fixed angles |
| `lidar_dropout_base_prob` | 0.005 | Baseline ray-failure rate (near-range) |
| `lidar_dropout_range_slope` | 0.03 | Range-weighted dropout: `p_drop = base + slope·(range/max_range)`. Far rays fail more (incidence angle, beam divergence, low reflectivity) — dropped rays return `max_range` |

**What we do not model** (acceptable for cylindrical-obstacle slalom):

- Vertical FOV (-7° to +52°) — assumed to be collapsed to a single horizontal slice after preprocessing
- Specular/mirror dropout
- Multi-return (we take first hit)
- Material reflectivity beyond the dropout proxy
- 3D point cloud → cylinder fitting noise on the obstacle list itself (modeled separately as `perception_noise_std=0.05` on `_obs_centers_w` for the **QP's SDF**, not the policy lidar)

**Ground-truth leak path (known, accepted for v7)**: the QP's perception SDF still uses `_obs_centers_w + noise` rather than `min(lidar_rays)`. The policy obs never sees this — only the QP behavior is conditioned on it. Killing this leak entirely (lidar-derived SDF with soft-min) is deferred until after v_max validation; see `cbf_action_term.py:760` and the §13 open work list.

### 6.3 Domain randomization (all channels)

Active in `CBFAdaptiveGo2UnifiedEnvCfg` and all subclasses (including SHIELD). Each per-episode channel is resampled on `env.reset()`; per-step channels resample every physics tick.

| Channel | Range | Cadence | Privileged? | Physics mechanism | α/φ target | Gate status |
| --- | --- | --- | --- | --- | --- | --- |
| `friction_range` | (0.3, 1.0) | per-reset | priv[1] | PhysX material write | φ | tested |
| `base_mass_range` | (-3, +3) kg | per-reset | priv[2] | rigid-body mass update | α | tested |
| `motor_strength_range` | (0.7, 1.3) | per-reset | priv[3] | torque multiplier in `process_actions` | φ | tested |
| `disturbance_force_range` | (0, 30) N | per-reset (+ dir resample every `disturbance_resample` steps) | priv[0] | `set_external_force_and_torque` on base | α/φ | tested |
| `actuation_noise_range` | (0, 0.05) rad | per-reset | priv[4] | additive Gaussian on joint targets | (none) | gate FAIL — keep priv slot for completeness |
| `com_offset_range` | (-0.05, +0.05) m | per-reset | priv[5] | shifted CoM via rigid-body API | (none) | gate FAIL — keep slot |
| **`v_max_range`** | (1.0, 2.0) m/s | per-reset | priv[6] | clamp on `u_nom` before QP | **α** | **PASS 92% bound span ([[alpha-channel-search]])** |
| `obstacle_pos_jitter_range` | ±0.5 m (SHIELD) | per-reset | no (read via lidar) | shift each obstacle xy from nominal | — | required to force lidar use (else proprio carries position) |
| `perception_noise_std` (QP SDF) | 0.05 m | per-step | no | additive on `_obs_centers_w` inside QP | — | SHIELD only |
| `perception_dropout_prob` (QP SDF) | 0.02 | per-step | no | Bernoulli mask on obstacle list | — | SHIELD only |
| `lidar_noise_std` | 0.02 m | per-step | no | additive on ray ranges | — | always on |
| `lidar_angle_jitter_std` | 0.0087 rad | per-step | no | additive on ray angles | — | always on |
| `lidar_dropout_*` | base=0.005, slope=0.03 | per-step | no | Bernoulli on rays, dropped → max_range | — | always on |

**Validated channels for adapting (φ, α)**: only `v_max` so far. `actuation_noise` and `com_offset` failed their gate sweeps but are kept in priv to give the encoder structure (they're cheap and the encoder can choose to ignore them).

**Not modeled (open work)**:

- Dynamic obstacles (moving cylinders / humans) — deferred to phase 7+
- Velocity gain mismatch (commanded vs realized loco velocity) — candidate φ-channel if v_max alone doesn't shift φ
- Action delay (1-step lag between policy output and loco input) — candidate φ-channel
- Terrain roughness (only used in the rough-terrain gate env, not Unified)

---

## 7. Reward function (inherited from RMAEnvCfg)

Effective weights for SHIELD (inherits `CBFAdaptiveGo2RMAEnvCfg.__post_init__` overrides):

| Term | SHIELD weight | Base weight | Function |
| --- | --- | --- | --- |
| `progress` | +1.0 | +1.0 | `prev_dist_to_goal - last_dist_to_goal` per step |
| `intervention` | **0.0 (overridden)** | -0.05 | `‖u_safe − u_nom‖` per step |
| `collision` | -1000.0 | -1000.0 | one-shot 1.0 when `h_realized < 0` (terminal — `collision_termination` fires same step) |
| `goal_reached` | +50.0 | +50.0 | one-shot 1.0 when `dist_to_goal < 0.4` (terminal — `goal_reached_termination` fires same step) |
| `action_smoothness` | -0.2 | (not in base) | `\|Δφ\|/φ_width + \|Δα\|/α_width` — **width-normalized L1**, not L2. Treats a 10% move in φ equally with a 10% move in α regardless of asymmetric ranges |
| `stuck` | **0.0 (wired, inactive)** | 0.0 | per-step 1.0 when `\|v_xy\| < 0.15 m/s` AND `dist_to_goal > 0.4`. Flip to ~-0.05 in a cfg post-init to discourage "freeze under disturbance" |

**Inheritance subtleties** (relevant if you wire a non-RMA env):

- Base `CBFRewardsCfg` has `intervention=-0.05`, NOT 0. The 0 is an experimental override at [cbf_adaptive_env_cfg.py:303](cbf_task/cbf_adaptive_env_cfg.py#L303) inside `CBFAdaptiveGo2RMAEnvCfg.__post_init__` — anyone using base / Phase2 directly gets -0.05.
- `action_smoothness` is **only** added in RMA post-init (not present in base).
- Per code comment, `intervention=-0.05` is "half the Phase 1 value" — Phase 1 was -0.10, not -0.3 as earlier docs claimed.

**⚠ `intervention=0` is experimental — see [[feedback_shared_repo_reverts]]. Revert to -0.05 (base default) if other people retrain on RMA cfg.**

**Magnitude rationale for `collision=-1000`**: at 50 Hz × ~10 s max episode × `progress=+1`, the max cumulative `progress` payoff is ≈ +500. Collision needs to dominate this so the policy doesn't trade a collision for early progress. -1000 is roughly 2× the payoff ceiling, the right inequality. Don't drop to -100 — it would flip the sign of `payoff − collision` and re-open the "collide early" basin.

---

## 8. Termination conditions

In `mdp.py`. Four termination terms, of which one is time-out:

| Term | Wired by default? | Condition |
| --- | --- | --- |
| `time_out` | yes | episode length exceeded (Isaac Lab default) |
| `collision` | yes | `last_h_realized < 0` (intersected obstacle) |
| `goal_reached` | yes | `last_dist_to_goal < 0.4` |
| `fall` | yes | `base_z < 0.15` AND `gravity_b[z] > -0.3` (AND-conjunction — OR with looser thresholds fired at spawn) |
| `stuck_termination` | **no (opt-in)** | sticky `episode_stuck_any` flag (>100 slow-not-at-goal steps, ~2 sec). Reads, does NOT increment — the increment lives only in `_ensure_post_physics`. Wire as `DoneTerm` in a subclass post-init together with a small `stuck_penalty` reward weight |

**Stuck tracking — single source of truth**. The increment for `episode_stuck_steps` / `episode_stuck_any` lives **only** in `_ensure_post_physics` ([mdp.py:48-49](cbf_task/mdp.py#L48-L49)). The previously-duplicate `stuck_check` function was renamed to `stuck_termination` and now only **reads** the flag — wiring it as a `DoneTerm` no longer double-counts. If you remove or change the increment in `_ensure_post_physics`, stuck tracking dies silently (no second source).

**Hardcoded `goal_tol=0.4`** appears in [cbf_adaptive_env_cfg.py:98](cbf_task/cbf_adaptive_env_cfg.py#L98), [cbf_adaptive_env_cfg.py:113](cbf_task/cbf_adaptive_env_cfg.py#L113), and [mdp.py](cbf_task/mdp.py) (default kwarg in `goal_reached_bonus`, `goal_reached_termination`, `stuck_penalty`, plus the literal 0.4 in `_ensure_post_physics:47`). Acceptable footgun for now — pull into a module constant if you ever sweep goal_tol.

**Speed threshold (0.15 m/s) vs `v_max_range` (1.0–2.0 m/s)**: the stuck floor is 7.5–15% of v_max. Any robot moving even at minimum-DR cruise speed clears it; the flag fires only when the policy effectively gives up. Strict but intentional.

---

## 9. PPO training

Config in `cbf_task/agents/rsl_rl_ppo_cfg.py:CBFAdaptiveGo2RMARunnerCfg`.

### 9.1 Hyperparameters

| Param | Value | Notes |
|---|---|---|
| `num_envs` | 1024 (recommended) | RTX 5090 has headroom for 32GB |
| `max_iterations` | 3000 | empirically converges before 1500 was tried |
| `num_steps_per_env` | 24 | per-iter rollout length |
| `entropy_coef` | **0.0** | recent change — was 0.002, blew up action_std |
| `value_loss_coef` | 1.0 | |
| `clip_param` | 0.2 | |
| `num_learning_epochs` | 5 | |
| `num_mini_batches` | 4 | |
| `learning_rate` | 3e-4 | adaptive schedule |
| `gamma` | 0.99 | |
| `lam` | 0.95 | |
| `desired_kl` | 0.01 | |
| `max_grad_norm` | 1.0 | |

### 9.2 Actor/critic

- Both use `RMAMLPModel` (class name resolved via `"cbf_task.agents.rma_actor_critic:RMAMLPModel"`)
- `hidden_dims=[128, 64]`
- `activation="elu"`
- `obs_normalization=True` (per-model EmpiricalNormalization)
- Distribution: `GaussianDistribution` with `init_std=0.3`, `std_type="scalar"`
- Uses NEW-format `actor`/`critic`/`obs_groups`, NOT deprecated `policy` field (silent fallback)

### 9.3 Throughput

Observed: ~15k env-steps/sec at 256 envs, ~40k at 1024 envs (Blackwell RTX 5090). 3000 iters at 1024 envs ≈ 30 min wall time.

---

## 10. Diagnostics

### 10.1 Health-of-the-model scripts

| Script | What it checks |
| --- | --- |
| `phase6_sanity_check.py` | obs dim, slice layout, vel_cmd leak, Lipschitz, perception SDF, lidar fidelity (17 checks) |
| `phase6_encoder_health.py` | architecture sanity, per-encoder activation health, sensitivity probes |
| `phase6_priv_attention.py` | per-channel priv sensitivity (Part A obs + Part B interventional) |
| `phase6_lidar_attention.py` | obstacle-distance-driven modulation (Part A obs + Part B interventional) |
| `phase6_decorrelation_test.py` | lidar vs goal-proxy split via partial regression (Decorr env) |
| `phase6_fixed_param_sweep.py` | "does scenario reward adaptation?" — varies fixed (φ, α) grid, compares per-distance-bin optima |
| `phase5_train_teacher.py --max_iterations=0` | disturbance sweep eval (Phase 5 standard) |
| `phase6_full_diag.sh` | chains encoder_health → priv_attention → lidar_attention → decorrelation → Phase5 sweep on one checkpoint |
| `phase5_baselines.py` | B0 (ECBF) / B1 (ECBF+fixed ISSf) / B2 (TISSf-CBF) sweep on any task |
| `phase5_train_teacher.py --diag_interval N` | in-training health snapshot every N iters with auto-WARN on degenerate signatures (peg detection, action_std collapse/blow-up, QP-not-firing, stuck/collision basin) |

### 10.2 Gate framework (channel-validation scripts)

**Strong-gate protocol** — run BEFORE training a new priv channel, to confirm the channel actually shifts the (φ*, α*) optimum. Without this, training silently bakes in a dead signal.

For each level `L` in the channel's range:

1. Instantiate the env with that level set on the candidate channel; **pin all other DR at nominal** so the result isolates this channel's effect.
2. Sweep fixed (φ, α) on a grid (typically φ pinned at 0 or const; sweep α — or vice versa for φ-channels).
3. Per cell: 8–64 episodes per (level, params), measure `collision_rate, reach_rate, fall_rate, stuck_rate, intervention_mean, tracking_err_mean`.
4. Per level, pick "best" param: lowest `intervention_mean` among **safe** cells (`coll ≤ 10%` AND `reach ≥ 80%`); fallback to lowest coll if no safe cell.
5. Compute **span** = `(max best_α − min best_α) / (α_hi − α_lo)` across levels.

**PASS gate**: span > 10% of bound width AND the curve is monotone in the channel direction predicted by CBF theory (e.g. lower α on rougher terrain). FAIL → drop the channel from priv before wasting a training run.

| Gate script | Channel tested | Verdict so far |
| --- | --- | --- |
| `phase6_vmax_gate.py` | `v_max` (α) | **PASS** — 92% span (best α: 4.0 @ v=1.0 → 2.5 @ v=2.0) |
| `phase6_alpha_gate.py` | terrain roughness (α) | FAIL — flat curve |
| `phase6_obstacle_pos_gate.py` | obstacle position (α) | FAIL — flat curve |
| `phase5_fingerprint_gate.py` | each priv channel via R²(z, priv) | encoder-side validation (R² ≥ 0.5 required) |
| `phase5_per_channel_sweep.py` | per-channel observability of the disturbance signal in proprio | early-Phase-1 tool, see [[alpha_channel_search]] |

### 10.3 Eval metrics

Computed per cell by `eval_cell` in [phase5_baselines.py](phase5_baselines.py) and `eval_teacher_at_disturbance` in [phase5_train_teacher.py](phase5_train_teacher.py):

| Metric | What it measures | Used for |
| --- | --- | --- |
| `collision_rate` | % episodes ended in collision | primary safety |
| `reach_rate` | % episodes reaching goal | primary task |
| `fall_rate` | % robot fell over | tip-over filter |
| `stuck_rate` | % robot froze (slow + not at goal >100 steps) | freeze-mode detection |
| `intervention_mean` | Σ‖u_safe − u_nom‖ per episode (CBF interference) | tie-breaker among safe cells (gate protocol) |
| `jitter_mean` | mean `\|Δφ\|/φ_width + \|Δα\|/α_width` | Lipschitz / smoothness |
| `min_h_mean` | per-episode min SDF (closest approach) | diagnostic only |
| `time_in_unsafe_frac` | fraction of steps where h < 0.2m | diagnostic only |
| `time_to_goal_mean` | mean steps to first goal reach (successful only) | diagnostic only |

**Modulation span** (`phi_range`, `alpha_range` reported in §12): computed at [phase5_train_teacher.py:323-326](phase5_train_teacher.py#L323-L326) as `max(phi_mean_per_d_bin) − min(phi_mean_per_d_bin)` across the eval disturbance grid. Verdict: `ADAPTS` if either span > 25% of its bound width. This is the "does the policy actually use priv?" check — flat span means the priv encoder is dead even if `phase6_priv_attention` shows sensitivity (a sensitive but unused channel is still a fail).

---

## 11. Baselines (`phase5_baselines.py`)

| Baseline | Form | Cells | Param grid |
| --- | --- | --- | --- |
| **B0 ECBF** (Ames et al. 2017) | `α=const, φ=0` | 3 | α ∈ {1, 2.5, 4} |
| **B1 ECBF + fixed ISSf** | `α=const, φ=const` | 6 | α ∈ {1, 2.5, 4} × φ ∈ {0.3, 1.0} |
| **B2 TISSf-CBF** (Cohen 2024 / Molnar 2023) | `α=const, φ(h) = (1/ε₀)·exp(−λh)` | 6 | α ∈ {1, 2.5} × **chosen (ε₀, λ) pairs**: `{(1, 0.5), (1, 1), (2, 1)}` |

**B2 pair grid (not a Cartesian product)** — at [phase5_baselines.py:81-86](phase5_baselines.py#L81-L86), `B2_EPS0_LAM` is three deliberately-chosen `(ε₀, λ)` points spanning (aggressive hedge, medium, conservative), not a sweep. The doc previously read like a `2×3×3=18` product; actual is `2×3=6`.

**Why no α=4 in B2**: TISSf with both high α AND a hedge `φ(h)` becomes near-infeasible (the QP slack term explodes when α tries to push toward the boundary fast while φ is also pulling). The asymmetry vs B0/B1 is deliberate, not a bug.

**Disturbance sweep grid**: d ∈ {0, 15, 30, 45} N (default). Training DR caps at **30 N** in `CBFAdaptiveGo2UnifiedEnvCfg`. **d=45 N is intentional out-of-DR extrapolation** — see [cbf_adaptive_env_cfg.py:381-383](cbf_task/cbf_adaptive_env_cfg.py#L381-L383): "45N collapses locomotion regardless of CBF". The d=45 column tests how gracefully the policy degrades past its training envelope. Report `worst_in_train` (max over {0, 15, 30}) **separately** from `worst_overall` (max over all four) — using only `worst_overall` compresses the in-distribution comparison signal because every controller collapses at d=45.

**Lipschitz fairness**: baselines emit *constant* (φ, α) every step (no per-step modulation), so the `action_max_step=0.05` rate-limit only matters for the first ~20 steps of each episode (ramp from 0 to the constant). The SHIELD teacher trains with rate-limit active and outputs time-varying (φ, α). For a strictly fair comparison, baselines could be wrapped with the same rate-limit — the effect is small but non-zero on episodes shorter than ~1 sec. Currently not wrapped; flag if you need a tight A/B.

**"Best" selection** (`pick_best` in [phase5_baselines.py:150-168](phase5_baselines.py#L150-L168)):

1. Aggregate each cell across the disturbance grid → `(worst_coll, worst_reach, mean_int)`
2. Among **safe** cells (`worst_coll ≤ 10%` AND `worst_reach ≥ 80%`), pick the one with lowest `mean_int`
3. If no cell is safe: fallback to lowest `worst_coll`, tie-break on highest `worst_reach`

The learned teacher must beat the per-family `best` aggregate. When reporting "best baseline" in §12, name the specific `(family, α, φ_or_ε₀_λ)` cell that won — `best.json` is written per run.

---

## 12. Current results state (Phase 6 in progress)

| Teacher | Reach @ d=0 | Coll @ d=30 | Lidar fraction | φ span (across d) | α span (across d) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Slalom (1500 iter, ent=-0.3) | 71% | 20% | n/a | **26%** | **32%** | Adapts via z; slow |
| Decorr (1500 iter, ent=0) | 96% | 18% | 54% | 3% (flat) | 4% (flat) | Adapts via lidar; not z |
| Unified (1500 iter) | 48% | 17% | ? | 14% | 27% | Undertrained |
| Unified (3000 iter, ent=0.002) | 78% | 17% | 4.8% | 5% | 1% | **Degenerate**: action_std blew up to 4.59 |
| Unified (3000 iter, ent=0) | **pending** | | | | | New build |
| **SHIELD (Lipschitz+perception)** | **pending** | | | | | Current run |

Best baseline (B2 TISSf): worst_coll=11%, worst_reach=89% on Decorr; 34%/51% on Slalom. Both numbers are on OLD envs — needs re-run on the SHIELD env (`phase5_baselines.py --task Isaac-CBF-Adaptive-Go2-UnifiedLidarSDF-v0`) before fair comparison.

### 12.1 Held-out eval scenes (E1–E4)

4 scenes registered as separate tasks (`Isaac-CBF-Adaptive-Go2-Eval{E1..E4}-v0`). All inherit the SHIELD env (full DR + Mid-360 lidar fidelity + Lipschitz). Only obstacles + goal + jitter differ:

| Scene | Obstacles | What it tests |
| --- | --- | --- |
| **E1 SINGLE** | one cylinder at (4.0, 0.0, r=0.7) on path | basic CBF avoidance; isolated from multi-obstacle SDF |
| **E2 SLALOM** | (2, 0.5), (4, -0.5), (5.5, 0.5), r=0.5 (training geometry) | in-distribution regression check |
| **E3 DENSE FIELD** | 5 obstacles staggered in a 3×2 grid | multi-obstacle SDF blending; replanning stress |
| **E4 NARROW GAP** | (3.5, ±0.6), r=0.5 (≈1.2m corridor) | tight-tolerance navigation; φ-hedging under proximity |

All four keep ±0.5m per-episode obstacle jitter (same as SHIELD training) so n=N envs give stat power, not n=1 deterministic scenes.

**E0 (empty) was dropped** — the CBF action term crashes on `K=0` obstacles. `torch.tensor([(o[0], o[1]) for o in []])` returns shape `(0,)` instead of `(0, 2)`, so the very first broadcast `env_origins_xy.unsqueeze(1) + centers_local.unsqueeze(0)` fails with "size 2 vs 0 at dim 2". A clean fix would also need K=0 branches in `_compute_sdf_smooth`, `_compute_sdf_smooth_perception`, and `_compute_lidar` (each does a `min(dim=-1)` over the obstacle dim that errors on empty input). The sanity-floor value didn't justify that refactor; **`B-trivial(φ=0, α=2.5)`** in the policy comparison serves as the no-tuning safety baseline.

### 12.2 Cross-scene comparison protocol (`phase6_eval_scenes.py`)

**Single-process limitation**: Isaac Lab allows only one sim context per Python process — `env.close()` does NOT release it fully, and a subsequent `gym.make()` fails with "Simulation context already exists." So the runner uses a **subprocess-per-scene** architecture:

- **DRIVER mode** (default invocation): orchestrates; spawns one CHILD process per scene via `subprocess.call`, then aggregates per-scene JSONs into one combined CSV and prints a summary table. Does NOT import Isaac itself — starts instantly.
- **WORKER mode** (`--scene X --json_out PATH`): does one scene end-to-end. Loads Isaac, runs all `(policy × disturbance)` cells for that scene, writes results to JSON, exits.

Driver streams worker stdout/stderr live to your terminal so per-cell rates appear as they're computed.

Policies tested per scene × disturbance ∈ {0, 15, 30} N:

1. **teacher** — `runner.get_inference_policy(device)` on the trained checkpoint
2. **B-trivial(φ=0, α=2.5)** — always-on safety, no tuning
3. **B0-best, B1-best, B2-best** — train-tuned on the SHIELD env (loaded from `phase5_baselines_summary.json`), FIXED across all scenes (tests generalization of the controller AND its selection method)

Workflow:

```bash
# step 1: tune baselines on SHIELD (one-shot, ~30 min)
~/IsaacLab/isaaclab.sh -p phase5_baselines.py \
    --task Isaac-CBF-Adaptive-Go2-UnifiedLidarSDF-v0 \
    --out_dir phase6_shield_baselines_outputs --headless

# step 2: cross-scene comparison (driver invocation -- spawns 4 child procs)
~/IsaacLab/isaaclab.sh -p phase6_eval_scenes.py \
    --teacher_ckpt phase6_shield_v7_teacher_outputs/rsl_rl/model_final.pt \
    --locomotion_ckpt /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \
    --baselines_summary phase6_shield_baselines_outputs/phase5_baselines_summary.json \
    --headless
```

Outputs: per-scene `scene_E{1..4}_results.json` (raw) + `phase6_eval_scenes_cells.csv` (every `(policy, scene, d)` cell) + `phase6_eval_scenes_summary.csv` (per `(policy, scene)` aggregate via `aggregate_across_d`) + a printed comparison table. `d=45` deliberately omitted from this grid (out-of-DR per §11; only compresses signal).

---

## 13. Open work

| Item | Status |
| --- | --- |
| SHIELD teacher train + diag | In progress |
| Tune baselines on SHIELD env | Not started — blocker for §12.2 step 1 |
| Cross-scene comparison (E0–E4) | Script ready, awaiting teacher checkpoint |
| Student distillation (proprio history → ẑ) | Not started; phase5_train_student.py docstring is stale (still says 4 priv channels, claims branched encoder was bypassed) |
| Real-robot deployment (Go2 + Mid-360) | Not started |
| Lidar-derived SDF (kill last GT leak in QP) | Deferred — see §6.2 |
| Dynamic obstacles | Deferred to Phase 7+ |
| Composite metrics + Pareto plot | Discussed, not built |

---

## 14. Critical gotchas to remember

1. **`class_name` silent fallback** — `RslRlPpoActorCriticCfg` (deprecated `policy` field) drops `class_name="..."` for custom models and falls back to vanilla MLP. Must use new-format `actor`/`critic`/`obs_groups`.
2. **`empirical_normalization` required** even in new format (set to False; per-model `obs_normalization=True` does the work).
3. **PhysX device coercion** — friction/mass/com_offset apply methods must coerce indices/values to `materials.device`.
4. **Isaac Lab RayCaster** doesn't support per-env-replicated mesh paths → we use analytic ray-cylinder lidar.
5. **`_r_safe` is a tensor** (1, K) not a scalar. Fallbacks must use `.min().item()`.
6. **Repo is shared** — `cbf_adaptive_env_cfg.py` and `rsl_rl_ppo_cfg.py` are touched by others. The experimental `intervention=0` override on `RMAEnvCfg` should be reverted when done.
7. **Isaac Sim 5.1 + Blackwell** requires the **open-source** driver variant (`nvidia-driver-580-open`), not proprietary.
8. **Bash wrapper script** (`phase6_full_diag.sh`) silently swallows Python errors via `tee` — exit codes are masked. If a diag is missing from the summary, run it alone to surface the actual error.

---

## 15. File map

```
go2/
├── cbf_task/
│   ├── __init__.py                    # task registrations
│   ├── cbf_action_term.py             # CBF QP, SDF, perception, Lipschitz, DR sampling
│   ├── cbf_adaptive_env_cfg.py        # env hierarchy (Phase 2 → RMA → Slalom → Unified → SHIELD)
│   ├── mdp.py                         # obs functions, reward terms, terminations
│   ├── locomotion_loader.py           # frozen Go2 actor loader
│   ├── terrain_helpers.py             # 7-level terrain generator
│   └── agents/
│       ├── rsl_rl_ppo_cfg.py          # PPO + actor/critic cfg
│       └── rma_actor_critic.py        # RMAMLPModel + branched encoder + LidarCNN
├── phase5_train_teacher.py            # PPO training entry + Phase5 eval
├── phase5_baselines.py                # B0/B1/B2 sweep
├── phase5_train_student.py            # (future) distillation
├── phase5_deploy_eval.py              # (future) student-substituted eval
├── phase6_fixed_param_sweep.py        # gate test: does scenario reward adaptation?
├── phase6_lidar_attention.py          # observational + interventional lidar usage
├── phase6_priv_attention.py           # per-priv-channel usage
├── phase6_encoder_health.py           # encoder activation health
├── phase6_decorrelation_test.py       # lidar vs goal-proxy split
├── phase6_sanity_check.py             # B.5 wiring validation (12 checks)
├── phase6_play.py                     # rollout + visualize / record video
├── phase6_full_diag.sh                # chained diagnostic pipeline
├── phase6_slalom_pipeline.sh          # retrain + diag chain
├── phase6_fixed_param_sweep.sh        # overnight sweep wrapper
└── play.sh                            # short play wrapper for desktop UI
```
