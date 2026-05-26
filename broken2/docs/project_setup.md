****# Project setup reference

Written 2026-05-22 to serve as a migration reference (Isaac Lab → MJLab candidate) and a complete snapshot of what V13.1 / V14.5 actually do. Covers env, policy, CBF math, DR pipeline, reward stack, training pipeline, eval pipeline, diagnostics, deploy story, and version history.****

---

## 1. Problem statement

Train an RL teacher that outputs CBF (Control Barrier Function) parameters per step for a quadruped (Unitree Go2). The CBF parameters feed a QP-based safety filter that sits between a commanded velocity stream and a frozen locomotion controller. Paper goal: an *adaptive* CBF teacher that beats best-fixed-parameter baselines on a combined safety + goal metric.

**Paper venue:** CoRL 2026, abstract due Sat 2026-05-25.

**Original codebase ancestry:** lab mate's `semantic-safety` repo; minimal walking + LiDAR was extracted into this project (`safety-go2`). The teacher RL training happens in Isaac Lab; deployment uses ROS 2 with the frozen locomotion controller running on a Jetson Orin onboard the Go2.

---

## 2. High-level data flow

```
                  ┌──────────────────────────────┐
                  │  cmd_vel stream              │
                  │  (planner / scripted / noisy)│
                  └──────────────┬───────────────┘
                                 │ u_des = (vx, vy, ω_yaw)
                                 ▼
   ┌─────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
   │  Priv obs (33 dims) │    │   CBF-QP filter    │    │ Frozen locomotion  │
   │   - hidden env (14) │    │  min  ‖u-u_des‖²    │    │ policy (48 obs →   │
   │   - proprio (19)    │    │  s.t. ḣ + α·h ≥ 0  │    │  12 joints)        │
   └─────────┬───────────┘    │   (one per K obs)  │    └─────────┬──────────┘
             │                └──────────┬─────────┘              │
             ▼                           │                        ▼
   ┌─────────────────────┐               │              ┌─────────────────────┐
   │ LiDAR grid          │               │              │ Isaac Lab physics   │
   │ (2 × 64 × 64        │               │              │ + reward + reset    │
   │  occupancy)         │               │              └─────────┬───────────┘
   └─────────┬───────────┘               │                        │
             │                           │                        │
             ▼                           │                        │
   ┌────────────────────────────────┐    │                        │
   │ Teacher policy π_θ             │    │                        │
   │ (two-stream encoder)           │    │                        │
   │ → outputs: α, φ, a, c, [b]     │────┘                        │
   └────────────────────────────────┘                             │
             ▲                                                     │
             │                                                     │
             └─────────────────────────────────────────────────────┘
                       (next-step obs)
```

The teacher chooses **α, φ, a, c** per timestep. The CBF-QP combines them with state-dependent terms (h, L_g h, L_f h) and the planner's `u_des` to compute `u_safe`. The locomotion controller takes `u_safe` and produces joint targets. Physics steps. Rewards fire. Repeat.

---

## 3. The CBF math

### Per-step signed distance to obstacles

For K obstacles indexed by i:

```
sdf_i(x)  = shape_sdf(robot_xy, obs_i_xy, OBSTACLE_SHAPES[i])
                                          ↑ cylinder or box, Minkowski-expanded by
                                          ROBOT_HALF_FOOTPRINT = 0.15 m
sdf_i(x) += δ_R[env, i]                   # per-episode obstacle-radius perception
                                          # error (DR axis, see §5)
sdf(x)    = min_i sdf_i(x)
```

`shape_sdf` is implemented in [cbf_go2_shapes.py](../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_shapes.py). It returns signed distance from `robot_xy` to the closest point on the obstacle surface, **after subtracting `ROBOT_HALF_FOOTPRINT`**. `sdf > 0` means the robot's body envelope is clear of the obstacle; `sdf = 0` means kissing contact; `sdf < 0` means the body envelope overlaps the obstacle.

### Smoothed barrier function

```
h(x) = LAMBDA_H · (1 − exp(−GAMMA_H · sdf(x)))     LAMBDA_H = 1.0
                                                    GAMMA_H  = 0.5  (half-saturation at sdf ≈ 1.4 m)
```

This is monotonic in `sdf`. Sign matches `sdf`: `h(x) ≷ 0 ⟺ sdf ≷ 0`. The exponential smoothing gives cleaner gradients near `sdf = 0` and saturates far from obstacles so the constraint is benign in free space.

### CBF constraint enforced by the QP

For an exponential class-K function `α(h) = α · h` (where α is the learned scalar param):

```
ḣ + α · (h − c) ≥ 0
```

Expanding `ḣ = L_f h + L_g h · u`:

```
L_f h + (L_g h) · u_safe + α · (h − c) ≥ 0
```

The QP solves per-step:

```
min   ‖u_safe − u_des‖²  + φ · ‖L_g h‖²  +  a · 1ᵀ slack    (cost)
s.t.  (L_g h) · u_safe + φ · ‖L_g h‖²
        + α · (h − c) ≥ 0 − slack     (per K obstacles)
      slack ≥ 0
```

The five RL action outputs and their roles:

| symbol | meaning | bounds | physical effect |
|---|---|---|---|
| α | class-K slope | [0.1, 5.0] | bigger α = constraint allows faster h decrease (less conservative) |
| φ | ‖L_g h‖² weight (Kolathaya ISSf) | [0, 5.0] | bigger φ = more cost on aggressive gradient response, absorbs actuation uncertainty |
| a | slack penalty weight | [0, 1] | bigger a = harder constraint, slack expensive |
| c | h offset (perception bias) | [-0.20, 0.20] | c < 0 = treat obstacle as smaller (less cautious), c > 0 = inflated (more cautious) |
| b | unused | — | reserved; enabling turns QP into SOCP (blocks OSQP/qpth batching) |

QP solver: **qpth** (differentiable batched QP) wraps OSQP-style internals; reverts to cvxpy for hardware deploy.

QP code location: [cbf_go2_env.py:_cbf_filter](../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py).

---

## 4. Observation space

### Priv obs (33 dims) — fed to teacher, NOT student

Layout matches `CbfLayer3ObservationsCfg_TwoStream_V13` (V13/V13.1) — HIDDEN-first ordering:

| slice | name | dim | per-episode? |
|---|---|---|---|
| HIDDEN env (14): | | | |
| 0 | friction_coef | 1 | yes |
| 1 | base_mass_offset | 1 | yes |
| 2-4 | applied_force_b | 3 | yes |
| 5-7 | applied_torque_b | 3 | yes |
| 8-10 | com_offset_b | 3 | yes |
| 11 | actuation_noise_sigma | 1 | yes (+ within-episode every 250 steps) |
| 12 | mean_signed_delta_R | 1 | yes |
| 13 | max_abs_delta_R | 1 | yes |
| OBSERVABLE proprio (19): | | | |
| 14 | base_height | 1 | per-step |
| 15-29 | tracking_err history (5×3) | 15 | per-step |
| 30-32 | base_ang_vel | 3 | per-step |

### Grid (8192 dims)

`2 × 64 × 64` occupancy grid in the robot's body frame, current + previous frame. Each cell is binary occupied/free based on the K obstacle positions. Cell resolution: ~10 cm. Half-width: 3.2 m. Stored as `priv_obs.occupancy_grid_b` from [cbf_go2_observations.py](../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_observations.py).

Total teacher obs: 33 + 8192 = 8225 dims per step per env.

### Sensor noise (V13.1 distinguishing feature)

Gaussian noise on the 3 observable proprio channels:
```
base_height    σ = 0.02 m       (matches Go2 leg-odom)
tracking_err   σ = 0.05 m/s     (per timestep, per axis)
base_ang_vel   σ = 0.015 rad/s  (matches Go2 IMU gyro)
```

V13 (no noise) and V8 (single-stream, no proprio passthrough) don't have this.

---

## 5. Domain randomization

### Per-episode (sampled at `_reset_idx`)

| axis | range | mechanism |
|---|---|---|
| friction (static) | (0.30, 1.20) | `events.physics_material.params["static_friction_range"]` |
| friction (dynamic) | (0.10, 1.00) | same, dynamic_friction_range |
| base_mass | (-1.0, +3.0) kg offset | `randomize_rigid_body_mass` |
| COM offset | magnitude ~0.04 m | `randomize_com_and_cache` (custom event) — caches per-env in `env.cbf_com_offset_b` |
| applied force | up to ~10 N body frame | external force/torque DR |
| applied torque | up to ~2 Nm | same |
| obstacle radius error δ_R | per (env, obstacle) ∈ (-0.15, 0.15) | env attr `cbf_obs_radius_error` |
| σ_act (V13.1) | U(0, 0.20) | env attr `cbf_actuation_noise_sigma` (curriculum: ramps 0.03→0.20 over 18000 steps) |
| σ_act (V14.5) | U(0, 0.40), no curriculum | same attribute |

### Within-episode (V5+; resamples every K control steps)

Currently only σ_act has within-episode plumbing. `dr_window_sigma_act = True` (inherited from PHIWIN_TIGHTCOR_V5) + `dr_resample_interval_steps = 250` (5s at 50 Hz). Every 250 steps, each env independently draws a new σ_act from `U(0, σ_max_now)`. **V14.5 planned next step was to extend this to friction + push — needs new code (~25 lines).**

### Curriculum

PHIWIN_TIGHTCOR family (V13.1, V13, V8, V7, V5) ramps σ_act from `curriculum_sigma_min = 0.03` to `actuation_noise_sigma_max` over `curriculum_warmup_steps = 18000` common-steps. V14.5 disables this.

**Known caveat:** diagnostic scripts (diagnose_alpha_corr / diagnose_phi_corr / probe_z_linear / diagnose_grad_sensitivity) create fresh envs starting at `common_step_counter=0`, so any diagnostic on a curriculum-enabled config sees σ_act at the cold-start floor (~0.03 max). Pre-V14.5 diagnostics underrepresent σ_act variance by ~7×.

---

## 6. Reward stack

Defined in [CbfRewardsCfg](../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py) (class declaration), overridden by ancestor configclasses for V13.1.

### Active terms (V13.1 effective weights)

| term | weight | per-step? | meaning |
|---|---|---|---|
| `collision` | −100 | per-step on `obstacle_contact_now` | physics contact sensor on obstacle |
| `base_contact_penalty` | **−500** | sparse, on fall | physics contact on base body (= robot tipped) |
| `infeasibility` | −10 | per-step if QP fails | QP returns no feasible solution |
| `stuck` | −1.0 | per-step if v_xy < 0.15 m/s | dense |
| `action_rate` | −0.05 | per-step | ‖Δ(α,φ,a,b,c)‖² — smoothness on CBF params |
| `tilt_penalty` | −2.0 | per-step | ‖proj_gravity_xy‖² — orientation deviation from upright |
| `velocity_tracking` | **+1.5** | per-step, bounded [0,1] | h-conditional v target match: `exp(-|v - v_target(h)|²)` |

### Disabled terms (weight = 0 for V13.1)

`u_safe_deviation`, `proximity`, `ttc_penalty`, `cbf_lhs_margin`, `velocity_along_cmd`, `action_rate_split`, `u_safe_rate`, `cbf_a_l1`, `cbf_phi_above_target`, `cbf_deflection_l2`.

### Empirical balance (V13.1 from training summary CSV, last 200 iters)

Per-step averaged across (step, env):

| term | contribution |
|---|---|
| (positive stack, dominantly `velocity_tracking + tilt`) | +12.90 |
| `r_collision` | −0.029 |
| `r_base_contact_penalty` | −0.056 |
| `r_stuck` | −0.129 |
| `r_action_rate` | −0.178 |
| **net mean_reward** | **+12.51** per step |

**Diagnosed structural problem:** safety penalty stack is ~3% of total reward magnitude. Ratio of velocity reward to per-event safety penalty is ~30:1. Break-even: policy will tolerate one collision (−100) for 66+ steps of slowdown lost (−1.5/step). Policy converges to "race-to-goal, accept collisions" because the math says it's optimal.

This is the root cause of V13.1's high collision rate (28-34% physical contact across distributions) despite high goal_reach (94-98%).

---

## 7. Policy architecture (two-stream, V13/V13.1/V14.5)

[cbf_go2_teacher_rma.py](../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py)

```
Input obs (8225 dims) split into:

  priv_hidden (14)       priv_proprio (19)      grid (2×64×64)
       ↓                       ↓                       ↓
  _PrivEncoder           _PrivRunningMeanStd    _GridEncoder
  (Welford normalize +   (Welford normalize)    (Conv2d 2→16→32, stride 2,
   Linear 14→64→64→8)    passthrough            two layers; flatten 32×16×16
       ↓                       ↓                  = 8192 → Linear → 64)
   z_env (8)             proprio (19)            z_grid (64)
       ↘                      ↓                       ↙
        ───────────► concat (91) ───────────────────
                            ↓
                       π_teacher MLP
                       (Linear 91→128→128→
                        2×5 for Gaussian mean+std)
                            ↓
                       5-dim action distribution
                            ↓
                       (sample/mean) → α, φ, a, c, b raw
                            ↓
                       (action mapping in cbf_go2_env._cbf_filter:
                        tanh + scale to bounds; α∈[0.1,5], φ∈[0,5], etc.)
```

### Single-stream variant (V8 and earlier)

All 33 priv dims fed to a single `_PrivEncoder` (no proprio passthrough). z_priv contains everything including tracking_err. This makes tracking_err Pearson-correlate strongly with α at the cost of "z_env is mixed env-class + state" semantics.

### Critic

Mirrors the actor structure with its own weights (no sharing). Output is scalar value, not action distribution.

### Encoder weight initialization

Torch default (Kaiming uniform for Linear, Conv2d default). `_PrivRunningMeanStd` accumulates per-feature mean/std during rollout collection via Welford's online algorithm — needed because raw priv values have ~190× scale variation (force ~5.5N std vs COM offset ~0.029m std).

---

## 8. Training pipeline

### PPO via rsl_rl

[agents/rsl_rl_ppo_rma_cfg.py](../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/agents/rsl_rl_ppo_rma_cfg.py).

Key hyperparameters (V13.1):
```
n_envs          = 4096
n_steps         = 24 per rollout
mini_batch_size = ~ standard PPO
clip_ratio      = 0.2
learning_rate   = adaptive (KL-targeted)
gamma           = 0.99
lambda          = 0.95
entropy_coef    = 0.001 (low; minimal exploration noise late training)
max_iterations  = 2500 (V13.1) — ~5-6 hr on RTX 5090
```

### Action mapping

After PPO outputs `raw_action ∈ ℝ⁵`, env scales via tanh:
```python
alpha = ALPHA_MIN + (tanh(raw[:, 0]) + 1) / 2 * (ALPHA_MAX - ALPHA_MIN)   # [0.1, 5.0]
phi   = PHI_MIN   + (tanh(raw[:, 1]) + 1) / 2 * (PHI_MAX - PHI_MIN)       # [0, 5.0]
# ... a, c, b similarly
```

### Frozen locomotion controller

The locomotion policy is a separately-trained 48-obs → 12-joint-position policy. It runs every CBF step and accepts `u_safe` as a velocity command, outputting joint targets. **The locomotion policy's weights are not updated during teacher training.** Implemented as a TorchScript module loaded from disk at env init.

---

## 9. Evaluation pipeline

### Tasks

Per training cfg, 4 paired deploy distributions:

| dist | task ID pattern (V13.1) | DR characteristic |
|---|---|---|
| indist | Isaac-CBF-Go2-RMA-Layer3-TwoStream-V13-1-v0 | training task, curriculum on |
| trainmatch | ...-Deploy-Realistic-FrozenAC-TrainMatch-V13-1-v0 | matches training distribution exactly |
| ood | ...-Deploy-Realistic-FrozenAC-V13-1-v0 | small distribution shift |
| stressor | ...-Deploy-Realistic-FrozenAC-Stressor-V13-1-v0 | σ_act bumped to 0.30, wider friction (0.10, 1.30) |

### Eval script

[scripts/eval_baseline.py](../scripts/eval_baseline.py).

### Baseline modes

| mode | what it is |
|---|---|
| B0 | fixed (α, c) — sweep over `α_grid` × `c_grid` |
| B1 | fixed (α, φ) — sweep over `α_grid` × `φ_grid` |
| B2 | fixed (α, ε0, λ) — sweep over a third grid |
| BR | RL-trained teacher (the policy under test) |
| BS-A | student adapter (week 4 distillation) |

A typical 4-eval run sweeps `α_grid="0.5,2.0,3.0"`, `φ_grid="0.5,2.0"`, `λ_grid="1.0,3.0"`, runs 64 envs × 1000 steps × all combos. Output: `baseline.csv` with one row per (mode, config) pair.

### Metrics (the important ones)

| metric | formula | what it means |
|---|---|---|
| `goal_reach_rate` | `displacement > 1.5 m` per episode | **wandering-distance**, not navigation |
| `collision_rate` | `h_min < 0` at any step | CBF safety-set violation (canonical CBF metric) |
| `collision_rate_actual` | physics contact sensor fired | physical mesh-mesh contact |
| `collision_rate_perceived_only` | h<0 but no physical contact | margin breach without contact |
| `fall_rate` | base_contact (robot tipped) | terminal failure |
| `stuck_rate` | v_xy < 0.15 m/s | low-speed terminations |
| `timeout_rate` | episode hit max length | neither failure nor success |
| `avg_cbf_alpha_mean` | mean α output across rollout | population stat |
| `avg_cbf_alpha_std` | per-episode within-ep std of α | adaptation magnitude |
| `avg_h_min` | min h reached per episode | how close to constraint boundary |
| `avg_qp_active_rate` | fraction of steps QP fires | how often constraint binds |
| `avg_deflection_mean` | ‖u_safe − u_des‖ averaged | QP intervention magnitude |

### Composite scoring

```
composite_perceived = goal_reach × (1 − collision_rate)        × (1 − fall_rate)
composite_actual    = goal_reach × (1 − collision_rate_actual) × (1 − fall_rate)
```

**Choose one for the paper claim and use it consistently.** They give different verdicts because `collision_rate` counts margin breaches as failures and is typically 2-4× higher than `collision_rate_actual` on V13.1-class teachers. Current project debate: paper memory says canonical (perceived); recent diagnostic work prefers physical.

### Multi-seed sweep

Runs each (mode, config, dist) tuple at 3 seeds (42, 123, 7) to estimate variance. Aggregation script averages across seeds + computes confidence intervals.

---

## 10. Diagnostics

### Pearson correlation diagnostics

[scripts/diagnose_alpha_corr.py](../scripts/diagnose_alpha_corr.py), [scripts/diagnose_phi_corr.py](../scripts/diagnose_phi_corr.py).

Runs N envs × K steps, captures per-step α/φ outputs alongside priv features. Computes Pearson correlation between α/φ and each priv feature, both between-episode (using per-episode means) and within-episode.

**Known unreliable for "does the policy use X?" claims.** Single-feature linear correlation misses nonlinear, multi-feature, and saturated couplings. See `memory/feedback_no_pearson_for_policy_use.md` for the V13.1 example where Pearson<0.13 looked like "policy ignores z_env" but ablation showed z_env was load-bearing.

### Linear probe on z latent

[scripts/probe_z_linear.py](../scripts/probe_z_linear.py).

Trains a linear regressor `z_env → priv_feature_i` for each feature i, reports R². Tells you what z_env **encodes**, not what the policy **uses**.

### Gradient sensitivity

[scripts/diagnose_grad_sensitivity.py](../scripts/diagnose_grad_sensitivity.py).

Computes `∂α/∂priv_i` and `∂φ/∂priv_i` per feature, standardized by feature std. Direct measure of how the head modulates outputs given input changes. Better than Pearson but still doesn't capture downstream behavior.

### Pathway ablation (the right test)

Env-var-gated hooks in [_SplitRMAMLP.forward](../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py):
```
CBF_ABLATE_Z_ENV    ∈ {mean, zero, unset}   # mute hidden-env latent
CBF_ABLATE_PROPRIO  ∈ {mean, zero, unset}   # mute observable passthrough
CBF_ABLATE_Z_GRID   ∈ {mean, zero, unset}   # mute LiDAR-grid latent
```

With `mean`, replaces the named pathway with its batch-mean every forward call. Eval the muted policy via standard `eval_baseline.py`; compare composite + α/φ stats to unmuted baseline. **This is the closed-loop ground-truth test for "does the policy use pathway X."**

Runner scripts:
- [scripts/ablate_z_env_v13_1.sh](../scripts/ablate_z_env_v13_1.sh) — z_env mean + zero, 4 dists
- [scripts/ablate_proprio_grid_v13_1.sh](../scripts/ablate_proprio_grid_v13_1.sh) — proprio + z_grid, mean, 4 dists
- [scripts/ablate_all_v13_1.sh](../scripts/ablate_all_v13_1.sh) — all three, mean, 4 dists

### Rollout dump + matplotlib animation (since Isaac UI is broken on 5090)

[scripts/dump_rollout.py](../scripts/dump_rollout.py) (runs headless on lab; produces `.npz` with pos/yaw/grid/α/φ/h/deflection/cmd_vel/obstacles).

[scripts/animate_rollout.py](../scripts/animate_rollout.py) (runs locally on Mac; matplotlib → mp4/gif).

**Known bug fixed 2026-05-22:** dump_rollout was capturing 0 obstacles due to a wrong attribute lookup; rendered videos showed the robot dodging invisible cylinders. Fix imports `OBSTACLE_NAMES` from the env module directly.

---

## 11. Hardware deployment story

### Architecture on Go2

```
   teleop ─────► walking_bridge ─────► SportClient.Move
                       │
                       ▼ (intercept)
                ┌─────────────────┐
                │ cbf_filter_node │
                │   - Livox PCL → │
                │     occupancy   │
                │     grid + obs  │
                │     cluster     │
                │   - cvxpy QP    │
                │   - student     │
                │     adapter →   │
                │     ẑ_env →     │
                │     α, φ, c     │
                └─────────────────┘
                       ↓
                   u_safe
```

ROS 2 Humble on Jetson Orin. Lab computer SSHes to Go2. Frozen locomotion stays untouched.

### Student distillation (planned, Week 4)

Student replaces `_PrivEncoder` with a history regressor `f(state_{t-K:t}, action_{t-K:t}) → ẑ_priv` trained via MSE against the teacher's z_priv. Grid CNN and π_teacher are inherited frozen from the teacher. Only the priv-inference network differs sim-to-real — minimum surface area for the domain gap.

Spec: [docs/student_distillation_spec.md](student_distillation_spec.md).

DAgger training skeleton: [scripts/train_distillation.py](../scripts/train_distillation.py).

Distillation dataset task: `Isaac-CBF-Go2-Distill-v0` (registered in `__init__.py`).

### Tier 3 hardware trial

Locked 2026-05-21 for 2026-05-22 execution. 3 conditions × ≥5 trials, cylindrical obstacle in lab space. Protocol: [docs/hardware_trial_protocol.md](hardware_trial_protocol.md).

---

## 12. Version history (recent)

### V8 (PHIWIN_TIGHTCOR_V8, 2026-05-19)

Single-stream architecture, all 33 priv dims → one encoder → z_priv (8-dim). No proprio noise. 2500 iters. **Most "adaptive" of recent versions by Pearson metrics** but those are unreliable; closed-loop wins are mild.

### V9 (RMA_CANONICAL_V9, 2026-05-19)

Single-stream with smaller priv obs (12-13 dims, dropped some channels). Closed-loop loses to baselines.

### V11 (RMA_V11, 2026-05-20)

Single-stream variant. 3000 iters. Worst closed-loop of the wk3 line.

### V13 (TWOSTREAM_V13, 2026-05-20)

First two-stream architecture. Hidden 14 + proprio 19 + grid. No proprio noise. **Introduces the priv split that V13.1 builds on.**

### V13.1 (TWOSTREAM_V13_1, 2026-05-20)

V13 + Gaussian noise on the 3 observable proprio channels (matches real Go2 sensor specs). 2500 iters. Closed-loop multi-seed: tied-or-close on trainmatch + OOD physical composite, clear loss on stressor, uniform loss on canonical composite. **Current "best" model — anchored for Tier 3 hardware deploy.**

3-axis ablation on V13.1:
- z_grid mute → 54-59% reduction in within-ep α std; +0.44 to +0.70 Δᾱ. **z_grid dominates α/φ adaptation.**
- proprio mute → small Δᾱ (0.03-0.18), moderate composite effect on OOD
- z_env mute → small Δᾱ (-0.07 to +0.16), small composite effect

### V14.5 (TWOSTREAM_V14_5, 2026-05-22)

V13.1 + wider σ_act (0.20→0.40) + curriculum off. Pure-config staged predecessor to V15. Tested whether σ_act regime amplitude alone would push z_env utilization above the success bar.

**Result: FAILED success criterion.** Δᾱ_z_env_mute remained < 0.20 on every distribution (−0.07 to +0.16); closed-loop composite_actual regressed on every distribution vs V13.1 (−1 to −13 pp); α dropped slightly (2.06 → 1.96), z_grid dominance increased (Δᾱ_z_grid_mute now +0.69 to +1.53 vs V13.1's +0.44 to +0.70). The wider DR didn't shift the policy toward env-class adaptation; it tightened reliance on perception.

### Diagnosed structural blocker

V14.5 falsified the σ_act-amplitude hypothesis. The deeper read is that **the reward stack is ~30:1 imbalanced toward velocity_tracking over safety**, so the policy will always converge to "race-to-goal, accept some collisions" regardless of what env class it perceives. Architecture and DR changes don't fix this; reward reweighting does.

Candidates for V14.6 (not yet drafted):
- bump `base_contact_penalty` to −2000 (4×) → contributes ~−0.22/step, same order as `action_rate`
- re-enable `cbf_lhs_margin` at −1.0 → dense h<0 penalty
- drop `velocity_tracking` weight from +1.5 to +0.5

---

## 13. Migration considerations for MJLab

### Portable

- **CBF math** (sdf computation, h, QP formulation). Pure tensor ops. ~200 lines.
- **Two-stream policy architecture.** Standard PyTorch modules. Replace rsl_rl with whichever JAX/PyTorch PPO MJLab provides.
- **Reward functions.** Pure functions of env state. Re-define as MJLab reward terms.
- **Diagnostics.** All operate on dumped numpy arrays. No Isaac dependency.
- **Eval baseline machinery.** Rewrite the env-stepping outer loop; metric computations are pure.
- **Animation pipeline.** Already matplotlib, already local.

### Needs rewrite

- **Asset import.** Go2 USD → MJCF. Convertible but requires care (collision geometry, joint limits, actuator gains, contact parameters).
- **Frozen locomotion controller.** Currently a TorchScript module trained in Isaac. Either (a) re-train in MJLab, (b) export inputs/outputs and run as a black box from Mujoco state, or (c) accept that "frozen Isaac controller" requires importing both the policy weights AND the obs format. Option (b) is probably right — treat it as a JIT'd function.
- **Event manager / scene management.** Isaac has a structured EventManager for DR (`randomize_rigid_body_mass`, `randomize_com_and_cache`, etc.). MJLab will have a different convention (likely simpler — set MJCF body params at reset).
- **Curriculum logic.** Currently lives in `_reset_idx` + step hooks. Direct port.
- **Within-episode DR plumbing.** Just `dr_window_sigma_act` for now. Easy port.
- **Obstacle scene generation.** Per-env random obstacle placement, corridor vs open scene types, kinematic obstacle motion. Re-implement as MJCF body manipulation.

### Lost (Isaac-specific)

- Omniverse renderer + Kit-based UI (the thing that's broken on 5090).
- Isaac's articulation/contact APIs (replaced by MuJoCo equivalents).
- rsl_rl's standard training loop (replaced by Brax PPO or similar JAX framework).

### Estimated migration cost

7-12 days minimum to recover parity with V13.1, before any new science. Recommended only post-abstract (after Sat 2026-05-25).

### What MJLab buys you

- Working renderer on 5090 (debug cycle 10× faster)
- Faster training iteration (JAX/MJX GPU steady-state speed)
- Cleaner hackability (smaller codebase, less Omniverse magic)
- Better mainline support for quadruped + safety work in 2026

### What MJLab does NOT buy you

- The reward-stack bottleneck is the same problem in any sim.
- The two-encoder design decision (z_env vs proprio passthrough) is independent of sim choice.
- The student distillation gap (sim-to-real perception) is the same problem.

---

## 14. Open questions / not-yet-resolved

1. **Reward stack imbalance.** V14.6 not yet drafted. Recommended single move: bump `base_contact_penalty` 4× and/or re-enable `cbf_lhs_margin` at moderate weight.
2. **Within-episode adaptation isolation.** Batch-mean z_grid ablation showed 54-59% within-ep α std reduction, but mean replacement kills BOTH per-env and time-variation signal. Cleaner isolation = `freeze_t0` ablation (cache z_grid at t=0, reuse all episode). ~20-line patch + one rollout per dist.
3. **Goal / task semantics.** The env has no fixed navigation goal — only velocity-tracking + obstacle avoidance + stay-upright. "goal_reach_rate" is misleadingly named (1.5m wandering threshold). Paper framing needs to acknowledge this or add a real navigation task.
4. **Collision metric headline.** Project memory used `collision_rate` (perceived h<0). Recent work prefers `collision_rate_actual` (physical contact). Lock one for paper, report both for completeness.
5. **MJLab migration timing.** Recommended post-abstract. Whether it gates Week 4 student work or sits until after CoRL deadline is a planning call.

---

## 15. Key file paths

```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
├── __init__.py                       # gym.register for all task IDs
├── cbf_go2_env.py                    # env class, CBF-QP filter, DR samplers
├── cbf_go2_env_cfg.py                # all configclasses (versions live here)
├── cbf_go2_observations.py           # priv obs term functions
├── cbf_go2_rewards.py                # reward term functions
├── cbf_go2_events.py                 # DR event handlers (COM, mass, friction)
├── cbf_go2_shapes.py                 # SDF computation (cylinder, box)
├── cbf_go2_perception.py             # SHIELD-style perception variants
├── cbf_go2_teacher_rma.py            # two-stream policy + critic
├── cbf_params_action.py              # 5-D action term wrapper
├── cbf_go2_locomotion_train_cfg.py   # frozen locomotion controller cfg
├── cbf_go2_student.py                # student distillation model
└── agents/
    ├── rsl_rl_ppo_cfg.py             # PPO config for single-stream teachers
    └── rsl_rl_ppo_rma_cfg.py         # PPO config for two-stream teachers

scripts/
├── eval_baseline.py                  # 4-eval B0/B1/B2/BR/BS-A pipeline
├── train_and_eval_twostream_v13_1.sh # V13.1 training+eval runner
├── train_and_eval_twostream_v14_5.sh # V14.5 (latest)
├── train_and_eval_twostream_v14.sh   # V14 (adaptive c)
├── dump_rollout.py                   # state→npz for matplotlib animation
├── animate_rollout.py                # npz→mp4 (local matplotlib)
├── play_render.py                    # Omniverse video (broken on 5090)
├── diagnose_alpha_corr.py            # Pearson(α, priv)
├── diagnose_phi_corr.py              # Pearson(φ, priv)
├── probe_z_linear.py                 # R²(z_env → priv_i)
├── diagnose_grad_sensitivity.py      # ∂(α,φ)/∂priv
├── ablate_z_env_v13_1.sh             # z_env mute closed-loop eval
├── ablate_proprio_grid_v13_1.sh      # proprio + z_grid mute
├── ablate_all_v13_1.sh               # 3-axis decomposition
├── extract_training_summary.py       # rsl_rl log → CSV per-iter
└── train_distillation.py             # DAgger student adapter

docs/
├── project_setup.md                  # this doc
├── paper_outline.md                  # paper claim + result tables
├── student_distillation_spec.md      # Week 4 student architecture
├── hardware_trial_protocol.md        # Tier 3 procedure
├── reward_structure.md               # reward stack design (somewhat dated)
├── env_runtime.html                  # env step diagram
├── teacher_architecture.html         # two-stream architecture diagram
└── per_axis_stress_eval_protocol.md  # OOD eval design

src/go2_walking_lidar/                # ROS 2 deploy package (on Jetson)
├── src/cbf_filter_node.cpp           # PCL → grid + cluster + QP + student
├── src/walking_bridge.cpp            # cmd_vel → SportClient.Move bridge
└── src/teleop.cpp                    # joystick interface
```

---

## 16. Magic numbers worth knowing

```python
ROBOT_HALF_FOOTPRINT       = 0.15 m       # Minkowski expansion
LAMBDA_H                   = 1.0          # h saturation magnitude
GAMMA_H                    = 0.5          # h smoothing rate (half-sat at sdf≈1.4m)
ALPHA_MIN, ALPHA_MAX       = 0.1, 5.0     # action bounds
PHI_MIN, PHI_MAX           = 0.0, 5.0
GRID_H, GRID_W             = 64, 64
GRID_HALF_WIDTH_M          = 3.2 m
N_OBSTACLES (K)            = variable per scene type
GOAL_REACH_DISTANCE        = 1.5 m        # wandering threshold for goal_reach
SHIELD_R_MIN, SHIELD_R_MAX = (perceived obstacle radius clamp range)
PI_TEACHER_HIDDEN          = (128, 128)
Z_PRIV_DIM (default)       = 8
Z_GRID_DIM (default)       = 64
PROPRIO_DIM (V13+)         = 19
PRIV_HIDDEN_DIM (V13+)     = 14
PRIV_DIM_TOTAL (V13+)      = 33
```

---

End of doc. Last updated 2026-05-22.
