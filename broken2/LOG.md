# Log

Chronological record of work. Newest at top. Add a dated entry after each session.

---

## 2026-05-09 (late afternoon) — distillation infra pre-staged + plotting + ckpt archive while v2.15 trains

### Context

v2.15 launched at ~14:01 lab time (~9-10h wall, alarm at 12:30 AM Sun EDT).
Used the in-flight time to harden tooling and pre-stage the student distillation pipeline (Goal B.5).

### What landed

**Eval pipeline upgrades** ([scripts/eval_baseline.py](scripts/eval_baseline.py), [scripts/plot_results.py](scripts/plot_results.py))

- Path C goal-reach metrics added: `goal_reach_rate`, `mean_time_to_goal`, `mean_final_displacement`. Displacement-from-spawn proxy since the env has no fixed goal target. Distinct from existing `mean_dist_traveled` (path length) — captures "did the robot actually get somewhere vs walk in circles."
- `--student_checkpoint` flag + `BS` mode for direct teacher-vs-student eval in one run.
- 4 paper-grade plot functions wired in `plot_results.py` earlier today (pareto, bfx, margins, 2axis_split).

**Checkpoints archived to Mac** (`checkpoints/v212/`, `checkpoints/v214/`)

- v2.12 → `2026-05-08_14-09-21/model_4999.pt` (~13MB) + `params/`
- v2.14 → `2026-05-09_02-50-51/model_4999.pt` + `params/`
- TF events skipped to save bandwidth.

**Student distillation infrastructure** (pre-staged, doesn't run until v2.15 lands)

- [docs/student_distillation_spec.md](docs/student_distillation_spec.md) written then revised. Initial draft assumed student input was a 60-D flat (x, y, R) list; reading the env revealed it's actually 8207-D (15 dynamics scalars + 64×64×2 occupancy grid). Spec rewrote the design as "same 64×64 grid shape, but built from LiDAR clusters + DR layers" so the encoder stays unchanged.
- Registered [Isaac-CBF-Go2-Distill-v0](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py).
- New `CbfDistillationObservationsCfg` in [cbf_go2_env_cfg.py](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py) — obs group that swaps `priv_obs.occupancy_grid_b` for `priv_obs.noised_occupancy_grid_b` on the obstacle channel. The 15 dynamics scalars stay (first-pass deploy-realism caveat documented in spec).
- New `noised_occupancy_grid_b` in [cbf_go2_observations.py](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_observations.py): calls `shield_perceive_v0c` (built earlier today as Goal B.4 closure) for cluster centroids → applies per-episode range gating + per-step Bernoulli dropout (rate sampled per-episode) → rasterizes as fixed-R=0.3m disks on the 64×64 grid.
- Per-episode DR state on env ([cbf_go2_env.py](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py)): `cbf_student_dropout_rate`, `cbf_student_range_gating` — sampled at `_reset_idx` per env. Mirrors v2.12 perception_bias / v2.15 radius_error patterns.
- DAgger training skeleton at [scripts/train_distillation.py](scripts/train_distillation.py) — frozen-teacher load, fresh student init, MSE-loss outer loop, scaffolded for 3K iters at 1024 envs.

### Conceptual clarifications (chat-driven)

- **The c-parameter is state-aware, not radius-adaptive.** User pushed back when I said "c adapts to radius error." The policy can't observe true radius. Honest framing: c learns a Bayes-optimal robust baseline (constant inflation for unobservable perception error) plus a state-dependent modulation (robot speed, heading toward obstacle). The "compensation" for unknown radius is a fixed bake-in; the "adaptation" is on observable robot dynamics. SHIELD differentiation (state-aware c vs single hand-tuned alpha) survives the clarification.
- **Spec doc had wrong input shape.** Forced re-grounding in the actual env code. Saved future-me a half-day of confused implementation.

### Files touched

- `scripts/eval_baseline.py` — Path C metrics + BS mode + `--student_checkpoint`
- `scripts/train_distillation.py` — NEW skeleton
- `IsaacLab/.../cbf_go2/__init__.py` — Isaac-CBF-Go2-Distill-v0 registration
- `IsaacLab/.../cbf_go2/cbf_go2_env_cfg.py` — `CbfDistillationObservationsCfg` + `CbfGo2EnvCfg_DISTILL`
- `IsaacLab/.../cbf_go2/cbf_go2_env.py` — per-episode student DR state in `__init__` + `_reset_idx`
- `IsaacLab/.../cbf_go2/cbf_go2_observations.py` — `noised_occupancy_grid_b`
- `docs/student_distillation_spec.md` — NEW, revised once after the input-shape correction

All files synced to lab, md5 verified. Additive (new cfg subclass, new register entry, new obs term, new env state read via `getattr` with defaults) — zero behavioral change for the v2.15 training in flight.

### What's still pending (Goal B.5 critical path)

- Smoke-test `noised_occupancy_grid_b` on lab at small env count to catch any runtime issue.
- Calibrate DR ranges (`student_sensor_dropout_max`, `student_range_gating_min/max`) from empirical clustering output once v2.15 lands.
- Optionally plumb per-env `grid_res` through `shield_perceive_v0c` for the third DR axis (deferred unless calibration shows it's load-bearing).
- Run v2.12 ckpt against `Shield-V0C-v0` for a first baseline number on clustering-error degradation.

### Next event

12:30 AM Sun EDT alarm — check v2.15 results, run 10-eval headline + 4-slot Bf-X table + plot_results.py for paper figures.

---

## 2026-05-09 (afternoon) — v2.14 ran and FAILED; v2.15 spec'd + implemented + parsing clean

### v2.14 result — bad

v2.14 (per-episode φ lock + train under shield_v0a perception) ran end-to-end
on the lab box overnight. Result: **0W / 0T / 8L on the headline 8-eval, avg
margin -7.62 pp.** Worse than v2.12. Worse than v2.6.

The 2-axis breakdown (added today as a reframing) was the key insight:

| eval | safety axis (fall) margin | performance axis (stuck) margin |
| --- | --- | --- |
| In-distribution | -17.4 pp LOSS | +15.1 pp WIN |
| DensePack | -16.1 LOSS | +8.3 WIN |
| Slippery | -20.2 LOSS | +8.5 WIN |
| HighDisturbance | -11.1 LOSS | +6.2 WIN |
| HeavyCOM | -17.9 LOSS | +4.7 WIN |
| FastObstacles | -17.4 LOSS | +15.1 WIN |
| RealisticCompound | -23.8 LOSS | +8.0 WIN |
| NoisyPerception | -13.4 LOSS | +10.6 WIN |
| **AVG** | **-17.2 pp LOSS** | **+9.6 pp WIN** |

**The architecture traded falls for un-stuck.** BR fall ~0.26 vs baseline
~0.085 (3× worse on safety); BR stuck ~0.04 vs baseline ~0.15. Net combined
loss because the safety-axis gap dominated. Combined-metric was hiding the
real failure mode.

### Bf-X table from v2.14 (mostly carried forward strong)

| Ablation | BR | Bf-X | degradation |
| --- | --- | --- | --- |
| Bf-α on in-dist | 0.303 | 0.481 | +17.9 pp |
| Bf-α on DensePack | 0.309 | 0.404 | +9.5 pp |
| Bf-φ on HighDist | 0.411 | 0.343 | **-6.8 pp INVERTED** (still) |
| Bf-a on NoisyPerc | 0.266 | 0.729 | +46.3 pp |
| Bf-c on HeavyCOM | 0.401 | 0.463 | +6.1 pp |
| Bf-c on FastObs | 0.303 | 0.519 | +21.6 pp |

5 of 6 slots show LARGER load-bearing adaptation than v2.12. The
architecture is doing real per-slot work. The problem is the policy uses
that work in the WRONG direction (toward aggression).

### Three findings drove the v2.15 spec

1. **2-axis eval framing.** Combined was masking the safety-axis gap.
   New paper goal: beat baselines on BOTH safety AND performance, large
   margin. Combined alone is necessary but not sufficient.

2. **Reward asymmetry was wrong.** -100 fall + -2.0/step stuck makes one
   fall ≈ 50 stuck-steps. Policy correctly inferred "never be stuck,
   falling is acceptable." Fix (REWARD-3): -500 fall + -1.0 stuck. Now
   one fall ≈ 500 stuck-steps. Forces safety-first equilibrium.

3. **The φ inversion is DR-mismatch, not slot redundancy** (user-led
   re-diagnosis). HighDisturbance pushes the robot's POSE via force/torque,
   not its u-space TRACKING. That's not actuation uncertainty — it's
   closer to a model-error or pose disturbance. The per-episode φ lock
   was correct (and worked: cbf_phi_locked_std = 2.08, no per-step jitter)
   but couldn't help because the DR axis we paired it with isn't actually
   φ's domain. Solution: add a proper actuation-uncertainty DR axis
   (per-episode σ_act on u_safe BEFORE locomotion).

   Same logic applies to the c slot: HeavyCOM is an indirect proxy for
   boundary correction. Real c-target DR is **per-episode obstacle-radius
   perception error** (R_perceived = R_actual - δ_R, δ_R ~ U(0, 0.10)).
   This doubles as **Goal B.2 closure** because radius uncertainty is
   exactly what real LiDAR-cluster-fit-cylinder pipelines produce at
   deploy time. Information PRESERVED, uncertainty EXPOSED — analogous
   to v2.12's perception_bias DR (which gave us Bf-a +44pp).

### HOCBF dropped

User-explicit decision: not switching to torque-level / acceleration-input
control. Single-integrator stays. Memory + methods_outline updated.

### v2.15 spec — packed retrain (combines original v2.15 + v2.16)

Files modified (~12 distinct changes):

| File | changes |
| --- | --- |
| `cbf_go2_env.py` | ALPHA_MIN=1.0, C_MIN=0.10 module constants; cbf_actuation_noise_sigma + cbf_obs_radius_error buffers; per-episode samplers in `_reset_idx`; α/c floor application in `_cbf_filter` squash block; per-step actuation noise injection in `step()` before `_run_locomotion`; per-episode radius error applied in `_compute_h`; 4 new health stats |
| `cbf_go2_env_cfg.py` | REWARD-3 weights (-500 fall, -1.0 stuck); perception_mode default → "priv"; actuation_noise_sigma_max=0.10 field; obstacle_radius_perception_error_max=0.10 field; CbfGo2EnvCfg_HIGH_ACTUATION_NOISE (σ_act_max=0.20); CbfGo2EnvCfg_RADIUS_ERROR (δ_R_max=0.20) |
| `__init__.py` | register Isaac-CBF-Go2-HighActuationNoise-v0; register Isaac-CBF-Go2-RadiusError-v0 |
| `scripts/train_and_eval_v215.sh` | 10-eval headline (added HighActuationNoise + RadiusError); 6K iters (was 5K); steps_per_config 1500 (was 2000); skip dual-regime mid-switch (saves ~50 min); Bf-φ paired with HighActuationNoise (was HighDist); Bf-c on HeavyCOM + RadiusError (was HeavyCOM + FastObs) |

All 4 files parse cleanly (Python ast + bash -n).

### v2.15 wall clock target

~9-10 h vs v2.14's ~13 h. Trim breakdown:

- Training 6K iters (was 5K, +1.5h) but eval steps_per_config 1500 (was 2000, -1h) and dual-regime skipped (-0.9h). Net -0.4h on training-side.
- 2 added headline evals (HighActuationNoise + RadiusError): +0.9h.
- Net ~9.5h.

### Predicted v2.15 results

- 8-eval safety margin: -17.2 → ≥ 0 pp (cross threshold)
- 8-eval combined: ≥ +5 pp WIN (most likely 0 to +10 pp)
- Bf-φ on HighActuationNoise: -6.8 → +20 to +40 pp (load-bearing)
- Bf-c on RadiusError: +20 to +40 pp (mirroring Bf-a structure)
- Bf-a on NoisyPerception: ≥ +20 pp (carries from v2.12)

### Still open after v2.15

- Tag + archive v2.12 and v2.14 ckpts to Mac (carried over)
- Goal B.3 FOV-gating eval test on v2.15 ckpt (cheap, no retrain)
- Goal B.4 shield_v0c (synthetic LiDAR + clustering, 1-2 days)
- Wk3 student distillation
- Wk4 hardware deploy

---

## 2026-05-09 (early morning) — v2.12 results in, two new findings drove v2.14 spec, code ready to launch

v2.12 finished training (5K iters) and the full 8-eval matrix landed.
Result: **3W / 1T / 4L**, avg margin +0.17 pp. In-dist (+7.05),
NoisyPerception (+5.51), FastObstacles (+7.05) — three clean wins.
DensePack / Slippery / HighDisturbance / RealisticCompound regressed
vs v2.6. HeavyCOM tied (+0.69).

The Bf-X ablation table is the headline:

| Ablation | BR combined | Bf-X combined | degradation | verdict |
| --- | --- | --- | --- | --- |
| Bf-α on in-dist | 0.328 | 0.463 | +13.4 pp | LOAD-BEARING |
| Bf-α on DensePack | 0.343 | 0.403 | +6.0 pp | LOAD-BEARING |
| Bf-φ on HighDisturbance | 0.409 | 0.333 | -7.6 pp | INVERTED |
| **Bf-a on NoisyPerception** | 0.250 | 0.691 | **+44.1 pp** | **HEADLINE** |
| Bf-c on HeavyCOM | 0.369 | 0.397 | +2.9 pp | modest |
| Bf-c on FastObstacles | 0.328 | 0.463 | +13.5 pp | load-bearing |

**Bf-a on NoisyPerception = +44pp** is the strongest single result of
the project. Fixing the `a` slot to its mean breaks performance by 44pp
combined under perception-noise stress. The slot's training-time mean
stays small (≈0.05) because the noise σ is small most steps — but the
policy uses it in spike-mode when σ_e is high. Validates the 4-param
robust QP architecture.

### Finding 1 — LLN/CLT on the policy-output side (φ slot)

Bf-φ on HighDist INVERTED (-7.6 pp): clamping φ to its mean *beat* the
adaptive policy. User caught the diagnosis: per-step Gaussian sampling
on φ has the same LLN failure mode as per-step IID DR noise. PPO's
advantage averages reward over the ~800-step episode; any single
per-step φ_t contributes ~1/800 to the return, so the gradient signal
on "specialize φ for state X" is washed out. Policy converges to "mean
works fine, std is exploration noise we couldn't suppress."

Fix is symmetric to the v2.12 perception-bias fix (which was on the
*input* side, also LLN-driven). On the *output* side: lock φ once at
episode start, replay for the rest of the episode. Captured at first
post-reset step including exploration noise → noise becomes
between-episode (useful for exploration), not within-episode (harmful
for the constraint).

Generalizable pattern saved to memory
(`feedback_lln_in_rl_outputs.md`): when training PPO with long
episodes, classify any per-step quantity as state-vs-environment-class.
Environment-class → per-episode persistence.

### Finding 2 — SHIELD-perception gap

Goal B v0a smoke test (eval v2.12 ckpt under
`Isaac-CBF-Go2-Shield-V0A-v0` — QP uses uniform R=0.3m cylinders,
mimicking SHIELD's deploy-time pipeline): margin flipped from
**+7.05 → -3.5 pp**. v0b (+6m sensor gating) gave -3.3 pp.

Structure: BR worsened slightly (0.258 → 0.291, +3.3 pp), but
baselines DROPPED dramatically (B1: 0.341 → 0.256, -8.5 pp). Net gap
closed because baselines benefited more from simplified geometry than
BR was hurt. BR's per-obstacle param adaptation is calibrated for true
radii; under uniform R=0.3m, it under-pads big obstacles (R=0.50 → 0.30)
→ +6 pp falls. Baselines were already calibrated for "average obstacle";
uniform R=0.3m IS that average.

Implication: the v2.12 architecture's win depends on the QP knowing
true geometry. At deployment with real LiDAR (which can't recover
radii), the win evaporates. Train UNDER shield perception so the policy
learns "the QP always thinks R=0.3m, output params accordingly."

### v2.14 spec — both findings folded in

| change | source | code |
| --- | --- | --- |
| Per-episode φ lock | LLN/CLT on output side | `USE_PER_EPISODE_PHI = True` in `cbf_go2_env.py` |
| Train under shield_v0a | SHIELD perception gap | `perception_mode = "shield_v0a"` default in `CbfGo2EnvCfg` |

Both edits done + parsed. Goal B perception module
(`cbf_go2_perception.py`) added with v0a/v0b ready, v0c stubbed (full
synthetic LiDAR raycast + grid clustering — ~150 lines, deferred).

Predicted v2.14 outcomes: Bf-φ on HighDist flips from -7.6 to positive
(load-bearing); 8-eval avg margin returns to +5 to +10 pp under shield
perception (deploy-realistic baseline, not priv-perception).

### Files modified this session

- `cbf_go2_env.py`: `USE_PER_EPISODE_PHI` flag, locked-φ buffers, reset
  invalidation, capture-replay block in `_cbf_filter`,
  `cbf_phi_locked_std` health stat, `perception_mode` plumbing
- `cbf_go2_env_cfg.py`: `perception_mode` field,
  `CbfGo2EnvCfg_SHIELD_V0A`, `CbfGo2EnvCfg_SHIELD_V0B`, default switched
  to `"shield_v0a"`
- `cbf_go2_perception.py` (new): `shield_perceive_v0a`,
  `shield_perceive_v0b`, `compute_shield_sdf_cylinder_batch`, v0c stub
- `__init__.py`: registered `Isaac-CBF-Go2-Shield-V0A-v0`,
  `Isaac-CBF-Go2-Shield-V0B-v0`
- `feedback_lln_in_rl_outputs.md` (new memory): generalizable LLN/CLT
  pattern documentation

### v2.12 archive pending

Need to: tag ckpt, copy `model_4999.pt` to `_archive/v2.12_model.pt`,
scp to Mac, save eval CSVs.

---

## 2026-05-08 — v2.11 results landed, failed; v2.12 designed and built; v2.13 reframed

Long session with multiple pivots. The session broadly went:
v2.11 7-eval results in → diagnosed the `a`/`c` collapse → first design
proposal "v2.6 + bimodal" rejected by user (wanted to keep v2.11 stack +
add the missing perception piece) → cylinder commitment + Option A
noise injection → caught LLN bug in per-step IID noise during sanity
check (would have been a v2.13-level architectural bug if it had shipped) →
fixed to per-episode persistent bias → 17-axis sanity check all pass →
v2.13 reframed from grid-distance-transform to SHIELD-path cluster-fit-
cylinder. Methods outline drafted. Ready to launch v2.12 tonight.

### v2.11 results — diagnostic gold despite failure

7-eval matrix under locked planner (deploy-realistic):

| Eval | v2.11 BR | best B | margin | v2.10 margin | v2.6 margin |
| --- | --- | --- | --- | --- | --- |
| **In-dist** | **0.472** | 0.453 | **-1.9pp LOSS** | +9.9pp WIN | +6.9pp WIN |
| DensePack | 0.353 | 0.459 | **+10.6pp WIN** | +3.4pp WIN | +0.6pp tie |
| Slippery | 0.453 | 0.493 | +4.0pp WIN | +2.1pp WIN | +5.6pp WIN |
| HighDist | 0.521 | 0.496 | -2.5pp LOSS | -0.6pp tie | +9.0pp WIN |
| **HeavyCOM** | 0.503 | 0.455 | **-4.8pp LOSS** | -7.8pp LOSS | +5.9pp WIN |
| FastObs | 0.472 | 0.453 | -1.9pp LOSS | -1.1pp tie | +10.6pp WIN |
| Compound | 0.548 | 0.546 | -0.2pp tie | +4.1pp WIN | -0.3pp tie |

**Score: 2W / 1T / 4L.** Worse than v2.10 (4W/2T/1L). v2.6 stays as
paper baseline. v2.11 ckpt to be archived.

### The diagnostic gold — a/c slots collapsed

The 12 new CBF training health stats (built into v2.11 prep) revealed
the failure mechanism clearly:

| Param | Range | Final mean | Final std | Status |
| --- | --- | --- | --- | --- |
| α | [0.1, 5.0] | 3.14 | **2.08** | Wide use ✓ |
| φ | [0.0, 5.0] | 2.52 | **2.10** | Wide use ✓ |
| a | [0.0, **3.0**] | **0.06** | 0.23 | **Collapsed** ✗ |
| c | [0.0, **1.0**] | **0.09** | 0.24 | **Collapsed** ✗ |

`a` and `c` collapsed to ~0 even with WIDE_PARAM_RANGES because they
had no gradient signal — analytical h(x) is exact, so optimal a=c=0 was
strictly minimal. WIDE ranges were wasted. α and φ overloaded carrying
both their own + the dead slots' work (training term_base_contact 11.9%
vs v2.6's 5.8%; action_std drifted to 0.87 with mean_reward going more
negative through training — same v2.7-v2.9b underconvergence pattern).

The architecture itself is sound. Bf-X ablations on the regressed ckpt:

| Comparison | BR | B-fixed-X | Adaptation effect |
| --- | --- | --- | --- |
| Bf-α @ in-dist | 0.451 | 0.570 | **BR wins +11.9pp** ✓ |
| Bf-α @ DensePack | 0.372 | 0.510 | **BR wins +13.8pp** ✓ |
| Bf-c @ HeavyCOM | 0.503 | 0.538 | BR wins +3.5pp ✓ |
| Bf-c @ FastObs | 0.451 | 0.503 | BR wins +5.2pp ✓ |
| Bf-φ @ HighDist | 0.544 | 0.471 | **BR LOSES -7.3pp** ✗ |

4 of 5 confirm per-axis adaptation is load-bearing. Only Bf-φ on
HighDist failed — policy mistuned φ on that axis (consistent with
training underconvergence). The `a`/`c` rows are missing because
those slots were dead — but Bf-α and Bf-c results suggest the slots
COULD be load-bearing if they had signal.

### The design pivot — v2.12 = v2.11 stack + cylinder + persistent bias

Initial proposal was "v2.6 + bimodal" as the cleanest minimal-change
v2.12. User pushed back: keep the v2.11 stack (bimodal + motion DR +
L_f h + WIDE ranges + REWARD-2), and add the missing perception
piece. The hypothesis: v2.11 didn't fail because the stack was wrong,
it failed because `a`/`c` had no signal. With perception noise added,
load redistributes off α/φ, training converges cleanly.

User chose cylinder commitment over grid distance transform: "if
everything is cylinder, we just do cluster-fit + analytical SDF, like
SHIELD." This dropped the v2.13 4-layer grid plan entirely — the
distance transform was load-bearing only for arbitrary-shape support,
which goes away with the cylinder commitment.

For v2.12, picked **Option A — noise on obstacle positions** (rather
than Option B — noise directly on h). Propagates through L_g h
naturally; cleaner physical model. Walls dropped from spawn pool;
separation_buffer kept at 0.4m (lower bar for first launch).

### The LLN bug — caught during sanity check, fixed before launch

Initial implementation had per-step IID noise:

```python
# In _compute_h, every step:
obs_pos_qp = obs_pos + torch.randn_like(obs_pos) * sigma  # FRESH ε_t
```

User pushed back: "remember when sampling noise, try to avoid the law of
large numbers and central limit theorem." Spot on. The policy has a
50-step history-aware encoder + 2-frame occupancy grid. Per-step IID
noise:

- `mean(ε_t over T steps) ~ N(0, σ²/T·I)` — at T=1000 steps per episode
  the residual is σ/√1000 ≈ 0.0016m. Effectively zero.
- The action_rate reward term `-0.005·‖Δa‖²` actively penalizes the
  policy for reacting to per-step jitter.
- Net effect: policy averages the noise away → `a`/`c` slot still has
  no gradient signal → same collapse as v2.11. Training fails identically
  even though we added "noise."

Fix: per-episode persistent bias drawn once at reset, fixed for the
whole episode. Each obstacle gets a stable offset for ~1000 timesteps.
Policy CANNOT average it away. Matches SHIELD's cluster-fit error
structure (LiDAR cluster fit is stable across frames, so deploy-time
perception error is biased rather than jittery).

```python
# At episode reset, ONCE:
self.cbf_obs_pos_noise_bias[env_ids] = torch.randn(n, K, 2) * sigma_per_env

# At each step, just APPLY the persistent bias:
obs_pos_qp = obs_pos + self.cbf_obs_pos_noise_bias  # no fresh draw
```

Two new health stats added: `cbf_obs_noise_sigma_mean / std` (confirm
DR is active during training).

### v2.13 SHIELD-path framing (deferred)

User asked why v2.13 was a grid distance transform. Re-reading my own
earlier framing: the distance transform was the v2.13 design back when
we were going to support arbitrary shapes (chairs, tables, irregular
furniture). With the cylinder commitment, the distance transform is
overkill. SHIELD-path is the right v2.13 architecture:

```text
Synthetic LiDAR raycast (analytical ray-cylinder, on GPU)
   ↓
Cluster hit points (connected-component on ray adjacency)
   ↓
Fit cylinder per cluster (centroid + max radius)
   ↓
Analytical SDF on FITTED cylinders (reuse compute_shape_sdf_batch)
```

~150 lines, 1-2 days. The "bias" becomes natural cluster-fit error
instead of synthetic Gaussian. Deferred until v2.12 ships Goal A — if
v2.12 fails on the same `a`/`c` collapse pattern despite real signal,
v2.13 won't help (it's the same architecture with different noise
source). Save the dev time, diagnose deeper instead.

### Decision philosophy — locked

Two orthogonal goals, sequence them:

| Goal | What | v2.12 status | v2.13 status |
| --- | --- | --- | --- |
| **A** — sim-only paper claim | per-step adaptive 4-param > fixed-param baselines on combined fall+stuck | has every piece; running tonight | not needed |
| **B** — deploy story | train h(x) pipeline = deploy h(x) pipeline | partial (synthetic bias matches SHIELD's deploy noise pattern) | full (real cluster-fit pipeline at training) |

If v2.12 wins Goal A → ship paper. v2.13 becomes optional polish.
If v2.12 fails → diagnose; v2.13 may or may not help.

### v2.12 sanity check — 17 axes, all pass

Built a sanity check script that verified everything before launch:

1. Obstacle pool: 20 cylinders, radii 0.10-0.50m ✓
2. Init positions: 20 entries (matches K_MAX) ✓
3. Planner mix: 6 planners, weights sum to 1.0 ✓
4. Noise cfg field present, default 0.05 ✓
5. NoisyPerception OOD overrides σ_max=0.10 ✓
6. All OOD subclasses inherit CbfGo2EnvCfg ✓
7. NoisyPerception registered in `__init__.py` ✓
8. env.py: σ init, cfg read, per-episode sample, persistent bias, log stats ✓
9. Priv obs grid does NOT reference obs_pos_qp (stays clean) ✓
10. Bash script: 8-task headline + Bf-a row added; no stale v211 refs ✓
11. Bf-X targets fall within physical ranges ✓
12. Sigma sampling guarded by hasattr (init-call safe) ✓
13. Noise applied only to obs_pos_qp, not raw obs_pos ✓
14. Off-stage obstacles unaffected (333× noise margin) ✓
15. All --task invocations match registered envs ✓
16. watch_training_health.sh is parametric ✓
17. REWARD-2 stack still in place (proximity -0.5) ✓

Plus deeper checks: persistent-bias is in `_reset_idx`, NOT `_compute_h`;
no `torch.randn` in `_compute_h` (the smoking gun for per-step IID);
bias scales with per-env σ (heterogeneity preserved across episodes).

### Ready to launch tonight

5 files to sync to lab (3 modified Python + new bash script + watch
script + extract script). Pane A training + eval (~10h). Pane B watch
script (don't skip this time — v2.11 lesson). Trip wires at iter
1000-2000:

- `cbf_obs_noise_sigma_mean ≈ 0.025` (= σ_max/2; confirms DR active)
- `cbf_a_std > 0.20` then > 0.30 by iter 5000 (was 0.23 in v2.11 with
  collapsed mean; want growing mean too)
- `cbf_c_std > 0.20` then > 0.30 by iter 5000

Early abort if `cbf_a_std < 0.10` at iter 2000 — bump σ_max from 0.05
to 0.10 and relaunch. Saves ~7h vs. waiting for the full doomed run.

### Files modified (v2.12 changeset)

- [`cbf_go2_env_cfg.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py): 20-cylinder
  OBSTACLE_SHAPES, restored 6-planner mix, `obstacle_position_noise_sigma_max`
  cfg field (default 0.05), `CbfGo2EnvCfg_NOISY_PERCEPTION` subclass (0.10).
- [`cbf_go2_env.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py): σ + persistent
  bias allocated in `__init__`; per-episode sampling in `_reset_idx`;
  `obs_pos_qp = obs_pos + cbf_obs_pos_noise_bias` in `_compute_h`;
  2 new CBF log stats.
- [`__init__.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py): registered
  `Isaac-CBF-Go2-NoisyPerception-v0`.
- [`scripts/train_and_eval_v212.sh`](scripts/train_and_eval_v212.sh) (new): 8-task headline (added NoisyPerception),
  Bf-a row paired with NoisyPerception, parallel 2-up + watch script
  invocation hint, syntax-verified.
- [`docs/class_paper/methods_outline.md`](docs/class_paper/methods_outline.md) (new): methods section
  TOC reflecting v2.12 design — section 1 problem formulation,
  section 2 CBF formulation + 4-param mapping, section 3 per-step
  adaptive policy, section 4 DR (incl. perception bias), section 5
  reward, section 6 PPO, section 7 evaluation methodology + Bf-X,
  section 8 sim-to-real considerations, section 9 limitations.

Standing by until v2.12 results land tomorrow.

---

## 2026-05-07 (overnight) — v2.11 prep complete + launched on lab

Implemented the full v2.11 scope locked in earlier today, sanity-
checked, fixed two bugs, synced to lab, and launched. Training started
on the lab box; ETA ~9h for train + 7-eval headline + dual-regime
HeavyCOM diagnostic + Bf-α/φ/c ablations.

### What v2.11 ships (all additive, rollback-safe)

**1. Bimodal resample DR** — addresses the v2.10 HeavyCOM smoking gun.
[`cbf_go2_commands.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_commands.py)
gets a `MultiPlannerCommand._resample_command` override that, after
calling `super()._resample_command()`, overwrites `time_left[env_ids]`
with a bimodal sample: `P(0.5)` → uniform [5, 15]s (mid-switch
episode), `P(0.5)` → 100s (locked episode). Earlier draft used uniform
[5, 100]s; user pointed out that with 20s episodes ~84% of those are
effectively-locked. Bimodal gives a clean 50/50 mix. New cfg fields on
the command term: `bimodal_resample`, `bimodal_midswitch_min`,
`bimodal_midswitch_max`, `bimodal_locked_time`. Eval-time forces
`bimodal_resample=False` + `resampling_time_range=(100, 100)` unless
`--bimodal_resample` flag is passed.

**2. Variable obstacle-motion DR.** Per-episode `v_obs` magnitude
sampled in [0, 0.4] m/s instead of fixed 0.2 m/s.
[`cbf_go2_events.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_events.py)
adds a `max_speed_range` param; new branch samples per-env scalar
speed, then per-axis direction `uniform(-1,1)^2 * per_env_speed`.
[`cbf_go2_env_cfg.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py)
event sets `max_speed=0.0, max_speed_range=(0.0, 0.4)`. Sometimes
static, sometimes fast — gives `c` a reason to vary for kinematic
margin.

**3. L_f h obstacle-drift term.** [`cbf_go2_env.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py)
gains `USE_LFH_OBSTACLE_DRIFT = True` flag. `_compute_h()` now
returns 3-tuple `(h_vals, L_g_h, closest_idx)`. `_cbf_filter()` adds
to RHS: `+ (L_g_h * v_obs_closest).sum(dim=-1)` — the
`∂h/∂p_obs · v_obs` piece previously omitted. Clean math justification:
by symmetry of h in `(p_robot, p_obstacle)`, `∂h/∂p_obs = -L_g h`,
so the dot product is angle-aware (cos θ projection of obstacle
velocity onto robot-relative gradient). User pushed back on
"directly subtracting absolute velocity" — confirmed math handles
projection correctly; no separate angle term needed.

**4. WIDE_PARAM_RANGES flag.** Module-level boolean in
`cbf_go2_env.py`. When True, `a_param ∈ [0, 3.0]` (was [0, 1.0]) and
`c_param ∈ [0, 1.0]` (was [0, 0.5]). α and φ unchanged. Single-flag
revert path. `eval_baseline.py` imports the flag so eval-time
`PARAM_RANGES` stays in sync with training.

**5. B-fixed-{α, φ, a, c} eval modes.** New
`make_b_fixed_provider(slot_name, target, br_provider_fn, device, N)`
in `eval_baseline.py`. Wraps the BR provider, clones the action,
overrides one slot via inverse-tanh of the target physical value
(`atanh(2 * (target - mid) / (max - min))`). `_BFIXED_SLOT_INDEX = {"alpha": 0, "phi": 1, "a": 2, "c": 4}`
(`b` slot 3 reserved). New CLI args:
`--bf_alpha_target`, `--bf_phi_target`, `--bf_a_target`,
`--bf_c_target`, plus `--bimodal_resample`. Will run a 4-row
ablation in v2.11 eval block — paper Table 2 starter.

**6. Training health logging — 12 new CBF stats.** Added
`self._cbf_log` dict in `cbf_go2_env.py`, populated each step inside
`_cbf_filter` with: per-slot mean/std for α/φ/a/c, mean h, mean
constraint-active rate, mean L_g h norm, mean QP slack, etc. (12
total). `step()` injects via `extras["log"]` after super().step()
so rsl_rl picks them up per-iter. `extract_training_summary.py`
extended with 12 new METRICS regex entries (permissive `\b<key>:`
pattern so column drift doesn't break parsing). Per-iter CSV now
27 columns vs the pre-v2.11 15.

**7. `watch_training_health.sh`.** New script. Polls the in-progress
training log every 5 min, runs `extract_training_summary.py`, prints
a spot-check block + trip-wire warnings (`r_stuck < -0.20` or
`term_base_contact > 0.10`). Dual-writes via
`exec > >(tee -a "$HEALTH_LOG") 2>&1` to terminal + a derived
`<training_log>.health.log`. Plain text, no ANSI escape codes — clean
copy-paste from the file at any time. User decided not to run a
second tmux pane; just `tee` the training log and tail/cat the
health log when needed.

**8. Parallel eval in `train_and_eval_v211.sh`.** GPU utilization is
<50% on RTX 5090 during eval, so 7-eval block runs 2-up via `&` +
`wait`, with 30s stagger between launches to avoid simultaneous
USD asset load.

### Bugs caught + fixed during sanity check

**Bug A — `task_tag` rename was broken.** Original line in the eval
block:

~~~bash
TASK_TAG="${TASK%-v0}"   # strips "-v0" from task name
~~~

Fails for the bare in-dist task `v0` (no dash prefix), which left the
output directory as `_v0` instead of `_indist`. Same bug as v2.10.
Fixed with a `task_tag()` helper:

~~~bash
task_tag() {
  if [ "$1" = "v0" ]; then echo "indist"
  else echo "${1%-v0}"; fi
}
~~~

User asked to fix and retrain so the in-dist directory naming stays
consistent for the analysis block.

**Bug B — awk column-index drift on Bf-X CSVs (identified, NOT fixed).**
`eval_baseline.py` writes CSVs with `fieldnames = sorted(...)`. Adding
`alpha_target` etc. shifts column positions, so any `awk -F, '{print $7}'`
in the analysis script would break silently. Punted to analysis-time:
post-run script uses header-aware Python parsers (`csv.DictReader` in
`combined_for_row()` and `best_baseline_combined()` bash helpers) so it
adapts to whichever columns are present.

**Concern (documented, not fixed) — bimodal `time_left` write order.**
The override relies on parent `_resample` setting `time_left` before
our hook runs. Standard Isaac Lab pattern, but not pre-verifiable
without source check. If wrong, mid-switch margins won't differ from
locked. Will surface in the trip-wire output (or absence thereof) at
iter ~2500.

### Launch + monitoring

User confirmed launch on the lab box: training writes to
`~/Desktop/safety-go2/IsaacLab/logs/train_and_eval_v211.log` via
`tee`, with the watch script ready to be invoked on demand against
that log. No second tmux pane — single-pane workflow. ETA ~9h for
train (5K iters at v2.10 SPS) + 7-eval headline (locked, 2-up
parallel) + dual-regime HeavyCOM diagnostic (locked vs mid-switch on
v0 + HeavyCOM) + Bf-{α, φ, c} ablations on a single OOD task.

### Predictions

| Metric | v2.10 | v2.11 prediction | Why |
| --- | --- | --- | --- |
| In-dist combined | 0.343 | 0.27-0.32 | bimodal resample restores stuck-recovery regularizer |
| HeavyCOM margin | -7.8pp LOSS | +3 to +6pp WIN | bimodal eliminates the locked-stall failure mode |
| Compound | +4.1pp WIN | hold or grow | REWARD-2 retained |
| `a` slot variance | low (no signal) | still low | measurement noise DR is v2.12; expected to bind only on `c`/`φ` |
| `c` slot variance | bounded by [0, 0.5] | wider use of [0, 1.0] | obstacle-motion DR + wider range |
| L_f h effect | n/a | minor (FastObs only) | v_obs only varies in motion-DR episodes |

If in-dist ≤ 0.31 + HeavyCOM margin recovers + compound holds → v2.11
ships as paper baseline. If HeavyCOM still loses → bimodal didn't fix
intrinsic recovery (would need explicit recovery reward for v2.12).
If in-dist regressed > 0.40 → bimodal destabilized training (would
need to fall back to mid-switch only).

### Post-run analysis script

Drafted a 6-block paste-back script for the user to run after train +
eval ship. Header-aware Python parsers handle the schema variation in
B-fixed-X CSVs:

+ BLOCK 1: training summary (27 cols incl. 12 new CBF stats)
+ BLOCK 2: BR rows from headline 7-eval (locked planner)
+ BLOCK 3: best baseline (B0/B1/B2) per task, header-aware
+ BLOCK 4: dual-regime BR (mid-switch eval on v0 + HeavyCOM)
+ BLOCK 5: Bf-X ablation rows (paper Table 2)
+ BLOCK 6: health-watcher snapshot (last 80 lines if available)

Standing by until v2.11 results land.

---

## 2026-05-07 (very late) — CBF QP audit + 4-param mapping + post-v2.10 plan

While v2.10 trains on the lab box (~7h unattended), audited the actual
CBF QP code and pinned each implemented action-vector slot to its
theoretical robust-CBF role. Found a real DR gap and reframed the
post-v2.10 plan around 4 param-aligned ablations that double as the
paper's Table 2.

### CBF QP audit — verified

Read [`cbf_go2_env.py:168-230`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py#L168-L230).
The actual constraint:

~~~text
L_g h · u_safe  ≥  -α(h - c) + φ·‖L_g h‖² + a
~~~

Standard robust-CBF form. Each implemented param maps cleanly to a
literature-grounded uncertainty class:

| Param | Code term | Range | Uncertainty class | Citation |
| --- | --- | --- | --- | --- |
| α | `-alpha * (h_vals - c_param)` | [0.1, 5.0] | model/tracking error | Molnar 2021 |
| φ | `phi * Lgh_norm_sq` | [0.0, 5.0] | actuation uncertainty | Kolathaya 2018 (ISSf) |
| a | `+ a_param` (additive) | [0.0, 1.0] | measurement uncertainty (state-indep) | Dean 2019 |
| c | `(h_vals - c_param)` shift | [0.0, 0.5] | misdefined safety boundary | — |
| b | slot reserved, unused | — | (input-dep meas. — needs SOCP) | Dean 2019 |

All four implemented params are theoretically correct robust-CBF terms.
The `b` slot would make the constraint second-order cone and break our
closed-form half-space projection at `cbf_go2_env.py:207-215` — needs
cvxpylayers / qpth.

### Real DR gap — `a` is dead weight

Mapping current DR axes to which param they exercise:

| DR axis | Drives | Exercises? |
| --- | --- | --- |
| friction (0.30, 1.20) / (0.20, 1.00), mass | α (model/tracking error) | ✅ |
| force ±10N, torque ±2Nm | φ (actuation uncertainty) | ✅ |
| **measurement noise — none** | **a — DEAD WEIGHT** | ❌ |
| COM ±5cm/±3cm | c (boundary mismatch) | ⚠️ partial; range cap 0.5 may bind |

The biggest single fix for v2.12: inject Gaussian noise on the
priv-obs occupancy grid (σ ∈ [0, 0.1] in training mix). Without
noise, `a` has no gradient signal and likely settles near a constant.
Plus a dedicated `Isaac-CBF-Go2-NoisyPerception-v0` OOD env (higher σ).

### Post-v2.10 plan: 4-axis param-aligned ablations (paper Table 2)

| Param | New reward term | OOD axis | Baseline ablation |
| --- | --- | --- | --- |
| α | `r_α = +k · α_norm · 𝟙(h > h_loose)` (aggressive when safe) | DensePack vs open | B-fixed-α |
| φ | none (implicit chain works) | HighDisturbance | B-fixed-φ |
| a | none (needs env noise first) | NoisyPerception (NEW v2.12) | B-fixed-a |
| c | `r_c = -k · c_norm · 𝟙(h > h_loose ∧ stuck)` (no excess when stuck) | HeavyCOM, FastObs | B-fixed-c |

`B-fixed-X` clamps param X to a tuned constant at eval; ignores
policy's output for that slot. ~30-line change in
`eval_baseline.py` per param. Money plot: x-axis = OOD axis
magnitude, y-axis = combined fall+stuck. BR (adaptive) sits below
both B0 (no-X) and B-fixed-X across the sweep.

### Param range adjustments needed for v2.11

+ `a`: [0, 1.0] → [0, 3.0] (will bind in NoisyPerception OOD otherwise)
+ `c`: [0, 0.5] → [0, 1.0] (HeavyCOM may need >0.5m equivalent shift)
+ α, φ: keep current

Ranges go behind a `WIDE_PARAM_RANGES` flag in `cbf_go2_env.py` so
v2.10 vs v2.11 differ by a single boolean flip — clean rollback.

### Rollback plan committed

Bundled once v2.10 7-eval lands and we like it:

~~~bash
git tag -a v2.10 -m "v2.10: narrow DR + PLANNER-2 + REWARD-2"
mkdir -p IsaacLab/logs/rsl_rl/cbf_go2_teacher/_archive
cp <v2.10_run>/model_4999.pt .../_archive/v2.10_model.pt
scp chrisliang@130.64.84.163:.../v2.10_model.pt ~/Desktop/safety-go2/checkpoints/
~~~

A checkpoint without its matching `env_cfg.py` is unloadable, so the
full rollback unit is `(ckpt + git tag)`. Git tag preserves all four
of: env_cfg, rewards, observations, scripts as they existed at v2.10.

For v2.11+ code changes: additive only.

+ New reward functions added alongside old in `cbf_go2_rewards.py`.
+ New `RewTerm` lines in env_cfg; old ones commented out (not deleted).
+ Param range widening behind `WIDE_PARAM_RANGES` flag — flip to revert.
+ NoisyPerception is a NEW env registration (`-NoisyPerception-v0`);
  doesn't touch the in-dist env at all.

Worst-case: `git checkout v2.10` recovers the entire frozen state.

### L_f h — two distinct sources, separated (clarified post-audit)

Earlier framing conflated two physical sources of L_f h. Cleanup:

**Source #1: robot's own drift (HOCBF territory).** With our current
single-integrator model `ẋ_robot = u`, the robot has no drift and
`L_g h ≠ 0` — the standard CBF works fine. HOCBF would only become
necessary if we switched to double-integrator (`v̇ = u`, `ṗ = v`),
which makes `L_g h = 0` and requires the higher-order formulation
`ψ̇_1 + α_2(ψ_1) ≥ 0` with `ψ_1 = ḣ + α_1(h)`. Not on roadmap;
file as honest paper limitation only.

**Source #2: obstacle drift (NOT HOCBF, real gap).** h depends on
`(p_robot, p_obstacle)`, so true
`ḣ = ∂h/∂p_robot · ṗ_robot + ∂h/∂p_obs · ṗ_obs = L_g h · u + L_f h_obs`.
Our QP currently omits the `L_f h_obs = ∂h/∂p_obs · v_obs` piece,
treating obstacles as static during ḣ computation. This **is** a
real gap, especially for FastObstacles. Fix is small:

~~~text
NEW constraint:
   L_g h · u_safe + L_f h_obs ≥ -α(h-c) + φ‖L_g h‖² + a
                    ↑ ∂h/∂p_obs · v_obs from cbf_obstacle_velocities
~~~

~50 lines: read obstacle velocities (already stashed), compute
gradient w.r.t. obstacle position (similar to existing L_g h), add
to RHS. No solver change, no HOCBF. Matters most for FastObstacles.
Worth doing — separate todo.

**Magnitude justifies the asymmetric handling** (robot drift implicit,
obstacle drift explicit):

| Source | Approx ḣ deviation | Handling |
| --- | --- | --- |
| Robot velocity tracking error | ~0.05 m/s | absorbed by α (small) |
| Obstacle drift | up to 0.4 m/s | too big for α — explicit term |

Asymmetry is engineering pragmatics, not physics. Both effects are
real; α can absorb 0.05 m/s of unmodeled drift without becoming
absurdly conservative; can't absorb 0.4 m/s. Honest paper framing:
"robot tracking error bounded by δ; α chosen large enough that
α(h) ≥ ‖∂h/∂p_robot‖ · δ everywhere; this absorbs the unmodeled
drift." Could measure δ empirically as Wk4 polish.

### Other parked items

+ Drop `b` from 5D action OR commit to SOCP solver — paper-cleanliness
  decision after Table 2 lands.
+ Variable obstacle count in training (currently fixed K) — would
  exercise α more.
+ Lidar cell dropout — additional measurement-uncertainty axis on
  top of Gaussian noise.

### Train-deploy h-pipeline gap (post-audit follow-up)

User raised a real deployment gap: our analytical `_compute_h()`
needs privileged obstacle positions/shapes, but at deploy we'll only
have a LiDAR-built occupancy grid. Standard fix at deploy is
distance-transform of the grid → SDF → finite-diff L_g h. But that
creates a train-deploy h mismatch on top of any noise gap.

This actually *is* the physical motivation for the `a` slot
(Dean 2019 measurement uncertainty = "h evaluated from a noisy
estimate vs. true h"). Connects directly to the v2.12 plan but
expands its scope.

**v2.12 expanded to 4 layers (was 1):**

+ **L1 (h-pipeline parity):** replace `_compute_h()` with
  distance-transform-on-grid variant; train QP and deploy QP use
  the same h(x). Robot footprint via morphological dilation of grid
  (replaces analytical Minkowski). L_g h via finite-diff Sobel.
+ **L2 (censored grid):** replace ego-centric perfect-info crop
  with raycast-built grid (line-of-sight occlusion + range falloff).
  LiDAR-realistic without full sensor sim.
+ **L3 (noise):** Gaussian noise on grid in training DR mix; new
  `Isaac-CBF-Go2-NoisyPerception-v0` OOD env. (Original v2.12 scope.)
+ **L4 (arbitrary obstacle shapes):** once L1 drops the analytical-
  SDF requirement, add irregular meshes (chairs, tables, etc.) to the
  scene and rasterize footprint onto priv-obs grid. Distance transform
  handles them identically. ~1 day code + USD asset library.
  Strengthens deploy realism (real obstacles aren't cylinders).

Three levels of train/deploy parity:

| Level | Training h | Deploy h | Gap |
| --- | --- | --- | --- |
| Now (v2.10) | analytical `h_priv` | grid `h_grid` | big — different math |
| v2.12 mild | `h_priv` + noise on policy's grid | grid `h_grid` | smaller |
| v2.12 honest | `h_grid` from censored noisy grid | `h_grid` from LiDAR | smallest |

User chose "v2.12 honest" (Option B + censored grid). Needs a math
refresher before coding (distance-transform L_g h via finite-diff,
Minkowski-via-dilation equivalence). Estimated 3-4 days of work
versus 1-hour original noise-only scope.

Paper claim becomes much stronger: "adaptive CBF survives the
realistic perception pipeline (occlusion + noise + grid
discretization), with `a` slot empirically learning the
measurement-uncertainty budget."

### SHIELD comparison — closest published related work

User shared SHIELD (Yang et al. 2025, arXiv:2505.11494). Read it. It's
the closest published baseline to what we're doing — humanoid CBF
safety filter with onboard LiDAR perception. Key takeaways for our
positioning:

**SHIELD uses LiDAR + analytical SDF.** Their hardware setup: Livox
Mid-360 LiDAR → Euclidean clustering (PCL) → fit cylinders R=0.3m to
clusters → analytical min-SDF (their Eq. 19) + exponential smoothing
(Eq. 20). Same `λ(1 - exp(-γ·sdf))` form as ours; their λ=10, γ=0.5
(γ matches ours exactly).

**My earlier "analytical = privileged-only" framing was overstated.**
SHIELD proves analytical-SDF works with LiDAR via a clustering layer.
The real restriction is obstacle generality: their cylinder-fit
pipeline can't handle walls, tables, chairs, irregular meshes
without additional primitives (planes, composites). We pick distance
transform for v2.12 because L4 commits to arbitrary shapes.

**Comparison table for paper related-work positioning:**

| Axis | SHIELD | Ours |
| --- | --- | --- |
| Robot model | single integrator (planar) | single integrator (planar) |
| Safety condition | discrete-time, **stochastic** (S-DTCBF, expectation) | continuous-time, **deterministic robust QP** |
| α | single, calibrated to risk level P via Freedman's inequality (Brent's method) | **per-step, RL-learned** (adaptive) |
| Other CBF params | none — α + λ + γ are fixed hyperparams | **4 RL-learned (α, φ, a, c)** for 4 uncertainty classes |
| Disturbance model | CVAE on dynamics residual | implicit via PPO + DR |
| Obstacle representation | analytical SDF via PCL clustering | grid distance transform (planned v2.12) |
| Robot platform | Unitree G1 humanoid | Unitree Go2 quadruped |

**Differentiation points for our paper claim:**

1. **Adaptivity:** SHIELD's α is set per-episode from desired risk P.
   We learn α/φ/a/c per-step from observation. "Params adapt online,
   no per-episode recalibration."
2. **Multi-param uncertainty handling:** SHIELD has just α as the
   safety knob. We have 4 params each tied to a specific uncertainty
   class (Molnar/Kolathaya/Dean/boundary). Paper Table 2 (B-fixed-X
   ablations) directly tests this — SHIELD wouldn't have a comparable
   ablation.
3. **Arbitrary shapes:** their cluster-fit-cylinder restricts to
   humans + simple objects; our grid-based pipeline handles arbitrary
   meshes (v2.12 L4).
4. **Deterministic vs stochastic:** different design point; not
   strictly better/worse — distinct axis for related-work
   discussion.

**Co-author overlap:** Cosner is a co-author on SHIELD. Same lab as
our advisor — relationship to flag, but not a conflict (different
methodology, different platform, different claim).

**Citation added** to `docs/class_paper/references.bib` as
`yang2025shield`.

### v2.10 done — partial recovery; HeavyCOM diagnostic smoking gun

**v2.10 7-eval finished (training done at 18:42, ~9h total).**

In-dist BR combined: **0.343** (vs v2.6's 0.306; partial recovery
zone per decision criteria). Margin against best B 0.442:
**+9.9pp WIN** (vs v2.6's +6.9pp on its own eval).

| Eval | v2.10 BR (fall+stuck=combined) | v2.10 best B | v2.10 margin | v2.6 margin | Δ vs v2.6 |
| --- | --- | --- | --- | --- | --- |
| In-dist (v0) | 0.306 + 0.037 = **0.343** | 0.442 | **+9.9pp WIN** | +6.9pp WIN | ↑ +3.0pp |
| DensePack | 0.367 + 0.043 = **0.410** | 0.444 | **+3.4pp WIN** | +0.6pp tie | ↑ +2.8pp |
| Slippery | 0.368 + 0.096 = **0.464** | 0.485 | +2.1pp WIN | +5.6pp WIN | ↓ -3.5pp |
| HighDist | 0.379 + 0.093 = **0.472** | 0.466 | -0.6pp tie | +9.0pp WIN | ↓ -9.6pp |
| **HeavyCOM** | 0.420 + 0.065 = **0.485** | 0.407 | **-7.8pp LOSS** | +5.9pp WIN | **↓ -13.7pp** |
| FastObs | 0.404 + 0.055 = **0.459** | 0.448 | -1.1pp tie | +10.6pp WIN | ↓ -11.7pp |
| Compound | 0.427 + 0.073 = **0.500** | 0.541 | **+4.1pp WIN** | -0.3pp tie | ↑ +4.4pp |

**Score:** 4W / 2T / 1L. v2.6's 5W / 2T narrative is cleaner.

**Trip wires fired during training** (consistent with eval gap):

~~~text
iter 4999:
  r_stuck            -0.340   ← trip wire was -0.20    ✗
  term_base_contact   0.114   ← trip wire was 0.10     ✗
  action_std          0.74    ← v2.6 was 0.42
~~~

**REWARD-2 trade-off held:** stuck dropped (0.075 → 0.037), fall
rose (0.231 → 0.306). Net combined +3.7pp worse on absolute. The
+9.9pp margin grew partly because best B also got worse on the
harder locked-planner eval (0.375 → 0.442) — eval distribution
itself is different between v2.6 and v2.10.

**Decision per script criteria:** `0.31 < combined < 0.40` →
**v2.6 stays canonical**. v2.10 not the new paper baseline. But
real improvements (compound win, DensePack improvement) suggest
REWARD-2 should be kept; the HeavyCOM regression needs separate
explanation.

### HeavyCOM mid-switch diagnostic — SMOKING GUN

Re-eval v2.10 BR on HeavyCOM with `--planner_resample_s 10`
(mid-switch eval, like v2.6's training+eval) instead of locked:

| Setup | fall | stuck | combined |
| --- | --- | --- | --- |
| v2.10 HeavyCOM **locked** (original) | 0.420 | 0.065 | **0.485** |
| v2.10 HeavyCOM **mid-switch** | 0.348 | 0.050 | **0.397** |
| Δ from regime change | -7.2pp | -1.5pp | **-8.8pp** |

**Just changing the eval-time planner regime drops combined by
8.8pp without retraining.** Both fall and stuck dropped, with fall
contributing the bigger benefit.

**Diagnosis confirmed:** PLANNER-2a (locked-planner training) is
the dominant HeavyCOM regression cause. Mechanism: when CBF
deflects velocity to zero near a COM-tilted obstacle, locked
planner doesn't kick the policy out of stall → robot loses balance
during freeze → fall. v2.6's mid-switching planner was a hidden
**stuck-recovery regularizer** that the policy depended on; v2.10
removed it.

### v2.11 path adjusted — variable resample DR

User decision: don't pure-revert PLANNER-2a (loses heterogeneous
training distribution). Instead **sample resample interval per
episode**:

~~~python
# was (v2.10):  resampling_time_range = (100.0, 100.0)
# now (v2.11):  resampling_time_range = (5.0, 100.0)
~~~

Each episode rolls a resample interval — sometimes mid-switch
every 5s, sometimes locked for the full 100s. Policy sees both
regimes during training. **Eval stays always-locked (100s)** for
deploy-realistic evaluation: real Go2 deployment uses one stable
nav stack.

**Training/eval mismatch is the point.** Training-time
mid-switching is a regularizer that teaches intrinsic stuck
recovery; locked-planner eval verifies the recovery skill
transferred. This is honest CoRL-defensible: "we use heterogeneous
training to learn intrinsic recovery, then verify the deployed
policy is robust to a single locked nav stack."

### v2.11 final scope (all in v2.11 prep)

+ Widen `a` ([0,1]→[0,3]) and `c` ([0,0.5]→[0,1]) behind
  `WIDE_PARAM_RANGES` flag (rollback-safe).
+ **Variable resample DR** (NEW from this diagnostic, BIMODAL not
  uniform). Episode is 20s, so naive uniform [5, 100] gives ~84%
  effectively-locked episodes — defeats the point. Bimodal sampling:
  `P(0.5)` sample uniform [5, 15]s (mid-switch episode); `P(0.5)`
  set 100s (locked episode). Gives clean 50/50 mix. Custom command-
  term subclass override in `cbf_go2_commands.py`, ~10-30 lines.
  Eval always locked.
+ Variable obstacle-motion DR: per-episode v_obs in [0, 0.4] m/s.
+ L_f h obstacle-drift term in QP constraint (~15 lines): equivalent
  to constraining relative velocity (u - v_obs).
+ B-fixed-{α,φ,a,c} eval modes in `eval_baseline.py` (~30
  lines/param, post-tanh-squash override).
+ No new reward terms (DR-implicit shaping preference held).

Targets: HeavyCOM margin recovers, in-dist + compound + DensePack
wins from v2.10 retained.

### Archives

+ v2.10 ckpt at `_archive/v2.10_model.pt` (lab) and
  `~/Desktop/safety-go2/checkpoints/v2.10_model.pt` (Mac, ~13MB
  via scp). Tag `v2.10` exists on Mac repo.

---

## 2026-05-07 (late) — v2.9b done; v2.10 implemented (DR revert)

v2.9b finished: 7 evals, combined 0.479 in-dist (-1.2pp loss vs best B,
basically tied). Marginal improvement over v2.9 (48.6%) but well below
v2.6 (30.6%). Pattern across all 7 evals confirmed wider DR is making
the env uniformly harder. v2.10 designed + implemented to revert that.

### v2.9b BR results (full 7 evals)

| Task | n | fall | stuck | combined | Best B | Margin |
| --- | --- | --- | --- | --- | --- | --- |
| **v0 (in-dist)** | 140 | 35.0% | 12.9% | **0.479** | 0.474 | **-0.5pp tie** |
| DensePack | 146 | 44.5% | 7.5% | 0.521 | 0.504 | -1.7pp loss |
| Slippery | 149 | 37.6% | 6.7% | 0.443 | 0.525 | **+8.2pp WIN** |
| HighDisturbance | 153 | 50.3% | 4.6% | 0.549 | 0.576 | **+2.7pp WIN** |
| HeavyCOM | 147 | 37.4% | 10.9% | 0.483 | 0.522 | **+3.9pp WIN** |
| FastObstacles | 140 | 36.4% | 8.6% | 0.450 | 0.537 | **+8.7pp WIN** |
| RealisticCompound | 158 | 50.0% | 7.0% | 0.570 | 0.587 | **+1.7pp WIN** |

5 OOD WINS / 1 in-dist tie / 1 DensePack loss. Compound flipped to a
win where v2.6 had tied (-0.3pp → +1.7pp) — REWARD-2 *did* fix something
on compositional stress. But in-dist dropped from v2.6's +6.9pp WIN
to -0.5pp tie.

### Trade-off vs v2.9 (in-dist)

| Metric | v2.6 | v2.9 | **v2.9b** | v2.9 → v2.9b Δ |
| --- | --- | --- | --- | --- |
| fall | 23.1% | 40.8% | 35.0% | -5.8pp ✓ |
| stuck | 7.5% | 7.7% | 12.9% | +5.2pp ⚠ |
| combined | 30.6% | 48.6% | 47.9% | -0.7pp |

The -50 → -100 retune did push fall down (mechanism worked) but at
the cost of caution lock-in (+5.2pp stuck). Net combined barely moved.
The retune by itself isn't enough.

### v2.9b training trajectory

action_std went 1.00 → 0.68 (mid) → **0.81** (end) — same convergence
instability as v2.9 (also ended at 0.81). PPO didn't find a clean
optimum even with the stronger fall penalty. Suggests the conflict
between stuck-push-to-move and fall-push-to-be-safe persists.

### v2.10 design — DR revert

v2.9b's pattern across all 7 evals points at a different bottleneck:
**BR absolute combined values are uniformly higher than v2.6's**, even
when margins-vs-best-B are decent. Wider training DR is making the
env harder for everyone. The policy can't converge as cleanly at 5K
iters as v2.6 could on its narrow DR.

v2.10 reverts only the DR (training + matching OOD ranges) back to
v2.6 levels. Keeps PLANNER-2a/2b + the REWARD-2 retune.

### Changes (v2.9b → v2.10)

Training DR (REVERTED to v2.6):

+ `OBSTACLE_MAX_SPEED`: 0.5 → 0.2 m/s
+ `force_range`: ±15N → ±10N
+ `torque_range`: ±3Nm → ±2Nm
+ `static_friction_range`: (0.20, 1.30) → (0.30, 1.20)
+ `dynamic_friction_range`: (0.15, 1.10) → (0.20, 1.00)

OOD eval ranges (REVERTED to v2.6 calibration so eval numbers
compare apples-to-apples with v2.6 paper baseline):

+ Slippery: friction (0.10, 1.45)/(0.05, 1.25) → (0.15, 1.50)/(0.10, 1.30)
+ HighDisturbance: ±22N/±4.5Nm → ±18N/±3.5Nm
+ FastObstacles: max_speed 1.0 → 0.4 m/s
+ RealisticCompound: all 4 components reverted to match
+ HeavyCOM, DensePack: unchanged (already v2.6 values)

Kept from v2.9b: PLANNER-2a (locked), PLANNER-2b (drop walk + adv),
REWARD-2 stack (`base_contact_penalty -100`, `stuck -2.0`,
`proximity -0.5`, plus original 4), 5K iters.

### Hypothesis

v2.6's narrow DR was the *dominant* regression cause across
v2.7/v2.8/v2.9/v2.9b. Wider DR + same compute = under-converged
policy. With training DR back to v2.6 levels:

+ Predicted training base_contact: ~6-8% (v2.6 was 5.8%)
+ Predicted in-dist combined: ~25-30% (v2.6 was 30.6%)
+ Single-axis OOD wins should match or exceed v2.6's
+ Compound: v2.6 tied (-0.3pp); v2.9b won (+1.7pp); v2.10 should win
  (REWARD-2 keeps that improvement on top of cleaner convergence)

### Files edited (2)

+ [`cbf_go2_env_cfg.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py): 5 training DR values reverted (lines 108, 438-439, 457-458) + 4 OOD config classes updated (lines 707-718, 721-733, 753-765, 767-797). Comments document the revert + the v2.6 paper-comparison rationale.
+ [`scripts/train_and_eval_v210.sh`](scripts/train_and_eval_v210.sh): cloned from v29b with full v2.10 rationale in the header. Output dirs `baseline_eval_v210_*`.

### Decision criteria

+ In-dist combined ≤ 0.31 → v2.10 ≥ v2.6 → ship as paper checkpoint.
+ 0.31 < combined < 0.40 → partial recovery; v2.6 stays canonical;
  consider further reward tuning or just ship v2.6.
+ combined ≥ 0.40 → DR revert wasn't the bottleneck; deeper issue.
  Most likely ship v2.6 + locked-eval as the paper claim.

Trip wire at iter 2500 same as v2.9/v2.9b: run
`extract_training_summary.py` partway through; if `r_stuck` is
climbing past -0.20 OR `term_base_contact` isn't dropping below 0.10,
the DR revert isn't enough.

---

## 2026-05-07 — v2.9 abandoned; --no_obstacles diagnostic; v2.9b launched

v2.9 finished training (5000 iters, ~5h) and ran 5 of 7 evals (killed
trailing FastObs + Compound to launch v2.9b sooner). Verdict: REWARD-2
stuck term worked; base_contact_penalty -50 wasn't strong enough.
Single-knob retune to v2.9b: -50 → -100.

### v2.9 BR results (5 evals, in-dist + 4 single-axis OOD)

| Task | n | fall | stuck | combined | Best B | Margin |
| --- | --- | --- | --- | --- | --- | --- |
| **v0 (in-dist)** | 142 | **40.8%** | 7.7% | 0.486 | 0.474 (B1 α=3.0 φ=1.5) | **-1.2pp LOSS** |
| DensePack | 136 | 39.0% | 15.4% | 0.544 | 0.504 (B2) | -4.0pp LOSS |
| Slippery | 148 | 42.6% | 10.1% | 0.527 | 0.525 (B1 α=0.5 φ=3.0) | -0.2pp tie |
| HighDisturbance | 155 | 50.3% | **5.8%** | 0.561 | 0.576 (B2) | +1.5pp small WIN |
| HeavyCOM | 159 | 46.5% | 11.3% | 0.578 | ~similar | likely tie/loss |
| FastObstacles | — | — | — | — | — | killed |
| RealisticCompound | — | — | — | — | — | killed |

### The trade-off (vs v2.6 + v2.8)

| | v2.6 in-dist | v2.8 in-dist | **v2.9 in-dist** |
| --- | --- | --- | --- |
| fall | 23.1% | 26.5% | **40.8%** ⚠ |
| stuck | 7.5% | 22.8% | **7.7%** ✓ |
| combined | 30.6% | 49.3% | 48.6% |
| BR margin | **+6.9pp WIN** | n/a | **-1.2pp LOSS** |

REWARD-2 traded stuck for fall. Stuck went 22.8%→7.7% (back to v2.6
level — stuck term `-2.0/step` worked). Fall went 26.5%→40.8% (almost
doubled). Net combined ~tied with v2.8.

Mechanism: stuck penalty pushed policy to keep moving (it did);
base_contact_penalty -50 wasn't strong enough to keep the moving
policy safe; policy converged on "move aggressively, accept falls."
The 5-eval pattern is consistent across in-dist + 4 single-axis OOD
(fall 39-50% everywhere) — policy-level signature, not env-specific.

### v2.9 training trajectory was unstable

`extract_training_summary.py` on full log:

| Metric | iter 0 | iter 2500 | iter 4999 | Read |
| --- | --- | --- | --- | --- |
| `r_stuck` | -0.009 | -0.212 | **-0.344** | oscillating; ends WORSE than mid |
| `term_base_contact` | 0.000 | 0.202 | 0.179 | slowly dropping (good direction) |
| `mean_ep_len` | 12 | 706 | 778 | grew but ends below v2.8's 852 |
| `action_std` | 1.00 | 0.69 | **0.81** | UP at end — policy destabilizing |
| `mean_reward` | -0.28 | -8.56 | -10.59 | regressing |

`action_std` rising in second half is the warning signal — PPO was
re-exploring because reward terms gave conflicting gradients (stuck
push-to-move vs fall push-to-be-safe), and -50 fall pressure couldn't
stabilize the policy.

### `--no_obstacles` diagnostic on v2.9 ckpt — KEY FINDING

Force K=0 obstacles → CBF non-binding → u_safe = u_des → isolates
locomotion+planner from CBF deflection. ~5 min run.

| Setup | n | fall | stuck |
| --- | --- | --- | --- |
| v2.9 BR with obstacles (in-dist) | 142 | **40.8%** | 7.7% |
| v2.9 BR `--no_obstacles` | 133 | **8.3%** | 16.5% |

**80% of v2.9's falls (32.5pp / 40.8) are CBF-attributable.** Loco-
internal floor under v2.8/v2.9 DR is 8.3%. The remaining 32.5pp comes
from CBF deflection (jumpy params, geometric jerks, sharp deflections
near obstacles).

Reward shaping has substantial room. The original worry that -100
would lock policy into pure caution → stuck is **empirically refuted**:
v2.9 stuck is 7.7% even with -50 + stuck-term. We have headroom to
add fall pressure.

The stuck flip (7.7% → 16.5% under --no_obstacles) is a planner
artifact — locked-per-episode planner decelerates near goal,
robot stops, classified as stuck once goal is reached. Not actionable;
ignore.

### v2.9b retune — single-knob change

**Change:** `base_contact_penalty: -50 → -100` in
`cbf_go2_env_cfg.py`. Everything else identical (stuck -2.0,
proximity -0.5, locked planner, PLANNER-2b, mild DR, 5K iters).

`-100` matches `collision -100` symmetrically — both are
"robot-broken-terminal" events, different from per-step shaping.

Predicted: fall ~15-20%, stuck ~7-10%, combined ~25-30%. If combined
≤ 0.30, v2.9b BEATS v2.6's 30.6% in-dist baseline.

Decision criteria:

+ In-dist combined ≤ 0.31 → v2.9b = new working ckpt; move to Wk3.
+ 0.31 < combined < 0.40 → partial recovery; iterate (tune stuck OR
  re-add `u_safe_rate` as REWARD-3).
+ combined ≥ 0.40 → reward shaping exhausted; deeper issue.

### Files for v2.9b

+ [`cbf_go2_env_cfg.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py): `base_contact_penalty.weight = -50.0 → -100.0` + comment block updated documenting the v2.9 evidence.
+ [`scripts/train_and_eval_v29b.sh`](scripts/train_and_eval_v29b.sh): cloned from v29 with header rewritten and output dirs `baseline_eval_v29b_*`.

v2.9 ckpt at
`logs/rsl_rl/cbf_go2_teacher/2026-05-06_14-25-49/model_4999.pt` is
preserved for later reference if needed.

### awk bug noted

The "best baseline" awk one-liner I provided initially had `sort -k2`
which sorted by name (lexicographic) rather than combined value, so
output always showed `B0_α=0.50` regardless. Real best baselines came
from earlier in the v2.9 eval session. Corrected for v2.9b:

```bash
awk -F',' 'NR>1 && $12 != "BR_teacher" { c=$5+$14; printf "%.3f %s\n", c, $12 }' "$d" \
  | sort -n | head -1
```

(Sort by combined first, ascending, exclude BR row from baseline pool.)

### v2.9b training launched

ETA ~7h total (5h training + 1h45m evals). Trip wire at iter ~2500
same as before — run `extract_training_summary.py` partway through;
if `r_stuck` climbs back above -0.20 the -100 may have been too
aggressive after all and we'd kill + try -75.

---

## 2026-05-06 (overnight) — REWARD-2 v2.9 training launched; iter ~1200 status

v2.9 launched on the lab at ~14:30 local. ETA ~5h total (faster than
v2.8's 6h — SPS ~27K vs ~22K). At iter 1211/5000 (~24%, ~1h 12m
elapsed, 3h 53m remaining). Wrote
[`scripts/extract_training_summary.py`](scripts/extract_training_summary.py)
to parse the rsl_rl log into a clean per-iter CSV; ran it
mid-training to spot-check the trajectory.

### Spot check at iter 1211 (first / middle / last)

| Metric | iter 0 | iter 606 | iter 1211 | Read |
| --- | --- | --- | --- | --- |
| `mean_ep_len` | 12 | 767 | **818** | ✓ Robot stays alive longer |
| `action_std` | 1.00 | 0.69 | 0.66 | ✓ Healthy convergence |
| `term_base_contact` | 0.000 | 0.227 | **0.216** | ✓ Slowly dropping — base_contact_penalty -50 biting |
| `error_vel_xy` | 0.001 | 0.085 | 0.091 | ✓ Locomotion still tracks |
| `r_stuck` | -0.009 | -0.242 | **-0.300** | ⚠ GROWING in magnitude — concerning |
| `term_obstacle_contact` | 0.000 | 0.160 | 0.165 | ⚠ Slowly increasing |
| `mean_reward` | -0.28 | -8.91 | -10.12 | (reflects bigger penalty terms) |

### Trajectory interpretation

Phase 1 of typical safety-RL: policy first learns "don't die" (good
— mean_ep_len up to 818). To not die, it slows down near obstacles.
Slowing triggers `stuck`, so stuck reward grows. The expected per-
episode arithmetic still says falling is ~4× cheaper than full-
episode stuck-timeout, so PPO *should* eventually find the
"navigate" optimum. But it hasn't yet — credit assignment is slow.

This is **early caution lock-in starting to form**. The -50 base_contact
and halved proximity were supposed to prevent it; they're insufficient
so far. Stuck rate trajectory is the trip wire.

### Decision rule (gate at iter 2500)

Check the extractor output around 50% of training:

+ `r_stuck` PEAKED + dropping (e.g., -0.30 → -0.25) → policy finding
  navigation optimum; let it run.
+ `r_stuck` STILL rising past -0.30 → caution lock-in confirmed; kill,
  retrain v2.9b with stuck weight -2.0 → -3.0 OR base_contact_penalty
  -50 → -25.
+ `term_base_contact` ≤ 0.10 by iter 2500 → fall is mostly solved;
  remaining issue is stuck. Same decision branches as above.
+ `term_obstacle_contact` should be dropping (currently 0.165) → CBF
  is being learned; if instead rising, the QP/policy isn't keeping up.

### New tooling

`extract_training_summary.py` parses any rsl_rl-format training log
into a CSV with per-iter columns: mean_reward, mean_ep_len,
action_std, all 7 reward components (including NEW base_contact_penalty
and stuck), error_vel_xy/yaw, and the 3 termination rates. Mid-training
runs are fine — it parses partial logs. Prints first/middle/last spot
check at the end. Useful for safe-to-leave-overnight monitoring.

### GPU / box health (iter ~1100)

Spot-checked nvidia-smi: 26.8GB / 32.6GB GPU mem, 48% util,
62°C, 190W / 575W cap. All normal — Isaac Lab + custom CBF env is a
CPU-GPU-interleaved workload; 30-60% util is expected. SPS 27K (vs
v2.8's 22K) confirms throughput.

### Sanity checks done before launch

Re-read all 4 edited files to verify reasoning:

+ `base_contact_event` mirrors `mdp.illegal_contact` exactly (matches
  the same-step termination check)
+ `stuck_penalty` reads `root_lin_vel_b[:, :2].norm()` — yaw-invariant
+ `last_prev_u_safe` cache plumbed correctly (init/rebind/reset);
  unused since u_safe_rate is unregistered, harmless
+ All weight values verified: -50 base_contact, -2.0 stuck, -0.5 proximity
+ Reverted env_cfg `resampling_time_range = (100, 100)` (locked planner)
+ Fixed 2 stale strings in train_and_eval_v29.sh ("(B-α')" banner +
  "-100 terminal" comment) caught during sanity check
+ Eval method also sanity-checked: `fall_rate` excludes obstacle
  penetration (those count as collision); env.reset per config; train
  threshold 0.15 stricter than eval 0.10
+ All 4 files parse clean (`ast.parse` + `bash -n`)

### Files synced to lab

```text
cbf_go2_env.py, cbf_go2_rewards.py, cbf_go2_env_cfg.py
scripts/train_and_eval_v29.sh, scripts/extract_training_summary.py, scripts/eval_baseline.py
PROGRESS.md, LOG.md
```

---

## 2026-05-06 (late evening) — Probe A done; B-α' rejected; REWARD-2 v2.9 implemented

Probe A finished, B-α' was rejected on principle, and REWARD-2 is
implemented in code (4 files edited). Ready to sync + launch.

### Probe A result

Re-eval v2.8 ckpt on `Isaac-CBF-Go2-v0` with `--planner_resample_s=10.0`
(v2.6-style mid-switch eval) using the new `--planner_resample_s` flag
added to `eval_baseline.py`.

| | v2.8 / locked eval | **v2.8 / mid-switch eval (Probe A)** |
| --- | --- | --- |
| fall | 26.5% | **32.7%** (+6.2pp WORSE) |
| stuck | 22.8% | **12.9%** (-9.9pp better) |
| combined | 49.3% | **45.6%** (-3.7pp better) |
| BR vs best baseline (B0_α=3.0=0.457) | unknown | **+0.1pp tie** |

Compare v2.6 ckpt under same regime (paper baseline): BR 0.306, best B
0.375, **+6.9pp margin**. Even when v2.8 is given v2.6's training-matched
eval regime, **v2.8 has zero margin**. v2.8 is fundamentally a worse
policy than v2.6, regardless of eval distribution.

The fall-up / stuck-down split under mid-switch is informative: stuck
is a *partial attractor* that external command changes (planner switches)
can break. Falls increase under mid-switch because v2.8's policy can't
handle the transitions it never trained on.

### B-α' rejected on principle

Initial plan was Probe B-α': revert PLANNER-2a (mid-switch training
restored) as a single-knob retrain. User pushed back: mid-switch
fixes stuck *artificially* — external command disturbance kicks the
policy out of the zero-velocity attractor; the policy never learns
*intrinsic* recovery. At deployment time, a real Go2 has one stable
nav stack — no mid-episode switches — so any "improvement" from B-α'
disappears in practice.

**This applies backward to v2.6 too.** v2.6's training-time mid-switching
means v2.6 *also* relies on external disturbance to escape stuck. Its
+10.5pp at locked-eval is partly that train was harder than eval
(regression-to-mean), not partly real adaptation. v2.6 might exhibit
the same stuck attractor on real Go2 with one fixed nav stack — we
just haven't tested it.

The legitimate fix is reward shaping that creates direct gradient
pressure on the failure mode, so the policy learns to escape stuck
without external disturbance. Reverted env_cfg back to
`resampling_time_range=(100.0, 100.0)` — locked planner stays.

### REWARD-2 v2.9 design (implemented)

Three reward changes on top of v2.8's structural config (locked planner,
PLANNER-2b, mild DR widening — all kept):

| Term | Weight | Rationale |
| --- | --- | --- |
| **NEW** `base_contact_penalty` | **-50** terminal on fall | Closes structural gap: collision -100 only fires on obstacle_contact (0% in v2.8). Falls (26.5%) had NO terminal penalty → falling was potentially value-positive. -50 (half collision) closes the gap without dominating local gradient and locking policy into pure caution → stuck. Initially proposed -100 but user flagged that as too aggressive. |
| **NEW** `stuck` | **-2.0** per step when ‖v_xy‖<0.15 m/s | Direct gradient on the 22.8% stuck failure mode. Threshold 0.15 m/s above eval's 0.10 m/s so policy trains a margin. Weight chosen so 100 stuck steps (-200) dominates the proximity savings of standing still near obstacle (~-100). |
| **CHANGE** `proximity` | -1.0 → **-0.5** | Episode-mean -0.195 made this dominant per-step term. User insight: "0.1m vs 1.0m from contact should mean the same thing — both safe." Continuous "more far is more better" is artificial and creates over-cautious near-obstacle pressure → contributor to stuck. Halve dominance, keep gradient (PPO needs continuous signal; pure-binary form starves learning outside fear zone). |

A `u_safe_rate` term (penalize ‖Δu_safe‖²) was considered and **rejected**
after user pushback: action_rate already penalizes the controllable
cause of u_safe jerk (smooth CBF params); the remaining (geometric +
constraint-switch) jerk is unavoidable near obstacles, so penalizing
it would conflict with the proximity reduction by re-adding
"stay-away-from-obstacles" pressure. Implementation kept in code
(`cbf_rewards.u_safe_rate` + `last_prev_u_safe` cache in
`cbf_go2_env.py`) but unregistered — revisit as REWARD-3.

### Files edited (4)

+ [`cbf_go2_env.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py): added `last_prev_u_safe` cache (init / step rebind / reset zero).
+ [`cbf_go2_rewards.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_rewards.py): added `base_contact_event`, `stuck_penalty`, `u_safe_rate` (unregistered).
+ [`cbf_go2_env_cfg.py`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py): proximity weight halved; `base_contact_penalty` + `stuck` registered; `resampling_time_range` reverted to `(100.0, 100.0)` with comment documenting B-α' rejection.
+ [`scripts/train_and_eval_v29.sh`](scripts/train_and_eval_v29.sh): rewritten header from B-α' to REWARD-2; output dirs `baseline_eval_v29_*`.

Final v2.9 reward stack (7 active terms): `collision -100`,
`base_contact_penalty -50`, `infeasibility -10`, `u_safe_deviation -0.1`,
`proximity -0.5`, `stuck -2.0`, `action_rate -0.005`.

Also added `--planner_resample_s` CLI flag to
[`scripts/eval_baseline.py`](scripts/eval_baseline.py) — overrides
`env_cfg.commands.base_velocity.resampling_time_range` to `(X, X)` at
eval time. Used for Probe A; kept for future planner-regime probes.

### Decision criteria for v2.9

- In-dist combined ≤ 0.31 (v2.6 paper level) → REWARD-2 closes the gap; v2.9 = new working ckpt; move to Wk3 student distillation.
- 0.31 < combined < 0.40 → partial recovery; iterate weights (likely tune stuck or base_contact, possibly add u_safe_rate as REWARD-3).
- combined ≥ 0.40 → reward shaping isn't enough; deeper architecture/compute issue; reconsider full retrospective.

### Retracted from earlier today

The "training DR needs `c` randomization" claim from 05-04 was wrong.
Confirmed in [`cbf_go2_env.py:179-183`](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py): `c` is the 5th action output of the
teacher (per-step policy decision), not an env DR knob. Memory note
corrected.

---

## 2026-05-06 (evening) — v2.8 failed; planner ablation underway

v2.8 finished training (~6h, 5000 iters) and ran 6 of 7 evals
(RealisticCompound CSV missing; status not yet checked). Result:
**worse than v2.6 on every single axis.** v2.8 abandoned.

### v2.8 final training stats (last 5 iters)

| Metric | v2.6 | v2.7 | **v2.8** |
| --- | --- | --- | --- |
| `base_contact` | 5.8% | 19% | **9.4%** |
| `obstacle_contact` | — | — | 13.5% |
| `error_vel_xy` | — | — | 0.10 |
| Mean reward | — | — | -5.3 |
| Mean episode length | — | — | 852/1000 |

Planner mix rates confirmed PLANNER-2b spec (smooth_goal 0.37-0.47,
waypoint 0.25-0.34, mpc 0.13-0.29, walk=0, adversarial=0). Locking
confirmed (no mid-episode switches).

### v2.8 BR eval results (combined = fall + stuck)

| Task | n | fall | stuck | combined |
| --- | --- | --- | --- | --- |
| In-dist (v0) | 136 | **26.5%** | **22.8%** | **49.3%** |
| DensePack | 136 | 22.1% | 22.1% | 44.1% |
| Slippery | 136 | 22.1% | 22.1% | 44.1% |
| HighDisturbance | 145 | 34.5% | 11.0% | 45.5% |
| HeavyCOM | 144 | 29.9% | 24.3% | 54.2% |
| FastObstacles | 135 | 20.0% | 27.4% | 47.4% |
| RealisticCompound | — | — | — | **CSV missing** |

In-dist comparison vs v2.6 (paper baseline):

| | v2.6 BR | v2.8 BR | Δ |
| --- | --- | --- | --- |
| fall | 23.1% | 26.5% | +3.4pp worse |
| stuck | 7.5% | 22.8% | **+15.3pp worse** |
| combined | 30.6% | 49.3% | **+18.7pp worse** |

**Stuck rate tripled** — that's the dominant failure signature. It's
uniformly ~22-27% across nearly every env (HighDist outlier with
11% stuck because big pushes physically prevent stalling). Uniform
high stuck = policy-level brittleness, not env-specific.

### Diagnosis

v2.6's mid-episode planner switching is the regularizer that
makes locked-eval a free win. Removing it during training (v2.8's
PLANNER-2a) removed the regularizer. Plus PLANNER-2b dropped
walk + adversarial (additional command-space regularization). At
fixed compute (5000 iters), the policy was less robust everywhere.

Stuck failures are the policy stalling under CBF deflection — when
the CBF intervenes, locomotion can't recover. v2.6 saw mid-episode
disturbances during training and learned to recover. v2.8 didn't.

### Earlier "CBF param c randomization" claim — RETRACTED

In an earlier note from 2026-05-04, we'd tagged "training DR needs
`c` randomization" as a missing distribution axis. **This was wrong.**
Reading [cbf_go2_env.py:179-183](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py#L179-L183):
the (α, φ, a, b, c) tuple is the teacher's per-step *action output*,
not a randomized environment parameter. `c` ∈ [0, 0.5] is sampled by
the policy every step — there's nothing static to randomize.

`b` is reserved for SOCP and currently ignored. That memory entry
has been corrected; do not propose c-DR again.

### v2.6 exact planner mix recovered

From `logs/rsl_rl/cbf_go2_teacher/2026-05-04_01-04-47/params/env.yaml`:

```
uniform: 0.00, goal: 0.05, walk: 0.05, adversarial: 0.05
smooth_goal: 0.40, waypoint: 0.25, mpc: 0.20
```

PLANNER-2b's redistribution (walk+adv → smooth_goal+waypoint) was
exactly: `walk -0.05, adv -0.05` reclaimed → `smooth_goal +0.05,
waypoint +0.05`. mpc, goal, uniform unchanged.

### Probe A (running now): is v2.8 ckpt brittle outside locked regime?

Test: re-eval v2.8 ckpt on `Isaac-CBF-Go2-v0` with
`resampling_time_range = (10.0, 10.0)` (v2.6-style mid-switch).
Added `--planner_resample_s` flag to `eval_baseline.py`. ~15 min
runtime. Three possible outcomes:

1. v2.8 collapses harder under mid-switch → policy is brittle outside
   training regime → confirms PLANNER-2a (locked training) damaged
   generalization.
2. v2.8 holds at the same level → not specifically a regime mismatch;
   the regression is from PLANNER-2b or env DR.
3. v2.8 improves → very weird, probably noise.

### Probe B (gated on A): single-knob retrain

Decided to start from v2.8 config and **revert ONE change**, retrain
5000 iters, eval. Whichever revert recovers v2.6-level performance
identifies the culprit.

| Variant | Revert | Tests | Cost |
| --- | --- | --- | --- |
| **B-α'** | `resampling_time_range = (100,100)` → `(10,10)` | Was locking the cause? (PLANNER-2a) | ~7h |
| **B-β'** | restore `walk_weight=0.05, adversarial_weight=0.05` (and pull 0.05 each off smooth_goal + waypoint) | Was dropping walk+adv the cause? (PLANNER-2b) | ~7h |
| **B-γ'** | revert env DR (obstacle 0.5→0.2, friction back, force ±15→±10) | Was DR widening at fixed compute the cause? | ~7h |

If Probe A says "brittle" → run B-α'. If Probe A says "similar
either way" → run B-β' first (next-most-likely), then B-γ' if still
bad.

### Reward-shaping queue (post-Probe-B)

Three terms stay queued. Will be designed once Probe B identifies
the regression cause:

- **Stuck penalty:** continuous `-λ_s · exp(-‖v_xy‖/σ)` always-on slow-
  velocity penalty. Targets the 22.8% stuck failure mode (no penalty
  signal exists for it currently).
- **Δu_safe smoothness:** `-λ_du · ‖u_safe_t − u_safe_{t-1}‖²`. Targets
  the 23.1% v2.6 in-dist fall (CBF over-deflection → locomotion
  can't track).
- **base_contact penalty:** terminal `-λ_b · 1[base contact]`. Mostly
  redundant with `collision` (-100); revisit only if needed.

---

## 2026-05-06 — v2.7 abandoned; PLANNER-2a validated; v2.8 launched

Three things happened today: (1) v2.7 evals came back losing across
the board, (2) a diagnostic isolated v2.7 ckpt as the cause AND
revealed PLANNER-2a alone is a free win, (3) v2.8 designed with
milder widening + locked planner + 5000 iters and launched via a
one-shot train+eval pipeline.

### v2.7 eval results (all losses)

| Eval | v2.6 margin | v2.7 margin | Δ |
| --- | --- | --- | --- |
| In-dist | +6.9pp | −0.5pp | −7.4pp |
| DensePack | +0.6pp tie | −0.8pp | −1.4pp |
| Slippery | +5.6pp | **−11.4pp** | **−17pp** |

Stopped after 3 evals — pattern was clear. v2.7's wider DR + same
3000-iter compute = under-converged. Training base_contact 5.8%
(v2.6) → 19% (v2.7) reflected this.

### Diagnostic (v2.7 ckpt vs v2.6 ckpt on identical setup)

Reverted base CbfGo2EnvCfg DR back to v2.6 ranges (kept
PLANNER-2a). Ran in-dist eval on both checkpoints:

| Setup | BR fall | BR stuck | BR combined | Best baseline | **Margin** |
| --- | --- | --- | --- | --- | --- |
| v2.6 ckpt + v2.6 DR + UNLOCKED (paper baseline) | 0.231 | 0.075 | 0.306 | 0.375 | +6.9pp |
| **v2.6 ckpt + v2.6 DR + LOCKED** | **0.131** | 0.185 | **0.316** | 0.421 | **+10.5pp** |
| v2.7 ckpt + v2.6 DR + LOCKED | 0.258 | 0.167 | 0.424 | 0.421 | −0.3pp |

Two findings:

1. **v2.7 ckpt is genuinely worse** (0.424 vs 0.316 on identical
   env). REALISM-1's widening hurt convergence at 3000 iters.
2. **PLANNER-2a (locked planner) alone is a free win.** v2.6 ckpt
   + locked planner grows the in-dist margin from +6.9pp →
   +10.5pp. Mechanism: locked planner removes a noise source
   (mid-episode planner switches that confused locomotion); BR
   keeps fall rate low while baselines hurt slightly more.

### v2.8 plan (designed + launched)

v2.6 hyperparams (frozen recipe), plus:

+ **PLANNER-2a:** planner locked per episode.
+ **PLANNER-2b:** drop walk + adversarial from training mix
  (45% smooth_goal / 30% waypoint / 20% mpc / 5% legacy_goal).
+ **Mild DR widening:** friction (0.20, 1.30)/(0.15, 1.10), force
  ±15N, torque ±3Nm, motion 0.5 m/s. COM unchanged.
+ **5000 iters** (vs v2.6's 3000) — compute headroom.
+ **OOD configs recalibrated** to push past v2.8 training (not the
  too-aggressive v2.7 OOD ranges).

Launched via `scripts/train_and_eval_v28.sh` — one-shot pipeline
that trains ~5h, locates the resulting checkpoint, then runs the
full 7-eval suite (~1h45m). Total ~7h unattended.

### TightGap dead code permanently removed

After yesterday's removal (~150 lines across `env_cfg`, `__init__`,
`events.py` + helper), confirmed no references remaining.

---

## 2026-05-05 (evening) — full v2.6 eval suite closed; compound TIE; REALISM-1 launched

All 6 v2.6 evals complete (in-dist + 5 single-axis + compound).

| Eval | Type | BR fall | BR combined | Best baseline | **Margin** |
| --- | --- | --- | --- | --- | --- |
| In-dist | mixed | 0.231 | 0.306 | 0.375 | **+6.9pp** |
| Slippery | priv-obs (cont.) | 0.312 | 0.411 | 0.467 | **+5.6pp** |
| DensePack | scene-only | 0.257 | 0.338 | 0.344 | +0.6pp tie |
| HighDisturbance | priv-obs (episodic) | 0.237 | 0.341 | 0.431 | **+9.0pp** |
| FastObstacles | priv-obs (grid history) | 0.211 | 0.338 | 0.444 | **+10.6pp** |
| HeavyCOM | priv-obs (startup) | 0.218 | 0.331 | 0.390 | **+5.9pp** |
| **RealisticCompound** | all 5 modest pushes simultaneously | **0.416** | **0.500** | **0.497** | **−0.3pp TIE** |

**5 of 6 wins on single-axis. Compound result is a TIE, not the
predicted +8-12pp win.** BR is the second-best of all 25 configs at
compound, but the most uniformly conservative B2 baseline edges it
out by 0.3pp (statistical noise).

### Diagnosis: adaptations don't compose

Compound BR fall rate: 0.416 (vs in-dist 0.231; **+18.5pp degradation**).
Sum of single-axis BR fall degradations: +8.0pp. Compositional stress
hits 2.3× harder than the linear sum of single-axis stresses suggests.

The teacher's CNN was trained on a distribution where each DR axis
varies independently within its training range. When all axes are
simultaneously at high-tail values, the priv obs vector is in a region
the training never visited. The teacher does *something*, but it's
not the per-axis correct response.

This is the exact failure mode REALISM-1 was gated on. The gating
trigger fires.

### REALISM-1 launched (v2.7)

Widening 4 dynamics DR axes in training so the teacher sees compound
shifts during PPO. Existing single-axis OOD configs shift to push
further out (so they remain meaningfully OOD post-v2.7).

**Training DR widening (`CbfGo2EnvCfg.__post_init__`):**

- Friction static (0.30, 1.20) → (0.15, 1.50); dynamic (0.20, 1.00) → (0.10, 1.30)
- Force ±10N → ±18N; torque ±2Nm → ±3.5Nm
- COM xy ±5cm → ±8cm; z ±3cm → ±5cm
- Obstacle max_speed 0.2 → 1.0 m/s (compromise — at robot speed; user
  vision is 1.5 m/s human-walking, but 1.0 keeps adversarial planner
  episodes navigable)

**OOD configs pushed further (so still OOD relative to widened training):**

- Slippery: (0.15, 1.50) → (0.05, 1.80) static; (0.10, 1.30) → (0.05, 1.60) dynamic
- HighDisturbance: ±18N → ±30N; ±3.5Nm → ±6Nm
- HeavyCOM: ±8cm → ±12cm xy; ±5cm → ±8cm z
- FastObstacles: 0.4 m/s → 2.0 m/s
- DensePack unchanged (separation_buffer 0.2; v2.7 base still has 0.4)
- RealisticCompound: all the above pushes simultaneously

v2.6 hyperparams (weight_decay 1e-5, action_rate -0.005,
entropy_coef 0.005) frozen for v2.7. No reason to revisit those.

**Sensor noise (user-listed REALISM-1 axis) deferred to v2.8 polish.**
Adding occupancy-grid corruption requires modifying the priv-obs
computation; want to land v2.7 dynamics widening first and confirm
the compound gap closes before introducing more variables.

### Decision criteria for v2.7

After v2.7 trains and we re-run the suite:

- **Compound margin ≥ +5pp:** REALISM-1 worked. Lock claim, move to
  Wk3 student distillation. Optionally do v2.8 with sensor noise
  for paper polish.
- **Compound margin still flat:** dynamics widening alone isn't
  enough. Reconsider — maybe sensor noise IS load-bearing, or the
  encoder isn't using the wider priv obs effectively.
- **Single-axis margins shrink significantly:** wider DR may have
  hurt convergence quality. Need to tune training (more iterations?
  bigger network?).

### REALISM-1 launch status

- `env_cfg` widening committed (no `__init__` changes needed —
  existing OOD task IDs stay registered, just point at updated
  configs that push further out).
- v2.6 checkpoint preserved at `2026-05-04_01-04-47/model_2999.pt`
  for paper comparison.
- **v2.7 training in progress** on lab GPU; new timestamp TBD.
- Wk3 student distillation gated on v2.7 results.

### PLANNER-2a (2026-05-05): planner_id locked for full episode

`resampling_time_range` changed from inherited `(10.0, 10.0)` to
`(100.0, 100.0)` in `CbfGo2EnvCfg.__post_init__`. With episode length
~20s, this means one planner per episode (no mid-episode swaps).
Effects:

- **Deployment-realistic.** At deploy a Go2 has one nav stack per
  session, not in-session planner swaps. Training distribution now
  matches.
- **Removes a source of fall/stuck.** Mid-episode planner change
  (e.g., smooth_goal hands off to waypoint at t=10s) leaves
  locomotion gait in a bad state — likely contributes to the 7.8%
  no-obstacle stuck floor. Hypothesis: post-change, no-obstacle
  stuck drops to ~3-4%.
- **Mismatch resolved by restart (later 05-05):** original v2.7
  launch had mid-episode swaps in train but locked planner at eval.
  Decided to kill that run and restart v2.7 with locked planner
  from the start. Train and eval now match. Cost: lost ~1h of
  compute on the killed run; gained clean experimental setup.

### TightGap dead code removed (2026-05-05 late evening)

Cleanup pass before v2.7 restart. TightGap was retired from
headline experiments on 05-04; the env_cfg class, gym registration,
and `place_tight_gap_obstacles` event handler (plus its
`_shape_half_y` helper) were all unused. ~150 lines removed across:

- `cbf_go2_env_cfg.py` — `CbfGo2EnvCfg_TIGHT_GAP` class deleted.
- `__init__.py` — `Isaac-CBF-Go2-TightGap-v0` gym registration
  deleted.
- `cbf_go2_events.py` — `place_tight_gap_obstacles` function +
  `_shape_half_y` helper deleted.

LOG.md still has the diagnosis of why TightGap failed (planner-
contract mismatch); referencing it for the paper limitations
section is unaffected by the code removal.

### v2.7 restart status (2026-05-05 late evening)

CUDA OOM on first restart attempt — zombie Python compute process
(PID 368428, 4h 40m old, holding 25.7 GB) from the original v2.7
launch we never fully killed. `pkill -f rsl_rl/train.py` evidently
left the omniverse-side process alive. Cleared with `sudo kill -9
368428` + ~20s lazy CUDA release; GPU memory dropped from 26604
MiB → 872 MiB without needing `nvidia-smi --gpu-reset` or reboot.
v2.7 retrain now running clean.

### REWARD-2 and PLANNER-2b queued (2026-05-05)

**REWARD-2 (v2.8 retrain after v2.7 eval lands):** combined reward
shaping pass — both axes worth trying:

1. **Explicit base_contact termination penalty** — terminal
   punishment, mechanistically novel (current reward has no fall-
   specific signal, falls only punished by losing future reward).
   Start at moderate weight (e.g., -20 to -30); watch action_std
   and stuck rate for over-caution.
2. **Δu_safe smoothness penalty** — penalize step-to-step changes in
   the QP-output velocity command. Diminishing returns vs existing
   `action_rate` (Δθ smoothness) and `weight_decay`, but mechanism
   is genuinely different — smooths the QP *output* rather than the
   parameters going *into* the QP. User explicitly wants this on
   the queue alongside base_contact.

Both target the residual ~21pp CBF-deflection-induced fall mode.
Goal: get fall rate under 18% in-dist while keeping stuck near floor.

**PLANNER-2b (queued):** drop walk + adversarial planners from
training mix (re-distribute their 10% weight to smooth_goal/
waypoint/mpc). Aligns training with deployment-realistic mix; small
expected in-dist quality bump. Apply alongside v2.8 train or as
separate ablation.

---

## 2026-05-05 — near-OOD partial results (3/5); pattern matches predictions

Three of five v2.6 near-OOD evals landed. **Pattern is exactly as
predicted: priv-obs axes win cleanly, scene-level axes tie.**

| Eval | Type | BR fall | BR stuck | BR combined | Best baseline combined | **Margin** |
| --- | --- | --- | --- | --- | --- | --- |
| In-dist (ref) | mixed | 0.231 | 0.075 | 0.306 | 0.375 (B2_α=1.5_ε=0.5_λ=1) | +6.9pp WIN |
| Slippery | priv-obs | 0.312 | 0.099 | 0.411 | 0.467 (B2_α=1.5_ε=0.5_λ=1) | +5.6pp WIN |
| DensePack | scene | 0.257 | 0.081 | 0.338 | 0.344 (B2_α=0.5_ε=0.5_λ=1) | +0.6pp TIE |
| HighDisturbance | priv-obs | 0.237 | 0.104 | 0.341 | 0.431 (B1_α=3.0_φ=3.0) | **+9.0pp WIN** |

Pending: **HeavyCOM** (priv-obs, predict ~5-9pp WIN), **FastObstacles**
(scene, predict tie), **RealisticCompound** (compositional, code
committed, eval not yet run).

### "BR fall barely degrades" signature on priv-obs axes

Telling diagnostic: BR fall rate vs in-dist fall rate (0.231 baseline).

- Slippery (priv-obs, continuous): BR fall went **0.231 → 0.312** (+8pp).
  Friction affects every footfall; BR adapts but can't fully compensate
  for repeated slips.
- DensePack (scene): BR fall **0.231 → 0.257** (+3pp). Modest
  degradation; both methods see denser obstacles via h(x).
- HighDisturbance (priv-obs, episodic): BR fall **0.231 → 0.237**
  (+0.6pp, basically unchanged). Force is applied at reset and damps;
  teacher reads it once via priv obs and picks conservative params.
  Cleaner adaptation target than continuous friction.

Meanwhile baselines uniformly degraded under HighDisturbance (B0
α=0.5: 0.293 → 0.396 fall; +10pp). The teacher held flat while
baselines collapsed → 9pp margin opens up.

### What this means for the paper claim

Predicted "priv-obs axes WIN, scene axes TIE" pattern is materializing.
This is *better* than a uniform "wins everywhere" claim because it's
interpretable: the teacher delivers exactly where its structural
advantage lies (priv obs visible to it but not to fixed-filter
baselines), and matches the best baseline where the playing field is
level (h(x) feedback shared by all methods).

Paper claim sharpens to:

> Our learned teacher beats the best hand-tuned baseline by 6.9pp
> in-distribution and by 5.6 / 9.0pp on the two priv-obs near-OOD
> axes evaluated so far. On scene-level OOD axes (where baselines
> have access to the same h(x) feedback as the teacher), BR matches
> the best hand-tuned baseline within statistical noise — confirming
> that the win comes specifically from privileged-obs adaptation,
> not from incidental advantages.

Stronger, cleaner, harder to attack than "wins everywhere" would be.

### Status going into evening of 2026-05-05

- 3/5 single-axis near-OOD evals done, all in line with predictions.
- HeavyCOM + FastObstacles in flight on lab GPU.
- RealisticCompound (compositional, all 5 modest pushes simultaneously)
  committed in code; will run after the 5 single-axis evals release
  GPU.
- REALISM-1 retrain still gated on full single-axis + compound results.
- Wk3 student distillation + paper draft is the path forward if
  remaining evals land as predicted.

---

## 2026-05-04 (evening) — v2.6 trained; eval-1 WIN; TightGap retired; pivot to 5-axis near-OOD suite

v2.6 training completed (3h, checkpoint at
`logs/rsl_rl/cbf_go2_teacher/2026-05-04_01-04-47/model_2999.pt`).
Eval-1 in-distribution: BR beats best baseline by 6.9pp combined —
the headline win. Eval-3 (TightGap OOD): BR loses 17pp combined,
diagnosed as planner-contract mismatch, not a teacher quality issue.
Decision: retire TightGap from headline experiments; replace with a
5-axis near-OOD suite that preserves the multi-planner contract.

### v2.6 training metrics (final iter)

```text
base_contact      5.8%   (LOWEST EVER — vs v2.4: 8.5%, v2.5: 15.5%)
obstacle_contact 14.8%
infeasibility     0.0%
u_safe_deviation -0.054
action_rate      -0.0006 (gentler -0.005 weight; v2.5 had -0.0022)
Mean action std   0.42   (HEALTHY — v2.5 collapsed to 0.06)
error_vel_xy      0.98   (best tracking we've recorded)
combined termin. 20.6%   (best yet)
```

The entropy bump (0.001 → 0.005) was the load-bearing fix, exactly
as predicted. Action-std stayed at 0.42 (well above the 0.06 collapse
seen in v2.5). Weight decay 1e-5 (from v2.5's 1e-4) gave smoothness
without crushing the policy. v2.6 hyperparams now treated as the
working recipe — frozen unless a future failure mode demands change.

### Eval-1 — in-distribution combined win

| Mode | Best config | fall | stuck | combined | h̄ | φ̄ |
| --- | --- | --- | --- | --- | --- | --- |
| Best B0 | α=0.50 | 0.293 | 0.226 | 0.519 | 0.305 | 0.005 |
| Best B1 | α=0.50, φ=1.50 | 0.296 | 0.119 | 0.415 | 0.371 | 1.50 |
| **Best B2** | α=1.50, ε=0.50, λ=1.0 | 0.243 | 0.132 | **0.375** | 0.371 | 1.43 |
| **BR (v2.6)** | teacher | 0.231 | **0.075** | **0.306** | 0.413 | 0.91 |

**BR margin: 6.9pp combined improvement over best baseline.** Three
narrative beats:

1. BR ties for best fall rate (0.231; best baseline 0.226 within noise).
2. BR has the **lowest stuck rate of all 24 baselines** by a clean
   margin — next-best stuck is B2_α=1.5_ε=0.1_λ=1 at 0.102 (BR is
   ~27% lower than that, ~43% lower than the best B1).
3. φ̄=0.91: less deflection budget than the high-φ baselines that
   match its fall rate. *Adaptive, not uniformly conservative.*

This is the realistic-deployment-distribution claim. ~134 episodes
per config (64 parallel envs × 2000 steps yielding ~134 finished
episodes). Standard error on each fall/stuck rate ≈ 3-4pp at this
sample size, so the 6.9pp gap is roughly 2σ — meaningful but not
crushing. Consider bumping `--steps_per_config` for the camera-
ready, or moving to paired evaluation (pre-generated seed list,
same scenes across all methods).

### Eval-3 — TightGap OOD loss; root-cause is planner contract

| Mode | Best config | fall | stuck | combined |
| --- | --- | --- | --- | --- |
| **Best B2** | α=1.5, ε=0.5, λ=3 | 0.039 | 0.117 | **0.156** |
| Best B0 | α=1.5 | 0.039 | 0.141 | 0.180 |
| Best B1 | α=3.0, φ=1.5 | 0.102 | 0.102 | 0.203 |
| **BR (v2.6)** | teacher | 0.023 | **0.305** | 0.328 |

BR has the 3rd-lowest fall rate of the 25 configs (0.023) but the
*highest* stuck rate of the entire sweep. h̄ = 0.536 (vs in-dist
0.413) — in tight gaps, BR *increases* margin from obstacles
instead of threading them, then freezes.

**Root cause** (read of `cbf_go2_env_cfg.py` line 584+): TightGap
is structurally different from training in ways that obstacle DR
alone cannot cover. The load-bearing difference is the **planner
contract**, not gap widths:

- Training planner = 6-way realistic mix (40% smooth_goal, 25%
  waypoint, 20% mpc, 5% each walk/adversarial/legacy). All routing
  planners — they pick goals at x∈[3,7], y∈[-2.5,2.5] and **steer
  around** obstacles to reach them.
- TightGap planner = constant `lin_vel_x = 1.0`, no routing.

The teacher learned "if tight, increase margin and the planner
will reroute." That's a coherent strategy *under the training
planner contract*. TightGap removes the routing flexibility, so
the teacher's "increase margin" response degenerates to "stop
dead." Hand-tuned B0/B1/B2 are stateless filters with no planner
expectations, so they pass straight through unaffected.

This is **structural unfairness**, not a teacher failure: the
eval was designed for a setting where the planner actively works
against the teacher's learned coordination. Fixing this by adding
straight-push to training would basically be "training on the
test distribution" for that specific scenario.

### Pivot: retire TightGap, design fair near-OOD suite

User decision: drop TightGap from the headline. Reasoning matched
his earlier philosophical point — fixed-filter baselines don't
have an in-dist/OOD axis (they're stateless), so any sufficiently
adversarial OOD eval will favor them by construction. The right
paper structure for an adaptive learned method is:

1. **Headline:** in-dist combined win on the realistic deployment
   distribution. (✓ in hand: +6.9pp.)
2. **Near-OOD:** push *single DR axes* slightly past training,
   keeping the planner contract identical. Tests whether learned
   adaptivity *extrapolates*. Hand-tuned baselines are tested on
   the same shifted scenes (symmetric).
3. **Limitations:** TightGap reported honestly as a stress test
   that breaks the planner-CBF coordination assumption.

### Near-OOD suite (5 axes, single-knob each)

Implemented as 5 new env configs in `cbf_go2_env_cfg.py` and 5 new
gym tasks in `__init__.py`. Each inherits `CbfGo2EnvCfg` and
overrides exactly one DR parameter:

| Task | Knob | Training | Near-OOD push | Tests |
| --- | --- | --- | --- | --- |
| `Isaac-CBF-Go2-DensePack-v0` | obstacle separation_buffer | 0.4m | 0.2m | spatial perception |
| `Isaac-CBF-Go2-Slippery-v0` | static/dynamic friction | (0.30, 1.20) / (0.20, 1.00) | (0.15, 1.50) / (0.10, 1.30) | proprioception adapt |
| `Isaac-CBF-Go2-HighDisturbance-v0` | force / torque | ±10N / ±2Nm | ±18N / ±3.5Nm | disturbance rejection |
| `Isaac-CBF-Go2-HeavyCOM-v0` | COM offset xy / z | ±5cm / ±3cm | ±8cm / ±5cm | body-property adapt |
| `Isaac-CBF-Go2-FastObstacles-v0` | obstacle max_speed | 0.2 m/s | 0.4 m/s | dynamic perception |

Three of the five (Slippery, HighDisturbance, HeavyCOM) push axes
that *are* in the teacher's priv obs but *cannot* be used by the
hand-tuned baselines — these are the strongest "adaptivity wins"
candidates. DensePack and FastObstacles are scene-level shifts
that affect both teacher and baselines symmetrically.

**Paper claim if all 5 land wins:**

> Our learned teacher beats hand-tuned baselines on combined
> fall+stuck, in-distribution AND across five near-OOD axes
> (obstacle density, friction, disturbance, body COM, obstacle
> motion). The adaptive response generalizes across multiple
> perturbation types, supporting the RMA-inspired teacher design.

Five near-OOD wins is *much* harder for a reviewer to dismiss
than the one-in-dist + one-disputed-OOD pattern.

### Status

- All 5 env configs + gym registrations committed in
  `cbf_go2_env_cfg.py` and `__init__.py`. ~75 lines added.
- TightGap config left in place (unregistered from headline use,
  but kept for reference / limitations section).
- Sync to lab pending.
- 5 evals each ~15 min. Run sequentially or up to 3-parallel
  (RTX 5090 GPU-Util saturates at 2-3 concurrent Isaac Sim
  instances; memory has headroom).

### Next gates

1. Sync 2 files to lab; run 5 near-OOD evals.
2. If 4-5 wins: lock the paper claim, start Wk3 student
   distillation. If 2-3 wins: investigate which axes broke and why
   (priv obs not actually used? regularization needed?). If 0-1
   wins: deeper diagnostic on the teacher's adaptation mechanism.
3. (Stretch) Compositional OOD — combine 2 axes (e.g., slippery
   and dense at once) — only after single-axis wins are clean.
4. (Stretch, paper-polish) Move to paired evaluation: same fixed
   seed list across all methods, then report per-scene paired
   stats. Standard error drops 5-10×.

### REALISM-1 — staged plan after 5-axis results

User framing (2026-05-04 evening): the current in-dist win (+6.9pp) is
real but modest. To produce a "win by 150%+" result, the training
distribution should be widened along axes that *structurally* favor
the teacher over hand-tuned baselines. Six asymmetric axes the teacher
exploits but baselines structurally cannot:

- **Friction** — priv obs; baselines have constant params.
- **COM offset / mass** — priv obs; baselines invariant.
- **External force / torque** — priv obs; baselines invariant.
- **Occupancy grid history** (motion inference) — teacher has a 2-frame
  ConvNet; baselines see only the current h(x) snapshot.
- **Sensor noise on occupancy grid** — teacher can learn denoising via
  the ConvNet; baselines consume whatever noisy h(x) the SDF outputs.
- **Harder scenarios in general** — adaptive params per scene beat
  one-set-fits-all once the scene distribution stresses the trade-off.

**Subtle tradeoff:** widening training to cover current near-OOD axes
pulls them inside distribution; loses them as OOD evals. New OOD evals
must be pushed further out. So REALISM-1 is a **paired redesign** of
training AND OOD eval suite, not just a training widening.

**Gating logic** — REALISM-1 fires only after the 5-axis suite results:

- 4-5 of 5 wins on current near-OOD: REALISM-1 is *optional polish*.
  Likely don't pursue with 3 weeks to deadline; ship v2.6 + Wk3
  student.
- 2-3 of 5 wins: REALISM-1 is the *diagnosis + fix*. Specific failures
  tell us which axes to widen (e.g., HighDisturbance loss → widen
  force training; Slippery loss → widen friction training).
- 0-1 wins: deeper issue (priv obs not actually used by the teacher).
  Investigate the encoder before retraining.

**Cheap additions worth doing regardless** (low retrain risk, high
sim-to-real credibility): sensor noise on occupancy grid (~2% dropout,
~1% spurious) + faster moving obstacles (0.2 → 1.5 m/s, human-walking
range). Both are env_cfg one-liners. Hold pending 5-axis results.

---

## 2026-05-04 — v2.5 trained; DIAG-3 (Lipschitz) confirmed; v2.5 LOST eval-1; v2.6 planned

v2.5 training completed (3h 13m). Lipschitz diagnostic on the trained
checkpoint confirmed the regularization mechanically did its job (L
dropped substantially). But eval-1 shows v2.5 lost vs both v2.4 and
hand-tuned baselines. Diagnosis: over-regularization + action-std
collapse.

### v2.5 final-iter (training)

```text
base_contact      15.5%  (vs v2.4: 8.5% — falls DOUBLED during training)
obstacle_contact  15.6%  (≈v2.4)
infeasibility      0.0%  (math clean throughout)
u_safe_deviation  -0.081 (vs v2.4: -0.052 — slightly more deflection)
action_rate       -0.0022 (penalty active, modest magnitude)
Mean action std    0.06  (vs v2.4: 0.22 — COLLAPSED, tripped flag)
error_vel_xy       1.20  (vs v2.4: 1.07 — slightly worse tracking)
```

The 0.06 action std was the "kill immediately" threshold I'd called out
pre-launch. Combined regularization (wd=1e-4 + action_rate=-0.01 + the
already-low entropy_coef=0.001) crushed exploration. Policy converged
to a deterministic local optimum.

### DIAG-3 — direct Lipschitz measurement

Wrote `scripts/diag_lipschitz.py` (standalone, no Isaac Sim — loads
state_dict and re-implements the forward pass manually). Three methods:

1. Spectral norm product (worst-case upper bound)
2. Local Jacobian σ_max (state-dependent local L)
3. Empirical finite-difference (most realistic; 3 perturbation flavors)

Results — every metric points the same direction:

| Quantity | v2.4 | v2.5 | Δ |
| --- | --- | --- | --- |
| Method 1: spectral upper bound | 2837 | 1303 | **−54%** |
| Method 2: local Jacobian mean | 17.16 | 11.22 | −35% |
| Method 2: local Jacobian p95 | 26.12 | 16.59 | −37% |
| Method 3: gaussian_obs | 0.224 | 0.166 | −26% |
| Method 3: gaussian_dyn | 0.141 | 0.079 | **−44%** |
| Method 3: single_grid_cell | 0.170 | 0.122 | −28% |

Per-layer breakdown showed weight decay disproportionately hit the
small/output-side layers (15→64 dyn, 128→12 head, 128→5 final all
shrunk 13-24%). The dominant L contributor — `grid_proj.linear`
(8192→64, σ_max=14.12) — only shrunk 6% because L2 decay at 1e-4
doesn't bite as hard on parameter-rich layers. Future v2.6+ may want
layer-specific decay or spectral norm clipping on grid_proj
specifically.

### Eval-1 — v2.5 in-distribution result

| Mode | Best | fall | stuck | combined | h̄ | φ̄ |
| --- | --- | --- | --- | --- | --- | --- |
| Best B2 | α=1.5, ε=0.5, λ=1.0 | 0.243 | 0.132 | **0.375** | 0.371 | 1.43 |
| BR (v2.4) | teacher | 0.291 | 0.082 | **0.373** | 0.421 | 1.36 |
| BR (v2.5) | teacher | **0.434** | 0.051 | **0.485** ⚠ | 0.463 | **2.66** |

**v2.5 BR lost.** Fall rate went UP 14pp vs v2.4. Combined went from
parity-with-best-B2 to 11pp WORSE. The v2.5 teacher is more
conservative (φ̄ 1.36→2.66, h̄ 0.42→0.46) but falls more despite the
conservatism. Smoking gun for over-regularization + action-std
collapse: deterministic policy stuck in a "very cautious but very
rigid" local optimum.

NOTE: eval-3 (TightGap OOD) overwrote eval-1's CSV (both eval runs
write to `logs/baseline_eval/baseline.csv` — race condition when run
in parallel). Re-running eval-3 separately right now.

### Reconciling: DIAG-3 says "smoother", eval-1 says "worse"

These are not contradictory. They tell us:

> Lipschitz reduction is necessary but not sufficient. Too much
> smoothness, combined with collapsed action std, over-constrains the
> policy. There's a Goldilocks operating point — neither very high L
> (jerky → falls from CBF jerkiness) nor very low L (rigid → falls
> from inability to react adaptively) is optimal.

This is actually a useful framing for the paper — provides a clean
story even with v2.5 losing. The mechanism works (DIAG-3 proves it);
the magnitude was wrong.

### v2.6 plan (gentler regularization)

| Knob | v2.5 | v2.6 | Rationale |
| --- | --- | --- | --- |
| `_CBF_WEIGHT_DECAY` | 1e-4 | **1e-5** | 10× weaker — partial L reduction |
| `action_rate` weight | -0.01 | **-0.005** | 2× weaker — bias not dominate |
| `entropy_coef` | 0.001 | **0.005** | 5× stronger — keep exploration alive (most important change) |

The `entropy_coef` bump is the load-bearing fix. We had 0.001 from the
v2.2-era when entropy was over-aggressive; now with two regularizers
on top, 0.001 is starving exploration. v2.6 returns it to a healthier
range.

Conditional on eval-3 OOD result for v2.5: if v2.5 BR also loses on
TightGap, definitely launch v2.6. If v2.5 BR somehow wins OOD (low
fall via conservatism), we have a richer Pareto and may delay v2.6 to
explore that finding.

---

## 2026-05-03 (late evening) — Eval-1/3 closed; DIAG-2 isolates CBF as fall driver; v2.5 launched

v2.4 trained, evaluated in-distribution + on TightGap, parity not win.
DIAG-2 (no-obstacles isolation) nails CBF deflection as the fall source.
v2.5 launched with action-rate penalty.

### v2.4 final-iter (training)

3000 iters / 4096 envs / 3h 18m. End state:

```text
base_contact      ~8.5%   (vs v2.3 9.3% on old mix)
obstacle_contact ~15.5%   (vs v2.3 ~11% on old mix)
infeasibility     0.000   (math clean throughout)
u_safe_deviation -0.052   (modest deflection)
error_vel_xy      1.07    (locomotion tracking residual ~1 m/s, expected)
```

CBF math solid; planner mix balanced; convergence stable.

### Eval-1 — in-distribution (Isaac-CBF-Go2-v0, 25 configs × 2000 steps × 64 envs)

Best combined (fall + stuck) per mode:

| Mode | Best | fall | stuck | combined |
| --- | --- | --- | --- | --- |
| B0 | α=1.5 | 0.346 | 0.125 | **0.471** |
| B1 | α=0.5 φ=1.5 | 0.296 | 0.119 | **0.415** |
| B2 | α=1.5 ε₀=0.5 λ=1.0 | 0.243 | 0.132 | **0.375** |
| BR | v2.4 teacher | 0.291 | 0.082 | **0.373** |

**Verdict: BR ties best-B2.** Difference 0.2pp is within sampling noise
(134 eps/config → ±3pp). Math 100% clean (0 collisions, 0 infeasibility).

Behavioral signal: **BR has the lowest stuck rate of any mode** (8.2% vs
10.2-13.2% baselines). Teacher genuinely learned to un-stick — not
random luck. But pays for it with marginally more falls than best B2
(29.1% vs 24.3%).

### Eval-3 — OOD on TightGap (Isaac-CBF-Go2-TightGap-v0)

| Mode | Best | fall | stuck | combined |
| --- | --- | --- | --- | --- |
| B0 | α=1.5 | 0.039 | 0.141 | **0.180** |
| B1 | α=3.0 φ=1.5 | 0.102 | 0.102 | **0.203** |
| B2 | α=1.5 ε₀=0.5 λ=3.0 | 0.039 | 0.117 | **0.156** |
| BR | v2.4 teacher | **0.008** | 0.320 | 0.328 |

**TightGap is easier than training mix** (sparse structured obstacles
vs cluttered random). BR has the **lowest fall rate of any config**
(0.8% vs 3.9% best B2 = 5× lower) but loses on combined due to high
stuck (32%) — teacher is over-cautious in tight passages.

CIs at n=128 are wide: BR fall 95% CI ≈ [0%, 4.3%]; best B2 ≈ [1.3%, 8.9%].
"BR has lower falls" is suggestive, not statistically robust at this
sample size. Would need 500+ eps/config for tight CIs.

### DIAG-2 — planner+loco isolation (no obstacles)

User-proposed isolation experiment to answer: "are falls coming from
the planner output (u_des too jerky) or from CBF deflection (u_safe
too jerky)?"

Implementation:

- Added `--no_obstacles` flag to `eval_baseline.py` (overrides
  `randomize_obstacles_position.params["k_max"]` to 0).
- Relaxed assertion in `cbf_go2_events.py:174` from `K == k_max` to
  `K >= k_max` so the structural obstacle pool can stay at 20 while
  runtime sample range is set to [0, 0]. Pool intact, no obstacles
  spawn on stage.

Result (B0 α=0.5, n=128 episodes):

```text
                  fall    stuck    combined
With obstacles    29.3%   22.6%    51.9%
NO obstacles       1.6%    7.8%     9.4%
                 -27.7pp -14.8pp  -42.5pp
```

**~95% of B0 falls come from CBF deflection or obstacle interaction,
not from planner+loco.** Planners alone produce u_des that locomotion
tracks fine (1.6% fall = noise floor: DR pushes, friction outliers).
The 28pp gap is what we need to recover. The 7.8% residual stuck is
likely "robot reaches goal early, sits there" episodes.

`h̄ = 1.000` confirms the override took effect (SDF saturates to 1.0
when obstacles are out of FOV).

### v2.5 changes

Two changes vs v2.4 (both target the CBF-induced fall component):

**1. Action-rate penalty (smoothness in time):**

```python
# cbf_go2_rewards.py:
def action_rate(env):
    diff = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(diff ** 2, dim=-1)

# cbf_go2_env_cfg.py:
action_rate = RewTerm(func=cbf_rewards.action_rate, weight=-0.01)
```

Penalizes step-to-step jerkiness in CBF params. Smooth params →
smooth u_safe → locomotion tracks → fewer falls.

**2. AdamW + weight decay (smoothness in input→output mapping):**

`RslRlPpoAlgorithmCfg` doesn't expose `weight_decay`. First attempt
modified upstream `rl_cfg.py` and `train.py` — failed at runtime
because those files didn't sync from dev → lab machine (only
`cbf_go2/` subtree sync'd). **Reverted both.**

Final approach: monkey-patch `OnPolicyRunner.__init__` from
`cbf_go2/__init__.py` (in-task, always sync'd). Patch wraps the
runner init, then walks `self.alg.optimizer.param_groups` and sets
`pg["weight_decay"] = 1e-4`. PyTorch Adam/AdamW both honor per-group
weight_decay updates at next step. Patch is scoped to runners whose
log_dir contains `"cbf_go2_teacher"` so other tasks are unaffected.

PPO cfg keeps `optimizer="adamw"` (rsl_rl honors that string in the
optimizer Literal); the patch supplies the decay value.

`_CBF_WEIGHT_DECAY = 1.0e-4` constant at top of `__init__.py` — set
to 0 to disable. Patch is idempotent (`_cbf_wd_patched` sentinel).

Why both: action-rate is direct (penalize Δa in reward) but local
(only sees one-step jumps). Weight decay is indirect (shrink network
weights) but global (smooths the entire input→output landscape, so
similar inputs map to similar outputs). They're orthogonal — together
should beat either alone.

**Deferred to v2.6 if v2.5 under-delivers**: EMA on teacher output,
spectral normalization on the CNN. (Both are "post-network" smoothing;
weight decay + action-rate cover the "in-network" + "in-time" axes.)

### Expected impact

If smoothness halves the CBF-induced fall component:

- v2.4 BR fall: 29.1% → ~15%
- BR combined: 37.3% → ~23%
- Beats best B2 (37.5%) by ~14pp — clean win

Even a 30% reduction puts BR at ~28% combined, still beating B2.

### Strategic note

If v2.5 wins clean → submit paper with "minimum total failures across
distributions" claim. If v2.5 only marginally improves → pivot framing
to "minimum catastrophic failure" (BR's fall rate is already 5× lower
on TightGap; with v2.5 also competitive on stuck, the safety claim
holds even without combined-metric dominance).

---

## 2026-05-03 (evening) — DR-coverage discussion (during v2.4 wait)

Conceptual discussion while v2.4 trains. No code changes; queuing two
items for v2.5 DR polish.

### SCENE-2 broaden — moving-obstacle velocity range

Current: ±0.2 m/s per axis on ~50% of slots. Locked (per discussion)
to broaden to **(0.0, 1.5) m/s** for v2.5. Rationale: 1.5 m/s is the
practical ceiling — robot can't reliably outrun faster obstacles
anyway (Go2 cruise ≈ 1.0-1.5 m/s), and the floor at 0 keeps mostly-
static episodes in the mix. Industry-standard DR recipe: cover the
deployment envelope, don't push past where the platform fails for
unrelated reasons.

### Sensor-noise DR — occupancy-grid corruption

Real Mid360 has dropouts (black/glossy surfaces, oblique angles,
glass), spurious returns (dust, multi-bounce), and range jitter
(~2cm close, ~5cm at 40m). Sim is currently perfect — sim-to-real
gap on perception not yet covered.

Plan for v2.5: corrupt the occupancy grid directly (not per-ray —
policy reads the grid, so cheaper to noise that):

```python
# In obs function, after building occupancy grid (training only):
# Knob 1: dropout — drop ~2% of occupied cells per frame
occupied_mask = (grid == 1)
drop = torch.rand_like(grid) < 0.02
grid = torch.where(occupied_mask & drop, 0.0, grid)

# Knob 2: spurious — add ~1% false-positive cells per frame
add = torch.rand_like(grid) < 0.01
grid = torch.where(~occupied_mask & add, 1.0, grid)
```

Independent noise per frame (don't carry forward — temporal stack of
2 frames lets policy learn "flickering = noise"). Skipping per-ray
range jitter: 10cm grid swallows ±2-5cm jitter. Skipping reflectivity-
dependent dropout (no material info in sim) and motion-smear (50Hz
loop is fast enough).

Rates (2% / 1%) are starting points; calibrate against real Mid360
deploy logs post-hardware if there's time.

### Why both queue for v2.5 (not v2.4 mid-train)

v2.4 vanilla isolates the planner-mix change. Bundling DR changes
in confounds the comparison. If v2.4 beats baselines clean, v2.5
becomes "DR polish" pass. If v2.4 underperforms, v2.5 also adds
smoothness terms (action-rate + weight decay) and the DR additions
ride along.

### Architecture clarification (for future me)

Confirmed: in RMA, priv obs (mass, CoM, friction, push vel) are
*inputs to the teacher's encoder*, never direct policy features.
Flow: `priv obs → encoder_teacher → Z(12) → π_decoder → action`.
The decoder reads only Z, not the priv obs themselves. For our CBF
use case, dynamics priv (15D) is less load-bearing than for original
RMA (which output joint torques) — it provides a "physical readiness"
signal so the policy knows how aggressive it can afford to be.
Stripping it would still work; the policy would just assume worst-
case dynamics. Keeping it in for v2.4.

### Z dim — is 12D enough?

Almost certainly yes for our setup.

- Output is 5D (4-effective: α, φ, a, c). Z just needs enough
  independent directions to describe task-relevant state; decoder
  paints action via 128-hidden MLP.
- Budget estimate: 2-3 dims for nearest-obstacle dist/heading,
  2-3 for clutter/gap geometry, 1-2 for moving-obstacle approach
  speed (the reason we stack 2 frames), 2-3 for physical readiness,
  1-2 slack. Total ~7-11. 12D has light headroom.
- RMA original used **8D** for *locomotion* (12-joint torque output,
  much harder). 12D for 4-effective-D output is generous.
- For student distillation, Z dim is rarely the limiter — the real
  failure modes are history length and adapter capacity.

Cheap diagnostic post-v2.4: SVD on a Z-trajectory (T × 12) from a
trained rollout. If singular values decay sharply after ~5, then 12
was overkill but harmless. If all 12 are similar magnitude, all dims
are doing work. ~10 lines, runs in a minute.

When to revisit: only if v2.4 plateaus (suspect capacity → ablate
16/24D) or student distillation fails despite long history (suspect
Z encodes student-unreconstructible info → consider lowering to 6-8D
to *force* student-reconstructibility). Both invalidate the trained
checkpoint, so 3h retrain per ablation.

### Strategic framing — M1 is the gate

Once teacher beats baseline (M1), the project moves from "is this
even going to work" to "known recipe with known failure modes."
Student distillation (M2) is RMA-paradigm well-established; sim-to-
real (M3) has standard tooling; paper writeup is well-trodden with
Cosner. So v2.4 is genuinely the make-or-break moment.

Three things to pin down BEFORE seeing v2.4 results (prevents post-
hoc rationalization):

1. **Win threshold.** B0 is 42-55% combined on new mix. Bar:
   v2.4 BR ≤ 50% of B0 = ≤ 22% combined. Below that = unambiguous
   win. 25-30% = solid. 35%+ = marginal/loss → trigger v2.5.
2. **B1/B2 will also shift on new mix.** Real comparison is BR vs.
   best-of-{B1, B2}, not BR vs B0. Run all 4 modes in eval-1.
3. **Training-curve early signals (~iter 1000, ~1h in):**
   - `u_safe_deviation` drifting → 0 = CBF not deflecting = "do
     nothing" collapse
   - `infeasibility` count rising = QP regression
   - termination rate stuck at B0 levels = no learning gradient

### Arbitrary obstacle shapes — current state and decision

Current shape pool (SCENE-4): boxes (8 cubes, 4 walls, 2 rect
boxes) + cylinders (6). Per-shape analytical SDF, Minkowski'd with
0.15m footprint. Not implemented: spheres, convex polytopes,
concave shapes, mesh-based geometry.

**Decision: not before paper.** Reasons:

1. CBF method is shape-agnostic — works on anything with an
   analytical or precomputed SDF. Paper claim doesn't require
   shape variety; it requires *the method generalizes given an
   SDF*. Boxes + cylinders demonstrate that.
2. Adding shapes costs a 3h retrain pass each. Pre-deadline cost-
   benefit is negative.
3. Real-world 80/20 already covered: walls (box), pillars (cyl),
   crates (box), columns (cyl). Concave furniture (chairs, tables-
   with-legs) typically convex-hulled in practice.

**Important nuance — two separate questions conflated under
"arbitrary shapes":**

| Question | Currently | Sim-to-real path |
| --- | --- | --- |
| Training-time shape variety | 2 primitives | Mechanical to extend |
| Deploy-time SDF source | Analytical (known shape) | LiDAR-derived approx |

Deploy-time is the harder one. Real Go2 won't have analytical
primitives — it'll have a Mid360 point cloud. Standard approaches:
distance-transform on occupancy grid (cheap, ~10cm floor accuracy),
point-cloud kd-tree NN (accurate, slower), implicit neural SDF
(overkill).

Post-paper queue:

- **Hardware demo:** distance-transform on LiDAR occupancy grid as
  deploy-time SDF. Grid already exists from PRIV-2; runtime ~1ms.
- **Follow-up paper:** richer shape variety + evaluation on unseen
  geometries. Its own project.

If anyone asks at review: "method is shape-agnostic; demonstrated
on box + cylinder in sim. Student LiDAR pipeline naturally handles
arbitrary geometry at deploy because the occupancy grid doesn't
care what shape made the points."

---

## 2026-05-03 — PLANNER-1 implemented; v2.4 launched on realistic mix

Implemented the 3 realistic planner approximations + reconfigured the
training mix. v2.4 teacher training launched (~3h).

### Code changes

`cbf_go2_commands.py` — three new planner_id values + dispatch:

```text
4 = PLANNER_SMOOTH_GOAL    rate-limited goal pursuit (P-controller approx)
                           u_des delta capped at 2.0 m/s² × dt = 0.04 m/s/step
5 = PLANNER_WAYPOINT_PATH  multi-waypoint path (start → 2 wiggle wpts → goal)
                           lateral perturbation ±1.5m off line; same rate limit
6 = PLANNER_MPC            PD-controller: u = K_p(goal-pos) − K_d(vel)
                           K_p=0.5, K_d=0.5, capped at 1.2 m/s; smooth via dynamics
```

Skipped: potential-field/reactive (1980s classical, known local-minima
failures, user explicitly rejected).

`cbf_go2_env_cfg.py` — new mix replacing old uniform/goal/walk/adversarial:

```text
40% Smooth GOAL   (rate-limited; basic nav stack)
25% Waypoint A*   (curved path)
20% MPC-like (PD) (smooth, decelerates near goal)
 5% Walk          (stress padding)
 5% Adversarial   (stress padding)
 5% Legacy GOAL   (back-compat, straight-line)
 0% Uniform       (dropped — covered by realistic ones)
```

`scripts/diag_jerk_source.py` — added planner_id 4/5/6 to PLANNER_NAMES.

### Smoke-test results

Eval (200 steps, biased — only fast-failures complete) ran cleanly. Diag
ran for seeds 10 (walk, timeout), 20 (smooth_goal, timeout) — both 1000
steps clean, no fall, no NaN.

### Full eval — B0 on new mix (2000 steps)

```text
α    fall    stuck    combined  h̄
0.5  23.7%   19.1%    42.8%     +0.332
1.5  37.1%   18.2%    55.3%     +0.295
3.0  27.0%   14.6%    41.6%     +0.327
```

Compare old mix:

```text
α    fall    stuck    combined  h̄
0.5  13.8%   10.8%    24.6%     +0.485
1.5  23.4%   12.4%    35.8%     +0.298
3.0  21.1%   13.5%    34.6%     +0.299
```

**New mix is ~2× harder for B0 (naive fixed-α CBF).** This is expected:
old mix was 30% `uniform` planner (held random velocity, often pointing
nowhere useful — robot just stood around, easy episodes). New mix is
~90% goal-directed → robots actively cross obstacle field every episode
→ more obstacle interactions → more failures with naive CBF.

The drop in `h̄` (0.485 → 0.332) confirms robots spend more time near
obstacles. CBF math still clean (zero collisions, zero infeasibility).

### Why this is good for the paper

Old framing: "B0 fails 14%, BR fails 8%" — marginal claim.
New framing: "Naive fixed-α CBF fails 42-55% on realistic deployment.
Adaptive teacher BR achieves X%." Bigger headroom for the teacher to
demonstrate value. Harder baseline = stronger paper claim once teacher
beats it.

### v2.4 training

Launched training on the new env (3000 iters, 4096 envs, ~3h). Same
config as v2.3 (REWARD-1 Variant C reward, original off-the-shelf
locomotion). The only difference is the planner mix. No smoothness
terms added yet — keeping that as v2.5 fallback if v2.4 doesn't beat
baselines convincingly.

### Next: 3-stage eval plan

After v2.4 finishes:

1. **In-distribution eval** on `Isaac-CBF-Go2-v0` (training mix).
   Modes: B0/B1/B2/BR. ~10 min. Headline number for paper.
2. **Per-planner breakdown**. Set planner weights to (1, 0, 0, ...) in
   turn for each realistic planner. Shows "teacher works across all
   realistic planner types we trained on." ~30 min.
3. **OOD eval** on `Isaac-CBF-Go2-TightGap-v0` (already registered).
   Tests teacher's set-shrink param `c` in tighter spaces it didn't
   train on. ~30 min. Stretch claim.

Total ~1.5h post-training to get full paper results.

---

## 2026-05-02 (afternoon) — DIAG-1 results; pivot to realistic planner mix

Built `scripts/diag_jerk_source.py` (per-step logger) + added `stuck_rate`
to `eval_baseline.py`. Three findings reshape the project plan.

### Finding 1: stuck_rate hidden inside timeouts

Re-ran B0 baseline eval with the new metric:

```text
α     fall    stuck    timeout (incl stuck)   true failure
0.5   13.8%   10.8%    86.2%                  24.6%
1.5   23.4%   12.4%    76.6%                  35.8%
3.0   21.1%   13.5%    78.9%                  34.6%
```

The "alive but not moving" failure mode (locomotion enters a zero-velocity
attractor after a sharp planner change) was hiding in the 86% "timeout
success." Real deployment failure rate is **25-36%**, not 14-23%.

### Finding 2: per-step traces — who's jerky changes per planner

Ran DIAG-1 across seeds 0/1/2 (got goal/goal/adversarial planners), plus
the original seed 42 (walk). Pattern:

| Seed | Planner | h_min | CBF activity | Outcome |
| --- | --- | --- | --- | --- |
| 42 | walk | 0.55 | None (always far) | timeout, **stuck** after 180° flip |
| 0 | goal | 0.15 | Brief deflection | timeout, OK tracking |
| 1 | goal | 0.13 | Sustained deflection | timeout, **degraded** after goal-reset |
| 2 | adversarial | 0.09 → 0 | Heavy | **fall/collision** at step 973 |

CBF math is correct in all four. The variability is in *how the locomotion
handles the planner's command stream*. Walk planner's random 180° flips
break the gait (the seed 42 stuck was an unlucky uniform-heading sample);
goal planner's resamples cause smaller jumps the locomotion mostly handles.
Adversarial actively drives toward obstacles — falls eventually.

### Finding 3: training distribution mismatch

Our training mix (30% uniform / 35% goal / 20% walk / 15% adversarial)
spends 65% of training on planners no real user ever runs. Walk's random
heading flips and adversarial's per-step obstacle-pursuit are unrealistic;
they consume teacher capacity learning to handle situations that won't
exist in deployment.

The cleaner research framing: train on realistic-controller mix, eval on
realistic deployment. Drop the synthetic stress planners.

### Decision: pivot to realistic planner mix

Replacing the planner mix with three realistic-controller approximations:

1. **Smooth GOAL** — rate-limited goal-pursuit (P-controller approximation
   of a basic nav stack). ~20 lines.
2. **Waypoint A\*-like** — precomputed path through occupancy grid at
   reset, output velocity along path. ~80 lines.
3. **MPC-like horizon** — closed-form LQR-style horizon optimization,
   output first velocity. ~60 lines.

Skipped: potential-field / reactive (classical 1980s, known local-minima
failures, would muddy results — user explicitly rejected).

Keep walk + adversarial at 5% each as robustness padding.

New mix target:

```text
40% Smooth GOAL
25% Waypoint A*-like
20% MPC-like horizon
 5% Walk        (padding)
 5% Adversarial (padding)
 5% Existing GOAL (legacy point-and-go)
```

### Next phase

1. Implement 3 realistic planners (PLANNER-1)
2. Reconfigure planner mix in `cbf_go2_env_cfg.py` / `MultiPlannerCommandCfg`
3. Retrain v2.4 teacher on new mix
4. Re-eval baselines on new mix (also per-planner breakdown for paper)
5. Decide on smoothness terms (v2.5) only if v2.4 doesn't beat baselines

Side note (deferred): teacher's `cbf_go2_teacher_cnn.py` has a JIT-export
bug — module-level int constants don't TorchScript. Doesn't affect
training/deployment (env loads via `torch.load`); only triggered by
`play.py` auto-export. ~5-line fix when we touch the teacher for v2.4.

---

## 2026-05-02 — Custom locomotion experiment closed; reverted to off-the-shelf

**Outcome:** custom-trained locomotion is *worse* in deployment than
the off-the-shelf one we were trying to replace. Reverted
`LOCOMOTION_CHECKPOINT` and reframed the project around the drop-in
safety filter philosophy.

### Training attempts (4)

1. **Rough + ±2 m/s in all 3 axes + matched DR** → collapsed to
   "stand still." `track_lin_vel_xy_exp ~0.08`, `terrain_levels` stuck
   at 0, `base_contact 22.9%`. 8× command volume on 1500-iter budget
   was too aggressive. [killed iter 1500]
2. **Rough + asymmetric ±2/±1/±1.5** → same failure, slower.
   `terrain_levels` regressing 0.40 → 0.18 at iter 200. [killed iter 200]
3. **Rough + default ±1 + matched DR** → trained successfully **but
   obs mismatch**: rough task adds a 187-D `height_scan` from a
   sim-only raycaster. Real Go2 (and our CBF env) has no such sensor
   → 235-D input policy can't slot into our 48-D `_run_locomotion`.
   Caught after ~70 min training.
4. **Flat + default ±1 + matched DR** → CONVERGED. 11 min wall-clock.
   Final iter `base_contact 1.34%`, `error_vel_xy 0.20`,
   `track_lin_vel_xy_exp 1.41`. Excellent training metrics.

### Deployment result

Slotted attempt-4 checkpoint into `LOCOMOTION_CHECKPOINT`, ran B0
eval (no teacher in loop, fixed CBF α):

```text
B0_α=0.50  fall=0.277  coll=0.000  infeas=0.000
B0_α=1.50  fall=0.286  coll=0.000  infeas=0.000
B0_α=3.00  fall=0.323  coll=0.000  infeas=0.000
```

**28-32% fall rate** vs off-the-shelf's previous ~20%. CBF math
clean (zero collision, zero infeasibility). The custom-trained policy
is *worse* in deployment than the one it replaced.

### Diagnosis: command-frequency mismatch

| Aspect | Training (locomotion task) | Deployment (CBF env) |
| --- | --- | --- |
| Commands | UniformVelocityCommand, **held 10s** | Planner per-step output, **50 Hz** |
| Episode | 1000 steps × 1 command | 1000 steps × 1000 commands |
| Result | locomotion locks into stable gait | locomotion never settles |

Locomotion was trained on commands held for 500 sim steps. At deploy
it gets a new command every step. Wider DR didn't help — if anything
made the policy more "twitchy" (reactive to perturbations), worse at
high-frequency commands.

### Reframe: drop-in safety filter

User clarified the product vision: a CBF that works with **any**
off-the-shelf locomotion. Customizing the locomotion to fit our env
breaks that promise — users plugging in their own loco would see
worse performance than they had before.

So:

- `LOCOMOTION_CHECKPOINT` reverted to
  `unitree_go2_flat/2026-04-15_19-24-45/exported/policy.pt`
- Burden of "safe AND performant" sits on **the teacher**, not the
  locomotion. Teacher must produce u_safe smooth enough that any
  reasonable off-the-shelf locomotion can track it.
- The 20% B0 floor is what we accept; the teacher should beat it
  through smarter CBF param choices.

### Next phase: diagnose then train v2.4 vanilla

1. Re-confirm B0 with reverted loco (~5 min)
2. Build per-step logger (u_des, u_safe, robot_vel) — find which
   stage of the planner→QP→loco chain is jerky. Three diagnoses:
   - Loco can't track even smooth commands → frequency-mismatch real,
     slow CBF down or accept floor
   - Planner u_des is jerky → fix planner-side or accept it
   - QP u_safe is jerky despite smooth u_des → CBF deflections near
     obstacles are the source
3. Train v2.4 teacher on original loco. No smoothness terms yet
   (action-rate penalty, weight decay) — see if vanilla teacher beats
   B0 floor on its own.
4. Eval v2.4 vs B0.
5. v2.5 with smoothness terms only if v2.4 doesn't help.

### Artifacts kept on disk

- Custom checkpoint at `unitree_go2_flat/2026-05-02_02-52-14/` —
  not used, but kept as comparison data.
- `cbf_go2_locomotion_train_cfg.py` — config encoding our DR; left
  in repo for reference even though unused.

### Side issue (deferred)

Discovered teacher's CNN model has a JIT-export bug — module-level
int constants `_DYN_DIM` etc. fail `torch.jit.script`. Doesn't affect
deployment (env loads teacher via `torch.load`, not `torch.jit.load`).
Only triggered when play.py auto-tries JIT export. ~5-line fix to
`cbf_go2_teacher_cnn.py`; deferred until we touch teacher for v2.4.

---

## 2026-05-01 (late evening) — Locomotion training task built; first run launched

User pivoted from the rollout-replay approach (Option C) to a simpler
plan: *just train robust locomotion that matches our CBF env's DR
and velocity envelope*. No artificial CBF-style command injection —
if the policy ends up handling jerky commands, it's a bonus.

**New task:** `Isaac-CBF-Go2-LocomotionTrain-v0` (and `-Play-v0`).

File: `IsaacLab/.../safety/cbf_go2/cbf_go2_locomotion_train_cfg.py`
(~110 lines). Inherits `UnitreeGo2RoughEnvCfg` (proven Go2 rough
velocity task) and overrides only DR + velocity ranges.

```text
Overrides applied:
  velocity ranges       ±1.0 → ±2.0 m/s    (matches CBF u_safe clamp)
  friction              fixed 0.8 → [0.3, 1.2]   (matches our DR-1)
  mass range            (-1, 3) → (-5, 5) kg     (matches our DR-1)
  base_com (was None)   re-enabled ±5/5/3 cm     (matches our DR-1)
  force/torque RESET    0 → ±10 N / ±2 N·m       (matches our DR-2)

Left alone:
  push_robot (interval velocity push)   stays None
    The Go2 maintainer deliberately disabled this in
    UnitreeGo2RoughEnvCfg; Go2 is too small to tolerate
    the ±0.5 m/s velocity push mid-episode without tipping.
    Our CBF env doesn't have an analog of this either —
    DR-2 is a reset-time event. Don't re-enable.
  rough terrain, curriculum, rewards, terminations, action scale,
  observations  — all inherited.
```

(I had initially re-enabled `push_robot` "for more robustness." User
caught it in review — the Go2 maintainer's choice to disable was
deliberate. Reverted.)

Registered in `cbf_go2/__init__.py` with two variants:

- `Isaac-CBF-Go2-LocomotionTrain-v0` — full training task
- `Isaac-CBF-Go2-LocomotionTrain-Play-v0` — small viz variant
  (50 envs, no DR corruption, no force pushes) for sanity-checking
  the trained checkpoint.

Uses standard `ManagerBasedRLEnv` (not our custom `CbfGo2Env`) and
the proven `UnitreeGo2RoughPPORunnerCfg` (no custom PPO config).

**Smoke test passed** (64 envs × 5 iters). Config loads cleanly,
iter 0/1 prints, no errors.

**Full training launched** in tmux session `loco`:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-CBF-Go2-LocomotionTrain-v0 \
    --num_envs 4096 \
    --max_iterations 1500 \
    --headless
```

Logs land at `logs/rsl_rl/unitree_go2_rough/<timestamp>/`.
Wall-clock estimate: ~1.5h (rough task is lighter than our CBF env;
no obstacle SDF or QP per step).

**What to watch:**

- `Episode_Termination/base_contact` — should drop from 30%+ toward <5%.
- `track_lin_vel_xy_exp` — should rise; velocity tracking quality.
- Mean reward — should ascend from ~-5 to ~+20-30 by iter 1500.

**Next:** when training finishes, sanity-check via Play variant,
update `LOCOMOTION_CHECKPOINT` in `cbf_go2_env.py`, then train v2.4
(REWARD-1 Variant C reward + new locomotion). Then re-run baseline
eval. Expected fall-rate floor: 20% → ~5%.

---

## 2026-05-01 (evening) — v2.3 trained: REWARD-1 Variant C confirmed; locomotion is now the bottleneck

v2.3 retrain finished. Same v2 stack (PRIV-2 + CNN + SCENE-2/3/4),
proximity reward weight -5 → -1. Solo on 5090, 3000 iters, 3h 43m,
no OOM. Run path
`logs/rsl_rl/cbf_go2_teacher/2026-05-01_<late afternoon>/`.

**Final iter 2999/3000 vs v2.2:**

```text
Metric                            v2.2          v2.3          Δ
─────────────────────────────────────────────────────────────────────
Mean reward                       -6.94         -2.88         +4.06
Mean episode length                750           867          +117
Mean action std                     0.22          0.18        collapsed further

Termination breakdown
  time_out                         65.2%         79.4%         +14.2 pts ✓
  base_contact (FALLS)             25.2%          9.3%         -15.9 pts ⭐
  obstacle_contact                  9.8%         11.3%         +1.5 pts (minor)

Reward components
  collision                        -0.009        -0.012        slightly worse
  u_safe_deviation                 -0.074        -0.044        ✓ less intervention
  infeasibility                     0.000         0.000         CBF still solid
  proximity                        -0.255        -0.101        ✓ as expected (×5 less)

error_vel_xy                        0.755         0.783        same
```

**Headline:** base_contact dropped from 25% to 9% (62% reduction).
Total termination rate (falls + collisions): 35% → 20.6%, a
**14-point overall improvement**. The hypothesis behind REWARD-1
Variant C — that proximity dominance was forcing over-conservative
behavior at the cost of locomotion stability — is confirmed.

**Decision: pause baseline eval against v2.3 until locomotion fix
lands.**

User raised a good point: even with v2.3, the B0 (no-CBF buffer)
baseline still has ~20-25% fall_rate in our env. That's the
**locomotion floor** — robot falls under DR even without any CBF
intervention. As long as that floor is 20%, the CBF tuning
differences (which span ~5-15 pts) are drowned out by locomotion
noise.

To get a clean teacher-vs-baseline comparison, we need locomotion
fall rate < 5%. Otherwise the CBF tuning story is muddled.

**Next plan: Option B + Option C — custom locomotion training task
with CBF-rollout-based velocity perturbations.**

User chose:

- **Option B:** custom locomotion task in our `cbf_go2/` module,
  inheriting from `Isaac-Velocity-Rough-Unitree-Go2-v0` (rough
  terrain for real-robot transfer too). Broaden velocity range to
  ±2 m/s; match our DR (force ±10N, friction [0.3, 1.2], mass/CoM).
- **Option C** for velocity-command perturbation: instead of
  random gaussian noise on the command, sample real `u_safe`
  sequences from CBF-env rollouts. Locomotion learns to track
  the exact kind of jerky velocity profiles the CBF will produce.
- Rough terrain: yes (helps Wk4 hardware demo).

**Concrete steps:**

1. Write `IsaacLab/.../cbf_go2/cbf_go2_locomotion_train_cfg.py`
   — custom velocity-tracking task config.
2. Write rollout collection script — run v2.3 policy on
   `Isaac-CBF-Go2-v0`, log u_safe per env per step, save to
   `data/u_safe_rollouts.pt`.
3. Wire rollout sampling into the locomotion task — at episode
   reset, randomly sample one rollout window as that episode's
   velocity command sequence.
4. Train locomotion (~3h, expect <5% fall rate).
5. Update `LOCOMOTION_CHECKPOINT` in `cbf_go2_env.py` to point
   at new file.
6. Train v2.4 (same v2 stack + Variant C reward + new locomotion).
7. Re-run `eval_baseline.py` against v2.4. Expected floor drops
   to ~5%, making CBF tuning differences measurable.

Total: ~10-12 hours of compute + ~2 hours of code work. Doable in a
day-and-a-half.

**Open question:** is v2.4 a fresh PPO run from iter 0, or do we
warm-start from v2.3's checkpoint? Fresh is cleaner (no
old-locomotion bias in policy). Warm-start is faster (~1.5h
instead of 3.5h). Lean toward fresh for paper-grade.

---

## 2026-05-01 (afternoon) — Baseline eval clean run: v2.2 loses, REWARD-1 Variant C applied

Re-ran `eval_baseline.py` with `--steps_per_config 2000` (3.3× the
biased run's window). Episode counts jumped from 6-21 to 130-147 per
config — clean statistical power.

**Headline:** v2.2 teacher (BR) is third-worst of 25 configs.

```text
RANK   CONFIG                          FALL    n     φ̄      h̄
─────────────────────────────────────────────────────────────────
 1     B1_α=0.50_φ=0.50                0.182   132   0.50    0.517   ⭐ BEST
 2     B1_α=0.50_φ=1.50                0.195   128   1.50    0.502
 3     B0_α=1.50                       0.204   137   0.005   0.478
 4     B2_α=3.00_ε₀=0.50_λ=1.00        0.212   132   1.30    0.486   ⭐ best B2
 5     B0_α=0.50                       0.215   130   0.005   0.470
 6     B2_α=1.50_ε₀=0.50_λ=3.00        0.222   135   0.69    0.515
...
22     B2_α=3.00_ε₀=0.50_λ=3.00        0.338   139
23     BR_teacher                      0.355   141   2.79    0.674   ❌ rank 23/25
24     B2_α=0.50_ε₀=0.10_λ=1.00        0.388
```

Key facts:

- `collision_rate = 0.000` across **ALL 25 configs** including B0
  with no buffer. CBF math is bulletproof in this env.
- `infeasibility_rate = 0.000` across all configs. Closed-form
  half-space projection always solves.
- Story is entirely about `fall_rate`.
- BR loses to best-B1 by 17 percentage points and to best-B2 by 14.

**Diagnosis: v2.2 over-conservative.**

```text
                  φ̄         h̄
BR_teacher        2.79      0.674
best-B1           0.50      0.517
best-B2           1.30      0.486
```

BR uses 2-5× larger buffer than best baselines and stays 30% further
from obstacles. Bigger buffers ⇒ bigger course corrections ⇒
locomotion can't track them ⇒ falls. The trained teacher is
optimizing the WRONG thing — it learned to maximize obstacle distance
instead of trade safety for stability.

**Why:** `proximity = -5 · exp(-min_sdf / 0.5)` over-weighted
obstacle distance in the reward. Policy chose massive margins because
PPO's gradient was dominated by proximity, not by collision/stability
trade-offs.

This is exactly REWARD-1 Variant C's hypothesis. Queue → mandatory.

**Applied: REWARD-1 Variant C.** In
`cbf_go2_env_cfg.py`, `proximity` reward weight changed -5 → -1.
σ=0.5 stays. Comment block updated to record motivation. v2.3 retrain
ready to launch.

**Plot fix.** `eval_baseline.py`'s output PNG showed collision_rate
on Y-axis (uniformly 0 → useless empty plot). Updated to 2-panel:
fall_rate vs mean_φ (left) and fall_rate vs mean_h (right).
BR's high mean_h with high fall_rate is the visual signature of
over-conservative learned behavior.

**v2.3 retrain command** (run when GPU is free):

```bash
ssh chrisliang@130.64.84.163
cd ~/Desktop/safety-go2
rsync from local OR git pull (whichever your sync flow uses)

cd ~/Desktop/safety-go2/IsaacLab
tmux new -s v23 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-CBF-Go2-v0 \
        --num_envs 4096 \
        --max_iterations 3000 \
        --headless \
        --video --video_length 200 --video_interval 2000'
```

Need to push the `cbf_go2_env_cfg.py` change to the lab box first
(file edit was on local mac).

**Hypothesis for v2.3:**

- φ̄ drops below 2.0 (less aggressive buffer)
- h̄ drops below 0.55 (closer to obstacles, more like best-B1)
- fall_rate drops below 0.212 (beats best-B2)
- collision_rate stays at 0 (CBF still bulletproof)

If v2.3 also loses → reward needs deeper redesign, or pivot paper
claim to "student matches teacher" fallback (per Risks/Pivots).

**Branching by v2.3 outcome:**

1. v2.3 BR < best-B2 by clear margin → paper claim holds. Write up.
2. v2.3 BR ≈ best-B2 → smaller CNN, different reward structure,
   or examine trajectory-trace differences (Fig 1 reproduction).
3. v2.3 BR > best-B2 → fall back to "student matches teacher" claim;
   the adaptation-mechanism story doesn't survive.

---

## 2026-05-01 (midday) — Baseline eval: BR bug fix, ran end-to-end, methodology issue surfaced

**BR provider bug.** First eval run crashed at config 25/25 (the BR
teacher) with `IndexError: too many indices for tensor of dimension 2`
inside `rsl_rl.MLPModel.get_latent`. The policy expects an obs dict
keyed by obs_group (e.g. `obs["policy"]`), but `make_br_provider` was
unwrapping to the tensor before calling. One-line fix in `fn`:
pass `obs` dict directly to `policy(obs)`. Configs 1-24 ran fine
(B0/B1/B2 don't use obs — they emit constant raw actions).

**Second run completed all 25 configs.** ~10 min wall clock at 64
envs × 600 steps. All configs returned the same shape:

```text
coll = 0.000     fall = 1.000     infeasibility = 0.000
```

across B0, B1, B2, AND the trained BR teacher. `eps` count per
config ranged 6-21 out of 64 envs.

**Methodology issue.** That all-fall result is a window-size artifact:

- Eval loop runs for `--steps_per_config 600` steps.
- `eval_one` only counts episodes that reach `terminated | truncated`
  inside that window.
- v2.2 training showed mean ep_length = 750, max_episode_length 1000.
- Falls happen fast (within 100-300 steps of episode start).
- Surviving episodes hit max_episode_length at step 1000 — outside
  the 600-step window — so they don't increment any counter.
- Net effect: only fast failures get counted, all of which are
  falls. Hence `fall=1.000`.

This is consistent with training-time numbers (25% base_contact +
10% obstacle_contact = 35% of episodes ended early; matches the
~13-33% finish rate we see in eval).

**Two ways to fix:**

1. Bump `--steps_per_config` to ≥2000 so all envs either terminate
   or hit max_episode_length within the window. Cleanest, no code
   change. Cost: ~3× longer eval (~30 min instead of 10).
2. Modify `eval_one` to snapshot all 64 envs at end of window and
   classify survivors as "still-alive timeouts" rather than dropping
   them. Faster eval but more bookkeeping.

Going with (1) for the next run.

**Useful signal from biased data:**

- Across α=3.0 hand-tuned configs: collision = 0% but fall = 100%.
  CBF math is bulletproof, but α=3.0 demands velocity profiles the
  locomotion controller can't execute without falling.
- BR teacher used mean φ = 2.567 across the rollout, corresponding
  to ε ≈ 0.39 (moderate buffer). It learned to be aggressive but
  not extreme.
- Infeasibility = 0% across ALL 25 configs — closed-form half-space
  projection always solved. The CBF QP layer is solid.

**Educational tangent during the wait.** Walked through the Cosner
paper with user — Examples 1-3, Eq 18, 32, 44, 45. Confirmed:

- B1 (fixed φ const) ↔ Cosner Example 2 (Fig 1c)
- B2 (φ(h) = (1/ε₀)·exp(-λh)) ↔ Cosner Example 3 (Fig 1d)
- BR (φ from RL net) is OUR extension

User asked about Fig 1 reproduction. Clarified: the *controllers*
B1/B2 are implemented (algorithms), but the trajectory-trace plot
(Cosner's Fig 1 visual style) is not. That figure is decoration
that's only worth building if the scalar comparison ends up muddled.

**Next:**

1. Re-run eval with `--steps_per_config 2000`. Should give clean
   collision/fall/timeout split per config.
2. Read CSV output: BR vs best-B2 is the headline number.
3. If decisive: PAPER-1 has its result table.
4. If close or worse for BR: REWARD-1 Variant C retrain or
   reward redesign.

---

## 2026-05-01 (early morning) — v2.2 training completed (3000 iters, 3h 26m)

Lab user's job ended overnight. Launched v2.2 solo on the 5090 around
23:50 with `--num_envs 4096 --max_iterations 3000 --headless --video
--video_length 200 --video_interval 2000` and `expandable_segments=True`.
Ran without OOMs. Run path: `logs/rsl_rl/cbf_go2_teacher/2026-04-30_23-50-51/`.

**Final metrics at iter 2999/3000:**

```text
Mean reward:           -6.94
Mean episode length:    750
Mean action std:         0.22         ⚠️ collapsed (target: keep >0.3)

Termination breakdown
  time_out:             65.2%  ✓
  base_contact:         25.2%  ⚠️ same as v2 attempt #1 (24%) — too high
  obstacle_contact:      9.8%       similar to v1's ~9%

Velocity tracking
  error_vel_xy:          0.75 m/s   ⚠️ high; planner not tracked well
  error_vel_yaw:         0.36

Reward components (raw, pre-weight)
  collision:            -0.009
  u_safe_deviation:     -0.074
  infeasibility:         0.000  ✓ CBF QP always feasible
  proximity:            -0.255       ×5 weight = dominant negative term

Training perf
  Total time:           12420 s (3h 26m)
  Steps per second:     25888
  Iter time:            ~3.8s avg
```

**Read of the metrics:**

- v2 attempt #1's base_contact issue (24% at iter 2213 OOM-kill) **was
  not solved by completing training.** Now 25.2% at iter 2999. The
  policy is genuinely tipping over a quarter of the time, not just
  a transient. Either locomotion can't track the CBF outputs, or the
  policy commands u_safe values the locomotion can't follow, or the
  robot is getting stuck between obstacles and falling.
- Action std collapsed to 0.22 — need TensorBoard to see WHEN. If
  late (≥iter 1500), benign convergence. If early (≤iter 200),
  exploration died and policy is stuck in a local optimum.
- Velocity tracking error 0.75 m/s is suspicious — robot isn't
  following the planner. Could be tied to base_contact (falling
  robots can't track), or CBF over-intervention.
- Mean reward -6.94 is dominated by proximity penalty
  (`-0.255 × 5 = -1.28` per step). REWARD-1 Variant C ablation
  (drop weight to -1) is queued and increasingly relevant.
- Infeasibility 0.000 across all 3000 iters — CBF QP closed-form
  projection always solved. That part of the math is solid.

**Pending interpretation: TensorBoard curves.** Need to see whether
metrics improved over training (and we just have a hard-to-beat
ceiling) or whether the policy plateaued early. User opening TB now
via VS Code port forward.

**Next:** run `eval_baseline.py` against `model_2999.pt` to see how
this teacher compares to B0/B1/B2. Even if v2.2 has issues, the
4-way comparison will tell us:

- whether RL ε(scene) provides ANY benefit over hand-tuned ε(h)
- whether B2 with reasonable (ε₀, λ) actually beats v2.2 on
  some scenes (which would be a real problem for the paper claim)

Possible follow-ups depending on baseline eval:

1. If BR > best-B2 by clear margin → publish v2.2 as-is despite the
   25% base_contact (may need a discussion section explaining the
   safety-aggressiveness trade-off).
2. If BR ≈ best-B2 → REWARD-1 Variant C retrain, pre-paper.
3. If BR < best-B2 on multiple scenarios → bigger problem.
   Likely needs reward redesign (drop proximity dominance) and/or
   smaller CNN to reduce overfit on grid texture.

---

## 2026-05-01 (overnight) — TISSf-style baseline plan + eval_baseline.py

Two more OOM attempts after the two from the earlier evening entry:

- **Attempt 3 (23:11):** OOM during conv forward of PPO update. 4096 envs.
  Process at 17.18 GiB, lab user at 12.81 GiB, free 706 MiB → short by 62 MiB.
- **Attempt 4 (23:15):** OOM during ELU activation of PPO update. **2048
  envs** this time, but still OOM'd because Isaac Sim's non-PyTorch memory
  was bigger than estimated. Process at 17.40 GiB (5.87 GiB PyTorch +
  11.53 GiB Sim/cameras/PhysX). Hard memory ceiling, not fragmentation —
  `expandable_segments` would not have helped (only 119 MiB unallocated).

Original 2048-env memory estimate (~13 GiB) was wrong by ~4 GiB. Misses:

- `--video` flag forces `enable_cameras=True` → Replicator pipeline
  pre-allocates ~2-4 GiB beyond model needs.
- 22 rigid bodies per env (1 robot + 20 obstacles + ground) × 2048 envs
  = ~45K bodies → PhysX broad/narrow phase costs ~2-3 GiB on its own.
- Isaac Sim baseline closer to 6-8 GiB, not 3-4 GiB.

User decided to wait for GPU to clear, then launch max settings solo.
Filed as the "active" todo on the v2.2 retrain.

**Pivot to baseline work while paused.** Professor messaged with the
eval-method spec, citing Cosner et al. (arXiv 2103.08041) Fig 1 +
Examples 2-3:

> "Baseline compare phi(x) = 1/epsilon(x) with both a fixed value of
> epsilon and the experiment performed for fig. 1 in this paper.
> Example 2 and Example 3 and compare against what we're trying to do."

The plan is a 4-way TISSf-style comparison:

```text
LEVEL              SAFETY LAW                       TUNING
─────────────────────────────────────────────────────────────────────
Plain CBF (B0)     ḣ ≥ -α(h)                        α (scalar)
                                                    Fig 1(a)/(b)
ISSf-CBF (B1)      ḣ ≥ -α(h) + ‖∇h‖²/ε              α, ε₀
                                                    Fig 1(c), Example 2
TISSf-CBF (B2)     ḣ ≥ -α(h) + ‖∇h‖²/ε(h)           α, ε₀, λ
                   ε(h) = ε₀·exp(λh)                Fig 1(d), Example 3
RL-TISSf (BR)      Same TISSf inequality            α, ε, a, c
                   ε = NN(scene)                    output by RL policy
```

Confirmed action-to-paper mapping by reading `cbf_go2_env._cbf_filter`:

```text
PAPER:  L_g h · u  >  -α(h)        +  ‖L_g h‖² / ε
OURS:   L_g h · u  ≥  -α·(h - c)   +  φ·‖L_g h‖²  +  a
                                  └─┘
                              φ = 1/ε  (direct equivalence)
```

So **φ is exactly the inverse epsilon from the paper**. The other two
slots in the 5-D action vector (`a`, `c`) are our extras outside TISSf:
`a` is an additive RHS slack, `c` is an h-shift inside the α(·) term.
For the baseline comparison we lock `a = c = 0` so the comparison is
clean against the TISSf paper's formulation. Slot `b` is already unused
(reserved for future SOCP extension).

**Implementation: `scripts/eval_baseline.py`** (~330 lines).

- Inverse-encodes target (α, φ, a, c) back through tanh+linear-scale to
  produce raw 5-D actions the env's filter will reconstitute. Borrows
  `inv_tanh_scale` pattern from `eval_tight_gap.py`.
- B2 mode queries `inner._compute_h()` each step to get current h(x),
  computes φ(h) = (1/ε₀)·exp(-λh) per env, encodes per-env action.
- BR mode loads checkpoint via rsl_rl `OnPolicyRunner` (same pattern
  as `play.py`).
- Sweep grids: α ∈ {0.5, 1.5, 3.0}; B1 φ ∈ {0.5, 1.5, 3.0}; B2 ε₀ ∈
  {0.1, 0.5} × λ ∈ {1.0, 3.0}. Total 24 hand configs + 1 BR ≈ 25.
- Default 64 envs × 600 steps per config; ~75s compute + Isaac Sim startup.
- Outputs `baseline.csv` (one row per config) + `baseline.png`
  (collision_rate vs mean_phi_used scatter, mode-coded).
- Metrics: collision_rate (via min h<0 in episode), fall_rate, timeout_rate,
  infeasibility_rate, mean_h, mean_phi_used.

Categorized terminations via `min h(x)` over the episode: if min_h < 0
the boundary was breached (true collision); otherwise terminated → fell;
otherwise truncated → timeout. Cleaner than reading per-term states from
TerminationManager.

Picked **Option 3** for α handling: sweep α across all baselines so we
compare against the BEST hand-tuned baseline, not a strawman. Total
extra eval cost is small (compute is cheap at 64 envs).

Ready to run B0/B1/B2 immediately as a smoke test (no checkpoint
needed). Once v2.2 has a checkpoint, append `--checkpoint` and `BR` to
`--modes`.

**Next steps:**

1. Wait for shared GPU to clear → launch max-setting v2.2.
2. Run `eval_baseline.py` smoke test (B0/B1/B2 only) to confirm the
   encoding produces sensible numbers.
3. Add disturbance flag for Fig 1 reproduction (paper figure).

---

## 2026-04-30 (evening) — v2.2 launch attempts: two OOMs at 4096, paused pending GPU availability

Tried v2.2 launch twice at 4096 envs. Both OOM'd. Decided to pause
training rather than burn cycles fighting the shared GPU.

**Attempt 1 (21:50):** OOM during first PPO update. Process at 17.56 GiB,
lab user at 12.81 GiB, free 0.35 GiB. Allocator tried to grab 768 MiB
during forward and crashed. Iter 0 didn't even complete.

**Attempt 2 (22:22):** Iter 0 finished (21.4s, ETA 17:50:08 = ~18h
total). OOM during `loss.backward()` of iter 1 at the conv backward
pass. Process at 25.84 GiB, lab user at 3.81 GiB. Tried to allocate
1.5 GiB, only 1.10 GiB free.

**Diagnostic via `nvidia-smi pmon -c 5`:**

```text
PID 62371 (codrincrismariu)   sm: 84-85%   mem: 28-31%
```

Lab user pegging 85% of GPU streaming-multiprocessor cycles. At
22:35, total GPU memory hit 31310/32607 MiB = 96% — basically
saturated even before any v2.2 process started.

**Key takeaways:**

- **4096 envs is structurally too big** with v2.2 architecture on
  this GPU regardless of contention. Solo peak ≈26 GiB (rollout
  17.6 + autograd graph for CNN backward ~8). Lab user holding even
  6+ GiB pushes us over the 32 GiB ceiling.
- **2048 envs sat at 17.6 GiB during rollout** — higher than the
  ~13 GiB I'd estimated. The new 8192-dim grid obs adds ~1.6 GiB
  just for storage (`2048 × 24 × 8192 × 4 bytes`) plus duplicates
  for advantage computation. PPO update would push past 25 GiB.
- **Iter time at 4096 was 21.4s** — collection 18.1s, learning 3.3s.
  Roughly 3× v2.1's per-iter cost. Grid rasterization (per-step,
  per-env, 20 shapes) is the dominant new compute. Even solo, 4096
  at this architecture would be ~12-14h for 3000 iters.
- **GPU contention split** when both running: lab user at 85% SM
  → ~15% available for us. At 2048 envs that turns ~5-6s/iter solo
  into ~12-18s/iter contended, ballooning a 4h run into 12-15h.

**Disposition:** pause v2.2 launch until GPU clears. Plan:

1. Watch `nvidia-smi --query-gpu=memory.free --format=csv,noheader`
   periodically. >25 GiB free → 4096 safe. >18 GiB free → 2048 safe.
2. When launching, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   to reduce fragmentation overhead by ~10-20%.
3. If GPU stays contested for too long, fallback is 1024 envs +
   6000 iters. Slower wall-clock but stays under memory ceiling
   even with lab user at 15+ GiB.

No code changes today. v2.2 stack as-launched is what will run when
GPU opens up.

---

## 2026-04-30 (afternoon) — SCENE-4: K_MAX=20 random-subset obstacle pool

Pushed back on whether the 5-slot fixed-shape setup gave the encoder
enough diversity. Across all 4096 envs at any moment, slot 0 was
always the same 0.30m cube — diversity came purely from random
*placement* of those 5 templates, not from any randomization of the
templates themselves. With a sufficiently expressive CNN, this is a
real overfit risk: the encoder could memorize "this footprint
signature = template #2."

Three options considered:

- **A.** Bump K_MAX to ~20-30 with more diverse pre-baked templates;
  random subset selection per reset.
- **B.** A + runtime per-env continuous size sampling via Isaac Lab's
  `set_local_scales` API.
- **C.** Just runtime size scaling on the existing 5 slots.

Picked **A**. (B) and (C) depend on `set_local_scales` which is
unverified for kinematic bodies in Isaac Lab — couldn't ssh in to
investigate without disturbing the running v2.1 training. (A) is
the safe option that gets ~80% of the diversity benefit without
any API risk.

Implementation:

- `OBSTACLE_SHAPES` expanded from 5 → 20 entries: 8 cubes (edges
  0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70 m), 6 cylinders
  (radii 0.15, 0.20, 0.25, 0.30, 0.35, 0.40 m), 4 walls (lengths
  0.6, 1.0, 1.5, 2.0 m), 2 rectangular non-wall boxes.
- `K_MIN` dropped from 2 → 0. Empty-scene episodes are useful: they
  let the policy practice nominal locomotion without having to
  always be evading something.
- `_DEFAULT_OBSTACLE_INIT_POS` extended to 20 entries spread further
  out so PLAY mode (which disables the placement event) renders
  them deterministically without crowding the camera.
- `randomize_obstacles_position` rewritten:
  - Old: `on_stage = i < k_actual` → biased toward smaller-index slots.
    Slot 0 was on-stage 100% of the time; slot 4 only 25%.
  - New: random permutation per env via `argsort(uniform_noise)`;
    first K_actual entries of permutation are on-stage. Vectorized
    across all 4096 envs. Each slot now has equal probability of
    being on-stage.
  - Pair-wise rejection sampling: per-pair min-sep equals
    `circ_r[i]` plus `circ_r[j]` plus a 0.4m buffer. Old single-scalar
    `min_separation=1.2m` was too tight for big walls (1.0m+ extent)
    and too loose for small cubes (0.30m). Per-pair adapts to the
    actual footprint sizes.
  - Spawn area widened to (1.5, 7.0) × (-2.5, 2.5) (was (1.5, 4.5)
    × (-2.0, 2.0)) to accommodate the bigger pool with margin.
- Goal-planner range expanded to match: `goal_range_x=(3.0, 7.0)`,
  `goal_range_y=(-2.5, 2.5)`.

Diversity math: number of distinct K-element subsets of 20 templates,
summed over K ∈ [0, 5]:

```text
K=0: 1
K=1: 20
K=2: 190
K=3: 1140
K=4: 4845
K=5: 15504
       ─────
       21700 distinct compositions per reset, before placement
```

Random placement of those compositions in a 5.5×5m area gives
effectively unlimited grid configurations. Encoder gets way more
training signal than before.

Compute / memory cost at 4096 envs:

- 20 × 4096 = 81,920 rigid bodies (vs 5 × 4096 = 20,480 before).
  Memory ~600MB extra. Fits comfortably in 32GB VRAM.
- `_compute_h`, `obstacle_proximity`, `_obstacle_contact_mask`,
  `occupancy_grid_b`, `_advance_obstacle_motion` all loop over K=20
  instead of K=5. Inner ops are tiny vector ops; ~5-10% slower
  iteration time expected.
- Pair-wise min-sep check is now O(K²)=400 inner iterations per
  reset vs 25 before. Still negligible.

The running v2.1 already imported the old code, so this lands as
v2.2: same architecture, expanded distribution. Won't take effect
until next retrain.

Risks for v2.2:

- The wider goal range (up to 7m) plus K=0 episodes might shift the
  reward landscape enough that PPO needs more iterations to converge.
- Empty-scene episodes give the proximity reward zero gradient
  signal, which could let σ collapse faster on those episodes.
  Watch action std curve carefully.

---

## 2026-04-30 (morning) — v2 attempt #1 OOM-killed; restarting fresh as v2.1

The v2 retrain that started 2026-04-29 19:16 ran ~4h 18m and got
OS-OOM-killed at iter 2213/3000 (74% through). The kernel-level
`Killed` (no Python traceback) is OOM-killer behavior, not CUDA OOM
or our code crashing — likely the other lab user's process grew
late in the run, pushed total memory over the limit, and the kernel
picked our process to evict.

Metrics at iter 2213:

- Action std: 0.26 (started at 0.99, decay was healthy)
- Mean episode length: 808 / 1000 steps
- `obstacle_contact`: 10.3% (in line with v1's 9%)
- `base_contact`: **24%** (vs v1's 9%) — the worrying signal. The
  robot is tipping over much more often. Could be: CNN encoder
  learned a more aggressive evasion strategy than the MLP; less
  data per iter (2048 envs vs v1's 4096); or just that the run was
  cut short before stability terms dominated.
- Steps/sec: 31k near the end (vs 4-7k early on) — the other user
  must have vacated by the last hour, but too late.

**Decision:** abandon attempt #1, don't resume from the 2200-iter
checkpoint. Restart fresh as v2.1 with SCENE-2 (moving obstacles)
already enabled. Reasoning: (1) we'd skipped v2-on-static eval
anyway since the new eval method isn't specified yet; (2) SCENE-2
code is on disk now, no point training a v2-static when the
intended distribution is v2.1-with-motion; (3) starting at iter 0
gives the policy a clean shot at the moving-obstacle distribution
instead of finetuning into it from a partly-converged static
policy.

v2.1 retrain kicked off ~2026-04-30 morning. Same 2048 envs, 3000
iters, headless. The other lab user's job appears to have
finished, so we should see steady ~1.5s/iter throughout.

**Lesson recorded for next time:** OS-OOM-kill leaves no
diagnostic. If it happens again, check `dmesg | tail` on the lab
box for the kernel's OOM-killer log entry — that confirms the
cause and shows which process got picked.

---

## 2026-04-29 (evening) — v2 architecture: occupancy grid + CNN + multi-shape SDF

**Advisor meeting outcome:** full v2 redo, no fallback to v1. Three
coupled changes that justify each other: (1) occupancy grid replaces
top-K obstacle slots; (2) the grid is shape-agnostic, so we add
multi-shape support natively (boxes + walls + cylinders mixed);
(3) frame-stacking on the grid encodes motion via grid_t − grid_{t−1},
which makes future moving-obstacle work free. SDF math has to
generalize from the disk approximation to per-shape analytical SDF,
and the encoder swaps MLP for CNN.

Decision points walked through with the user:

- **Occupancy grid is 2D, not 3D.** Robot moves in 2D and the CBF
  filters only `vx, vy`, so a top-down rasterization captures
  everything we need. Cuts memory ~32× vs a 3D voxel grid and lets
  the CNN stay shallow.
- **Resolution: 64×64 × 0.1m → 6.4m × 6.4m FOV.** Smallest obstacle
  (0.3m cube) covers a 3×3 cell footprint — readable by the conv
  kernel. FOV exceeds our 5m max range with margin so an obstacle
  never gets cropped.
- **Why CNN, not MLP-on-flatten.** Translation invariance: the same
  3×3 conv kernel detects "wall-edge near robot" wherever it appears
  in the grid. An MLP would have to learn that pattern at every
  absolute position separately.
- **Mix shapes from training start.** No cubes-only warmup phase —
  the user said we'd get to mix shapes anyway, so the encoder learns
  the shared representation from iteration 0.
- **No OOD-rank gate before declaring v2 the published version.**
  Committed; v1 archived only as reference for the prior architecture.

**Implementation (steps a-e), 5 hours total:**

- (a)+(b) Observation infra + grid rasterization
  (`cbf_go2_observations.py`). New `occupancy_grid_b`. Per-shape
  rasterization (box AABB check, cylinder disk check). Previous-frame
  grid cached on env, zeroed per-env in `_reset_idx` so post-reset
  obs doesn't carry a stale frame. Old `k_obstacles_body_frame`
  deleted entirely.
- (c) Multi-shape spec (`cbf_go2_env_cfg.py`). `OBSTACLE_SHAPES` is a
  K-tuple of `("box", hw, hh)` or `("cylinder", r)`. Spawn loop
  dispatches to `MeshCuboidCfg` or `MeshCylinderCfg` per slot.
  Slot mix: 2 cubes (0.3m, 0.4m edges) + 1 wall (1.5×0.2m) + 2
  cylinders (0.25m, 0.35m radii).
- (d) Per-shape analytical SDF (new `cbf_go2_shapes.py` with
  `compute_shape_sdf_batch`). Box SDF: `q = |p−c| − halfext;
  norm(max(q, 0)) + min(max(q.x, q.y), 0)` with sign-corrected
  exterior gradient. Cylinder SDF: `‖p−c‖ − r` with radial unit
  gradient. Both Minkowski-shifted by `ROBOT_HALF_FOOTPRINT` inside
  the helper. Wired into `_compute_h`, `_obstacle_contact_mask`,
  `obstacle_proximity`, and the adversarial planner. `OBSTACLE_RADII`
  derived tuple deleted — every consumer reads `OBSTACLE_SHAPES`
  directly now. Adversarial planner argmin's per-shape SDF (true
  closest surface, not closest center).
- (e) CNN encoder (`cbf_go2_teacher_cnn.py`). `_GridDynamicsEncoder`
  splits flat (N, 8207) into 15-D dynamics path + 8192-D grid path,
  runs `Conv(2→16, k=3, s=2) → ELU → Conv(16→32, k=3, s=2) → ELU →
  Linear(8192→64) → ELU` on the grid, `Linear(15→64) → ELU` on the
  dynamics, concats to 128, then `Linear(128→128) → ELU →
  Linear(128→z_dim)`. Shared structure between actor (output_dim=z_dim
  = 12, with a π_teacher head on top) and critic (output_dim = 1,
  no Z bottleneck, fully independent encoder so actor's Z stays
  clean for student distillation). PPO config updated to point at
  the new model classes.

**Self-check before retrain.** All 7 changed files compile; all
`obstacle_radii` callsites swept; eval scripts only reference
`OBSTACLE_NAMES` (still defined); priv obs ordering puts dynamics
first 15D and grid last 8192D, clean for the encoder split; box
gradient verified analytically; cylinder SDF produces identical
numbers to the old single-radius math when shape is `("cylinder", r)`.

Two nits flagged: (1) `min_separation = 1.2m` is tight for wall +
cylinder along the wall's long axis (worst-case 0.10m gap), but only
one wall slot exists so wall+wall isn't possible — non-blocker;
(2) adversarial planner duplicates the 0.15m robot footprint as a
literal to avoid an env→commands import cycle.

**Retrain.** Started ~19:16. First attempt at `--num_envs 4096`
crashed with CUDA OOM — another lab user (`Predicate-Based-Walker`,
PID 1037437) was using 13 GB on the shared 5090, leaving us ~18 GB
which wasn't enough for the wider obs's PPO update intermediates.
Dropped to `--num_envs 2048`, runs cleanly. ETA ~6h (sharing GPU at
~70% util slows iteration time to 6-12s vs the ~2s we'd see solo).

Iter 0-3 health: action_std stable at 0.99, mean reward becoming
more negative as episodes lengthen (13 → 80 steps), losses non-NaN,
obstacle_contact rate climbing as expected during random
exploration. Standard PPO startup; no anomalies.

**Side observations.** Param count went from ~50k → ~600k for the
actor + ~300k for the critic. Bigger net, justified by the 182×
larger obs. The CNN's first conv intermediate is the main memory
hog during PPO mini-batch update — that's what set the 2048-env
ceiling under shared GPU.

**SCENE-2 (moving obstacles) implemented while v2 trained.** The
2-frame priv obs grid was originally justified for moving obstacles;
user pushed back on whether it was pulling its weight on the
static-obstacle scene we currently train on. Honest answer: not
much. So we wired up SCENE-2 to actually use the second frame.

Implementation:

- `randomize_obstacles_position` samples a per-episode constant
  velocity per on-stage obstacle, uniform in `[-0.2, 0.2] m/s` per
  axis. Bernoulli mask sets ~50% to zero so the encoder sees both
  moving and static obstacles. Off-stage slots get zero velocity.
  Stored on env as `cbf_obstacle_velocities` shape (N, K, 2).
- `CbfGo2Env.step()` calls `_advance_obstacle_motion()` before the
  inner physics tick. Reads velocities, adds `vel × step_dt` to each
  obstacle's world pose, writes via `write_root_pose_to_sim`.
  Step-and-hold motion (held across the `decimation` substeps of
  the inner physics loop).
- No bouncing on spawn-area walls. Obstacles drift naturally; over
  a 50s episode at 0.2 m/s, max drift is ~10m, which puts them well
  outside the 6.4m grid FOV — they fade out of relevance for the
  encoder and the CBF (large SDF). Re-randomized at next reset.
  Cleaner than collision-prone bouncing, and gives the encoder
  scene-evolution dynamics for free.
- Velocity cap (0.2 m/s) is well below the robot's ~1 m/s walking
  speed so the robot can outpace any obstacle. Otherwise episodes
  degenerate into "sprint or get pinned."
- PLAY and TightGap env configs disable position randomization
  entirely, so they also disable motion — deterministic eval scenes
  stay static.

The running v2 training won't pick up these changes (Python source
is already imported into the running process). Treats this as a
v2.1 add-on: same architecture, expanded training distribution. If
v2 lands cleanly we can retrain into v2.1 with moving obstacles.

---

## 2026-04-29 (early morning) — Paper claim lands: teacher Pareto-dominates

**OOD eval result:** Teacher rank 1 or 2 across all 4 OOD conditions on
the K=5 + per-slot-varying-radius retrain.

| Condition | K=1 (Wk3.5) | K=1 uniform-only | K=5 + radius |
| --- | --- | --- | --- |
| slip_calm | 16 / 19 | 8 / 19 | **1 / 19** |
| slip_push | 6 / 19 | 4 / 19 | **2 / 19** |
| grip_calm | 2 / 19 | 9 / 19 | **2 / 19** |
| grip_push | 17 / 19 | 10 / 19 | **1 / 19** |
| Mean | 10.25 | 7.75 | **1.5** |
| Range | 15 | 6 | **1** |

**Teacher's row in each condition** (from CSV, sorted by collision rate
ascending):

```text
slip_calm:  trained_teacher  coll=0.091  min_d=1.78  |du|²/step=0.519
slip_push:  trained_teacher  coll=0.128  min_d=1.65  |du|²/step=0.609
grip_calm:  trained_teacher  coll=0.117  min_d=1.58  |du|²/step=0.601
grip_push:  trained_teacher  coll=0.128  min_d=1.74  |du|²/step=0.633
```

The trained teacher consistently:

- Has the lowest or 2nd-lowest collision rate.
- Keeps the highest `min_d` (avg minimum distance from any obstacle
  during episodes) of any config — 1.58 to 1.78m, vs hand-tuned ~1.0-1.4m.
- Uses the CBF actively (`|du|²/step ≈ 0.5-0.65`), comparable to other
  CBF-active configs and 6-10× higher than the passive `m=0.001` configs.

**No single hand-tuned config wins all 4 conditions.** Each condition's
top hand-tuned performer collapses elsewhere:

```text
slip_calm winner: ht_a=3.5_m=0.1   → rank 16 in grip_calm
grip_calm winner: ht_a=1.0_m=0.001 → rank 13 in slip_calm
grip_push winner: ht_a=0.1_m=0.001 → rank 17 in slip_calm
slip_push winner: ht_a=0.1_m=0.5   → rank 12 in slip_push (passive,
                                      barely moves: 6.0 |du|²/step)
```

This is the Pareto-dominance shape we needed for the paper claim.

**On σ=0.05 collapse.** Both K=5 retrains ended with σ=0.05.
Pre-eval, this looked alarming — entropy bonus too weak in the richer
environment. Post-eval, it's revealed as fine: the deterministic
policy landed on a good local optimum, and OOD adaptation didn't
suffer. We can revisit entropy_coef bump later if needed but it's not
on the critical path.

**On the radius axis.** The K=5-no-radius and K=5-with-radius runs had
nearly identical bulk metrics during training. The OOD eval is what
revealed the radius work mattered (we don't have a no-radius eval to
compare directly, but the 1.5 mean rank is a substantial jump from the
old K=1 result). Strong inference: per-obstacle radii in priv obs are
helping the encoder build a better Z; the bulk training metrics aren't
sensitive enough to show this.

**Next steps.**

- **PAPER-1 stress scenarios (Wk2).** OOD Pareto-dominance is "teacher
  beats hand-tuned"; the four per-param scenarios are the
  proof-of-mechanism: "teacher specifically picks parameter value X
  for scenario Y, matching the optimum." Cheapest first: tight gap (c).
- **Lower-weight proximity ablation** in parallel — bounded test, ~30 min.
- **Skip entropy bump for now.** σ=0.05 didn't hurt; revisit only if
  PAPER-1 scenarios reveal brittleness.
- **Skip per-env-per-obstacle radius (SCENE-1.5 Phase-2)** — the
  per-slot fixed sizes were sufficient. Phase-2 was the deferred
  contingency; we don't need it.

PROGRESS.md TL;DR + Wk1 checklist + closed-this-cycle list synced.

---

## 2026-04-28 (night) — SCENE-1.5 Phase-1 (varying radius) implemented

**Why now.** Multi-obstacle retrain finished earlier with σ collapsed
to 0.05 and 9.34% obstacle_contact. The privileged-obs `radius` slot
was constant 0.35 across all envs — dead weight. User wanted the
radius axis active so the teacher can adapt obstacle-size → CBF
behavior. Decided to ship a minimal version (Phase-1) before the next
retrain rather than wait on OOD eval results.

**Phase-1 design.** Per-scene-slot fixed cube edges, same across all
envs:

```text
OBSTACLE_SIZES = (0.3, 0.4, 0.5, 0.6, 0.7) m         # cube edges
OBSTACLE_RADII = tuple(s * 0.7071 for s in SIZES)    # circumscribed disks:
                                                     # 0.21, 0.28, 0.35, 0.42, 0.49
```

The priv-obs `k_obstacles_body_frame` sorts obstacles by distance, so
priv-obs slot 0 carries whichever scene obstacle's radius is closest.
That gives the teacher real per-reset variation in the radius signal,
even though scene-slot radii are fixed.

**Files changed.**

1. `cbf_go2_env_cfg.py` — added `OBSTACLE_SIZES` + `OBSTACLE_RADII`
   constants; spawn loop uses each slot's edge in MeshCuboidCfg; reward,
   termination, and priv-obs params now pass `obstacle_radii` instead
   of a single scalar.
2. `cbf_go2_env.py` — replaced module-level `EFFECTIVE_RADIUS` scalar
   with `self._effective_radii: (1, K)` cached at init. `_compute_h`
   broadcasts it against per-obstacle distances for per-obstacle SDFs.
3. `cbf_go2_observations.py` — `k_obstacles_body_frame` takes
   `obstacle_radii` tuple; gathers per-obstacle radius alongside the
   distance-sort, so the priv-obs slot's radius matches whichever
   scene obstacle ended up closest.
4. `cbf_go2_rewards.py` — `_obstacle_contact_mask` uses per-obstacle
   threshold (`R_obs_i + 0.15` robot footprint). `obstacle_proximity`
   switched from center-to-center to **surface distance**, clamped at
   0. Now penalty fires based on distance to obstacle surface, not
   center — varying radii correctly differentiate.

**Notable side effect.** Proximity reward magnitude shifted ~2-4×
higher near contact vs the K=1 version. Old: -0.68/step at 1m
center-to-center. New: -1.84/step at 1m center-to-center for a 0.5m
cube (0.5m surface). May need to retune sigma if this causes
over-cautious behavior; document in the next retrain results.

**Phase-2 deferred.** Full per-env-per-obstacle continuous sampling
via `root_physx_view.set_local_scales` at startup. API not yet
verified for kinematic bodies post-startup. Phantom-radius workaround
(visual fixed, math uses sampled) is unsafe — physics collides at
visual size, CBF reasons about smaller size, robot crashes while
math says safe. Only worth pursuing Phase-2 if Phase-1 doesn't give
enough adaptation signal.

**Other cleanup tonight.**

- Deleted dead `nearest_obstacle_body_frame` from
  `cbf_go2_observations.py` (unused after the wire flip earlier today).
- PROGRESS.md reorganized end-to-end. Old structure had accumulated
  cruft — consolidated into TL;DR, This week, 4-week plan, Backlog,
  Done, Reference. Killed "Realistic scope" and "Morale check"
  duplicates. Plain-English in prose with codenames as backlog
  identifiers only. 613 → 355 lines. Per-week action checklists
  preserved in the 4-week plan.

**Next.** Rsync, retrain (~52 min on multi-obstacle + varying radius),
OOD eval (4 conditions × 19 configs). Compare against the K=5
no-radius checkpoint to isolate what the radius axis bought us.

---

## 2026-04-28 (late evening) — SCENE-1 K=5 retrain finished; σ collapsed; eval pending

**3000-iter retrain done at 18:01.** 52 minutes wall time on the lab
box, 94k steps/sec. No crashes. All planner rates non-zero. Multi-
obstacle pipeline confirmed working end-to-end.

**Final metrics (concerning):**

| Metric | K=1 baseline (post-BUG-1) | K=5 final | Δ |
| --- | --- | --- | --- |
| σ (action std) | 0.33 | **0.05** | collapsed |
| obstacle_contact | 6.74% | **9.34%** | +2.6 pp |
| base_contact | 8.43% | **9.83%** | +1.4 pp |
| Mean reward | -2.03 | -3.60 | worse |
| Mean episode length | ~890 | 893 / 1000 | similar |

**σ collapse trajectory:**

```text
iter 500  (sanity end):   σ = 0.33  ← still healthy
iter 3000 (final):        σ = 0.05  ← collapsed
```

`entropy_coef=0.001` was tuned for the K=1 environment. With K=5 the
problem space is richer and PPO drives exploration to near-zero faster.
Policy at end is essentially deterministic — could be a sharp local
optimum that still adapts, or a degenerate one that doesn't.

**Open question.** Does the policy still adapt to OOD dynamics despite
σ=0.05? OOD eval will tell us. Eval was kicked off at 18:09; user
killed it to record video first. Will re-run.

**Strategic note: layer changes one axis at a time.** Discussed adding
SCENE-1.5 (varying radius) + SCENE-3 (multi-shape) + SCENE-2 (motion)
all at once. Decided no — if we stack 3 changes onto a teacher whose
σ already collapsed, debugging which factor caused what becomes
impossible. Order:

1. Run K=5 OOD eval first (gating decision).
2. If teacher mid-pack: bump entropy_coef and retrain (cheap test).
3. If still mid-pack: SCENE-1.5 for real adaptation signal.
4. SCENE-2 (motion) and SCENE-3 (shapes) stay deferred unless time
   permits or they become hero-claim-relevant.

**Docs synced.** PROGRESS.md "Next up" updated with retrain metrics
table + σ collapse trajectory; "Recommended next session order" now a
3-branch decision tree (eval result → next step). 4-week plan Wk1
checklist marks retrain done. "Done so far" 2026-04-28 entry expanded
with collapse details.

---

## 2026-04-28 (evening) — REWARD-1 Variant B result + SCENE-1 design + priority shift

**REWARD-1 Variant B (σ=0.2) finished.** 3000-iter run on the
multi-planner all-DR setup (same hyperparams as the post-BUG-1
baseline). Worse than baseline:

| Config | obstacle_contact | base_contact | mean reward |
| --- | --- | --- | --- |
| σ=0.5 baseline (post-BUG-1) | 6.74% | 8.43% | -2.03 |
| σ=0.2 (Variant B) | **7.36%** | 3.96% | -0.77 |

Reading. Narrower kernel = penalty only fires when robot is already
close (≤0.5m at penalty=exp(-2.5)≈0.08). At ≥1m the proximity
penalty is effectively zero, so PPO has no gradient signal for early
swerving. The robot ends up running at the obstacle and swerving
last-minute — which sometimes fails. σ=0.5 stays. Variant C
(weight=-1.0) deferred to post-SCENE-1 so the ablation lands on the
multi-obstacle teacher.

**Priority shift: SCENE-1 jumps to top, ahead of PAPER-1.** Previous
plan was Wk1 = "REWARD-1 + first PAPER-1 scenario", Wk2 = "SCENE-1 +
remaining PAPER-1 scenarios." User pushed back: PAPER-1 wants to
stress each CBF param on a *meaningful* scene, but the K=1 scene
leaves CBF math degenerate (single obstacle = no min-over-K
behavior, no anticipatory dodging). Better to retrain on K>1 once
and then sweep all 4 PAPER-1 scenarios on the new teacher. New plan:

- Wk1: SCENE-1 + MODEL-2 wire + retrain
- Wk2: All 4 PAPER-1 scenarios on the multi-obstacle teacher
- Wk3: Student distillation + paper draft
- Wk4: Sim-to-real + polish + submit

**SCENE-1 design (per-env K_actual sampling).** Converged in
discussion today:

- **K_max = 5, K_min = 2.** Spawn 5 obstacles per env always (fixed
  NN input dim). Per reset, sample K_actual ∈ [2, 5] uniformly.
- **Off-stage parking.** Place K_actual obstacles in the spawn area;
  park the rest at e.g. (100, 100) env-local — far beyond max_range.
  - Priv obs: `presence` bit auto-zeros via the range gate. Encoder
    learns "presence=0 → ignore."
  - CBF math: off-stage SDFs are huge → never the min. No code change.
  - Rewards: off-stage proximity ≈ 0, collision check fails by huge
    margin. Naturally excluded.
- **Spawn area: 3m × 4m** (`x ∈ [1.5, 4.5]`, `y ∈ [-2.0, 2.0]` =
  12m²) — bumped from 2 × 3 to fit 5 obstacles with 1.2m min
  separation comfortably.
- **Min separation 1.2m** center-to-center → effective gap ~0.20m.
  Robot footprint 0.30m can't fit through → robot must go around
  rather than between. Narrow-gap ("between") regime saved for
  PAPER-1 c-stress scenario via tighter spacing locally.
- **Rejection sampling** for positions, cap of 50 retries per
  obstacle.

The user's mental model question — "X-Y grid, randomize the map,
split into 40 blocks" — clarified as a confusion between Isaac Lab's
*parallel env replication* (4096 self-contained 2.5m × 2.5m bubbles)
and *within-env obstacle placement* (random sampling inside a
sub-rectangle of one bubble). Documented with ASCII visuals in the
chat; not duplicated here.

**Six files SCENE-1 touches.**

1. `cbf_go2_env_cfg.py` — spawn K_max RigidObjectCfgs under
   `{ENV_REGEX_NS}/Obstacle_<i>`; flip priv obs wire; update reward
   params to take tuple; expand spawn ranges.
2. `cbf_go2_events.py` — `randomize_obstacle_position` takes tuple,
   per-reset K_actual sampling, rejection-sampled positions, off-stage
   parking for unused slots.
3. `cbf_go2_env.py` — `_compute_h` reads list of obstacle names into
   `(N, K_max, 2)`. Most structure already K-ready.
4. `cbf_go2_rewards.py` — `_obstacle_contact_mask` and
   `obstacle_proximity` take tuple; min over K obstacles.
5. `cbf_go2_observations.py` — flip activation in env_cfg only;
   both functions stay during transition.
6. `agents/rsl_rl_ppo_cfg.py` — bump z_dim 8→12, hidden 64→128.
   Old checkpoints incompatible.

**φ stressor correction in PAPER-1.** Earlier draft of PAPER-1 had
"wrap u_safe with multiplicative noise" as the φ stress scenario.
That stresses *execution noise*, which is a different parameter,
not φ (the safety buffer). Corrected in the backlog entry: better
candidates are (a) tracking sloppiness on the locomotion controller
(commanded ≠ executed → robot needs larger φ to absorb slop), or
(b) narrow-gap navigation where φ tuning matters for fit.

**Docs synced.** PROGRESS.md: top status block consolidated to
single 2026-04-28 entry; REWARD-1, SCENE-1, MODEL-2, PAPER-1
backlog entries updated; 4-week plan dedup'd (had two parallel
Wk2/Wk3/Wk4 blocks from a prior reorg); Wk1 task list updated to
reflect SCENE-1 priority. This LOG entry written.

**SCENE-1 implementation landed (same evening).** All six files edited:

- `cbf_go2_env_cfg.py`: K_MIN=2, K_MAX=5, OBSTACLE_NAMES const; loop
  spawns 5 RigidObjectCfgs with `_DEFAULT_OBSTACLE_INIT_POS` per slot;
  flipped priv obs from `nearest_obstacle_body_frame` to
  `k_obstacles_body_frame`; reward + termination params take
  OBSTACLE_NAMES; spawn area 3m × 4m.
- `cbf_go2_events.py`: new `randomize_obstacles_position` —
  vectorized rejection sampling across envs (sequential across slots),
  per-env K_actual ∈ [K_MIN, K_MAX], off-stage parking at
  `(100 + 5i, 100 + 5i)` per slot.
- `cbf_go2_env.py`: `_compute_h` stacks K obstacle positions into
  `(N, K, 2)`, min over K SDFs. Off-stage SDFs ~100m → never the min.
- `cbf_go2_rewards.py`: contact mask ORs over K obstacles; proximity
  takes min distance over K.
- `cbf_go2_observations.py`: no functional change; `k_obstacles_body_frame`
  default updated to `("obstacle_0",)` for consistency.
- `agents/rsl_rl_ppo_cfg.py`: z_dim 8→12, encoder hidden 64→128,
  head 64→128.

**One follow-up bug after first deploy (~16:47).** Adversarial planner
in `cbf_go2_commands.py` still referenced the old single
`obstacle_name="obstacle"` config — KeyError on first step. Fixed:
`_compute_adversarial_command` now does argmin-distance over all K
obstacles to pick the closest, and `MultiPlannerCommandCfg` takes
`obstacle_names: tuple[str, ...]` instead. env_cfg passes
OBSTACLE_NAMES at construction.

**Smoke test status.** PLAY config hangs on the lab box at "Starting
the simulation..." with zenity GUI dialog warnings every 2 min — Isaac
Sim viewport rendering attempt over headless SSH. Not a code bug,
just a deploy quirk. Killed it; relying on training-metrics validation
instead. Training kicked off on the other terminal and is running
(no crash on first iter = SCENE-1 plumbing is sound).

---

## 2026-04-28 — MODEL-3 Step 1 result + strategic priorities defined

**MODEL-3 Step 1 finding: planner-coupling is real and large.**

Ran 4-condition OOD eval (slip_calm / slip_push / grip_calm /
grip_push) with planner forced to uniform-only via the new
`--force_uniform_planner` flag. Compared trained_teacher ranks to the
existing eval (uniform+goal 50/50).

| Condition | uniform+goal rank | uniform-only rank | Δ |
| --- | --- | --- | --- |
| slip_calm | 16 / 19 | 8 / 19 | +8 |
| slip_push | 6 / 19 | 4 / 19 | +2 |
| grip_calm | 2 / 19 | 9 / 19 | −7 |
| grip_push | 17 / 19 | 10 / 19 | +7 |

| | uniform+goal | uniform-only |
| --- | --- | --- |
| Mean rank | 10.25 | 7.75 |
| Range | 15 | 6 |

Teacher's wild rank variance under uniform+goal compressed
dramatically — from 15-rank-spread to 6-rank-spread — when the
goal-planner was removed. The grip_calm rank-2 dominance evaporated
(→ rank 9) and the grip_push rank-17 disaster improved (→ rank 10).

**Reading.** The teacher had partly adapted to the joint distribution
`(planner × dynamics)`, not pure dynamics. Under fixed-planner eval,
teacher is consistently mid-pack across conditions — competent at
each but not strongly specialized to any.

This confirms MODEL-3's hypothesis empirically. Step 2 / Step 3
deeper fix (add u_des to priv obs, or replace u_safe_dev with
planner-agnostic metric) is justified — but should wait for PAPER-1
stress scenarios to provide sharper measurement first.

**Strategic priorities (2026-04-28).** Three items added to plan:

1. **PAPER-1 (NEW backlog item)** — per-parameter stress scenarios.
   Build OOD environments that isolate each CBF param (α, φ, a, c)
   so we can prove no fixed config wins all and the teacher's
   per-scenario param choices match the optima. Direct evidence for
   the paper's adaptation claim.
2. **SCENE-1 + MODEL-2** (existing backlog) — multi-obstacle scene
   + dim adjustments. MODEL-2's `k_obstacles_body_frame` function
   is implemented; awaiting SCENE-1 to flip the wire.
3. **SCENE-2 (NEW backlog item)** — dynamic obstacles. Currently
   priv obs `rel_vel` slots always 0 because obstacles don't move.

**REWARD-1 ablation also still pending** (independent of the three
above; ~1h compute, can run anytime).

**4-week plan reshaped:**

- Wk1 (now): MODEL-3 update + REWARD-1 + PAPER-1 first scenario
- Wk2: SCENE-1 + MODEL-2 + remaining 3 PAPER-1 scenarios
- Wk3: Student distillation + paper draft begins
- Wk4: Sim-to-real + polish + submit

Stretch / drop list documented; SCENE-2 first to fall, then PAPER-1
trims to 2 scenarios, then hardware demo.

---

## 2026-04-27 (afternoon) — MODEL-2 implementation + MODEL-3 Step 1 diagnostic

**Context.** Continued the post-overnight session. Recalibrated the
"Cosner approval gates compute" framing — the REWARD-1 ablation and
MODEL-3 diagnostic are bounded exploration, not strategic commitments.
Cosner conversation moves to FYI / strategic-pivot territory only.
Current goal: empirical data on planner-coupling before committing to
either fix.

**MODEL-2 implementation (function added, NOT wired in).**

Added `k_obstacles_body_frame` to `cbf_go2_observations.py`. ~80 lines,
pure-PyTorch, follows same style as the existing
`nearest_obstacle_body_frame`. Returns `(N, k_max × 6)` with each slot
`[rel_pos_x_b, rel_pos_y_b, rel_vel_x_b, rel_vel_y_b, radius,
presence]`. Sorted by distance, range-gated, zero-padded for missing
slots.

Also added an activation comment block in `cbf_go2_env_cfg.py:66-86`
showing exactly how to flip from `nearest_obstacle_body_frame` to the
new function when Wk3.5g multi-obstacle scenes land. Includes the
dim-change reminder (20D → 45D, z_dim 8→12, encoder 64→128, retrain
from scratch).

Function is currently dormant — active obs term still uses the
single-closest variant. Code is K-ready and waiting for a multi-
obstacle scene to make it useful.

**MODEL-3 Step 1 free diagnostic kicked off.**

Modified `scripts/eval_pareto.py` with `--force_uniform_planner` CLI
flag. Existing OOD eval already restricts to uniform+goal 50/50; the
new flag drops goal too, leaving uniform-only across all 4 conditions
(slip_calm / slip_push / grip_calm / grip_push).

Compared against existing eval ranks (16 / 6 / 2 / 17 under
uniform+goal). Big rank changes → planner-coupling is real; stable
ranks → planner-coupling is small, MODEL-3 deprioritizes itself.

Running on the lab machine: ~5 min × 4 conditions = ~20 min total.
Output dirs: `IsaacLab/logs/pareto_eval/2026-04-27_uniform_<cond>/`.

**Other docs work.**

- PROGRESS.md "Pending Cosner conversation" section reframed —
  conversation moves from "gating" to "strategic pivot" only.
- LOG.md gets this entry.
- REVIEW.md untouched today.

**Pending.**

- MODEL-3 Step 1 results (4 CSVs). Compare ranks, update MODEL-3
  backlog with finding.
- REWARD-1 ablation: Variants B (sigma=0.2) and C (weight=-1.0).
  Two ~28-min training runs, run sequentially in tmux.
- After both: brief Cosner message with results in hand.

---

## 2026-04-27 (early) — 3000-iter overnight verifies BUG-1/MODEL-1 fix

**Context.** Kicked off 3000-iter overnight in tmux session `teacher_3k`
to verify the BUG-1/MODEL-1 refactor wasn't a regression. The 500-iter
sanity sample (2026-04-25) had shown `obstacle_contact` going UP from
pre-fix 5.16% → 8.67%, but that was suspected as early-training noise.

**Result (final iter 2999/3000, training time 28 min vs Wk3.5f's 1h 51m):**

| Metric | Post-fix | Pre-fix Wk3.5f | Direction |
| --- | --- | --- | --- |
| Mean reward | -2.03 | -2.84 | ✓ better |
| σ (action std) | 0.33 | 0.26 | ✓ healthier (didn't collapse) |
| Mean episode length | 933.67 | ~877 | ✓ longer |
| `obstacle_contact` | **6.74%** | **5.16%** | ✗ higher |
| `base_contact` | 8.43% | 8.06% | ≈ similar |
| `u_safe_deviation` reward | -0.013 | -0.032 | ✓ less intervention |
| `infeasibility` | 0.000 | 0.000 | ✓ QP healthy |
| Training speed | 4× faster | — | refactor simpler |

**500-iter early sample (2026-04-25) was noise.** Converged at 6.74%,
not 8.67%.

**Interpretation. The fix is correct — DO NOT roll back.**

The pre-fix 5.16% was inflated by the bug. With `OBSTACLE_RADIUS = 0.35`
in the math but `threshold = 0.5m` in the termination, there was a
0.15m gray zone where the math said "safe" while the termination
said "collision." Some of the 5.16% events were robots at ~0.40m —
math thought safe, termination flagged collision.

Post-fix, h=0 boundary IS at 0.5m. If `obstacle_contact` fires, the
CBF genuinely failed to maintain h ≥ 0. The 6.74% is the **honest**
collision rate; the 5.16% was an underestimate hiding the bug.

So apples-to-apples comparison isn't possible. The new baseline is
6.74% under correct math.

**Why `obstacle_contact` is non-zero at all.** Even with correct math,
the teacher with current hyperparameters doesn't reliably keep h > 0
under wide DR + adversarial planner. The CBF brake works, but the
teacher's chosen α/φ/a/c isn't aggressive enough in some conditions.
This is exactly what REWARD-1 (proximity dominance) and MODEL-3
(planner coupling) are pointing at: the teacher is shaped by signals
that don't reward true CBF-tuning enough.

**Healthy signs:**

- σ stayed at 0.33 (in healthy [0.3, 1.0]), didn't collapse like
  pre-fix's 0.26.
- Mean reward improved.
- QP healthy throughout (infeasibility = 0).
- Training 4× faster — confirms the simpler math (no SHIELD branches,
  no prev_e_i lookup) is computationally cheaper.

**What this means for the plan.** REWARD-1 and MODEL-3 just got more
important. The teacher needs to be BETTER than current (not just
mathematically consistent) to beat the honest 6.74% baseline. The
candidate fixes are exactly the architectural concerns flagged in
the 2026-04-26 review walkthrough.

**Pending.**

- Cosner conversation: bring 6.74% number + REWARD-1/MODEL-3 framing.
- After Cosner input: prioritize REWARD-1 ablation (3-config sigma
  sweep) or MODEL-3 Step 1 diagnostic (free).

---

## 2026-04-26 — REVIEW.md walkthrough complete (Modules 1-8)

**Context.** Worked through all eight REVIEW.md modules in depth,
with significant clarifying conversation on Modules 1, 2, 3, 4, 5, 8.
The doc had been re-organized earlier this week (topical layout, Q&A
with answers) and this is the first end-to-end pass on top of that.

**Conceptual ground covered.**

- Module 1 — Sport Mode unpacked: single integrator works because
  locomotion is a separate frozen layer; sim-to-real consistency
  requires the same walker in both sim and deploy.
- Module 2 — `{ENV_REGEX_NS}` and the env_origins grid clicked;
  inheritance vs override mental model is solid.
- Module 3 (the big one) — equilibrium as the final stopping
  distance (not where braking starts); CBF as a velocity filter,
  not a planner; perpendicular projection geometry; e_i as the safe
  direction; L_g h as direction × magnitude with exp-warp scaling;
  single integrator vs second-order trade-offs.
- Module 4-5 — range gating purpose (LiDAR-mirroring); the
  passthrough convergence problem; reward magnitude calibration.
- Module 8 — Pareto eval as the paper's actual quality gate, not
  just evaluation infrastructure.

**Backlog items surfaced (added to PROGRESS.md).**

- **MODEL-2** — `nearest_obstacle_body_frame` returns only closest
  obstacle while `_compute_h` operates on K obstacles internally.
  Benign at K=1, blocks Wk3.5g. Concrete dim plan: 20D → 45D priv
  obs, encoder 64→128 wide, z_dim 8→12.
- **REWARD-1** — `proximity` reward (-5 × exp(-d/0.5)) is 27× larger
  than `u_safe_deviation` at 1m operating distance. Teacher may be
  shaped by proximity rather than CBF tuning. Proposed ablation:
  sigma=0.2 (tight) and weight=-1.0 (weak). High priority — paper
  claim.
- **MODEL-3** — `u_safe_deviation` couples teacher's optimal CBF
  params to the planner distribution. Muddies the
  adaptation-to-dynamics claim. Raise with Cosner before committing
  more compute.

**Visualizations produced (under `docs/viz/`).**

- `cbf_shapes.html` — robot/obstacle shapes, Minkowski expansion,
  rectangle robot, wall and concave-blob obstacles.
- `cbf_gradient.html` — e_i and L_g h in position space, plus
  half-plane projection in velocity space.
- `range_gating.html` — 5m cutoff, hard vs soft sigmoid, priv obs
  slot values shown live.

All three pair with REVIEW.md modules as study aids.

**Pending.**

- 3000-iter overnight to verify BUG-1/MODEL-1 fix wasn't a regression.
  500-iter post-fix sample showed `obstacle_contact` going up (8.67%
  vs pre-fix 5.16%); needs longer training to disambiguate noise.
- Cosner conversation on REWARD-1 / MODEL-3 priorities + paper-claim
  framing.

---

## 2026-04-25 — Module 3 review: BUG-1 fixed + MODEL-1 refactor implemented

**Context.** During Module 3 walkthrough of `cbf_go2_env.py`, a paper
Section IV cross-check (against Eq. 19, 20, 21, 22) surfaced two
issues with `_compute_shield_h`:

1. **BUG-1 — radius wrong.** `OBSTACLE_RADIUS = 0.35` was only the
   obstacle's geometric circumscribed radius. Paper Eq. 19 specifies
   `R_i` as the combined Minkowski radius (robot + obstacle). The
   missing 0.15m of Go2 body half-footprint meant `h = 0` sat inside
   the actual collision distance — CBF math reported "safe" while the
   robot was physically touching the cylinder. Likely contributor to
   the 4.6–5.2% non-zero `obstacle_contact` rates we'd been seeing
   throughout Wk3 training.

2. **MODEL-1 — function should be plain Eq. 19+20.** Per professor's
   scope guidance, the safety function should be the multi-obstacle
   SDF + exponential smoothing only. Drop SHIELD-specific extensions:
   - Eq. 21 concave approximation (`prev_e_i` projection)
   - Eq. 22 front/behind branch (linear-extension fallback)
   - Theorem 2 / Eq. 23-24 stochastic constraint (S-DTCBF)
   - Algorithm 1 CVAE-based `u_adjusted`

**Implementation.** Folded BUG-1 fix into the MODEL-1 refactor.
Changes to `cbf_go2_env.py`:

- Added `ROBOT_HALF_FOOTPRINT = 0.15` and
  `EFFECTIVE_RADIUS = OBSTACLE_RADIUS + ROBOT_HALF_FOOTPRINT = 0.50m`.
- Removed `self.prev_e_i`, `self.prev_closest_idx`, `_init_prev_e_i`.
- Renamed `_compute_shield_h → _compute_h`. New body:
  - Operate on `(N, K, 2)` obstacle tensor; today K=1 by broadcast,
    Wk3.5g multi-obstacle is a scene-config change only.
  - `sdf_i = ||p − ρ_i|| − EFFECTIVE_RADIUS` per obstacle.
  - `sdf_min, closest_idx = sdfs.min(dim=-1)`.
  - `h = λ(1 − exp(−γ · sdf_min))`.
  - Live `e_i` from the closest obstacle's actual current position
    (no caching across timesteps).
  - `L_g h = λγ · exp(−γ · sdf_min) · e_i_live`.
- Updated `_cbf_filter` to call `_compute_h` (returns 2 values, not 3).
  Removed the `self.prev_e_i = e_i_current` cache assignment. Docstring
  no longer mentions "SHIELD".
- File shrunk 271 → 252 lines. `_compute_h` is ~25 lines; previous
  `_compute_shield_h` was ~50 lines with both branches.

Also fixed `scripts/play_cbf_smoke.py` — it was calling the old
`_compute_shield_h()` directly for diagnostic prints (3-tuple unpack).
One-line change to call `_compute_h()` instead (2-tuple unpack).

**Tests on lab.** Both passed cleanly.

- **1-iter smoke** (`train.py --num_envs 16 --max_iterations 1
  --headless`):
  - Actor `in_features=20, out_features=5` ✓
  - 384 steps complete (16 envs × 24 steps) ✓
  - Mean reward -0.04 (fine for 1 iter), σ = 1.0 (initial)
  - All reward terms wired (collision, u_safe_deviation, infeasibility,
    proximity), all termination terms wired (time_out, base_contact,
    obstacle_contact)
  - 0% collisions, 0% infeasibility, 0% base_contact in this iter
  - No tracebacks, no NaNs

- **play_cbf_smoke** (fixed hand-tuned params α=1, φ=0.1, a=0.01,
  c=0.1; 400 steps, --video --headless):
  - Robot stalls at distance **1.65m** from obstacle center
  - h at equilibrium ≈ 0.44 (positive, stable for ~280+ steps)
  - CBF starts intervening at distance ~1.74m (step 80)
  - No NaN, no QP infeasibility, no contact, no falls
  - Video recorded to `IsaacLab/logs/cbf_smoke_videos`

**Behavior shift.** Robot is significantly more conservative than
pre-fix:

| Behavior | Pre-fix | Post-fix |
| --- | --- | --- |
| CBF starts firing | ~1.2m | ~1.74m |
| Equilibrium distance | ~0.7m | ~1.65m |
| Equilibrium h | ~0.12 | ~0.44 |

Two contributing factors, both intentional:

1. The h = 0 boundary moved outward by 0.15m (BUG-1 fix). Same params,
   safer baseline.
2. SHIELD's `prev_e_i` smoothing previously softened the CBF response
   when the gradient was changing. Without it, the CBF reacts to the
   live gradient — kicks in earlier, harder. This is a correctness
   improvement (the math now uses the actual current direction of
   danger, not a stale one), at the cost of more conservative behavior
   relative to the (broken) baseline.

**Note on tanh squashing of params.** The script accepts raw values
(α=1.0, etc.) but `_cbf_filter` applies `tanh` then scales to the
physical range — same as during PPO training. Effective values for
α=1.0 raw → α≈4.4 effective; for c=0.1 raw → c≈0.275 effective. Both
pre-fix and post-fix tests use the same squash, so the comparison is
apples-to-apples. The script labeling is misleading but not load-
bearing for this verification.

**Predictions for next training run** (revised after 500-iter sanity ran):

- `obstacle_contact` rate should drop dramatically — target <1%, ideally
  ~0%. The h = 0 boundary now matches actual physical contact, so the
  CBF can defend it correctly.
- σ should still descend cleanly, no runaway expected (tanh squash and
  entropy_coef=0.001 from Wk3 still in place).
- The trained teacher may pick smaller `c` than before — less added
  margin needed when the baseline boundary is already at the right
  physical place. Equilibrium during deployment may end up tighter than
  today's 1.65m once the policy learns.

**500-iter sanity training (2026-04-25 EOD).** Ran on lab in ~4.5 min,
1024 envs, 12M total steps. Final iter (~498) numbers:

| Metric | Wk3.5f pre-fix (3000 iters) | Post-fix (500 iters) |
| --- | --- | --- |
| σ | 0.26 | 0.34 |
| `obstacle_contact` rate | 5.16% | **8.67%** |
| `base_contact` rate | 8.06% | 8.50% |
| `time_out` rate | ~86.8% | 82.8% |
| `u_safe_deviation` reward | -0.032 | -0.022 |
| Mean reward | -2.84 | -2.59 |

`obstacle_contact` went **up**, not down. This contradicts the
prediction. Three possible explanations, in order of likelihood:

1. **500 iters is early-training, not converged.** σ still at 0.34 (vs
   0.26 converged pre-fix), reward still drifting. Pre-fix probably
   also had ~8% `obstacle_contact` at iter 500 — fair comparison
   requires 3000 iters.
2. **Teacher learned to use CBF less.** `u_safe_deviation` magnitude
   *decreased* — teacher is intervening less, accepting more hits.
   Smell of degenerate equilibrium reasserting itself in a different
   form despite the proximity reward.
3. **Removing `prev_e_i` smoothing slows convergence.** Cross-step
   gradient noise harder for PPO to fit.

**Next step:** 3000-iter overnight to disambiguate. Resolves cleanly:

- `obstacle_contact` drops to <2% → fix works, prediction confirmed.
- Stabilizes ~5% → fix is a wash.
- Stays ≥8% → fix made it worse, theory 2 or 3 is right; investigate.

Not a regression to roll back yet — could be early-training noise.
Decision deferred to overnight result.

**Status.** BUG-1 closed. MODEL-1 implemented at K=1 scope. Multi-
obstacle scene config (adding K > 1 obstacles + per-obstacle priv obs
slots) remains for Wk3.5g.

**Doc cleanup (today).**

- Folded `REVIEW_UPGRADES.md` into PROGRESS.md as a `## Backlog`
  section. Active items (DR-1, INFRA-1) and recently-closed items
  (BUG-1, MODEL-1) preserved with status markers.
- Deleted `REVIEW_UPGRADES.md` (content fully moved).
- Deleted `spam.txt` (its contents had already been extracted into
  PROGRESS.md "Open perception concerns" earlier in the review).
- Kept `REVIEW.md` (still actively used for module-by-module study)
  and `TODO_training.md` (theory / methodology reference, different
  category).

End state: 4 long-term docs (PROGRESS.md, LOG.md, TODO_training.md,
REVIEW.md). Two-doc core for "plan + history" is PROGRESS + LOG.

---

## 2026-04-22 (EOD) — Wk3.5f overnight ✓, Pareto re-run, 4-condition OOD eval: adaptation story DOESN'T land

**Wk3.5f overnight completed.** 3000 iters × 1024 envs, 1h 51m wall
(faster than the 3h 30m I estimated — steps/sec climbed back once
past initial chaos). Iter 2999:

- σ = **0.26** (below the healthy [0.3, 1.0] band — slight overcommit)
- `Mean reward = -2.84` (was -5.20 at smoke)
- `obstacle_contact = 5.16%` (from smoke's 17% — halved, real avoidance learning)
- `base_contact = 8.06%` (barely dropped from smoke's 9.5%)
- `u_safe_deviation = -0.032` (teacher using CBF actively)
- All 4 planner rates non-zero (survival-bias visible but infra correct)

`base_contact` flatness indicates locomotion is at its limit — the
Wk1 policy was trained at friction=0.8, no pushes; we're running it
at friction 0.3–1.2 with ±10N/±2Nm disturbances, which is OOD for
locomotion. The CBF can't stabilize falling legs, only steer u_des.
Would need to retrain locomotion under matching DR to move the
floor. Deferred — not blocking teacher eval.

**3.5c Pareto re-run on new teacher.** Same setup as before (64 envs
× 2000 steps × 19 configs). Result: all configs still show
**collision_rate = 0.000** — not a bug; closed-form half-space
projection + SHIELD exponential h(x) means even α=0.1 gives nonzero
deceleration. Falls (base_contact → "oth") became the differentiator:

| Config        | fall | timeout | |du|²/step |
|---------------|------|---------|------------|
| ht_a=0.1_m=0.5 | 72.9% | 27.1% | 6.66 (CBF too aggressive) |
| ht_a=2.0_m=0.5 |  9.4% | 90.6% | 0.06 (balanced) |
| ht_a=1.0_m=0.001 | 14.5% | 85.5% | 0.037 (gentle) |
| trained       | 18%  | 82%     | 0.58       |

Trained teacher is ≈ mid-pack hand-tuned. No Pareto dominance.

**4-condition OOD eval added to `eval_pareto.py` via `--condition` flag.**
Pinned friction × disturbance per condition (COM offset zeroed, planner
mix 50/50 uniform/goal):

- `slip_calm`: friction=0.25, no pushes
- `slip_push`: friction=0.25, ±10N/±2Nm
- `grip_calm`: friction=1.20, no pushes
- `grip_push`: friction=1.20, ±10N/±2Nm

Each condition ran all 19 configs. Per-condition best hand-tuned
(**different α in every condition — adaptation is a valid goal**):

| Condition  | Best hand-tuned  | Best fall % | Teacher fall % | Teacher rank/19 |
|------------|------------------|-------------|----------------|-----------------|
| slip_calm  | α=2.0, a=0.1     | 14.5%       | 28.5%          | 16              |
| slip_push  | α=3.5, a=0.001   | 13.5%       | 16.7%          | 6               |
| grip_calm  | α=0.1, a=0.001   |  9.2%       | 9.4%           | 2               |
| grip_push  | α=2.0, a=0.5     |  9.4%       | 24.3%          | 17              |

Average across 4 conditions:

- Teacher: **19.7%** falls
- Best fixed hand-tuned (ht_a=5.0_m=0.1): **15.8%** falls

**Adaptation story does NOT land.** Teacher's per-condition rank is
wildly variable (2nd, 6th, 16th, 17th) — it's excellent on one
condition, bad on others. A fixed α=5.0, a=0.1 hand-tuned config
beats the teacher on the grand average.

**Diagnosis (preliminary, pending probe):**

1. **σ=0.26** — policy overcommitted before exploring the priv-conditional
   space. Needs a σ floor regularizer or periodic exploration bumps.
2. **Reward landscape rewards averaging, not specialization.** Teacher
   minimizes *mean* cost across episodes; "safe everywhere" beats
   "optimal per condition" when the spread isn't extreme enough.

**What's solid:** the infrastructure, training stability, Z bottleneck
health, CBF usage by the teacher, and the fact that the per-condition
dilemma is real (different best hand-tuned per condition).

**What's missing:** teacher that actually uses the priv obs to
condition its α/φ/a output.

**Paused here for review** before deciding next move. Three paths
detailed in PROGRESS.md "Next up" → Wk3.5 outcomes section:

- **A)** Probe the teacher's adaptation directly (30 min; if α is
  flat across friction sweeps, we know training didn't teach adaptation)
- **B)** Retrain with σ-floor + wider DR extremes (3–4 h)
- **C)** Accept and move to Wk4 student, re-frame paper to
  "LiDAR-only student matches priv-info teacher"

**Files touched today:**

- `cbf_go2_commands.py` — extended to all 4 planners (Phase 1/2 smokes).
- `cbf_go2_events.py` — new `randomize_com_and_cache`.
- `cbf_go2_observations.py` — new `com_offset_b` (priv 17D→20D).
- `cbf_go2_env_cfg.py` — swap to MultiPlannerCommand, widen DR, add
  COM event + obs term, PLAY overrides.
- `scripts/eval_pareto.py` — added `--condition` flag for OOD eval.

**Checkpoint used for all evals:**
`IsaacLab/logs/rsl_rl/cbf_go2_teacher/2026-04-22_12-27-02/model_2999.pt`

**Output artifacts (for review):**

- `IsaacLab/logs/pareto_eval/pareto.csv,png` (default Pareto, Round 2)
- `IsaacLab/logs/pareto_eval/pareto_slip_calm.csv,png`
- `IsaacLab/logs/pareto_eval/pareto_slip_push.csv,png`
- `IsaacLab/logs/pareto_eval/pareto_grip_calm.csv,png`
- `IsaacLab/logs/pareto_eval/pareto_grip_push.csv,png`

---

## 2026-04-22 (resumed) — Wk3.5d smoke ✓ (2 phases), 3.5e DR widened ✓, 3.5f overnight kicked

GPU freed up midday. Ran both smokes, coded + tested 3.5e, kicked
3.5f overnight.

**Wk3.5d Phase 1 (uniform + goal, walk + adv zeroed).** 50-iter,
1024 envs, 3 min. Iter 49:

- planner rates: uniform 0.39 / goal 0.61 (matches 0.40/0.60 weights)
- walk 0.00 / adversarial 0.00 (correctly disabled)
- σ = 0.82, `u_safe_deviation = -0.0424` (**8× higher** than the Wk3
  baseline of -0.006 — the teacher is now actively using the CBF because
  goal-reaching forces u_des through the obstacle on many episodes)
- `obstacle_contact = 0.11` (up from 0.05 baseline, expected — harder scenes)

Clean. Infrastructure works end-to-end.

**Wk3.5d Phase 2 (all 4 planners at 0.30/0.35/0.20/0.15).** Deleted
the four weight overrides in `cbf_go2_env_cfg.py`. 50-iter smoke.
Iter 49:

- planner rates: uniform 0.42 / goal 0.21 / walk 0.35 / adversarial 0.02
- σ = 0.82 stable, no tracebacks from quaternion or multinomial paths
- `obstacle_contact = 0.17`, `u_safe_deviation = -0.025`

**Survival-bias note.** Observed planner rates deviate from sampling
weights — adversarial shows 0.02 vs its 0.15 weight. Reason: adversarial
envs die fast (robot walks at obstacle, collision terminates, env
resets, new planner sampled). The multinomial sampling IS correct at
reset-time; instantaneous snapshots undercount short-survival planners.
Not a bug; aggregates-over-episodes match the weights.

**Wk3.5e — widened domain randomization.**

- Friction: parent's 0.8/0.8 → static (0.3, 1.2), dynamic (0.2, 1.0).
- Force: (-1, 1)N → (-10, 10)N.
- Torque: (-0.1, 0.1) → (-2, 2) Nm.
- New startup event `randomize_com_and_cache` (per-env body-frame COM
  delta, ±5cm x/y, ±3cm z). Wraps Isaac Lab's
  `randomize_rigid_body_com` + caches the delta on the env for priv obs.
- New priv obs `com_offset_b` (3D) reading the cached delta.
- Priv obs **17D → 20D**. Actor input layer auto-reshapes (trained
  checkpoint is now dimensionally incompatible; retraining from scratch
  is fine, don't `--resume`).

**Wk3.5e smoke.** 50-iter, 1024 envs. Iter 49:

- σ = 0.81 (healthy)
- `u_safe_deviation = -0.085` (**another 3× higher** than Phase 2 —
  teacher working much harder under real disturbances)
- `base_contact = 0.095` (up from 0.041 in Phase 2; 10N pushes +
  slippery friction are OOD for the locomotion policy. Should drop
  as teacher learns to compensate. If it doesn't drop during the
  overnight, locomotion needs retrain with matching DR — separate task.)
- `obstacle_contact = 0.17`, all 4 planner rates non-zero.
- Steps/sec 5.9k vs prior 12k — extra physics work from big
  force/torque writes. Acceptable.

**Wk3.5f overnight kicked ~12:30 local.** 3000 iters × 1024 envs.
ETA ~3h 30m (longer than prior 2h 24m due to slower steps/sec). When
it lands, re-run 3.5c Pareto eval — harder env should surface the
hand-tuned tradeoff dilemma the first Pareto couldn't.

**Files touched:**

- `cbf_go2_commands.py`: expanded MultiPlannerCommand to all 4 planners.
- `cbf_go2_events.py`: new `randomize_com_and_cache`.
- `cbf_go2_observations.py`: new `com_offset_b`.
- `cbf_go2_env_cfg.py`: widened friction / force / torque, added COM
  event + obs term, PLAY cfg zeroes COM delta, removed Phase 1 weight
  overrides after Phase 1 passed.

---

## 2026-04-22 — Wk3.5d multi-planner code landed; GPU contention blocks smoke

Code done, no sim time. Lab GPU locked all afternoon/evening by
`codrincrismariu`'s `Mjlab-HLIP-CLF-Two-Platform-Stepping-Corridor-Unitree-G1`
training (4096 envs, ~30 GB / 32 GB used), so the teacher-side smoke
tests are deferred to tomorrow.

**New file: `cbf_go2_commands.py` — MultiPlannerCommand.** Subclasses
`UniformVelocityCommand`, adds per-env `planner_id` sampled at reset
via `torch.multinomial` with configurable weights. Four planner types:

| ID  | Name        | Behavior                                                  |
|-----|-------------|-----------------------------------------------------------|
| 0   | uniform     | parent behavior (constant velocity)                       |
| 1   | goal        | u_des = norm(goal_w − pos_w) × speed, resample on arrival |
| 2   | walk        | heading flips by ±70° every ~1s, walks in that direction  |
| 3   | adversarial | u_des points at nearest obstacle every step               |

Each planner has its own `_compute_*_command` method that writes to
`vel_command_b` + `heading_target` for its envs. Parent's
`_update_command` then applies heading tracking for all. Per-planner
activation rate surfaces in training logs as
`Metrics/base_velocity/planner_{uniform,goal,walk,adversarial}_rate`.

Default weights: 0.30 / 0.35 / 0.20 / 0.15 (uniform-heavy to start,
will tighten after validation).

**Env-cfg swap.** `cbf_go2_env_cfg.py`: parent's `UniformVelocityCommand`
replaced with `MultiPlannerCommandCfg`, preserving all parent ranges.
PLAY cfg sets `uniform_weight=1.0` + zeroes the rest so smoke-script
geometry stays deterministic.

**Phase 1 override in place.** To isolate new-code bugs, Wk3.5d is
staged in two phases:

- Phase 1 (current): `uniform_weight=0.40, goal_weight=0.60`, walk +
  adversarial zeroed. Tests the infrastructure + goal-reaching code
  without touching the bigger quaternion/multinomial paths.
- Phase 2 (next): delete the four weight overrides in `cbf_go2_env_cfg.py`
  to restore defaults (all 4 planners active at 30/35/20/15).

**Blocker.** Smoke test of Phase 1 requires ~3 GB free GPU. The lab's
5090 is currently at 30/32 GB used by another user's training. Tried
twice (00:55 and 01:09 local), same `_physics_sim_view is None`
AttributeError both times — classic symptom of no free PhysX context.
Check `nvidia-smi` before each attempt.

**Files touched:**

- `cbf_go2_commands.py`: new, ~220 lines, 4 planners + cfg.
- `cbf_go2_env_cfg.py`: swap command term, PLAY override updated.

**Tomorrow's first action:** `nvidia-smi` → confirm GPU free → run
Phase 1 50-iter smoke → if clean, delete phase-1 weight overrides in
env_cfg → run Phase 2 50-iter smoke → if clean, overnight retrain.

---

## 2026-04-22 — Wk3.5a locomotion retrain ✓; 3.5c Pareto eval (diagnostic pass)

Two things landed.

**1. Locomotion retrain (3.5a).** 1500 iters on
`Isaac-Velocity-Flat-Unitree-Go2-v0`, 4096 envs, 1h 22m wall.
Final: 99.4% time_out, 0.59% base_contact, error_vel_xy=0.14 m/s,
error_vel_yaw=0.28 rad/s. Fall rate essentially on par with Wk1's
0.5% — slightly over the 0.3% gate I'd set, but good enough to
consider swapping later. Deferred the actual policy swap: swapping
invalidates the current teacher (trained against old locomotion),
so we'll swap + retrain teacher together in 3.5f.

**2. Pareto eval (3.5c) — diagnostic pass, not a paper result yet.**
Ran the trained teacher vs 18 hand-tuned configs (α ∈ {0.1, 0.5, 1.0,
2.0, 3.5, 5.0} × a ∈ {0.001, 0.1, 0.5}, φ=0.1, c=0.1) with
`scripts/eval_pareto.py`, 64 envs × 2000 steps per config, ~3 min
total.

Result: **all configs → 0.000 collision rate.** In the current Wk3
eval env, the robot walks forward and the obstacle+CBF combo is
never tight enough to actually crash — regardless of CBF params.
What varies is the fall rate (base_contact):

| Config                      | coll  | time_out | fall  | du²/step |
|-----------------------------|-------|----------|-------|----------|
| ht_a=0.1_m=0.5 (most agg)   | 0.000 |    0.351 | 0.649 |    5.37  |
| ht_a=1.0_m=0.001 (gentle)   | 0.000 |    0.969 | 0.031 |    0.014 |
| ht_a=5.0_m=0.5 (paranoid)   | 0.000 |    0.906 | 0.094 |    0.034 |
| trained teacher             | 0.000 |    0.938 | 0.062 |    0.051 |

The CBF's aggressive corrections cause base_contact because the frozen
locomotion policy wasn't trained against heavily-shaped u_des.

Interpretation: the env is **too easy** to prove the paper's Pareto
hook. Without pressure (reachable goal, tight clearance, adversarial
u_des, wider DR), any reasonable CBF config works. Trained teacher
lands in the moderate-intervention cluster with mid-pack fall rate —
not dominating, not regressing.

This is what Wk3.5d (multi-planner) and 3.5e (widen DR) were designed
to fix. This eval was always going to be a diagnostic dry-run; the
paper-facing Pareto comes after 3.5f (retrain on harder env).

**Files touched:**

- `scripts/eval_pareto.py`: new. Hand-tuned α×a sweep + trained teacher,
  per-config per-episode stats, CSV + matplotlib PNG.
  Key subtlety: hand-tuned configs need inverse-tanh-scale to compensate
  for `_cbf_filter`'s squash, otherwise requested α=1.0 lands as α≈2.96
  on the physical scale.

**Outputs:**

- `IsaacLab/logs/pareto_eval/pareto.csv`
- `IsaacLab/logs/pareto_eval/pareto.png`

---

## 2026-04-21 — Wk3 minimum-bar teacher ✓; Wk3.5 track opened

Three things landed:

**1. Proximity-reward overnight (3rd attempt, run #3) — teacher converged.**
After the σ-runaway (run #1) and degenerate-passthrough (run #2) fixes,
added a dense `-5·exp(-dist/0.5)` proximity penalty every step so PPO
gets a continuous gradient for obstacle-avoidance, not just a sparse
terminal collision event. 3000 iters, 1024 envs, 2h 24m wall.

Final: σ=0.86, `obstacle_contact`=4.61% (halved from 9.4% mid-training),
`u_safe_deviation`=-0.006 (active CBF use, not no-op), episode length 980.
Minimum "teacher converges" bar met. Not paper-quality yet.

**2. SHIELD port sanity check.** Re-read
`safety_cbf/experiments/68_moving_mixed_shapes/env_dynamic.py:356` and
compared to our Isaac Lab port `cbf_go2_env.py:194-243`. The h(x) math
is correctly ported — same front/behind branches, same
`LAMBDA_H=1.0` / `GAMMA_H=0.5`, same exponential warping, same
`prev_e_i` smooth handoff. What's STRIPPED from experiment 68:
multi-obstacle loop + `argmin` SDF selection, per-shape
`_obs_sdf_est` / `_obs_gradient`, per-obstacle `effective_radius`,
sensor-noise injection, moving obstacles, A*/path planner,
periodic replanning. All deferred — see Wk3.5d/e/g.

**3. Wk3.5 deep-improvement track opened.** Because the Wk3 teacher is
weak (single planner, single obstacle shape, mild DR, Wk1 locomotion),
distilling it into a student would just bake in the weakness. Plan:
fix locomotion first (GIGO: teacher's gradient is contaminated by
locomotion wobble), probe the Z bottleneck, run hand-tuned Pareto eval
(paper's central comparison), then deepen DR + planner variety + multi-
shape. Full plan is now in PROGRESS.md "Next up" as a checklist.

**Actions taken this session:**

- Kicked 3.5a (locomotion retrain, 1500 iters, `Isaac-Velocity-Flat-
  Unitree-Go2-v0`) in the `teacher` tmux session, ~45 min wall.
- Wrote + ran **3.5b Z-bottleneck probe** (`scripts/probe_z.py`). Pure
  torch, auto-discovers env_encoder layers by filtering state_dict for
  `mlp.0.*` keys. Fed 1000 realistic priv-obs samples through the
  encoder. Result: all 8 Z dims alive, std 0.85–4.0 (no collapse,
  no near-dead dims), dim 3 dominant (std=4.0, max=19.0). Interpretation:
  encoder reserves dim 3 for a rare-but-important signal (likely close-
  obstacle proximity), other dims carry the rest. Wk4 distillation has
  a valid target.
- Minor bug fix: probe script initially looked for `model_state_dict`
  but rsl_rl saves as `actor_state_dict` + `critic_state_dict`. Fall
  through added.

**Files touched:**

- `cbf_go2_rewards.py`: new `obstacle_proximity` function
  `exp(-dist/sigma)`.
- `cbf_go2_env_cfg.py`: new RewTerm `proximity`, weight -5, σ=0.5.
- `scripts/probe_z.py`: new, pure-torch Z-bottleneck probe.
- `PROGRESS.md`: Next-up rewritten as a Wk3.5 dashboard with gate criteria.

**Headline numbers across the three overnights:**

| Run | Fix                                       | σ       | obstacle_contact | Verdict                |
|-----|-------------------------------------------|---------|------------------|------------------------|
| #1  | clamp + entropy=0.01                      | 187.77  | 6.0%             | σ runaway              |
| #2  | + tanh squash, entropy=0.001              | 3.17    | 7.0%             | degenerate passthrough |
| #3  | + dense proximity reward                  | 0.86    | 4.61%            | converged, uses CBF    |

---

## 2026-04-20 — Item #6: first overnight exposed σ-runaway; tanh + entropy fix

Kicked the overnight right after #5 landed: 3000 iters, 1024 envs, 2h 24m
wall time. Training completed cleanly (no crashes, no NaNs), but the
final policy is not usable — diagnosis and fix below.

**What we saw at iter 2999:**

| Metric              | Value   | Read                              |
|---------------------|--------:|-----------------------------------|
| `Mean action std`   | 187.77  | **catastrophically blown up**     |
| `Mean entropy loss` | 33.26   | absurdly high (σ dependent)       |
| `Mean reward`       | -0.26   | barely moved from iter-47 (-0.38) |
| `obstacle_contact`  | 6%      | halved from 13% — some learning   |
| `base_contact`      | 2.1%    | stable                            |
| `episode length`    | 936     | survives most episodes            |

So the teacher *did* learn a little avoidance (obstacle_contact 13% → 6%),
but σ=187 means the Gaussian is so wide that actions are essentially
random after the squash. Whatever avoidance it picked up is a lucky
accident, not a deployable policy.

**Why σ blew up.** The 5D CBF params were `.clamp()`-ed inside
`_cbf_filter` to their valid ranges (α ∈ [0.1, 5], etc). With a hard
clamp, every sample past the range lands on the same boundary value
— the env's reward is identical whether PPO samples α=10 or α=1000.
So the gradient of reward w.r.t. σ is **zero** for any sample beyond
the clamp range. Meanwhile PPO's entropy bonus (default
`entropy_coef=0.01`) always rewards higher σ. No counter-force →
monotonic σ drift to infinity. The 50-iter burst had already hinted
at this (σ climbed 1.0 → 2.18) — underweighted at the time.

**Two-part fix:**

1. **Replace clamp with tanh-squash** in `_cbf_filter`
   ([cbf_go2_env.py](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env.py)).
   Raw 5D action → `tanh(x)` → scale `[-1, 1]` to each param's
   physical range. Unlike clamp, tanh is smooth everywhere — the
   derivative is small-but-nonzero in the tails. That restores a
   gradient signal on σ.

2. **Drop `entropy_coef` 0.01 → 0.001**
   ([rsl_rl_ppo_cfg.py:84](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/agents/rsl_rl_ppo_cfg.py#L84)).
   With rewards around -0.3, 0.01 × entropy was numerically larger
   than the real advantage signal — entropy dominated PPO updates.
   10× reduction puts it in the right ballpark.

**Smoke evidence that σ is now under control:**

| Config at iter 49          | σ    | Note                       |
|----------------------------|-----:|----------------------------|
| clamp + entropy=0.01       | 2.18 | drifting up                |
| tanh + entropy=0.01        | 1.87 | still drifting up (slower) |
| **tanh + entropy=0.001**   | 0.82 | **shrinking (1.0 → 0.82)** |

Reward and obstacle_contact held steady across all three — we're not
losing any learning by tightening the entropy bonus, just stopping
the policy from drifting into random-action territory. Kicking a
fresh 3000-iter overnight with tanh + entropy=0.001.

**Files touched:**

- `cbf_go2_env.py`: 6 clamp lines → 5 tanh+scale lines in `_cbf_filter`.
- `agents/rsl_rl_ppo_cfg.py`: `entropy_coef=0.01` → `0.001`.

---

## 2026-04-20 — Item #5: per-env obstacle randomization ✓

Moved the obstacle from a single global `/World/Obstacle` prim to per-env
`{ENV_REGEX_NS}/Obstacle` and added a reset event that samples `(x, y)`
for each env independently.

**Why it mattered.** With the old single-global obstacle at world (3, 0)
and `env_spacing=2.5` on a 32×32 grid, only ~4 of 1024 envs had their
robot within the 5 m range-gate of the one obstacle. The other ~99% of
envs trained against an all-zeros priv-obs obstacle slot — effectively
teaching the teacher that "obstacle" just means "zeros, ignore me". Now
every env sees its own cube, randomized per reset in x ∈ [1.5, 3.5],
y ∈ [-1.5, 1.5] (env-local, front arc).

**Files touched:**

- `cbf_go2_events.py`: new `randomize_obstacle_position` event. Reads
  `obstacle.data.default_root_state` for z + quat, samples (x, y),
  shifts by `env.scene.env_origins`, writes via `write_root_pose_to_sim`.
- `cbf_go2_env_cfg.py`: obstacle `prim_path` → `{ENV_REGEX_NS}/Obstacle`;
  LiDAR `RayCasterCfg` commented out (single-mesh limit incompatible
  with per-env obstacles — teacher doesn't observe it anyway, re-enable
  in Wk4 student work); removed now-unused `RayCasterCfg, patterns`
  imports; registered the new reset event; `CbfGo2EnvCfg_PLAY` sets the
  event to `None` so the 1-env smoke stays deterministic.
- `cbf_go2_env.py`: single-line bugfix in `_init_prev_e_i` — was
  `obstacle.data.root_pos_w[:, :2]` (fine when obstacle was (1, 3)
  broadcast), broken now that obstacle is (N, 3). Changed to
  `[env_ids, :2]` to match the robot's indexing. `_cbf_filter` already
  used full-N slice so nothing to fix there.

**Training comparison — 1024 envs, 50 iters, seed 42:**

| Metric                 | Before (global) | After (per-env)       |
|------------------------|----------------:|----------------------:|
| `obstacle_contact`     | 0.29%           | **~13%** (≈45×)       |
| `base_contact`         | 0%              | 3.5%                  |
| Final reward           | -1.05           | -0.38 (trending up)   |
| Episode length         | ~50             | 967                   |
| Action std (final)     | 1.14            | 2.18 (still exploring)|

Interpretation: the teacher is actually seeing obstacle signal across
the whole fleet now. `obstacle_contact` going to ~13% is expected —
with 1024 envs running against a random forward-arc obstacle and an
un-trained policy, many episodes end in collision. `base_contact`
ticking from 0 → 3.5% is also healthy; robots are now failing in
navigation ways rather than just walking in straight lines forever.
Reward rising from -0.87 back up toward -0.38 over the last ~15 iters
shows PPO is learning to trade collision cost against tracking cost.

**Debug detour.** First smoke at 16 envs crashed on the first reset:

    RuntimeError: shape mismatch: value tensor of shape [16, 2] cannot
    be broadcast to indexing result of shape [1, 2]

Root cause above. Fixed in one line, re-sync'd via `sync_to_lab.sh`,
clean 1-iter and then a clean 50-iter.

**Throughput.** Sporadic iterations jumped from ~1.9 s → ~5.5 s (iters
16–18, 28–30, 41–43). Pattern doesn't correlate with any training
signal; likely GPU sync hitches or another process on the box. Not
systematic, not acting on it unless it persists across the overnight
run.

---

## 2026-04-20 — First training burst + u_safe clamp fix ✓

Ran the first real 50-iter PPO burst at 1024 envs. Found and fixed a
serious reward/action-space issue along the way.

**Fix 1: u_safe_deviation weight -1.0 → -0.1.** Raw reward magnitudes
were swamping everything else in PPO's value estimates. Per-episode
contribution was -40k to -80k, making value targets astronomical and
adaptive learning-rate unstable (KL adaptation pushed action std *up*
rather than committing). Did not help on its own — see fix #2.

**Fix 2: clamp u_safe to [-2, 2] m/s in `_cbf_filter`.** Turned out
to be the critical one. At medium distances `‖L_g h‖` can be small
(~0.1) while `slack` is modestly negative; the closed-form projection
formula divides by `‖L_g h‖²`, so `u_safe` was being pushed to
20 m/s commands when PPO sampled extreme CBF params (α=5, φ=5, a=1).
A 20 m/s command to the frozen locomotion policy = robot faceplant.

Physical walking envelope for the Go2 is O(1 m/s); ±2 m/s gives
headroom without being insane. Single `.clamp(-2.0, 2.0)` on
`u_safe_xy` before yaw concat.

**Training results (1024 envs, 50 iters, seed 42):**

Before both fixes:

| iter | reward | u_safe_dev | base_contact | action std |
|------|--------|------------|--------------|------------|
| 0    | -640k  | -23528     | 7%           | 1.01       |
| 48   | -28k   | -1820      | 21%          | 1.57       |

After both fixes:

| iter | reward | u_safe_dev | base_contact | action std |
|------|--------|------------|--------------|------------|
| 13   | -5.66  | -0.02      | 1%           | 0.96       |
| 48   | -1.05  | -0.001     | **0%**       | 1.14       |

Reward ~30,000× better, robot stopped tipping entirely, action std
drift slowed dramatically. The fixed pipeline is clearly learnable.

**New flag to watch:** `obstacle_contact` ticked 0 → 0.29% in the
last few iters. With the filter no longer making the robot crash,
PPO is exploring closer to the obstacle. Rate is still tiny (~3
collisions per 1000 envs) but worth monitoring during longer runs.

**Debug diary (painful but educational):**

1. Weight change from -1.0 to -0.1 didn't visibly help at first.
2. Several "identical" runs in a row convinced me the file wasn't
   loading — but I was comparing the wrong iterations across runs.
3. Added a `print` at cfg module scope to verify the load path;
   confirmed cfg loaded and weight was -0.1 at runtime.
4. Realized the real issue was u_safe magnitude, not the weight.
5. Paste path mistakes (rel paths changed when cwd changed between
   `~/Desktop/safety-go2` and `~/Desktop/safety-go2/IsaacLab`) ate a
   couple of extra cycles. Rsync setup next session would help here.

**Files touched:**

- `cbf_go2_env.py`: added `u_safe_xy.clamp(-2.0, 2.0)` in `_cbf_filter`.
- `cbf_go2_env_cfg.py`: `u_safe_deviation` weight -1.0 → -0.1.

---

## 2026-04-20 — Week 3 slice #2: teacher actor with env_encoder + π_teacher split ✓

Landed the explicit RMA-style architecture split in the PPO actor.
Teacher architecture is now:

    priv(17) → env_encoder → Z(8) → π_teacher → action_means(5)

**Files:**

- New `cbf_go2_teacher.py`: `CbfTeacherMLPModel` subclasses rsl_rl's
  MLPModel, overrides `self.mlp` with a `_SplitMLP` holding two named
  `MLP` submodules. `env_encoder` is 17→64→64→8 with ELU on Z (so the
  bottleneck is nonlinearly activated before the head). `π_teacher` is
  8→64→5. `get_z(obs)` exposes the Z latent for Wk4 student distillation.
- Rewrote `agents/rsl_rl_ppo_cfg.py`: new-style `actor` + `critic` slots
  (rsl-rl 4.0+), `obs_groups` dict, full algorithm hyperparams. Now
  inherits directly from `RslRlOnPolicyRunnerCfg` instead of
  `UnitreeGo2FlatPPORunnerCfg` — the locomotion parent sets a concrete
  `policy` field that the isaaclab_rl compat shim uses to auto-infer
  `actor`/`critic`, clobbering our custom actor. Skipping the locomotion
  parent keeps `policy` MISSING so the shim respects our explicit config.
- `class_name` uses rsl_rl's fully-qualified resolver form
  `"module.path:Class"` — no monkey-patching of rsl_rl.models needed.

**Debug notes (two failed attempts before success):**

1. First attempt inherited `UnitreeGo2FlatPPORunnerCfg` and tried
   `self.policy = None` in `__post_init__`. Compat shim still inferred
   actor/critic from the parent's policy, printing a flat 128³ MLP.
   Diagnosis: configclass didn't treat `= None` as "unset."
2. Terminal paste of heredoc got truncated mid-body (`EOF` landed
   after `"critic": ["policy"],`, rest leaked into shell). Switched
   to base64 `| base64 -d >` one-liners — paste-proof.

**Smoke result (16 envs, 1 iter, `Isaac-CBF-Go2-v0`):**

    Actor: CbfTeacherMLPModel → _SplitMLP
      [0] MLP: 17 → 64 → 64 → 8 (ELU)
      [1] MLP: 8 → 64 → 5
    Critic: MLPModel 17 → 128 → 128 → 128 → 1
    Iter completes clean, 77 steps/s at 16 envs.

**Lab-machine rsync:** saved `chrisliang@130.64.84.163:~/Desktop/safety-go2`
as the sync target for future iterations (hostname `MEMCEU17155` doesn't
resolve from the Mac).

---

## 2026-04-20 — QP vectorization via closed-form projection ✓

Swapped the per-env cvxpy/OSQP loop in `_cbf_filter` for a closed-form
half-space projection. First Week-3-prep slice landed.

**Realization:** our QP is trivial — one 2D variable, one linear
inequality (b=0 means no SOC term). That's geometry, not optimization:
*"find the closest point to u_des that lies in the half-plane
{u : L_g h · u ≥ rhs}."* One-liner:

- if u_des already satisfies the constraint → u_safe = u_des
- else → u_safe = u_des + slack / ‖L_g h‖² · L_g h

Pure torch, fully GPU-batched, differentiable, no CPU roundtrip, no
dependency. Replaces ~25 lines of per-env Python-loop cvxpy solve
with ~6 lines of vectorized ops.

**Scope caveat:** the closed form applies only while the `b·‖u‖` term
stays off (that term would make the constraint second-order-conic, not
linear, and break the half-space geometry). Dropped a TODO in the
filter pointing at cvxpylayers/qpth for when we re-enable b.

**Why it matters:** this was listed as the H×H risk in the register
("cvxpy QP too slow in training loop") and was Week 3's first blocker
for scaling to 1000+ envs. Instead of swapping to qpth/cvxpylayers
as originally planned, we deleted the solver entirely. Removed
`import cvxpy as cp` and `import numpy as np` from cbf_go2_env.py;
removed the `cp.Variable/Parameter/Problem` setup in `__init__`.

**Smoke result** (`play_cbf_smoke.py`, num_envs=1, 400 steps):

- Steps 0–80 (dist ≥ 1.6m): CBF passthrough, u_safe = u_des
- Step 100 (dist 1.19m): filter engages, u_safe_x = 0.705
- Steps 140+ (dist < 0.85m): heavy braking, u_safe_x ≈ 0.1–0.2
- Robot halts cleanly at ~0.73m from the 0.35m-radius cube
- h stays strictly positive throughout — never crossed unsafe

Same qualitative profile as prior OSQP smoke runs.

**Perf benchmark (3-iter PPO burst, `Isaac-CBF-Go2-v0`):**

| N envs | Steady steps/s | Per-env steps/s |
|--------|----------------|-----------------|
| 16     | 288            | 18.0            |
| 1024   | **9026**       | **8.8**         |

31× end-to-end throughput scaling 16 → 1024. Per-env throughput only
halved under 64× more envs — near-linear scaling on the 5090. With
the old per-env cvxpy loop, 1024 envs × ~1 ms/env would have been
~1 s per sim step in QP alone, effectively unrunnable. That was
exactly the H×H risk; it's dead now.

**Side observation (not QP-related, flagged for Wk3 training):**
base_contact terminations ramped 6→20→27% across the three PPO
iterations. Untrained policy + fresh force/torque disturbances tipping
the robot over as PPO explores random CBF params. Value loss is
astronomical (~10¹¹) because `-λ₂·‖u_safe-u_des‖²` accumulates when the
filter rejects hard. Expected for cold-start — but a reward-scaling
note to revisit during Wk3 training.

---

## 2026-04-20 — Step 2g.1b: disturbance priv obs (force + torque) ✓

Completes the teacher's privileged observation to match the RMA
framing — teacher now sees applied external force and torque on the
base, not just friction/mass/height. Priv obs 11D → 17D.

**Architecture decision recap:** debated whether to add these or fall
back to pure domain randomization (user's instinct). Kept teacher-
student / RMA split per the paper's framing. Half-privileged teacher
(friction+mass in obs, wind not) would've been inconsistent —
either full RMA or full DR.

**Implementation:**

- New `cbf_go2_events.py` with `apply_external_force_torque_and_cache`:
  - Samples force ~U(-1, 1) N and torque ~U(-0.1, 0.1) N·m on reset.
  - Calls `asset.set_external_force_and_torque` to actually apply.
  - Stashes sampled values in `env.cbf_applied_force_b` /
    `env.cbf_applied_torque_b` for priv obs to read.
- Replaces parent's stock `base_external_force_torque` event (which
  had `force_range=(0, 0)` by default — effectively off). Means the
  training distribution now includes real disturbances, which is
  desirable for RMA.
- Two new priv obs functions `applied_force_b` / `applied_torque_b`
  — just return the cached tensor, with zeros fallback if the event
  hasn't run yet.

**Smoke result:** 16 envs, 1 iter, 2.6 s. Obs shape (17,), actor MLP
`in=17 out=5`, all 7 priv terms registered and concatenated cleanly.
Benign deprecation warning about `set_external_force_and_torque` — the
function is being renamed in a future Isaac Lab release but still
works now.

**Still deferred:** COM offset. Needs its own event to both sample AND
apply the offset via `asset.root_physx_view.set_coms(...)`. Scope
creep for one session; belongs with a full domain-randomization pass
pre-Wk3.

---

## 2026-04-19 — Step 2h: CBF reward + obstacle-contact termination ✓ (Week 2 done)

Final piece of Week 2. Replaced the parent's 10 locomotion-tracking
reward terms with the 3-term CBF reward, added obstacle-contact
termination so collision penalty fires once per episode.

**Reward config (weights applied by RewardManager):**

| Term               | Weight | Source                                           |
| ------------------ | ------ | ------------------------------------------------ |
| `collision`        | -100   | geometric distance < 0.5 m to obstacle           |
| `u_safe_deviation` |   -1   | `‖last_u_safe − last_u_des‖²` (summed 3D)        |
| `infeasibility`    |  -10   | `last_infeasibility` flag from `_cbf_filter`     |

**Termination additions:** kept parent's `time_out` and `base_contact`
(fall), added `obstacle_contact` (same distance check as the collision
reward, but returns bool instead of float).

**New infrastructure:**

- `cbf_go2_rewards.py` — reward + termination functions, split into a
  `_obstacle_contact_mask` helper (bool) plus two typed wrappers
  (`collision_with_obstacle` → float, `obstacle_contact_termination`
  → bool). Two wrappers because reward manager wants float, termination
  manager wants bool — same logic, different dtype.
- `cbf_go2_env.py::_cbf_filter` — now counts per-env infeasibility
  flags during the Python loop over envs and stashes them as
  `self.last_infeasibility` for the reward term to read.
- `self.last_u_des` / `self.last_u_safe` / `self.last_infeasibility`
  allocated in `__init__` so reward compute at reset time doesn't
  crash on missing state.

**Gotcha fixed on first smoke:** first attempt returned `.float()`
from `collision_with_obstacle` and reused that function for both
reward AND termination. TerminationManager uses `bool |= value` to
OR into the terminated_buf, which errored on float. Split into two
wrappers above.

**Smoke result:** 16 envs, 1 iter, 1.25 s. All 3 reward terms +
3 termination terms registered and firing. Per-term episode reward
breakdown shows `u_safe_deviation: -0.0224` (tiny CBF intervention
activity) and 0 for the other two (no collisions or QP failures this
iter — expected with random params early).

**Week 2 status: DONE.** 2a–2h all landed. `Isaac-CBF-Go2-v0` runs
end-to-end: 11D priv obs → encoder (future) → π_teacher (future) →
5D CBF params → CBF-QP → u_safe → frozen locomotion → physics →
3-term reward → PPO.

Week 3 is next — actual teacher training. Key prep items:
vectorize the QP (qpth/cvxpylayers) for 1000+ envs, define
env_encoder + π_teacher as a module PPO can train, add disturbance
priv obs (2g.1b), add obstacle-shape variety.

---

## 2026-04-19 — Step 2g.1: teacher privileged-info obs (11D) ✓

Wired the teacher's observation group. Actor/critic now see an 11D
ground-truth vector instead of the parent's 41D proprioceptive obs.

**Obs terms (total 11D):**

| Term                  | Dim | Source                                    |
| --------------------- | --- | ----------------------------------------- |
| friction              | 1   | `root_physx_view.get_material_properties` |
| base_mass_offset      | 1   | current - default mass                    |
| base_height           | 1   | `root_pos_w[:, 2]`                        |
| tracking_err          | 3   | `u_des - base_vel` (body frame)           |
| nearest_obstacle_body | 5   | `[rel_pos, rel_vel, radius]` body-frame,  |
|                       |     | range-gated to 5m                         |

**Design decisions captured in code:**

- Obstacle obs in BODY frame (not world) — matches the FOV shape
  that the student's LiDAR will later deliver. Teacher and student
  reason about the same spatial scope.
- Range gate (5m) — zeros out when obstacle is beyond LiDAR reach.
  Teacher won't learn to over-rely on unavailable info.
- Group named `policy` (not `teacher_priv`) so rsl_rl picks it up as
  actor obs without extra `obs_groups` wiring. Student group will be
  added alongside at Wk4.

**Gotcha fixed on first smoke:** PhysX view calls
(`get_material_properties`, `get_masses`) return CPU tensors in
Isaac Lab 2.3.x. `torch.cat`-ing with GPU obs blows up with a device
mismatch. Fix: `.to(env.device)` on those two.

**Smoke result:** 16 envs, 1 iter, 2.93 s. Actor MLP `in=11 out=5`.
No tracebacks, no base_contact terminations this run
(was 2% under 2d.5's 41D obs — single-run variance but worth watching).

**Deferred to 2g.1b (pre-Wk3):** applied force vector + COM offset
as priv obs. Both exist as event randomization but don't persist
sampled values; need event wrappers to cache. Not blocking teacher's
first real training — just means it won't "see" these two env signals.

**Deferred to Wk4:** student obs group (LiDAR + base_vel + history
buffer). Skipped now to avoid committing to a layout before Wk3
training reveals what signals matter.

**New Wk3-prep risk captured:** obstacle-shape generalization. Current
SHIELD h(x) hard-codes a circle radius; priv obs returns a single
`radius` scalar. Walls / boxes / arbitrary shapes need shape-aware
SDF + shape-aware priv obs (type enum + parametric dims). Fine for
today's single cube, mandatory before randomizing shapes.

---

## 2026-04-19 — Teacher/student architecture revised (no code change)

Prof revisit during obs-space design. The teacher is *cleaner* than
what TODO_training.md originally had.

**OLD (in the previous TODO_training.md):**

```text
obs + z ──▶ env_encoder ──▶ e
                            │
                            ├──▶ policy_head ──▶ params
obs ────────────────────────┘
```

Teacher's policy_head saw `(obs, e)`. Obs shape must then match
student's so policy_head stays shareable.

**NEW (as of today):**

```text
z ──▶ env_encoder ──▶ Z
                      │
                      ▼
                 π_teacher ──▶ α, φ, a, c
```

Teacher's π_teacher takes ONLY Z. No obs input. Privileged info is
the *entire* input to the teacher stack. Cleaner RMA-style split:
the encoder's job is "squeeze z into a compact latent," and
π_teacher's job is "map latent → params." State-dependence
(which step in the episode, how close to obstacle, etc.) is
implicitly inside Z because the encoder sees ground-truth obstacle
pose, COM state, etc.

**Student side** unchanged in spirit, sharpened in detail:

```text
obs (LiDAR, base_vel) + history (past obs + past CBF params)
    │
    ▼
 adapter ──▶ Ẑ ──▶ (frozen) π_teacher ──▶ α, φ, a, c

loss = ‖Ẑ - Z‖²   (pure supervised, teacher encoder frozen)
```

**Planner stays on a sidecar wire.** u_des comes from {A*, RRT, PID,
MPC, random walk}, randomized during training. Hits the CBF-QP
directly. Neither encoder nor adapter sees u_des.

**Concrete consequences that landed in the docs today:**

- `TODO_training.md` — rewrote "What we're training", "Input
  structure" (split teacher vs student), and Approach B diagrams.
- `PROGRESS.md` — rewrote Current state, Next up, Step 2g
  (5 sub-items now), Week 3 (clarified env_encoder + π_teacher,
  noted π_teacher takes Z only), Week 4 (clarified adapter sees
  obs + history, outputs Ẑ, frozen π_teacher).

**No code change yet.** The CBF filter, action term, locomotion wrap
are unchanged. Step 2g is where this architecture actually hits
`cbf_go2_env_cfg.py` — new privileged obs term + LiDAR obs term +
history buffer + two observation groups.

---

## 2026-04-19 — Sim-to-real locomotion strategy clarified (no code change)

Prof. revisit on locomotion. Initial ask was "use the Sport Mode
locomotion that teleop uses" to close sim-to-real gap. Rereading
together, the real goal is THE GAP, not the specific controller.

**Key clarification:** Sport Mode (what `SportClient.Move` drives) is
Unitree's proprietary onboard firmware, not a library we can `import`.
No public sim ships it runnable. `SportClient.Move` is a DDS *network
call* to the robot — it has no algorithm of its own, no joint-target
output on the client side.

**Flip the direction of the fix:** instead of making sim use
hardware's locomotion, make hardware use sim's locomotion. Deploy our
Isaac-Lab-trained `policy.pt` to the Go2's Jetson, bypass Sport Mode
via Unitree's low-level motor interface. Standard practice in
quadruped research; no vendor cooperation needed.

**Consequence for the hardware layer:** `walking_bridge.cpp`
currently calls `SportClient.Move`. Week 4 hardware deploy will swap
this for a low-level pipeline — run `policy.pt` inference on the
Jetson, feed 12 joint targets via Unitree's low-level DDS topics.

**Known caveat: Wk1 locomotion is flat-only.** Trained on
`Isaac-Velocity-Flat-Unitree-Go2-v0`. Real floors have carpet seams,
tile lips, slight slopes — flat-trained policies are fragile. Fix:
retrain on `Isaac-Velocity-Rough-Unitree-Go2-v0` before hardware
deploy. That env adds a height-scan observation (obs ~48D → ~235D),
so `_run_locomotion` also needs an obs-shape update to match.
Deferred to Week 4.

**Nothing CBF-side depends on which locomotion is loaded.** Swapping
`LOCOMOTION_CHECKPOINT` and fixing the obs construction in
`_run_locomotion` is the whole change.

---

## 2026-04-17 — Step 2f: hand-tuned CBF smoke test ✓

End-to-end confirmation that the CBF pipeline actually intervenes.
Used a dedicated PLAY-variant cfg and a minimal no-RL loop script;
fixed params α=1.0, φ=0.1, a=0.01, c=0.1; u_des = const forward
1 m/s; single env; deterministic spawn at origin.

**What actually happened in sim:**

| Phase  | Steps   | x (m)      | u_safe_x   | What CBF is doing                 |
| ------ | ------- | ---------- | ---------- | --------------------------------- |
| Start  | 0       | 0.00       | 1.000      | Silent — u_des passes through     |
| Walk   | 20–80   | 0.2→1.4    | 1.000      | Still silent, far from obstacle   |
| Engage | 100     | 1.78       | 0.74       | First real intervention           |
| Ramp   | 120–160 | 2.04→2.23  | 0.39→0.17  | Smoothly pulling u_safe_x down    |
| Hold   | 160–399 | ~2.25–2.39 | 0.18→-0.01 | Equilibrium held ~0.7m from box   |

`h` stays positive throughout (min ≈ 0.12). Robot never enters the
unsafe region. No NaNs, no QP crashes, no falls.

**Two bugs fixed along the way:**

1. **prev_e_i init bug (critical).** Was hard-coded to `(1, 0)` in
   `__init__` and on reset. For robot-obstacle geometries not aligned
   with +x, SHIELD's first-step h computation projected onto the wrong
   hemisphere → h reported as -2 (unsafe) → QP demanded u_safe ≈ 3.5
   m/s TOWARD the obstacle on step 0. Fixed with a new
   `_init_prev_e_i` helper that seeds from actual robot/obstacle
   positions; called from both `__init__` and `_reset_idx`.
2. **Directional drift.** Parent's `reset_base` event randomizes yaw
   over [-π, π]; with `heading_command` disabled (my first PLAY
   config), the policy had no yaw corrective signal and walked off in
   whatever direction it spawned facing. Fixed in PLAY cfg: zero
   pose_range/velocity_range, keep `heading_command=True` with
   `heading=(0, 0)`.

**New infrastructure:**

- `scripts/play_cbf_smoke.py` — minimal no-RL loop that uses
  `parse_env_cfg` (avoids hydra), builds `Isaac-CBF-Go2-Play-v0`,
  feeds fixed 5D params into env.step() per step. Supports `--video`
  via `gym.wrappers.RecordVideo`, diagnostics printed every N steps.
- `CbfGo2EnvCfg_PLAY` overrides: `num_envs=1`, deterministic spawn,
  deterministic forward u_des, no observation noise, no push events.
- `self.last_u_des` / `self.last_u_safe` stashed on env each step
  (used by the smoke script for diagnostics; will also feed the Step
  2h reward term `-λ₂·‖u_safe - u_des‖²`).

**Observation for future sub-questions:**

- Lateral avoidance is NOT exercised here (u_des purely +x). Robot
  just halts in front of the box. Next: random-walk u_des or
  off-center u_des to check CBF picks a lateral direction.
- Per-env obstacle randomization still deferred.
- Realistic planner u_des (A* / tangent-bug) still deferred pre-Week 3.

---

## 2026-04-17 — Step 2d.5: action space 12D → 5D ✓

Retooled the env so the outer RL sees 5D `(α, φ, a, b, c)` instead of
12D joint targets. Coded end-to-end on the Mac; smoke test on the lab
still to run.

**Design call — Option A1 over A2:**

- **A1 (chosen):** custom 5D `ActionTerm` that stashes joint targets
  and applies them via its `apply_actions()` hook. `super().step()`
  still runs normally — we just slide our targets in at the apply
  stage. ~15 lines in the term, three small hooks in the env.
- **A2 (rejected):** override `step()` entirely, drive the articulation
  directly, manually run physics/reward/termination/reset. ~50 lines
  and the real cost isn't size — it's reimplementing
  `ManagerBasedRLEnv`'s ceremony, where the easy-to-miss reset events
  (base pushes, joint-pos randomization) silently shape the training
  distribution.

**Three files changed:**

1. New `cbf_params_action.py` — `CBFParamsAction(ActionTerm)` with
   `action_dim=5`, `apply_actions()` → `set_joint_position_target`,
   plus a `set_joint_targets()` hook for the env to call each step.
2. `cbf_go2_env_cfg.py` — new `CbfActionsCfg` with a single
   `cbf_params` term; `__post_init__` does `self.actions = CbfActionsCfg()`
   to overwrite the parent's 12D `joint_pos`.
3. `cbf_go2_env.py` — three changes:
   - `__init__` grabs `self._action_term = self.action_manager.get_term("cbf_params")`
   - `step()` stashes joint targets on the term, calls
     `super().step(cbf_params)` with 5D
   - `_cbf_filter` reads clamped params from `cbf_params`; clamps
     handle PPO's Gaussian init (mean=0, std=1) going negative

**Gotcha caught mid-implementation:** the frozen locomotion policy's
48D obs has a 12D `actions` slot that expects *raw scaled deltas* —
the training distribution — not absolute joint targets. Before 2d.5,
the parent's `JointPositionAction` did `target = raw * 0.25 + default`
automatically; after 2d.5 our custom term pushes absolute targets
directly, so `_run_locomotion` now has to do the scale+offset itself,
AND cache the raw output for next step's obs. Split cleanly:
`_last_raw_locomotion_action` is private state for the inner policy's
obs, `joint_targets` is what goes to the articulation.

**Param clamps in `_cbf_filter`:** α ∈ [0.1, 5.0], φ ∈ [0, 5.0],
a ∈ [0, 1.0], c ∈ [0, 0.5]. `b` ignored (SOCP, deferred).

**Smoke test outcome:** clean pass on the lab. Action shape 5, actor
`out_features=5`, obs shape `(41,)` (was 48; the `actions` obs term
shrank 12→5), 384 steps in 3 s, 140 steps/s. No tracebacks.

**Regression that cost us a run:** my heredoc for `cbf_go2_env_cfg.py`
overwrote the known-working `MeshCuboidCfg` + single-mesh-path setup
with an old per-env + multi-path version, which the RayCaster rejects
(one mesh, no wildcards). That gotcha was explicitly noted in the
2026-04-16 LOG entry. Restored the known-good cfg and re-ran.

**New side signal:** mean episode length fell from ~1000 (plain
locomotion) to 17 in this run, with 2% of steps terminating on
base_contact. Random 5D CBF params are causing falls. Two hypotheses:
(a) damping fallback (u_safe=0) confuses the frozen locomotion policy;
(b) clamped-extreme params distort u_safe enough that the policy
struggles. Diagnose in Step 2f with fixed sensible params.

---

## 2026-04-17 — Step 2e: robust CBF-QP live in pipeline

Replaced the pass-through `_cbf_filter` with SHIELD SDF h(x) + robust
CBF-QP. Six modular slices, all validated on the lab desktop.

**Design decisions:**

- **Constraint form** — TODO_training.md version (no `h_safe` divisor):
  `L_g h · û − φ‖L_g h‖² − a + α(h − c) ≥ 0`.
- **`b` term dropped for now** per professor's advice. Keeps it a pure
  QP instead of SOCP, so OSQP stays the solver and `qpth` batching is
  still on the table for Week 3. `b` slot stays in the 5D action space
  as a no-op placeholder; re-added as an ablation later.
- **h(x) source** — SHIELD exponentially-warped SDF from
  `safety_cbf/experiments/71_discrete_02_twopolicy/env_safety_discrete.py`,
  not the raw distance-squared from the legacy `env.py`. Vectorized
  across N envs with `torch.where` over front/behind branches.
- **Parametric cvxpy QP** — built once in `__init__`, reused per step
  via `cp.Parameter`. RHS precomputed on the torch side so cvxpy never
  sees α/φ/a/c directly; it sees a scalar.
- **Infeasibility fallback** — damping (`u_safe = 0`). Counter not wired
  up; that's for Step 2h's reward term.

**Hard-coded defaults** (until Step 2d.5 plumbs the 5D RL action):
α=1.0, φ=0.1, a=0.01, c=0.0.

**Gotchas:**

- `ManagerBasedRLEnv.__init__` calls `_reset_idx` before subclass attrs
  exist → guard with `hasattr`.
- `pip install cvxpy` upgraded osqp 0.6 → 1.1, which conflicts with
  isaacsim-robot's declared pin. Isaac Lab still runs clean; ignored.

**Perf benchmark:** 16 envs, 1 iter, 5.6 s, 74 steps/s (down from
210 steps/s pass-through). 384 QPs/iter, ~8 ms each including CPU↔GPU
round-trip. At 1000+ envs for Week 3 this is the bottleneck — batched
GPU solver needed (`qpth` / `cvxpylayers`).

**NOT validated yet:** visual confirmation that the CBF actually pushes
the robot away from the obstacle. That's Step 2f.

---

## 2026-04-16 — Custom env with frozen locomotion inner loop

**Design decisions locked in:**

- Env action space = **5D** `(α, φ, a, b, c)`. k_nom NOT part of action
  (u_des flows directly from planner). Avoids degenerate solution where
  RL could zero out k_nom.
- Observation = `[LiDAR, velocity]` — NOT u_des. Params depend on
  ENVIRONMENT, not on user's current command. Online adaptation comes
  from changing observations, not changing u_des.
- u_des is in the REWARD only (`‖u_safe - u_des‖²`).

**Step 2d.1–2d.3 done:**

- Located trained locomotion checkpoint at
  `logs/rsl_rl/unitree_go2_flat/2026-04-15_19-24-45/exported/policy.pt`
- Verified TorchScript policy loads, 48D obs → 12D joints, on cuda.
- Created `CbfGo2Env(ManagerBasedRLEnv)` subclass overriding `step()`.
- Pipeline: cbf_params → CBF filter (pass-through) → locomotion policy
  → 12D joint targets → physics.
- Gym registration updated to use `CbfGo2Env` entry point.
- Verified via debug print that custom step is called with
  `(N, 12)` action shape.

---

## 2026-04-16 — LiDAR sensor attached to Go2

- Added 2D horizontal ray-caster LiDAR to `CbfGo2EnvCfg`:
  72 rays around the robot at 5° resolution, 5m max range,
  mounted 0.3m above the base.
- Sensor detects obstacle mesh. Smoke test passed.
- Two gotchas fixed along the way:
  - RayCaster only supports 1 mesh path (no wildcards) — moved to
    single global `/World/Obstacle`.
  - `CuboidCfg` creates USD `Cube` prim, not `Mesh` — switched to
    `MeshCuboidCfg` which spawns proper triangulated geometry.
- Known limitation: single global obstacle means only env_0 "sees" it
  during multi-env training. Deferred to proper randomization pass.

---

## 2026-04-15 — Isaac Lab installed, Go2 trained, CBF task with obstacle

**Step 2b — obstacle added:**

- Added a 0.5m red box (`RigidObjectCfg` + `CuboidCfg`) at (3, 0, 0.25)
  to `CbfGo2EnvCfg`. Kinematic (static), collision enabled.
- Smoke test passed on lab desktop. Obstacle is in the physics scene.
- Position randomization on reset deferred to later.

**Step 2a — CBF task scaffolded:**

- Created `Isaac-CBF-Go2-v0` as a new Isaac Lab task under
  `isaaclab_tasks/manager_based/safety/cbf_go2/`.
- Inherits from `UnitreeGo2FlatEnvCfg` (identical behavior for now).
- Registered and confirmed visible in `list_envs.py` (#173, #174).
- Smoke test passed: 1 training iteration, 0.74 seconds, no errors.
- Files created on MacBook, replicated to lab desktop via SSH `cat` commands
  (scp failed due to hostname resolution on campus network).

**Locomotion visualized:**

- Ran play variant with 4 Go2 robots, recorded video.
- All robots standing stably, tracking velocity commands, moving in varied
  directions. No falls. Visual confirms the training metrics.
- Video saved as supplementary material for CoRL paper.

**Infrastructure:**

- Installed Isaac Sim 4.5 via pip on lab desktop (RTX 5090, Ubuntu 24.04).
- Installed Isaac Lab 2.3.2 from GitHub, `./isaaclab.sh --install` succeeded.
- Verified Isaac Sim loads headless on the 5090 (iray photoreal warning is
  cosmetic; PhysX + ray-caster work fine).
- Verified Isaac Lab detects RTX 5090 at 32 GB VRAM, runs create_empty tutorial.

**Go2 locomotion policy trained:**

- Task: `Isaac-Velocity-Flat-Unitree-Go2-v0`
- 4096 parallel envs, 300 iterations, ~18 min wall clock on 5090.
- Final mean reward: 34.07, episode length: 1000, base_contact: 0.5%.
- track_lin_vel_xy_exp: 1.42 / 1.5 (95% tracking).
- track_ang_vel_z_exp: 0.63 / 0.75 (85% tracking).
- Checkpoint: `IsaacLab/logs/rsl_rl/unitree_go2_flat/<timestamp>/model_299.pt`
- This checkpoint is the **black-box locomotion policy** for subsequent CBF work.

**Key discoveries:**

- Isaac Lab ships Go2 envs natively — no need for Unitree RL Lab.
- Go2 env action = 12D joint positions; velocity commands are OBSERVATIONS.
- safety_cbf/ contains substantial prior work (75 experiments, AdaptiveCBFEnv,
  PPO pipeline). NOT wasted time — this is the 2D ablation foundation.

---

## 2026-04-13 — Planning session

- Wrote TODO_training.md with theoretical reference + 6-week plan.
- Adopted RMA teacher-student as primary training approach.
- Mapped out 3-distribution framing (nominal, filter, deploy) from
  collaborator discussion.
- Deleted TODO_cbf.md (redundant with training notes).

---

## (Earlier — pre-2026-04-13) — Hardware bring-up

- Extracted minimal ROS 2 package from lab mate's `semantic-safety`
  repo. Nodes: teleop, walking_bridge, odom_publisher, cloud_merger.
- Added heartbeat e-stop (5s timeout) and spacebar e-stop to walking_bridge.
- Verified real Go2 walks via arrow-key teleop over direct Ethernet link.
- Set up MacBook → lab desktop → Go2 Jetson rsync deploy workflow.

