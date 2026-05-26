# Phase 10 / V2 — Unified Adaptive-CBF Comparison

This phase trains five outer-policy architectures on the **same** Isaac
Lab Go2 + CBF distribution, then evaluates them across four held-out
scenes with controlled per-axis DR sweeps. Headline question:

> Does the adaptive CBF policy beat the best fixed (φ, α) baseline, and
> which observation contract (priv vs proprio vs history) is necessary?

## 1. Architectures (5 × 2 intervention costs = 10 policies)

| arch tag         | obs layout (dim)               | model class             | what it tests                                        |
|------------------|--------------------------------|-------------------------|------------------------------------------------------|
| `V2Full`         | priv(7) + proprio(45) + 2 + 144 = **198** | `RMAMLPModel`           | full information, upper bound on adaptation         |
| `V2NoPriv`       | 198, priv slot zeroed at every step | `RMAMLPModel`     | can the policy learn safety + reach with no priv? |
| `V2NoProprio`    | 198, proprio slot zeroed       | `RMAMLPModel`           | does z become informative when proprio is removed? |
| `V2RMAClassic`   | priv(4) + proprio(45) + 2 + 144 = **195** | `RMAClassicMLPModel`    | RMA-paper canonical 4 priv channels |
| `V2History`      | proprio_history(50×45) + 2 + 144 = **2396** | `RMAHistoryMLPModel` | end-to-end "blind" via proprio history, no priv slot |

Each is trained twice (intervention cost ∈ {0.0, −0.05}) → **10 trainings total**.

## 2. Training distribution (`CBFAdaptiveGo2UnifiedV2EnvCfg`)

Identical across all 5 architectures (this is the whole point).

**Obstacles.** K=3 random per episode (`random_topology=True`), radius
0.5 m, sampled in the corridor x∈[1.5, 6.0], y∈[−1.5, 1.5] with rejection
sampling enforcing 0.8 m start/goal exclusion and 1.4 m pairwise separation.
Goal at (7.0, 0.0).

**DR (7 channels active simultaneously, per-episode random sample):**

| channel              | range            | theoretical pairing |
|----------------------|------------------|---------------------|
| friction             | (0.3, 1.0)       | φ — control-effectiveness |
| motor_strength       | (0.7, 1.3)       | φ — control-effectiveness |
| v_max                | (1.0, 2.0)       | α — kinematic urgency (validated 92 % bound span) |
| base_mass_delta      | (−3.0, 3.0) kg   | (mostly inertial; weak gate) |
| com_offset           | (−0.05, 0.05) m  | α — tracking residual |
| actuation_noise_std  | (0.0, 0.05) rad  | φ — input uncertainty |
| disturbance_force    | (0.0, 30.0) N    | (constant XY wrench; mostly absorbed by loco) |

**Clumsy-human u_nom.** Active at both train and eval so the policy sees
this nominal distribution in both. Training uses the **"child" preset**
(5-year-old: bigger wobble, slower base speed, more swerves) — harder
than deployment. Eval uses **"teen"** (10-year-old: smaller wobble, faster,
fewer swerves). Both still try to reach the goal but make mistakes
occasionally.

| param            | child (train) | teen (eval) |
|------------------|--------------|-------------|
| lat OU σ (m/s)   | 0.40         | 0.20        |
| speed-mult OU σ  | 0.25         | 0.12        |
| speed-mult mean  | 0.65         | 0.85        |
| swerve prob/step | 0.008        | 0.003       |
| swerve duration  | 30 steps     | 20 steps    |

The u_nom_w output is **rate-limited** (`unom_max_step=0.15 m/s/step`)
so the CBF sees a smooth reference — no jitter from OU jumps or swerve
onsets propagating into the QP.

**Rewards (`CBFV2RewardsCfg`).**

| term              | weight   | function            |
|-------------------|----------|---------------------|
| progress          | +1.0     | `mdp.progress_reward` |
| goal_reached      | **+100** | `mdp.goal_reached_bonus` |
| collision         | −1000    | `mdp.collision_penalty` |
| fall (NEW)        | **−100** | `mdp.fall_penalty` |
| stuck             | −0.05    | `mdp.stuck_penalty` (per-step) |
| action_smoothness | −0.2     | `mdp.action_smoothness_penalty` |
| intervention      | **0.0 or −0.05** | `mdp.intervention_penalty` (the swept variant) |

**Terminations.** `collision`, `fall`, `goal_reached`, `time_out`. Stuck
is **not wired** — the env keeps running so the policy can wiggle out.
`episode_stuck_any` only latches after **250 continuous slow-and-not-at-goal
steps (5 s @ 50 Hz)**, configurable via `cfg.stuck_threshold_steps`.

**Other inherited from `CBFAdaptiveGo2UnifiedLidarSDFEnvCfg`:**
SHIELD perception SDF (5 cm position noise + 2 % dropout + 20 m range),
Mid-360 lidar fidelity (72 rays, 2 cm noise, 0.5° jitter, range-weighted
dropout), Lipschitz rate-limit on (φ, α) (`action_max_step=0.05`).

## 3. Evaluation

**Four held-out scenes** (each in 3 obs layouts: Full / RMAClassic /
History — NoPriv & NoProprio reuse the Full layout with masking flipped
at eval time):

| scene     | obstacles                                       | sweep axes                                  |
|-----------|------------------------------------------------|---------------------------------------------|
| E1 GAP    | 2 × r=0.5 at (3.5, ±1.0)                       | friction {0.3, 0.6, 1.0} + v_max {1.0, 1.5, 2.0} |
| E2 SLALOM | 3 × r=0.5 at (2.0,0.7), (4.0,−0.7), (5.5,0.7)  | motor {0.7, 1.0, 1.3} + v_max {1.0, 1.5, 2.0}   |
| E3 WALL   | 4 × r=0.5 wall at x=2.5 (y ∈ ±0.8/±1.6) + 1 mid | friction {0.3, 0.6, 1.0} + v_max {1.0, 1.5, 2.0} |
| E4 FIELD  | 7 × r=0.4 scattered (dense)                     | motor {0.7, 1.0, 1.3} + v_max {1.0, 1.5, 2.0}   |

Per scene: **2 axes × 3 values = 6 cells per scene, 24 cells per policy**.
All **other DR channels are pinned to nominal** per cell so the swept
axis is the only variable. Per cell: 512 envs × 1250 steps × 512 episodes
(≈±3pp CI on rates).

**Outcomes are mutually exclusive** (first-termination latching with
priority `collide > fall > reach`):

`reach_rate + collision_rate + fall_rate + timeout_rate = 1.0`

`stuck_rate` is a **subset of timeout** (env never terminated AND latched
the stuck flag at 5 s).

`safe_reach = reach_rate` (collisions already excluded by the priority rule).

**Three fixed baselines** evaluated on every scene with the same DR cells:
- B-trivial: (φ=0.0, α=2.5) — nominal CBF, no margin
- B-α-max: (φ=0.0, α=4.0) — most aggressive α
- B-both-max: (φ=1.0, α=4.0) — most conservative

Total eval invocations: 10 policies × 4 scenes + 3 baselines × 4 scenes = **52 (policy/baseline × scene) cells**, each containing 6 DR sub-cells = **312 measured cells**.

## 4. File map

| file                                       | purpose |
|--------------------------------------------|---------|
| `cbf_task/cbf_action_term.py`              | adds `unom_clumsiness` preset, `unom_max_step` Lipschitz rate-limit, `stuck_threshold_steps`, fix to always-write friction/mass/COM and external force |
| `cbf_task/mdp.py`                          | adds `fall_penalty`, reads `_stuck_threshold_steps` from action term |
| `cbf_task/cbf_adaptive_env_cfg.py`         | `CBFV2RewardsCfg` + 5 train env cfgs + 12 eval scene cfgs |
| `cbf_task/__init__.py`                     | registers `Isaac-CBF-Adaptive-Go2-V2*-v0` tasks (5 train + 12 eval) |
| `phase10_train_unified.py`                 | one (arch × intervention_cost) per invocation; optional video |
| `phase10_eval_unified.py`                  | one (policy × scene) per invocation; iterates 6 DR cells; optional video |
| `phase10_aggregate.py`                     | walks per-cell JSONs → one flat CSV |
| `phase10_overnight.sh`                     | shard-based sequential orchestrator (run twice in parallel) |
| `phase10_smoke.py`                         | **pre-flight** sanity test (~3 min); run BEFORE the overnight |

## 5. How to run

### 5.1 Pre-flight smoke test (mandatory, ~3 min)

```bash
cd ~/Desktop/cbf_rl_mvp/go2
~/IsaacLab/isaaclab.sh -p phase10_smoke.py \
    --checkpoint /path/to/locomotion/model.pt --headless
```

Exit code 0 = safe to launch the overnight. Non-zero = stop and read the
listed failures.

The smoke test verifies (1) all 5 train envs construct; (2) obs dims
match per-arch contract; (3) a 50-step rollout produces sane (phi, alpha,
intervention) values; (4) u_nom rate-limit holds (||Δu_nom_w|| ≤ cap +
ε); (5) DR pinning round-trips correctly through PhysX (catches both
bug classes that were fixed in this round); (6) one eval scene of each
of the three obs layouts constructs.

### 5.2 Overnight pipeline (shard-based, two terminals)

```bash
# Terminal A (shard A = even job indices)
bash phase10_overnight.sh /path/to/locomotion/model.pt phase10_outputs A

# Terminal B (shard B = odd job indices)
bash phase10_overnight.sh /path/to/locomotion/model.pt phase10_outputs B
```

Each terminal runs its assigned jobs **sequentially** with a per-job
ETA header. Two terminals = two GPU jobs at once.

`ALL` runs everything in one terminal (~2× longer wall-clock).

### 5.3 Video recording (opt-in)

Recording adds ~10-30% runtime per affected stage. Toggle via env var:

```bash
# both train and eval videos
VIDEO_TRAIN=1 VIDEO_EVAL=1 bash phase10_overnight.sh /path/to/loco.pt phase10_outputs A
VIDEO_TRAIN=1 VIDEO_EVAL=1 bash phase10_overnight.sh /path/to/loco.pt phase10_outputs B

# eval-only videos (cheaper)
VIDEO_EVAL=1 bash phase10_overnight.sh /path/to/loco.pt phase10_outputs A
```

Tweakable: `VIDEO_TRAIN_INTERVAL` (default 15000 env steps ≈ every 5-10 min
during training), `VIDEO_LENGTH` (default 1250 steps = one full 25 s
episode).

## 6. Outputs

```
phase10_outputs/
├── V2Full_int0_0/                  # one per (arch × cost)
│   ├── manifest.txt                # arch, cost, iters, train_seconds
│   ├── rsl_rl/
│   │   ├── model_final.pt
│   │   ├── model_<iter>.pt         # periodic checkpoints
│   │   └── ...tensorboard logs...
│   └── videos/train/               # if VIDEO_TRAIN=1
│       └── rl-video-step-<N>.mp4   # ~5 videos per training
├── V2Full_intn0_05/
├── ...
├── eval_results/
│   ├── eval_V2Full_int0_0_E1Gap.json    # per (policy × scene), 6 cells inside
│   ├── eval_V2Full_int0_0_E2Slalom.json
│   ├── ...
│   ├── eval_baseline_phi0.0_alpha2.5_E1Gap.json
│   └── videos/                          # if VIDEO_EVAL=1
│       └── <policy_label>_<scene>-step-0.mp4
└── phase10_summary.csv             # one row per (policy, scene, dr_axis, dr_value)
```

`phase10_summary.csv` is the one table to plot from — columns:
`policy, arch, scene, dr_axis, dr_value, n, reach_rate, collision_rate, fall_rate, timeout_rate, stuck_rate, safe_reach, time_to_goal_mean, min_h_mean, intervention_mean, time_in_unsafe_frac, phi_mean, phi_std, alpha_mean, alpha_std, jitter_mean`.

## 7. Environment-variable knobs

| var                     | default | meaning |
|-------------------------|---------|---------|
| `NUM_ENVS_TRAIN`        | 2048    | parallel envs at train time |
| `MAX_ITERS`             | 3000    | PPO iterations per training |
| `NUM_ENVS_EVAL`         | 512     | parallel envs at eval |
| `EVAL_STEPS`            | 1250    | env.step calls per eval cell (= 1 episode @ 25 s) |
| `EPS_PER_CELL`          | 512     | episodes averaged per cell |
| `VIDEO_TRAIN`           | 0       | 1 = enable train videos |
| `VIDEO_EVAL`            | 0       | 1 = enable eval videos |
| `VIDEO_TRAIN_INTERVAL`  | 15000   | env.step interval between train videos |
| `VIDEO_LENGTH`          | 1250    | env.step length per video |
| `ISAACLAB`              | `~/IsaacLab/isaaclab.sh` | path to the Isaac Lab launcher |

## 8. Expected wall-clock (RTX 5090, 32 GB)

| stage             | per-job | jobs per shard | shard wall-clock |
|-------------------|---------|----------------|------------------|
| 1 — train         | ~70 min | 5              | ~5.8 h           |
| 2 — eval (policy) | ~10 min (6 cells × ~25 s + ~30 s sim startup) | 20 | ~3.3 h |
| 3 — eval (baseln) | ~10 min | 6              | ~1.0 h           |
| 4 — aggregate     | <1 min  | 0 or 1         | ~0               |
| **total / shard** |         | 31–32          | **~10 h**        |

With video enabled, add ~10-30% per stage.

## 9. What to look for if something looks off

- **All four outcome rates ~0 except timeout=1.0**: env never terminates.
  Check that `CBFTerminationsCfg` was inherited (it is, via parent chain).
  If a future cfg change wiped it, terminations vanish.
- **`reach_rate + ... ≠ 1.0`**: regression in the first-termination
  latch logic. Check the `pending & ... & ...` ordering in
  `phase10_eval_unified._roll_out`.
- **Per-cell `phi_mean` doesn't move across the swept axis**: either the
  policy doesn't adapt, OR the DR pinning isn't reaching PhysX. The smoke
  test's "DR pinning round-trip" check verifies the latter; if smoke
  passes but `phi_mean` is flat across cells, it's a policy issue.
- **`jitter_mean > 0.2`**: Lipschitz rate-limit on (φ, α) overrun —
  check `cfg.action_max_step`.
- **Multiple sticky flags fire per env in a cell BUT outcome buckets
  still sum to 1**: that's expected. Sticky flags are within-cell
  cumulative across multiple auto-resets; the latched outcome reflects
  the FIRST termination only.
- **Eval video shows the robot teleporting / glitching**: render fidelity
  issue, often safe to ignore (cameras can lag physics).
- **Training diag shows `[WARN] phi pegged at LO/HI`** for many
  consecutive prints: policy converged to a degenerate corner. Check
  intervention cost and entropy — the rsl_rl cfg sets `entropy_coef=0`
  intentionally (see the comment in `rsl_rl_ppo_cfg.py`); if you see this
  repeatedly, consider bumping it slightly via cfg override.

## 10. What to redo if something fails

The orchestrator is **idempotent**: rerunning either shard skips trainings
whose `manifest.txt` exists and skips evals whose result JSON exists. Safe
to interrupt and resume. To force a re-run of one cell, delete just its
output and rerun the matching shard — the rest is skipped.
