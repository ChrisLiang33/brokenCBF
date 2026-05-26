# CoRL 2026 paper outline (V13.1)

**Deadlines**: Abstract 5/25 · Body & finalize 5/25–5/28

## Working title (pick one to refine)

- *Two-Stream Adaptive Control Barrier Function Parameter Learning for Sim-to-Real Quadruped Navigation*
- *Adaptive CBF Parameters via RMA-Style Distillation: Sim-to-Real Transfer on a Quadruped*
- *Learning Adaptive Robust-CBF Parameters with a Two-Stream Encoder and Proprio-Noise Sim-to-Real Bridge*

## Headline claim

A learned policy outputs per-step adaptive CBF parameters (α, φ) that
**(a)** beat hand-tuned fixed parameters on deploy-realistic distributions in
simulation with statistical confidence, **(b)** can be distilled into a student
that requires only deployable sensors at inference, **(c)** transfers to a
Unitree Go2 quadruped and triggers correct safety behavior on real LiDAR data.

## One-paragraph abstract draft (revise after multi-seed)

> Standard Control Barrier Functions (CBFs) require hand-tuned hyperparameters
> (recovery rate α, ISSf margin φ, perception correction c) that trade off
> safety for completion across deployment conditions. We learn α and φ as
> per-step outputs of a reinforcement-learning policy with a **two-stream
> privileged encoder**: truly-hidden environmental state (friction, mass,
> applied disturbances) is bottlenecked through a small latent z_env, while
> observable proprioceptive signals (base height, tracking error, IMU
> angular velocity) pass through directly. A LiDAR occupancy grid is encoded
> separately. We train under domain randomization with realistic
> proprioceptive sensor noise; this regularization both prevents
> overfitting to ground-truth values and bridges the sim-to-real gap. The
> teacher is distilled into a temporal-convolution student that estimates
> z_env from a history of observable proprio + actions. We show, with
> multi-seed evaluation on three deploy distributions, that the adaptive
> policy beats hand-tuned baselines on trainmatch and OOD distributions
> while the distilled student matches the teacher closed-loop. We deploy
> the student on a Unitree Go2's Jetson Orin at 50 Hz with real Livox LiDAR
> and show the safety filter intervenes correctly on a real obstacle.

(~180 words; CoRL allows ~250.)

## Contributions

1. **Two-stream privileged encoder** that separates hidden env state from
   deployable proprio. Cleaner distillation target (only the truly hidden
   part needs to be estimated by the student).
2. **Proprio-noise regularization at training time** matched to real Go2
   sensor specs. Acts as a sim-to-real bridge AND improves closed-loop
   performance vs. clean-input training.
3. **Multi-seed evaluation protocol** that controls for episode randomness
   via per-run env seeding (we found 5–15 pp eval-to-eval drift without
   this; common pitfall in CBF + RL papers).
4. **Hardware demonstration** on a Unitree Go2: distilled student runs at
   50 Hz on Jetson Orin, real LiDAR-driven, with verified CBF intervention.

## Method

### 1. CBF formulation
- ḣ + α·(h − c) ≥ φ·‖L_g h‖² + a
- (α, φ) adaptive per-step; (a, b, c) fixed hyperparameters for this paper
  (acknowledged as a limitation — future work)
- Closed-form half-space projection in 2D `u` — no QP solver needed
  (single linear constraint)

### 2. Two-stream architecture
- `priv_obs[:14]`  → MLP encoder → `z_env ∈ R^16`     (HIDDEN env)
- `priv_obs[14:33]` → running-mean-std normalize → `z_proprio ∈ R^19`  (OBSERVABLE, passthrough)
- `grid` (2×64×64) → CNN encoder → `z_grid ∈ R^64`    (LiDAR perception)
- π_teacher: MLP `(z_env ⊕ z_proprio ⊕ z_grid) → action`

### 3. Training regimen (PPO)
- ≈4096 envs, ~2500 iterations, Isaac Lab + rsl_rl
- Domain randomization: friction, mass, applied force/torque, com_offset,
  σ_actuation, δR (radius error)
- **Proprio noise** on observable channels matching Go2 sensor specs:
  - base_height σ=0.02 m
  - tracking_err σ=0.05 m/s
  - base_ang_vel σ=0.015 rad/s
- Per-step φ, per-step α, c fixed at −0.05 (hyperparameter)

### 4. Student distillation
- Architecture: Kumar 2021-style 1-D conv over 50-step history of
  (proprio, prev_action). Output ẑ_env (16-D).
- Loss: MSE(ẑ_env, z_env_teacher) on stored rollout trajectories.
- Closed-loop test: BS-A bridge swaps z_env for ẑ_env, runs same teacher
  pi_teacher + grid_encoder + proprio passthrough.

## Experiments

### Sim distributions (per multi-seed × 3 seeds)
- **trainmatch**: deploy-realistic env, same DR as training
- **OOD**: deploy-realistic env, DR shifted (narrower σ_act, low friction, frozen c)
- **stressor**: adversarial DR (wider than training)
- *(in-distribution dropped from headline — too easy, not deploy-relevant)*

### Baselines (sim & hardware)
- **B0**: fixed α (3 values), no φ
- **B1**: fixed α + φ (3×2 grid)
- **B2**: state-conditional α/φ via exp decay (hand-tuned)
- **BR**: V13.1 teacher (ours, with ground-truth z_env)
- **BS-A**: V13.1 student adapter (ours, deploy-realistic, ẑ_env from history)

### Headline metrics
- **Composite**: goal_reach × (1 − collision) × (1 − fall)
- **Pareto breakdown**: safety vs. completion vs. efficiency vs. h-margin

### Diagnostics (paper supplement)
- Head sensitivity per priv channel (α/φ vs. each priv feature)
- z_env linear probe R² per priv channel (encoder quality)
- Grid-to-priv gradient ratio (the "36% / 21% / 36%" gating breakdown)

### Hardware demonstration
- Go2 + Mid360 LiDAR + Jetson Orin
- Static obstacle (cardboard box) in clear hallway
- Three conditions:
  - **Raw teleop** (no CBF, unsafe baseline)
  - **Fixed-param CBF** (B1: α=2.0, φ=0.5)
  - **V13.1 BS-A student** (ours)
- ~5–10 trials each. Metrics: closest distance to obstacle, completion
  rate, velocity along cmd, deflection magnitude.

## Results to fill in (after multi-seed + hardware)

| Distribution | B0 best | B1 best | B2 best | BR (ours, μ±σ) | BS-A (ours, μ±σ) |
|---|---|---|---|---|---|
| trainmatch   |   |   |   | **0.86 ± ?** | **? ± ?** |
| OOD          |   |   |   | **0.80 ± ?** | **? ± ?** |
| stressor     |   |   |   | 0.66 ± ?    | ? ± ?     |

| Hardware (per condition) | Raw | B1 fixed | V13.1 student |
|---|---|---|---|
| Closest distance (m)  |   |   |   |
| Completion rate       |   |   |   |
| Mean v_along_cmd (m/s)|   |   |   |
| Mean deflection       |   |   |   |

## Limitations (be honest, lands better with reviewers)

- σ_actuation gating barely emerges (∂α/∂σ ≈ 0.003) despite being the
  textbook target for φ via ISSf theory. Framed as "multiple valid
  solutions" — closed-loop performance is sufficient validation.
- (a, b, c) kept as hyperparameters. Wider adaptive parameter set is future
  work and requires SOC-capable QP solver (currently closed-form 2D).
- Stressor distribution: BR loses to best fixed by ~13 pp. Wider DR
  during training (V12/V14 attempts) was deprioritized to focus on
  hardware deployment.
- In-distribution: BR loses to best fixed by ~8 pp. Adaptive parameters
  cost a small amount of "easy case" performance for big gains on
  deploy-realistic conditions.

## Figures plan

- **Fig 1**: System diagram (two-stream encoder + LiDAR grid + student distillation)
- **Fig 2**: Multi-seed headline bar chart (composite BR vs. best fixed, 3 distributions, error bars)
- **Fig 3**: Adaptation traces (α/φ/h over time during a single rollout with push events) — use the side-by-side comparison plot for 4 envs
- **Fig 4**: Head sensitivity breakdown (the 36/21/36% gating)
- **Fig 5**: Hardware setup photo + trajectory plot (Go2 with vs. without filter, top-down)
- **Fig 6 (supplement)**: z_env linear probe R² per priv channel

## Schedule

| Day | Task |
|---|---|
| **Today (5/21)** | Multi-seed V13.1 sweep overnight (~2h) |
| **Wed 5/22** | Pull multi-seed; hardware Tier 2 (robot walks) + Tier 3 (obstacle test) |
| **Thu 5/23** | Lock MVP experiment; draft abstract; first plots |
| **Fri 5/24** | Refine abstract; figure-quality plots |
| **Sat 5/25** | **Submit abstract** |
| **Sun 5/26 – Tue 5/28** | Body sections, related work, polish, supplementary |

## Open questions

- **Hardware experiment details**: 1 obstacle or multiple? Hallway with end-goal or open lab? Confirm by Wed.
- **Student vs. teacher hardware demo**: only the *student* is deployable (teacher needs hidden priv). Should hardware section show student only, or also offer a "if you had ground-truth perception" teacher comparison via Vicon?
- **Real-LiDAR sim-to-real bridge**: V15 (noised LiDAR in training) was on backlog. Now deprioritized. Acknowledge gap in limitations.
