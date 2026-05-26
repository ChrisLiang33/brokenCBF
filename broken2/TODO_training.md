# CBF Tuner RL — Training Reference

Reference doc covering methodology, theory, and architectural decisions for
the CBF tuner RL training. **For progress tracking and the week-by-week
plan, see [PROGRESS.md](PROGRESS.md).**

Sections:

1. What we're training + observation vector + two training approaches
2. Deploy pipeline + open questions + relevance of RMA
3. Deeper theoretical notes — the 5 CBF parameters, three distributions,
   Lyapunov tube argument, other approaches to uncertainty, in-distribution
   vs OOD

---

## What we're training

A two-network system that outputs 4 CBF parameters (α, φ, a, c).
Parameters feed a CBF-QP filter (cvxpy) that takes a planner's `u_des`
and produces a safe `u_safe`. The CBF math is fixed — only the
parameter-tuning networks are learned.

The 5th slot (b) is reserved in the action space but unused — enabling
it would turn the QP into an SOCP, blocking the OSQP / qpth batching
path. Deferred as an ablation.

## Input structure (teacher vs student see different things)

**Teacher (Week 3)** has ground-truth access to the simulator's
privileged info — nothing else. No LiDAR, no proprioception.

| Component | Example values |
|---|---|
| Friction coefficient | 0.1 … 1.0 per episode |
| Applied push / wind force | 3-vector in world frame |
| Center-of-mass disturbance | 3-vector offset on the base |
| Ground-truth obstacle pose(s) | position + velocity + size |
| Tracking-error statistics | running μ, σ of `‖u_des - base_vel‖` |

These go through an `env_encoder` → latent `Z` → `π_teacher` → 4 CBF
params.

**Student (Week 4)** sees only what the deployed Go2 can measure, plus
history:

| Component | Purpose |
|---|---|
| LiDAR scan (egocentric) | Current environment snapshot |
| base_vel (vx, vy, ω_yaw) | Proprioceptive tracking quality |
| History: past N LiDAR + base_vel | Observable trajectory reveals z |
| History: past N CBF params (actions) | What the policy has been doing |

These go through an `adaptation` module → estimated `Ẑ` → the frozen
`π_teacher` → 4 CBF params.

**Neither network sees `u_des`.** The planner's command reaches the
CBF-QP on a separate wire. CBF params depend on environment, not on
what the user is asking for — the CBF math itself handles u_des.

## Two training approaches

### Approach A — End-to-end domain randomization (simpler)

One policy, trained directly from noisy observations via RL.

Per episode, randomize:
- **Planner:** teleop-jerky, A*, RRT, tangent-bug, adversarial, random noise
- **Environment:** open, corridor, cluttered, dynamic obstacles
- **Disturbances:** wind, slip, push, LiDAR noise, sensor dropout

Reward:
```
reward = -λ₁·collision -λ₂·‖u_safe - u_des‖² -λ₃·infeasibility
```

Algorithms: PPO or SAC.

Policy is "planner-agnostic" because it never sees the same planner twice.

### Approach B — Teacher-Student (RMA-style)

Two-phase training. The teacher learns the "what CBF params should
this environment get?" mapping using ground-truth info; the student
then learns to infer that mapping from observable history alone.

**Phase 1: Train the teacher.**

Teacher gets privileged info directly. That is its *entire input* —
no LiDAR, no proprioception.

```text
privileged z ──▶ env_encoder ──▶ Z (latent)
                                  │
                                  ▼
                            π_teacher ──▶ α, φ, a, c
                                           │
     u_des (from planner) ──▶ CBF-QP ◀────┘
                                │
                                ▼
                             u_safe → locomotion → physics → reward
```

Both env_encoder and π_teacher are trained end-to-end with PPO on the
reward from Approach A. Because the teacher sees ground truth, it
converges fast and stably.

**Phase 2: Train the student.**

Student doesn't have z. It gets the observable state the real Go2 has,
plus a history buffer. An adaptation module infers Ẑ.

```text
obs + history (of obs + past actions)
       │
       ▼
    adapter ──▶ Ẑ                  (frozen) ──▶ α, φ, a, c
                 │                     ▲
                 └──▶ π_teacher ───────┘   (same π_teacher,
                                            frozen from Phase 1)


reference: z ──▶ env_encoder (frozen) ──▶ Z

loss = ‖Ẑ - Z‖²          ← latent loss (RMA standard, supervised)
```

Only the adapter's weights update. Both env_encoder and π_teacher stay
frozen. Pure supervised learning — no reward, no RL instability.

**Deploy:** just the adapter + π_teacher. env_encoder is discarded
(no privileged info exists at deploy).

## Latent loss vs action loss

For Phase 2, two loss choices:

| Loss | Formula | Trade-off |
|---|---|---|
| Latent | `‖e_hat - e‖²` | Direct, stable, sometimes over-penalizes irrelevant latent differences |
| Action | `‖student_params - teacher_params‖²` | Only penalizes behavior differences, but gradient flows through frozen teacher — noisier |

**Use latent loss.** That's what RMA does and what works in practice.

## Which approach first?

**Primary: Approach B (teacher-student RMA-style).** This is the intended
training architecture for the paper. The teacher-student split gives faster
and more stable training when the sim has privileged info (which Isaac Lab
provides), and directly matches the RMA citation in the paper's related work.

Approach A (end-to-end) stays as a fallback if the teacher-student setup has
implementation issues. Phase 2 (supervised adapter training) is much simpler
than full RL, so adding it costs little extra time vs. Approach A.

## Deploy pipeline (on real Go2)

Same for either approach — frozen policy runs each timestep:

```
Go2 sensors ──▶ obs ──▶ policy ──▶ params ──▶ CBF-QP ──▶ u_safe ──▶ walking_bridge
                                              ▲
                                              │
                                             u_des
```

Policy is frozen (weights don't update). Observations still flow in each step.

## Open questions

1. **Which simulator?** Isaac Gym / Isaac Lab / MuJoCo / PyBullet.
   Needs Go2 URDF + dynamics. Lab may already have a setup.
2. **Grid patch size** — 20x20? 40x40? Tradeoff between context and policy size.
3. **Adapter architecture** — LSTM vs 1D conv over history. Either works.
4. **History length N** — how many past steps does adapter need? RMA uses ~50.
5. **Policy head architecture** — MLP after combining grid encoder + velocity + u_des + e.
6. **Real-time inference on Jetson** — must run at ≥10 Hz. Benchmark PyTorch + cvxpy on Orin.
7. **QP infeasibility at deploy** — if cvxpy can't solve, what do we do? Damp?
8. **Scale of z** — what privileged info goes in z? Start minimal (friction,
   disturbance vector, maybe obstacle shapes), add more only if needed.

## Relevance of RMA

RMA (Kumar et al. 2021, Rapid Motor Adaptation for Legged Robots) is a
**locomotion policy** — it replaces Unitree's built-in locomotion, not the CBF
filter. We are NOT building a locomotion policy.

But RMA's teacher-student structure is exactly Approach B above. So when
Professor Michael mentioned RMA, he was pointing to the training recipe, not
the policy itself.

---

# Deeper theoretical notes (from collaborator discussion)

## What the 5 CBF parameters actually mean

From the robust CBF formulation being used:

```
∂h/∂x f̂(x̂) + ∂h/∂x ĝ(x̂)û  -  φ‖L_ĝ h(x̂)‖²  -  a  -  b‖û‖₂  +  α(h(x) - c)  ≥  0
└──────────┬───────────────┘  └─────┬─────┘  └┬┘  └──┬──┘    └────┬────┘
    Nominal CBF term              ACTUATION  MEAS   MEAS       TRACKING /
                                  UNC (φ)   UNC(a) UNC(b)      MODEL ERROR
                                                               (α, c)
```

| Param | Uncertainty type | Intuition |
|---|---|---|
| **α** | Tracking / model error | CBF decay rate. Smaller α = slower decay = more conservative, survives worse tracking. Directly tied to 3DoF↔16DoF gap. |
| **φ** | Actuation uncertainty | Actuators don't perfectly execute commands. Tightens proportionally to how much control influences safety. |
| **a** | Measurement uncertainty (constant) | State estimate x̂ ≠ true x. Fixed offset penalty. |
| **b** | Measurement uncertainty (input-scaled) | State estimation errors amplify at high velocity. Penalty scales with ‖u‖. |
| **c** | Misdefined safety boundaries | h(x) function might not perfectly capture "safe". Shrinks the safe set by c. |

Why hand-tuning fails: worst-case bounds → extremely conservative or infeasible.
That's exactly why we learn these parameters dynamically.

## Three distributions (the framing that matters for your paper)

Not two distributions, three:

1. **Training distribution of the NOMINAL CONTROLLER** — what A*/RRT/teleop
   were designed/tuned for. Usually assumes clean world.
2. **Training distribution of the SAFETY FILTER** — what the RL policy sees
   in sim. Should include disturbances, tracking error, sensor noise.
3. **DEPLOYMENT distribution** — real world at test time.

**Core hypothesis:** the filter has value if (2) adds significant uncertainty
beyond (1). Case A: (2) ≈ (1) → filter does nothing → pointless.
Case B: (2) ⊃ (1) → filter learns to cover what nominal misses → contribution.

**Deployment condition:** (2) and (3) need to be similar. If (3) is static/
known (e.g., always indoor carpet, always Go2), tailor (2) specifically to (3).

## Robot-agnostic architecture, robot-specific training

| Component | Robot-agnostic? |
|---|---|
| CBF-QP solver (cvxpy) | ✓ pure math |
| The 5 params (α, φ, a, b, c) | ✓ same meaning |
| Nominal controller interface | ✓ (vx, vy, ω_yaw) |
| Locomotion policy | ✗ Unitree for Go2, different for G1 |
| **RL teacher (param tuner)** | ✗ must be retrained per robot |

Why teacher retrains: Go2 and G1 differ in **tracking error distribution**,
not just raw dynamics. Locomotion policy + full-body dynamics combine to
produce a robot-specific command-to-motion mapping. Teacher learns params
that account for THAT robot's tracking error pattern.

Summary phrasing: **"robot-agnostic in structure, robot-specific in training."**

## The Lyapunov tube / 3DoF vs 16DoF argument

Professor's formal argument for why the architecture works:

```
3DoF model:   (vx, vy, ω_yaw) — what CBF reasons about
16DoF model:  base pose + 12 joints — the actual robot
Bridge:       locomotion policy (imperfect tracking)
Tube:         bounded region of possible 16DoF states given 3DoF command
```

Two rates compete:
- **α** = CBF decay rate (how fast safety margin allowed to shrink)
- **λ** = Lyapunov tracking rate (how fast 16DoF catches up to 3DoF command)

**Theorem:** if **λ > α**, tube closes faster than safety decays → 16DoF
robot is provably safe despite using 3DoF CBF.

```
h_nom (blue)   = safety value CBF thinks we have (3DoF model)
h_low (red)    = worst-case actual safety (16DoF given tracking error)
Tube (shaded)  = space of possible actual states
```

If tracking is poor, need small α. If tracking is clean, can afford bigger α.
RL learns to make α responsive to observed tracking quality.

## Other approaches to dynamics uncertainty (not competing, complementary)

Four categories of fixes for tracking error / dynamics uncertainty:

1. **RL-tuned CBF params** (what we're doing)
2. **Robust CBF formulations** (ISSf-CBF, tube-based CBF) — bake disturbance
   bounds into the math for formal guarantees
3. **Online adaptation** — RMA-style adapter inferring env params from history
4. **Dynamics model learning** (system ID) — learn f(x, u) online, feed into CBF

State estimation (Kalman, particle filters) is a DIFFERENT problem — it fixes
SENSING uncertainty, not DYNAMICS uncertainty. Still relevant because CBF
consumes state estimates, so better estimates → better CBF decisions.

The Go2's built-in `sportmodestate` already gives a decent state estimate
(IMU + leg odometry + SLAM fusion). Probably don't need our own Kalman filter.

## Training method refinements (from collaborator)

**Discretized parameter actions [up, stay, down]:**
- Clear RL signal, easy to debug
- MUST clamp params to valid range (all 5 are strictly positive)
- Implement: `new_param = clamp(current + delta, min, max)`

**PPO actor-critic:**
- Standard default. Actor = param output. Critic = value estimation.

**Curriculum strategies:**
- Environment difficulty: start no-disturbance → ramp up
- Parameter learning: try freezing/unfreezing individual params (5 is small
  enough to explore orderings). Joint learning probably fine first attempt.

## Core paper-level framing

> "CBF layer's value = what distribution (2) covers beyond distribution (1)."

This is the one-line justification for the whole pipeline. If (2) doesn't
extend (1), there's no contribution. The paper's job is to show that
learned, dynamically-tuned params operate effectively in the (2) \ (1)
region — handling disturbances the nominal controller can't.

## In-distribution vs OOD — what it actually means

Common confusion: "in-distribution" does NOT mean `u_des = u_safe`. It
means: the current situation looks like what the RL policy saw during
training, so the policy trusts its parameter output. CBF may still
intervene in-distribution (if u_des would cause collision). OOD means
parameters may be wrong — risk of under- or over-correction.

At deploy: you cannot detect OOD reliably. The point of robust training is
graceful degradation without needing detection. Validation: hold out
specific conditions (e.g., friction outside training range), measure
performance gap vs. in-distribution baseline.

---
