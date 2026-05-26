# Methods section outline — v2.14 baseline

Planning doc for the CoRL submission's methods section. Reflects design
decisions locked in through 2026-05-09. Update inline as v2.14 results
land.

**v2.14 deltas vs v2.12 (the 2 fixes folded in 2026-05-09):**

1. **Per-episode φ lock** (`USE_PER_EPISODE_PHI = True`). Caught from
   v2.12 Bf-φ inversion on HighDisturbance: per-step Gaussian sampling
   on φ creates LLN/CLT-style jitter without state-conditional gradient
   signal (PPO advantage averages over 800-step episode → individual
   per-step φ_t contributes ~1/800 to return → policy converges to
   "noisy mean"). Lock φ at episode start, replay for episode. Same
   shape of fix as v2.12 perception-bias fix on the input side; this
   one is on the output side.
2. **Train under shield_v0a perception** (`perception_mode =
   "shield_v0a"` default in `CbfGo2EnvCfg`). Caught from Goal B v0a
   smoke test: v2.12 ckpt under SHIELD-style perception flipped from
   +7.05 → -3.5 pp because policy outputs were calibrated for true
   radii but QP saw uniform R=0.3m. Train under deploy-side perception
   so policy learns to compensate for the R=0.3m QP assumption.

The existing `main.tex` has a methods skeleton (Section III) — this doc
expands and re-sections it to match what we actually have implemented
post-v2.12. Map: each section here maps to a subsection in `main.tex`
under "Technical Approach / Methodology / Theoretical Framework" or
"Experimental Setup."

---

## 1. Problem formulation

**1.1 Setting.** Quadrupedal safety filter in front of a black-box
locomotion policy. Drop-in philosophy — the locomotion controller is
fixed, frozen, and not retrained per robot.

**1.2 Three-distribution framing.** Following the framing from the
project's deeper theoretical notes (TODO_training.md):

- **D_1 — nominal controller distribution:** what A*/RRT/teleop were
  designed and tuned for. Clean world.
- **D_2 — safety filter training distribution:** what the RL teacher
  sees in sim. Includes friction, force, COM, motion, and now
  perception-uncertainty disturbances.
- **D_3 — deployment distribution:** real Go2 at test time.

Hypothesis: filter has value if D_2 ⊃ D_1; filter generalizes if
D_2 ≈ D_3. v2.12's perception-noise DR is the design decision that
expands D_2 along the perception-uncertainty axis.

**1.3 Robot model.** Single-integrator `ẋ_robot = u` for the CBF math
(L_g h ≠ 0 → standard CBF applies). Not switching to torque-level /
acceleration-input model in this work.

**1.4 Locomotion abstraction.** Pre-trained Isaac Lab flat policy,
frozen black box. CBF outputs `u_safe ∈ R³` (vx, vy, ω_yaw); locomotion
policy maps to 12-D joint targets. Native fall rate 0.5%; in CBF env
at B0 ~20% due to 50 Hz vs 10 s training distribution.

---

## 2. CBF formulation

**2.1 Safe set construction.**

- Per-shape analytical SDF (cylinder-only, post-v2.12 commitment).
- Multi-obstacle min-SDF (Eq. 19).
- Exponential smoothing (Eq. 20): `h(p) = λ · (1 − exp(−γ · sdf(p)))`.
- Minkowski-expanded by robot half-footprint (0.15 m).

**2.2 Robust QP constraint.** Closed-form half-space projection on GPU:

$$L_g h \cdot u_\text{safe} \;\geq\; -\alpha\,(h - c) \;+\; \varphi\,\lVert L_g h\rVert^{2} \;+\; a$$

**2.3 L_f h obstacle-drift augmentation (v2.11 onward).** True ḣ has
two contributions:

$$\dot h \;=\; L_g h \cdot u \;-\; L_g h \cdot v_\text{obs}
        \;=\; L_g h \cdot (u - v_\text{obs})$$

where the second equality uses the symmetry $\partial h / \partial p_\text{obs} = -L_g h$
for our $h(p_\text{robot} - p_\text{obs})$. Substituting into $\dot h \geq -\alpha(h-c) + \varphi\,\lVert L_g h\rVert^2 + a$ and rearranging gives the QP constraint:

$$L_g h \cdot u_\text{safe} \;\geq\; -\alpha(h-c) + \varphi\,\lVert L_g h\rVert^2 + a + L_g h \cdot v_\text{obs}$$

**2.4 4-parameter action space.** $(\alpha, \varphi, a, c)$ active;
$b$ reserved for input-dep slack (would require SOCP solver, deferred).

**2.5 Uncertainty class mapping.** Each parameter is grounded in a
specific robust-CBF literature uncertainty class:

| Param | Range (training) | Uncertainty class | Citation |
|---|---|---|---|
| α | [0.1, 5.0] | Model / tracking error | Molnar 2021 |
| φ | [0.0, 5.0] | Actuation uncertainty | Kolathaya 2018 (ISSf) |
| a | [0.0, 3.0] | Measurement uncertainty (state-indep) | Dean 2019 |
| c | [0.0, 1.0] | Boundary correction | — |

---

## 3. Per-step adaptive policy

**3.1 Action space.** Policy outputs $(\alpha_t, \varphi_t, a_t, c_t)$
**every timestep** at 50 Hz. Distinguishes from per-episode α-tuning
baselines (e.g., SHIELD's Freedman/Brent calibration).

**3.2 Privileged observations (teacher only).** PRIV-2: 8207-D total.
- 15-D dynamics block (proprioception, base velocity, COM offset, mass,
  friction estimate).
- 8192-D occupancy grid: 2 frames × 64 × 64 ego-centric, 0.1 m cells,
  6.4 m FOV. Two-frame stack lets the encoder infer obstacle motion.

**3.3 Network architecture.**
- Encoder: priv-obs → CNN [Conv 2→16 s=2, 16→32 s=2, Linear→64], dyn
  MLP [15 → 64] → concat → Linear 128 → 12 → Z(12).
- Bottleneck: $Z \in \mathbb{R}^{12}$ (RMA-style; `get_z(obs)` exposed
  for Wk3 student replay).
- Policy head: $\pi_\text{teacher}: Z \to (\alpha, \varphi, a, b, c)$,
  128 hidden, output dim 5, tanh-squash → physical-range scale.

**3.4 `u_des` is sidecar, not input.** Planner command never enters the
network. Used only in reward via $\lVert u_\text{safe} - u_\text{des}\rVert^2$.
CBF params should depend on environment, not on user's current command.

**3.5 Per-episode φ lock (v2.14).** Despite the policy outputting
$(\alpha, \varphi, a, c)$ at every timestep, the φ output specifically
is **frozen for the full episode** at first-step capture. Captured
value $= \pi(z_0) + \varepsilon_0$ (priv-obs latent at reset + Gaussian
exploration noise from action sampling), then replayed unchanged for
the remaining ~800 steps. Same exploration-noise structure as the
unlocked policy ($\varepsilon \sim \mathcal{N}$); the difference is
that the noise is between-episode (good for exploration), not
between-step (harmful for the constraint).

*Why φ specifically.* φ scales the actuation-uncertainty robust-CBF
term $\varphi \lVert L_g h\rVert^2$. Actuation uncertainty (motor lag,
friction, locomotion-policy tracking error) is an
**environment-class** property — it doesn't change per step within an
episode. So per-step adaptation of φ has no useful gradient signal
(PPO's advantage averages reward across the episode → any one φ_t
contributes ~1/800 to the return). The other slots (α model error, a
measurement bias, c boundary correction) all have state-conditional
adaptation gradients, so they stay per-step.

Diagnostic that motivated this: v2.12 Bf-φ on HighDisturbance lost by
−7.6 pp (clamping φ to its mean *beat* per-step adaptive φ).
Predicted v2.14 outcome: Bf-φ flips to positive (load-bearing).

---

## 4. Domain randomization

Per-episode unless noted.

**4.1 Friction.** Static / dynamic ranges $(0.30, 1.20) / (0.20, 1.00)$.
Exercises α (model error).

**4.2 External force / torque.** $\pm 10$ N / $\pm 2$ Nm. Exercises φ
(actuation uncertainty).

**4.3 COM offset.** $\pm 5$ cm xy / $\pm 3$ cm z. Indirectly exercises α
via tracking-error coupling.

**4.4 Obstacle motion (variable v_obs).** Per-episode max-speed sampled
in $[0, 0.4]$ m/s, then per-obstacle random direction. Sometimes static
scenes, sometimes fast-moving. Pairs with L_f h drift term (§2.3).

**4.5 Per-episode persistent perception bias (v2.12 — load-bearing for
`a`/`c`).** Each episode, each env samples:

- $\sigma_e \sim \text{Uniform}(0, \sigma_\text{max})$ — episode noise level
- $\varepsilon_{e,k} \sim \mathcal{N}(0, \sigma_e^2 \cdot I_2)$ for each
  obstacle $k$ — **persistent** for the whole episode

The QP sees biased obstacle positions $p_k^{QP} = p_k + \varepsilon_{e,k}$;
the priv-obs grid stays clean. Per-episode persistence (NOT per-step
i.i.d.) is critical: i.i.d. noise would average to zero under LLN over
1000-step episodes, leaving `a` slot with no gradient signal.

This noise model matches the **biased** error structure of a real
LiDAR + cluster + analytical-SDF pipeline (cf. Yang et al. 2025, SHIELD,
Section V): SHIELD pipes Livox Mid-360 returns through Euclidean
clustering (PCL) and treats each cluster as a cylinder of FIXED
$R = 0.3$ m, then computes analytical min-SDF (their Eq. 19) on those
cluster centers. The cluster's center estimate is stable across LiDAR
frames, so deploy-time perception error is a temporally coherent bias
on cluster centroid (which is what `a` and `c` are designed to absorb)
rather than per-step jitter.

**4.6 Bimodal planner-resample DR.** Each episode rolls:
- $P=0.5$: mid-switch every $\text{Uniform}(5, 15)$ s — restores
  v2.6's stuck-recovery regularizer.
- $P=0.5$: locked for full 100 s — deploy-realistic single-nav-stack.

Eval-time always forces locked; the train/eval mismatch is the design
choice. Training mid-switch teaches intrinsic recovery; locked eval
verifies it transfers.

**4.7 Multi-planner mix.** Per-episode planner sampled from
{smooth_goal 0.40 / waypoint 0.25 / mpc 0.20 / legacy_goal 0.05 /
walk 0.05 / adversarial 0.05}. Restored to v2.6 mix in v2.12 after
PLANNER-2b ablation showed no benefit from dropping walk + adversarial.

**4.8 Obstacle scene composition.** $K_\text{actual} \in [0, 20]$
uniform per reset; UNIQUE indices drawn from a 20-cylinder pool, radii
$0.10$–$0.50$ m. SHIELD-style commitment to cylinders for analytical
SDF + train/deploy parity.

**4.9 SHIELD-style perception model (v2.14 default).** The QP's view
of the world is forced through a SHIELD-equivalent simplification:
**every obstacle is modeled as a fixed-radius cylinder $R = 0.3$ m,
regardless of its true radius.** The policy still sees true geometry
via priv-obs (8192-D occupancy grid, §3.2); only the QP loses the
radius information.

*Why force this at training.* SHIELD's deploy-time pipeline does
exactly this: LiDAR returns are clustered (PCL Euclidean clustering),
each cluster is treated as a $R = 0.3$ m cylinder. Real LiDAR cannot
recover obstacle radius reliably, so SHIELD avoids per-cluster fitting
and applies a uniform fixed radius. Training under
`perception_mode = "shield_v0a"` means the policy learns CBF-param
outputs that are calibrated for the QP using R=0.3m everywhere — exactly
the assumption that holds at deployment.

*Empirical motivation.* v2.12 ckpt evaluated under shield_v0a perception
flipped from +7.05 (priv) to −3.5 pp (shield) — under-padding big
obstacles because the policy expected the QP to know R=0.50m geometry
that the QP no longer sees. v2.14 closes that gap by training under
the deploy-side assumption.

---

## 5. Reward shaping

| Term | Weight | Trigger | Type |
|---|---|---|---|
| collision | $-100$ | per-shape SDF $< 0$ | terminal |
| base_contact_penalty | $-100$ | fall (REWARD-2 NEW post-v2.6) | terminal |
| stuck | $-2.0$/step | $\lVert v_{xy}\rVert < 0.15$ m/s (REWARD-2 NEW) | shaping |
| infeasibility | $-10$/step | QP infeasible | shaping |
| u_safe_deviation | $-0.1\cdot\lVert u_\text{safe}-u_\text{des}\rVert^{2}$ | per-step | shaping |
| proximity | $-0.5\cdot\exp(-\text{min\_sdf}/0.5)$ | per-step (REWARD-2 halved from $-1.0$) | shaping |
| action_rate | $-0.005\cdot\lVert\Delta a\rVert^{2}$ | smooths CBF params | shaping |

**Design principle: DR-implicit parameter shaping.** No reward terms
explicitly target $(\alpha, \varphi, a, c)$ behavior. The DR axes (§4)
provide the disturbance; the parameters learn to absorb it. Cleaner
paper claim: "DR creates the disturbance, the parameter learns to
compensate" rather than hand-crafted reward signals per slot.

---

## 6. Training procedure

**6.1 PPO recipe.**
- AdamW optimizer, weight_decay $= 1\mathrm{e}{-5}$ (monkey-patched from
  rsl_rl PPO; smooths input→output map).
- entropy_coef $= 0.005$ (5× v2.5; load-bearing — prevented
  action-std collapse).
- Action-rate penalty $-0.005 \cdot \lVert \Delta a \rVert^2$ (smooths
  in time; orthogonal to weight-decay).
- 4096 envs, 5000 iterations (~5 h on RTX 5090, SPS ~27K).
- Episode length 20 s.

**6.2 Episode reset hook.** Per-episode bias and σ sampling (§4.5)
happen in `_reset_idx` so they persist for the whole episode.

**6.3 Training health logging.** 14 CBF stats surfaced into rsl_rl per-iter log:
- 8: $(\alpha, \varphi, a, c)$ mean/std — diagnose slot adaptation.
- 2: $h_\text{min}$, $h_\text{mean}$ — closeness-to-obstacle distribution.
- 2: `qp_active_rate`, `u_safe_clamp_rate` — QP + actuator behavior.
- 2: noise-σ mean/std — confirm DR active.

---

## 7. Evaluation methodology

**7.1 Baselines.**
- **B0**: fixed $(\alpha, \varphi, a, b, c) = (0.5, 0, 0, 0, 0)$.
  Plain CBF, no robust slacks.
- **B1**: B0 + tuned $\varphi \in \{0.5, 1.5, 3.0\}$. ISSf form.
- **B2**: TISSf form, varying $(\alpha, \varepsilon_0, \lambda)$.
- All baselines hand-tuned over a grid; reported = best per task.

**7.2 BR (ours).** Per-step adaptive 4-param policy from RL teacher.

**7.3 B-fixed-X ablations (paper Table 2).** Lesion study: at eval
time only, clamp ONE slot to a fixed value (mean of training-time
output for that slot); other 3 still adaptive. Tests whether per-slot
adaptation is load-bearing on each OOD axis.

| Bf-X | Fixed value | Paired OOD axis |
|---|---|---|
| Bf-α | 2.5 | DensePack, in-dist |
| Bf-φ | 2.5 | HighDisturbance |
| Bf-a | 1.5 | NoisyPerception (v2.12 NEW) |
| Bf-c | 0.5 | HeavyCOM, FastObstacles |

**7.4 OOD eval suite.** 7 single-axis pushes (each one DR axis past
training edge) + 1 compositional. Plus NoisyPerception (v2.12).

| Eval | Type |
|---|---|
| In-distribution | mixed |
| DensePack | scene-only (h(x) symmetric) |
| Slippery | priv-obs (continuous friction) |
| HighDisturbance | priv-obs (episodic force/torque) |
| HeavyCOM | priv-obs (startup COM bias) |
| FastObstacles | priv-obs (motion DR) |
| RealisticCompound | all 5 single-axis pushes |
| NoisyPerception | priv-obs (perception σ) — v2.12 |

**7.5 Metric.** Combined fall + stuck rate (lower is better). 64 envs
per config × 2000 sim steps ≈ ~150 episodes per row. Same `--num_envs`,
`--steps_per_config`, seeded RNG across configs for reproducibility.

---

## 8. Sim-to-real considerations (Wk4 / future work)

**8.1 Train/deploy h(x) parity (v2.14 onward).** v2.14 trains under
`perception_mode = "shield_v0a"` — the QP is given uniform $R = 0.3$ m
cylinders at training time, matching SHIELD's deploy-time choice.
This closes the *radius-mismatch* component of the sim2real perception
gap. Combined with v2.12's per-episode persistent perception bias
(§4.5), the QP-side training distribution now matches SHIELD's
deploy-time perception in two ways: (1) cluster centers are biased by
a temporally-coherent offset (`a` slot absorbs this), (2) all clusters
get fixed $R = 0.3$ m regardless of true geometry (policy learns to
output radius-mismatch-robust params).

**What's still missing.** Currently v2.14 uses ground-truth obstacle
*positions* (only the radii are SHIELD-ified). Real LiDAR also
introduces:
- Cluster-merge errors (two close obstacles → one cluster)
- Cluster-split errors (one obstacle → two clusters at different LiDAR
  aspects)
- FOV gating (obstacles outside ~6 m disappear) — partially modeled by
  shield_v0b
- Occlusion (distant obstacles hidden behind near ones)

These are addressed by `shield_v0c` (full synthetic LiDAR raycast +
grid clustering) — a stub in `cbf_go2_perception.py` for follow-up.
~150 lines, 1-2 days. Not blocking for paper submission since v0a/v0b
already capture the radius-mismatch structural gap.

**Note vs. SHIELD's pipeline.** SHIELD applies the same fixed
$R = 0.3$ m we now use (arXiv:2505.11494 Section V). Differentiation
vs SHIELD: (1) per-step 4-param adaptive CBF vs SHIELD's per-episode
single-α calibration, (2) we add `a` and `c` slots tied to
measurement-uncertainty and boundary-correction literature classes,
(3) we additionally augment with $L_f h$ obstacle-drift term (§2.3)
that SHIELD does not include.

**8.2 Student distillation (Wk3).** Frozen $\pi_\text{teacher}$ + adapter
network mapping (LiDAR + base_vel + history) → $\hat Z \in \mathbb{R}^{12}$.
Latent loss against frozen env_encoder. Standard RMA pattern.

**8.3 Hardware deployment (Wk4 stretch).** TorchScript export + Jetson
Orin onboard inference. Drop-in safety filter wraps Unitree's existing
locomotion stack via `walking_bridge`.

---

## 9. Honest limitations / future work

- **`b` slot reserved but unused.** Input-dependent slack would require
  SOCP solver; OSQP closed-form half-space projection breaks. Keep as
  ablation row in revised version.
- **HOCBF.** Only relevant if we switch to torque-level control. Current
  single-integrator model has $L_g h \neq 0$ so standard CBF applies.
- **Real LiDAR-cluster perception pipeline (`shield_v0c` — partial).**
  v2.14 trains under shield_v0a (radius mismatch + ground-truth pos);
  shield_v0c would add the full raycast → cluster → cluster-pos →
  analytical SDF pipeline that introduces cluster-merge / cluster-split
  / FOV-gating / occlusion failure modes. Stubbed in
  `cbf_go2_perception.py`; deferred for follow-up. Per-cluster radius
  fit (going beyond SHIELD's fixed $R = 0.3$ m) is a further extension.
- **Arbitrary obstacle shapes.** Cylinder commitment loses generality vs
  v2.13 grid distance transform (also deferred). Restriction matches
  SHIELD's deploy assumption; future work (post-camera-ready) extends
  to general shapes.
- **Tracking error δ.** Empirical measurement of robot velocity tracking
  error pending (Wk4 polish). Currently absorbed implicitly by α.
