# Learned Adaptive CBF — MVP & Handoff

A minimal, runnable testbed for one research claim:

> A reinforcement-learned, **state-adaptive** parameterization of a
> robustified Control Barrier Function is minimally invasive in normal
> conditions and more robust out-of-distribution than any single
> hand-tuned (fixed) parameterization.

This repository is an **MVP** — it deliberately strips the problem to
its core so the *idea* can be validated or falsified in seconds, not
weeks. There is no legged robot, no Isaac Lab, no learned dynamics, no
navigation planner here. The last section explains how what was learned
here should shape the full Isaac Lab + Go2 pipeline.

---

## 1. The idea in one paragraph

The robustified CBF constraint has free parameters — in the full project
`(phi, a, b, c, alpha)`. Normally these are hand-tuned constants. The
proposal is to learn a *policy* that emits them as a function of the
current situation, so conservatism adapts: tight near obstacles or under
high uncertainty, relaxed when the nominal command is already safe. The
policy is trained with PPO. Crucially the policy does **not** drive the
robot — it parameterizes a QP safety filter, the QP produces the command,
the robot executes it. The QP and the dynamics together are the RL
environment; the policy acts *through* the optimizer, so no gradients
ever flow through the QP and PPO only needs sampled rollouts.

---

## 2. What this MVP models

- **Robot**: a 2D single integrator, `x_dot = u + d`, in a 10x10 world.
  A stand-in for *command-space control* — the CBF lives in command
  space, so nothing in the contribution depends on what executes the
  velocity command. (On the Go2, the locomotion policy executes it.)
- **Barrier**: per obstacle `h_i = ||x - p_i|| - r_safe_i`. The gradient
  is a unit vector, so `||L_g h_i||^2 = 1` and the robustified constraint
  collapses to one linear inequality per obstacle:
  `grad_h_i . u  >=  phi - alpha*(h_i - c) + a + grad_h_i . v_i`
  where `v_i` is the obstacle velocity (zero for static obstacles; the
  `grad_h_i . v_i` term is the time-varying `dh/dt` compensation).
- **Safety filter**: a per-constraint-slack-relaxed QP,
  `min ||u - u_nom||^2 + M * sum(delta_i^2)`, built once with cvxpy
  Parameters and re-solved each step. Slack guarantees feasibility.
- **What the policy learns**: `pi(obs) -> (phi, alpha)`, emitted every
  step. (The MVP learns 2 of the 5 parameters; `a, b, c` are fixed at 0.)
- **Disturbance** `d`: the out-of-distribution dial. Fixed for
  in-distribution work; sampled per-episode from a range when the policy
  must learn to condition on it.
- **One moving obstacle**: optional (config toggle). Oscillates faster
  than the robot's top speed; the QP carries the correct `dh/dt` term.

---

## 3. Quickstart

```
pip install -r requirements.txt

python phase0_fixed_cbf.py       # instant -- QP/dynamics sanity, no RL
python phase1_train_constant.py  # ~120k steps -- reward + PPO loop check
python phase2_train_adaptive.py  # ~200k steps -- efficiency (Pareto)
python phase3_ood.py             # ~200k steps -- robustness (the headline)
```

Every training script accepts `--quick` (a few-second smoke test) and
`--timesteps N`. `train_ppo` supports `init_from=<path>` to resume a
saved model, so long runs can be split across time limits. Outputs are
PNGs and saved `.zip` models in the working directory.

---

## 4. File map

```
config.py                 all hyperparameters + obstacle layout (one dataclass)
cbf_qp.py                 robustified multi-obstacle CBF-QP filter (cvxpy)
env.py                    Gymnasium env: QP + dynamics + reward
utils.py                  rollout / evaluation / PPO training / plotting
phase0_fixed_cbf.py       Phase 0 -- fixed CBF, no learning
phase1_train_constant.py  Phase 1 -- PPO recovers the best constant params
phase2_train_adaptive.py  Phase 2 -- state-conditioned policy, Pareto + traces
phase3_ood.py             Phase 3 -- out-of-distribution robustness
results/                  reference figures (alpha_diagnostic, phase0)
requirements.txt
```

---

## 5. The phase ladder (and why it is a ladder)

Each phase isolates exactly one failure mode, so a bad result is
unambiguous. **Do not skip Phase 1**: if PPO cannot even recover a good
*constant* parameter set (independently checkable by grid search), a bad
Phase 2 result cannot distinguish "adaptivity does not help" from "the
reward or training loop is broken."

| Phase | Question it answers |
|-------|--------------------|
| 0 | Do the dynamics and QP work -- does a fixed CBF stay safe? |
| 1 | Are the reward and PPO loop sound -- does PPO find the grid-best constant? |
| 2 | Does a state-conditioned policy beat the fixed-parameter Pareto frontier? |
| 3 | Does the adaptive policy degrade gracefully past the training disturbance? |

---

## 6. CHECKPOINT -- what is established vs. what is shown

This separates **theory** (mathematical / structural facts, true beyond
this MVP) from **empirical results** (demonstrated in this MVP, with the
honest magnitude). Treat the empirical numbers as MVP-scale evidence,
not final claims.

### 6a. Established -- theory / structural

1. **Command-space reduction.** With a distance barrier the robustified
   CBF constraint reduces, per obstacle, to a single linear inequality
   in the command `u`. Exact; this is why the toy is faithful to a
   controller-agnostic design -- the constraint structure does not
   depend on the robot underneath.

2. **QP-as-environment is well-posed.** The policy parameterizes the QP;
   the QP (non-differentiable, slack-relaxed) plus the dynamics form the
   MDP transition. PPO optimizes a sampled return; no gradient passes
   through the optimizer. Validated end-to-end; transfers unchanged to
   the Go2.

3. **phi and alpha are different *kinds* of parameter.** The single most
   important finding for the 5-parameter plan:
   - `phi` (and by the same logic `a`, `b`) is a **robustness margin** --
     a hedge against uncertainty (`phi ~ 1/epsilon`, the `L_g-hat h`
     error). How much the robot does not know genuinely varies with
     state, so these have a real **state-dependent optimum** and a
     policy can learn to adapt them.
   - `alpha` is the **class-K gain**. For a *correctly formulated* CBF
     (including the `dh/dt` term for moving obstacles), `alpha` does not
     affect safety or feasibility -- only conservativeness, and
     **monotonically**. There is no state in which a different `alpha`
     wins, so a policy correctly leaves it pinned at a bound.
   Verified empirically (6b.4); a CBF-theory fact, not an artifact.
   **Expect it to recur in the 5-parameter policy: hedge-like parameters
   adapt, gain-like parameters park** -- that is correct, not a bug.

4. **`alpha` needs a *plant/tracking-error* signal to adapt.** Because a
   correct CBF makes `alpha` safety-irrelevant, the only thing that
   gives it a state-dependent optimum is a gap between *commanded* and
   *executed* motion (tracking error), plus an observation feature
   exposing it. A toy where the command *is* the motion cannot produce
   this. Directly motivates the Go2 experiment in Section 7.

### 6b. Demonstrated -- empirical (MVP-scale)

1. **Phases 0-1 pass.** Dynamics + QP keep a fixed CBF safe; PPO recovers
   the grid-best constant parameters -- reward and loop are sound.

2. **Robustness (Phase 3) -- clear win.** Trained on disturbances up to a
   ceiling and tested past it, a fixed parameter collides 34-54% while
   the adaptive policy holds near 0%. The headline result: the value of
   a learned adaptive CBF is primarily **robustness across varying
   conditions**.

3. **In-distribution efficiency (Phase 2) -- real but modest.** On a
   4-obstacle scatter layout the fixed-parameter grid traces a genuine
   2-D Pareto frontier and the learned policy is **non-dominated**. The
   `phi`-over-time trace shows clear spatially-adaptive conservatism.
   Honest magnitude: ~3% lower intervention at equal safety vs. the best
   matching fixed parameter; against a dense frontier the learned policy
   sits roughly *on* the line. At a single disturbance level a well-tuned
   fixed CBF is already very good.

4. **`alpha` does not adapt -- confirmed.** With a fast moving obstacle, a
   correct time-varying QP, and slack + closing-rate fed to the policy,
   the fixed-`alpha` sweep shows zero collisions at every `alpha` and
   strictly monotonic intervention cost. Nothing to adapt. See
   `results/alpha_diagnostic.png`. Confirms 6a.3.

5. **Negative result -- symmetric geometry kills the efficiency story.** A
   symmetric corridor layout produced *no* efficiency win: integrated
   intervention there is dominated by the mandatory go-around every safe
   policy pays equally, and the robot is only constraint-active in the
   gap where it needs high `phi`, so cost and benefit are co-located. The
   scatter layout (near-miss obstacles + one tight pinch) was adopted to
   break this. **Lesson: efficiency experiments need environments where
   conservatism is wasteful *somewhere* the policy can detect and avoid.**

### 6c. Honest framing for the paper

The defensible headline is **robustness**, not in-distribution
efficiency: "a learned adaptive CBF that holds safety across
out-of-distribution conditions where a fixed parameter fails, at no
in-distribution efficiency cost." Every result here supports that
without overreach. If the efficiency claim must be load-bearing, the
margin needs to be grown deliberately (Section 8).

---

## 7. How this should inspire the Isaac Lab + Go2 pipeline

The MVP is not a throwaway -- it is a **specification and a set of
de-risked design decisions** for the real pipeline.

### 7a. Component translation

| MVP component | Go2 / Isaac Lab counterpart |
|---|---|
| single integrator `x_dot = u` | Go2 + RL locomotion policy executing `(vx, vy, omega)` |
| `u_nom` from a P controller | nominal command from the navigation layer (e.g. A*) |
| CBF-QP in 2-D command space | the *same* QP, in 3-D command space `(vx, vy, omega)` |
| `disturbance d` | terrain, friction, pushes, model mismatch in sim |
| `phi` adapts | `phi` adapts to **state-estimation / `L_g-hat h` uncertainty** |
| `alpha` (parked) | `alpha` adapts only if **tracking error** gives it a signal |
| QP-as-environment + PPO | unchanged pattern; QP must become batched/differentiable |

The command-space CBF is the bridge: the constraint the MVP validated is
the *same* constraint on the Go2. The single integrator was a faithful
stand-in for "something tracks the command" -- on the Go2 the locomotion
policy is that something, imperfectly, which is exactly the interesting
part.

### 7b. Keep the phase ladder

- **Go2 Phase 0** -- QP + Isaac Lab sim sanity with a fixed CBF.
- **Go2 Phase 1** -- PPO recovers a good *constant* parameter set;
  certifies reward + loop before any adaptivity claim.
- **Go2 Phase 2** -- state-conditioned policy vs. fixed-parameter frontier
  in cluttered terrain.
- **Go2 Phase 3** -- OOD robustness across terrain/friction not seen in
  training.
Skipping Phase 1 on the Go2 would make every later result ambiguous.

### 7c. The `alpha` experiment, designed properly

The MVP proved `alpha` only adapts given a genuine tracking-error signal.
The Go2 is where that signal exists -- but it must be *engineered and
instrumented*, not hoped for. Separate the two error channels cleanly:
- **Tracking-error channel -> `alpha`.** Domain-randomize friction and
  inject disturbance torques so the *executed vs. commanded* velocity gap
  varies with terrain and maneuver. Add an observation feature exposing
  it -- recent tracking residual `||v_cmd - v_meas||`, or contact/slip
  state. Then test whether `alpha` adapts to *that*.
- **Estimation-error channel -> `phi`.** Vary sensor / state-estimator
  noise. Add an estimate-uncertainty feature. `phi` should adapt to that.
If `alpha` still parks even with a tracking-residual feature present,
that is itself a strong, publishable result (the policy is genuinely
1-D adaptive-`phi`). If it adapts, you have the clean two-parameter
story. Either way, design the randomization ranges and observation
features *for this question* -- do not expect `alpha` adaptation to fall
out of a generic "add model error" setup.

### 7d. Engineering carry-overs and what the MVP did NOT touch

- The cvxpy QP is fine for one environment but will bottleneck Isaac
  Lab's thousands of parallel environments. Move to a batched /
  differentiable QP (`cvxpylayers`, `qpth`) before scaling.
- Reward structure transfers directly: `progress - intervention -
  collision + goal`. The collision:intervention balance is a real knob
  (the MVP settled near 15:1); re-tune on the Go2.
- The MVP deliberately does **not** validate: locomotion-controller
  coupling, sim-to-real transfer, QP latency at the ~50 Hz control rate,
  the `phi = 1/epsilon` numerics near singularities, or the real
  `epsilon`. These are the genuine Go2 risks -- but the MVP means you
  attack them against a *de-risked idea*, not an unproven one.

### 7e. Negative results that save Go2 time

- Do **not** expect an efficiency win where intervention is dominated by
  an unavoidable maneuver. Design cluttered terrain with near-miss
  structure where conservatism is sometimes wasteful.
- Do **not** add all 5 parameters at once. Expand from the working
  2-param policy one parameter at a time, each with a scenario that gives
  that parameter a genuine signal -- otherwise you get extra parked
  dimensions and a much harder debugging surface.

---

## 8. Open questions / next steps

1. **Decide the paper's load-bearing claim.** Robustness (solid now) or a
   bigger efficiency number (more work)? This decides where effort goes.
2. **Grow the efficiency margin, if needed.** The policy learned
   *proximity* adaptation but not *pinch vs. near-miss discrimination*.
   The observation already exposes `h_second` for exactly this; push the
   policy to use it via a near-miss curriculum or reward shaping.
3. **Run the full ladder at full training budget** on real hardware
   (MVP runs here were compute-limited).
4. **Unicycle dynamics** as an intermediate step -- nonholonomic
   constraints make the CBF non-trivial without any legged-robot
   complexity. Small change: only the nominal controller and integration.
5. **Then the Go2 pipeline**, following Section 7.

### config toggle to be aware of

`config.py: moving_obstacle` -- currently set (one oscillating obstacle),
which affects all phases. Set it to `None` to revert phases 0-3 to a
fully static scatter layout.
