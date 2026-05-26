# CBF parameter learning — MVP

A minimal, runnable testbed for the core research claim:

> **An RL-learned, state-adaptive CBF parameterization is minimally
> invasive in-distribution and more robust out-of-distribution than any
> fixed hand-tuned parameterization.**

This MVP deliberately strips out the engineering so it can de-risk the
*idea*. There is **no legged robot, no Isaac Lab, no learned dynamics,
no navigation policy**. The robot is a 2D single integrator; because the
CBF lives in command space, that substitution is faithful — nothing in
the contribution depends on what is underneath the velocity command.

## The setup

- **Robot**: single integrator, `x_dot = u + d`, in a 10×10 workspace.
- **Task**: go-to-goal P controller produces the nominal command `u_nom`.
- **Obstacle**: one circle, placed so the CBF constraint always binds
  (and off the start–goal line, to avoid the colinear CBF deadlock).
- **Barrier**: `h(x) = ||x - p_obs|| - r_safe`. The gradient is a unit
  vector, so `||L_g h||^2 = 1` and Lucas's robustified constraint
  collapses to a single linear inequality:

  ```
  grad_h . u  >=  phi - alpha * h
  ```

  - `phi`   — constant rate margin; **directly the disturbance hedge**.
  - `alpha` — how fast `h` may decrease (the standard CBF gain).

- **Safety filter**: a slack-relaxed QP, `min ||u - u_nom||^2 + M*delta^2`,
  built once with cvxpy Parameters and re-solved each step.
- **What RL learns**: a policy `pi(obs) -> (phi, alpha)`, emitted every
  step. The policy acts *through* the QP — no gradients flow through the
  optimizer; PPO only needs the sampled rollout.
- **Disturbance** `d` is the OOD dial: fixed for in-distribution work,
  sampled per-episode from a range when we want the policy to learn to
  condition on it.

## The phase ladder

Each phase isolates one failure mode, so a bad result is unambiguous.

| Phase | Script | Validates |
|-------|--------|-----------|
| 0 | `phase0_fixed_cbf.py` | dynamics + QP; a fixed CBF avoids the obstacle. No RL. |
| 1 | `phase1_train_constant.py` | reward design + PPO loop — by checking PPO recovers the grid-search-best **constant** params. |
| 2 | `phase2_train_adaptive.py` | a **state-conditioned** policy beats the fixed-parameter Pareto frontier (in-distribution). |
| 3 | `phase3_ood.py` | the core claim: the adaptive policy degrades gracefully when the test disturbance exceeds the training range. |

Do not skip Phase 1. If PPO cannot recover a good constant, a bad Phase 2
result cannot tell you whether adaptivity fails or the loop is broken.

## Running it

```bash
pip install -r requirements.txt

python phase0_fixed_cbf.py            # instant, no training
python phase1_train_constant.py       # ~120k steps
python phase2_train_adaptive.py       # ~200k steps  -> phase2_pareto.png
python phase3_ood.py                  # reuses/ trains -> phase3_ood.png
```

Every training script accepts `--quick` (a few-second smoke test) and
`--timesteps N` (override the budget). Outputs: PNG plots and saved
`.zip` PPO models in the working directory.

## Reading the results

- **Phase 2 Pareto** — fixed `(phi, alpha)` grid traces a frontier in
  (intervention cost, safety) space. The learned policy should land
  *up and/or left* of it: same safety for less intervention, or more
  safety for the same intervention.
- **Phase 3 OOD** — collision rate vs test disturbance. The fixed
  parameter, tuned on the training range, should start colliding once
  the disturbance exceeds it; the adaptive policy reads its disturbance
  estimate, raises `phi`, and the curve stays low.

## What this MVP does NOT validate

It validates the *idea*, not the *Go2 system*. Out of scope here, and
genuine separate risks: locomotion-controller coupling, the real
`epsilon` you cannot dial, sim-to-real transfer, QP latency at 50 Hz,
and `phi = 1/epsilon` numerics near singularities. A clean Phase 3 means
the Go2 work becomes engineering against a de-risked idea rather than a
bet on an unproven one.

## Files

```
config.py                 all hyperparameters (one dataclass)
cbf_qp.py                 robustified CBF-QP safety filter (cvxpy)
env.py                    Gymnasium env: QP + dynamics + reward
utils.py                  rollout / evaluation / PPO training / plotting
phase0_fixed_cbf.py       Phase 0
phase1_train_constant.py  Phase 1
phase2_train_adaptive.py  Phase 2
phase3_ood.py             Phase 3
```

## Natural next steps

- Swap the single integrator for a **unicycle** — nonholonomic
  constraints make the CBF behavior non-trivial without adding any
  legged-robot complexity. The QP and reward are unchanged; only
  `_nominal` and the integration step move.
- Re-introduce the `a, b, c` parameters (the QP already supports `a`,
  `c`; `b||u||` needs the norm term re-enabled) once the two-parameter
  version is solid.
- Replace cvxpy with a batched/differentiable QP (`cvxpylayers`, `qpth`)
  before scaling to Isaac Lab's parallel envs — per-solve cvxpy overhead
  is fine here but will bottleneck thousands of parallel environments.
