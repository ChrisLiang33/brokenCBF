# Student Distillation Spec

**Goal**: train a deploy-realistic policy ("student") that matches the v2.15
teacher's performance using only LiDAR-derived perception. The teacher knows
true obstacle positions and radii from the simulator; the student must work
with only what a real robot's LiDAR can see — the output of the
synthetic-LiDAR + clustering perception pipeline.

## Inputs / outputs

### Teacher (frozen, v2.15 weights)

- input: 8207-D privileged observation
  - 15 dynamics scalars (friction, base_mass, base_height, applied_force,
    applied_torque, tracking_err, com_offset)
  - 8192-D = 64 × 64 × 2 occupancy grid built from **true** obstacle
    positions and shapes (current frame + previous frame)
- output: 5-D CBF parameter vector (alpha, phi, a, b, c)
- state: frozen — no gradient.

### Student (trained from scratch)

- input: same 8207-D shape as teacher, but the 8192-D grid is built from
  the **noised LiDAR cluster output** instead of true positions. The
  15 dynamics scalars stay (first-pass — see "Deploy-realism caveat"
  below).
- output: 5-D CBF parameter vector (same as teacher).
- state: trained via DAgger.

The shape match means the same MLP architecture works for both — only the
content of the obstacle channel differs.

## How the student's grid is built

```text
                    True obstacles in sim
                              |
                              v
              synthetic LiDAR rays cast outward
                              |
                +-------------+----------------+
                |             |                |
                v             v                v
         dropout DR     occlusion DR     range-gating DR
        (drop rays    (closer cylinders   (obstacles past
         randomly)     block farther)      X m vanish)
                |             |                |
                +------+------+----------------+
                       |
                       v
              remaining hits clustered into centroids
                       |
                       v
       centroids + fixed 0.3m radius rasterized into a
              64x64x2 occupancy grid
                       |
                       v
              student policy sees this grid
```

Compared to the teacher pipeline:

```text
              True obstacles in sim
                       |
                       v
           rasterized into 64x64x2 grid
                       |
                       v
              teacher policy sees this grid
```

The student's grid has the same shape but degraded content.

## Domain-randomization layers on the student input

Each layer is per-episode persistent (drawn at reset, held for the
episode) so a long-episode RL pass actually sees a stable corruption
pattern rather than per-step jitter that averages out.

| Layer                   | Effect                                              | Range (placeholder)        |
| ---                     | ---                                                 | ---                        |
| Sensor dropout          | Random fraction of LiDAR rays return nothing        | 0–10%                      |
| Occlusion               | Already inherent to raycast (closer cylinders block)| n/a                        |
| Range gating            | Effective sensor range varies per episode           | uniform in (5.0, 7.0) m    |
| Cluster grid resolution | Bin size for clustering                             | uniform in (0.30, 0.50) m  |

These mirror the teacher's existing DR axes (position noise for the
QP, radius perception error, actuation noise) — same playbook on the
student-input side instead of the QP / actuator sides.

## Method: DAgger (Dataset Aggregation, Ross & Bagnell 2011)

1. Roll out the student in sim with noised obstacle grid in its obs.
2. At every visited state, query the frozen teacher with the **clean**
   privileged grid — record its action as the label.
3. Train student to match teacher's labels via mean-squared-error loss.
4. Repeat.

Student trains on its own visited states (not the teacher's), preventing
the compounding-error failure mode of plain behavior cloning.

## Architecture

Same MLP as the teacher — only the obstacle channel's content differs.
Both inputs are 8207-D; the policy network can stay identical.

## Training environment

- env: same as v2.15 (`Isaac-CBF-Go2-v0`), reusing the 6K-iter domain
  randomization setup (REWARD-3, alpha/c floors, phi-DR, c-DR, etc.).
- num envs: 1024 (down from v2.15's 4096). DAgger is matching, not
  exploring.
- compute budget: ~3K iters (~5h wall clock).

## Loss

Mean-squared-error between student's 5-D output and teacher's 5-D output:

```text
L = ||student_action - teacher_action||^2
```

No KL term — teacher is deterministic at inference (matches deploy use-case).

## Eval

Same 10-task suite used for v2.15 plus a teacher-vs-student comparison
on each task: same env, same seeds, log fall-rate / stuck-rate /
mean_dist_traveled / goal_reach_rate / etc.

## Success criterion

Student fall-rate within +5 percentage points of teacher across all 10
eval tasks, on both axes (safety + performance).

## Deploy-realism caveat

For the first-pass distillation experiment, the student keeps access
to the 15 privileged dynamics scalars (friction, base_mass, etc.). This
is **not** deploy-realistic — a real robot can't measure these. But
keeping them in for the first pass isolates one failure mode at a time:
"can DAgger transfer the teacher's CBF-param policy to a perceptually-
degraded version of the same input?" If that works, a second-pass
experiment replaces the privileged dynamics scalars with deploy-realistic
estimates (or zeros) and measures the additional gap.

## Implementation checklist

Pre-staging (doable now, before v2.15 finishes):

- [ ] Add `train_distillation.py` skeleton: loads frozen teacher
      checkpoint, instantiates student with same architecture, runs
      DAgger loop. Stub the obstacle-grid noising step with a clear
      TODO until the noised-grid pipeline is implemented.
- [ ] Register `Isaac-CBF-Go2-Distill-v0` in the manager_based safety
      task `__init__.py`. Points to a `CbfGo2EnvCfg_DISTILL` that's
      a thin subclass of the base env (same recipe; placeholder for
      noised-grid cfg fields).
- [ ] Add `--student_checkpoint` flag to `eval_baseline.py` so we
      can run the same 10-task suite on a trained student.

After v2.15 lands:

- [ ] Calibrate the noised-grid pipeline: pick representative ranges
      for sensor dropout / range gating / cluster grid resolution.
- [ ] Implement the noised-grid obs term: a new function
      `cbf_go2_observations.noised_occupancy_grid_b` that runs the
      synthetic-LiDAR + clustering pipeline + DR layers and rasterizes
      the result. Bypasses true positions.
- [ ] Add a student-only obs config that uses the noised grid in place
      of `priv_obs.occupancy_grid_b`.
- [ ] Train v3.0 student via DAgger.
- [ ] Eval student vs teacher on the 10-task suite.

## Related-work positioning

The SHIELD paper (Yang et al. 2025) uses a similar synthetic-LiDAR +
clustering perception pipeline but trains a single hand-tuned alpha
(per-episode constant). Our distinction:

- We train a 4-parameter adaptive policy (alpha, phi, a, c) that
  learns to compensate for perception-induced radius error via the
  c-parameter (boundary correction) and for measurement uncertainty
  via the a-parameter (Dean 2019).
- Our student inherits this adaptive behavior from the teacher via
  DAgger. The deploy policy is also 4-parameter adaptive, not single
  alpha.

That's the differentiation point: SHIELD assumes perception error
away. We adapt to it per step, and the student inherits the adaptation
through distillation.
