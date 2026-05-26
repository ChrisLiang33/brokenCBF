# go2 — clean restart of the Go2 adaptive-CBF pipeline

This is a deliberate restart of the Go2 + learned-adaptive-CBF research project.
The parent MVP (`../`) validated the *idea* (command-space CBF +
QP-as-environment + PPO works; `phi` adapts as hedge against uncertainty);
this subproject is where that idea is built up on a real legged robot in
Isaac Lab, **from scratch and following a strict phase ladder**.

The phase ladder exists for one reason: if you cannot show that a *fixed*
CBF works, a bad result with a *learned* CBF is ambiguous between "adaptive
parameterization does not help" and "something upstream is broken." Skipping
phases makes every later finding unfalsifiable. The MVP work surfaced
exactly this kind of bug (a misweighted reward that made colliding pay
better than fighting through a constraint) — caught only because Phase 1
isolated the loop from the adaptivity question.


## Phase ladder

| Phase | Question it answers | What's in the loop |
|---|---|---|
| **0 (this phase)** | Does the plumbing work? Can a stock Go2 locomotion policy reach a goal under a nominal P controller, with the realized velocity tracking the commanded velocity within a tolerable gap? | Stock `Isaac-Velocity-Flat-Unitree-Go2-Play-v0` + a P-controller-to-goal command source. **No obstacle. No CBF. No learning.** |
| 0.5 | Does a *fixed* CBF safety filter keep the Go2 out of a static obstacle when the QP can only constrain *commanded* velocity but the body executes a *realized* one? **Pass = realized `h > 0` every step**, not just commanded. | Phase 0 + one static obstacle + CPU-side cvxpy QP (3-D command space with yaw rotation) + hand-tuned `(phi, alpha)`. |
| **0.6 (GATE)** | Does optimal-φ actually *move* with its channel? Sweep fixed φ across a friction (or motor-strength) continuum. If the per-friction optimal φ doesn't shift, no policy architecture can learn to adapt it and no scene design will fix that. | Phase 0.5 + friction-randomization DR knob + grid of fixed φ values. **No RL.** If this gate fails, stop and rethink the channel before any training. |
| 1 | Are the PPO loop and reward correct? Can PPO recover a good *constant* `(phi, alpha)` (matching the grid optimum), with observation zeroed? | Phase 0.5 + zeroed-observation PPO. **Do not skip.** This is where reward bugs are surfaced before adaptivity makes them invisible. |
| 2 | Does a state-conditioned policy beat the fixed-parameter Pareto frontier? | Phase 1 + deployable obs (proprio history, lidar — NOT privileged z). |
| 3 | Does the adaptive policy degrade more gracefully than fixed past the training-channel ceiling? (**The headline claim.**) | Phase 2 + per-episode randomization across the φ-channel (and later α-channel). |


## Phase 0 deliverable

A single standalone script, `phase0_plumbing.py`:

- Loads the stock `Isaac-Velocity-Flat-Unitree-Go2-Play-v0` task with `num_envs=1`.
- Loads a trained Go2 velocity-tracking PPO checkpoint (stock recipe — see
  "Running on the remote box" below).
- Replaces the task's random velocity command with a **P controller toward a
  fixed goal waypoint**, in the robot's base frame.
- Steps the simulator for a fixed number of seconds.
- Logs per step: timestamp, base xy, base yaw, commanded velocity,
  realized velocity, distance to goal, reached-flag.
- Saves a CSV and an optional plot.

### Phase 0 pass criteria

A passing Phase 0 demonstrates **three** things, in order:

1. **Plumbing works at all.** Sim launches, env makes, policy loads, step
   loop runs to completion without crash.
2. **Go2 reaches the goal.** Distance to goal drops below 0.4 m at some
   step before the time budget expires.
3. **Realized velocity tracks commanded velocity acceptably.** Define
   tracking RMSE = `mean(||v_realized - v_commanded||) / mean(||v_commanded||)`.
   Phase 0 passes if RMSE < 0.30 averaged across the last 80% of the
   trajectory (initial transient excluded). If the gap is much larger
   than that, the locomotion policy cannot execute the safe command well
   enough — that is the kind of thing the MVP **structurally could not
   surface**, and Phase 0 is the cheapest place to find it.

If (3) fails, the cost of finding out at Phase 0 (one fixed waypoint, no
CBF) is hours; the cost of finding it out at Phase 2 (learned adaptive
policy, full training) is days.


## Running on the remote box

Phase 0 cannot run locally on macOS — Isaac Sim requires Linux + NVIDIA
GPU. Develop here, run on the lab Linux machine.

### One-time setup (on the remote)

**Use a fresh Isaac Lab clone — NOT the one inside `safety-go2/`.** That
fork carries weeks of customizations (custom command terms, DR, task
config overrides, the `manager_based/safety/cbf_go2/` extension, etc.)
that would silently contaminate Phase 0. The whole point of the restart
is to run against stock behavior; the fork makes "stock" undefined.

On the remote, install Isaac Lab in its own directory (e.g.
`~/IsaacLab/`), unrelated to `safety-go2/`. The quickest path on
Linux+NVIDIA is the pip workflow:
<https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>.

`phase0_plumbing.py` imports `isaaclab` from whichever Python it runs
under — so the only thing that matters is which Isaac Lab's
`isaaclab.sh` you invoke it through.

1. Train a stock Go2 velocity-tracking policy (skip this step if you can
   use Isaac Lab's published pretrained checkpoint via
   `--use_pretrained_checkpoint`, but for clean provenance training fresh
   is recommended for the writeup):

   ```bash
   cd /path/to/IsaacLab
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
       --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
       --num_envs 4096 \
       --headless
   ```

   This trains the stock policy; takes on the order of an hour on a
   single GPU. The checkpoint ends up under
   `IsaacLab/logs/rsl_rl/unitree_go2_flat/<timestamp>/model_*.pt`.

2. Sync this directory onto `labbox` (the SSH alias is configured
   locally; see `~/.ssh/config`). The destination on the remote defaults
   to `~/cbf_rl_mvp/go2/`. From local:

   ```bash
   ./scripts/sync_push.sh                                # default dest
   LABBOX_PATH=~/some/other/path/ ./scripts/sync_push.sh   # override
   ```

   `sync_push.sh` excludes `__pycache__/`, run artifacts (CSV/log/PNG/PT/ONNX),
   and uses `--update` so remote-only files survive. The script imports
   `isaaclab`, so on the remote it must run under Isaac Lab's Python
   (i.e. via `./isaaclab.sh -p ...`).

3. After a Phase 0 run on the remote, pull the artifacts back:

   ```bash
   ./scripts/sync_pull.sh
   ```

   This brings down only CSVs, logs, PNGs, and checkpoints — it never
   overwrites source files, so local edits can't be clobbered.

### Run Phase 0

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/cbf_go2/phase0_plumbing.py \
    --num_envs 1 \
    --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
    --checkpoint /path/to/logs/rsl_rl/unitree_go2_flat/<timestamp>/model_*.pt \
    --goal 5.0 0.0 \
    --max_time 15.0 \
    --log_csv phase0_log.csv
```

Or, equivalently, with the published pretrained checkpoint:

```bash
./isaaclab.sh -p /path/to/cbf_go2/phase0_plumbing.py \
    --num_envs 1 \
    --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
    --use_pretrained_checkpoint \
    --goal 5.0 0.0
```

Pass criteria evaluation is printed at the end of the run.


### Phase 0.5 (after Phase 0 passes)

Adds one static obstacle and a fixed CBF safety filter. **Pass = realized
`h > 0` every step.** This is the unicycle-rung check the MVP could not
make.

```bash
./isaaclab.sh -p ~/Desktop/cbf_rl_mvp/go2/phase0_5_fixed_cbf.py \
    --checkpoint /path/to/model_*.pt \
    --goal 6.0 0.0 \
    --obstacle 3.0 0.7 \
    --obstacle_radius 0.8 \
    --phi 0.25 --alpha 1.5 \
    --max_time 25.0 \
    --log_csv phase0_5_log.csv \
    --headless
```

Note: the obstacle is intentionally laterally offset from the start→goal
line. With a dead-on collinear obstacle, a pure P controller has no
lateral pressure and the QP stalls the robot directly in front — that
is a geometry pathology, not a CBF bug.


### Phase 0.6 — fixed-φ sweep GATE (before any RL)

For each value of the channel under test, sweep fixed φ over a grid and N
seeds. The analyzer aggregates and finds optimal φ per channel value.
**Pass = optimal φ moves across the channel (range > 0.15).** If flat,
φ has no signal on this channel — stop and rethink before any RL.

Two channels are supported: **friction (`--friction_mu`)** and **external
disturbance force (`--disturbance_force`)**. Sweep one at a time, holding
the other at a sane default. Empirically friction alone gets laundered by
the stock locomotion (the Go2 just walks more conservatively at low μ);
external velocity disturbance is the canonical φ-stressor (it mirrors the
MVP's `d`).

**Disturbance sweep (recommended first):**

```bash
for D in 0 5 10 15 20; do
    ./isaaclab.sh -p ~/Desktop/cbf_rl_mvp/go2/phase0_6_phi_sweep.py \
        --checkpoint /path/to/model_*.pt \
        --friction_mu 0.6 \
        --disturbance_force $D \
        --out_csv ~/Desktop/cbf_rl_mvp/go2/phase0_6_d${D}.csv \
        --headless
done

python ~/Desktop/cbf_rl_mvp/go2/analyze_phi_sweep.py \
    ~/Desktop/cbf_rl_mvp/go2/phase0_6_d*.csv \
    --channel disturbance_force \
    -o ~/Desktop/cbf_rl_mvp/go2/phase0_6_gate.png
```

**Friction sweep (for completeness, but expect a flat optimum):**

```bash
for MU in 0.2 0.4 0.6 0.8 1.0; do
    ./isaaclab.sh -p ~/Desktop/cbf_rl_mvp/go2/phase0_6_phi_sweep.py \
        --checkpoint /path/to/model_*.pt \
        --friction_mu $MU \
        --disturbance_force 0 \
        --out_csv ~/Desktop/cbf_rl_mvp/go2/phase0_6_mu${MU}.csv \
        --headless
done

python ~/Desktop/cbf_rl_mvp/go2/analyze_phi_sweep.py \
    ~/Desktop/cbf_rl_mvp/go2/phase0_6_mu*.csv \
    --channel friction_mu \
    -o ~/Desktop/cbf_rl_mvp/go2/phase0_6_friction_gate.png
```

Defaults: `phi_values = {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}`,
`seeds_per_cell = 5`, `alpha = 1.5`, same geometry as Phase 0.5.
~5 min per invocation, ~25 min per channel sweep.

The disturbance is implemented as an external horizontal force on the
robot's base, direction resampled every `--disturbance_resample` steps
(default 50 = 1 s at dt=0.02). For Go2 (~15 kg), 20 N pushes the body by
~1.3 m/s per epoch — comparable to v_max.


### Phase 1 — PPO recovers grid-best constant (zeroed obs)

**Do not skip this.** PPO trains an outer policy whose **observation is
zeroed**, so it can only emit a state-independent (φ, α). After training,
the script grid-searches the same scenario and reports the best fixed
(φ, α) as a baseline. **Pass = PPO lands close to the grid optimum.** If
it does, the reward + RL loop are sound and we can advance to Phase 2.
If it doesn't, debug reward weights or training before adding adaptivity
— anything wrong here will silently corrupt every later result.

Implemented as a canonical Isaac Lab manager-based task at
`cbf_task/` (registered as `Isaac-CBF-Adaptive-Go2-v0`):

- Custom **action term** that takes `(φ, α)` in, runs the closed-form
  CBF + frozen locomotion in batched torch across all envs, applies
  the disturbance force, and writes joint targets — exactly the same
  pattern as the stock Go2 velocity task.
- Custom **reward / termination / observation** managers.
- Trained with **rsl_rl OnPolicyRunner** (Isaac Lab's native trainer)
  on `num_envs=64` parallel environments by default.

```bash
./isaaclab.sh -p ~/Desktop/cbf_rl_mvp/go2/phase1_train.py \
    --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/<ts>/model_*.pt \
    --num_envs 64 \
    --max_iterations 200 \
    --out_dir ~/Desktop/cbf_rl_mvp/go2/phase1_outputs \
    --headless
```

Outputs `phase1_outputs/`:

- `rsl_rl/` — Isaac Lab / rsl_rl logging directory (checkpoints, tensorboard).
- `phase1_grid.csv` — every (φ, α) cell with collision/reach/intervention.
- `phase1_summary.json` — learned constant + grid best.

The script prints a PASS/REVIEW verdict comparing PPO's learned constant
to the grid optimum. REVIEW = investigate reward weights or training
budget before Phase 2.

**Throughput note:** with `num_envs=64` and batched closed-form CBF,
expect ~20–30 min for the full Phase 1 run (training + grid). The
locomotion policy is loaded once from its rsl_rl `.pt` checkpoint via
a small helper (no second Isaac Lab env required).


### Phase 1.5 — fingerprint sweep GATE (before Phase 2)

Phase 0.6 proved the env has a φ-channel signal (optimal fixed φ moves
with the stressor). Phase 1.5 asks the next-level question: **is that
signal visible from a deployable observation?** If yes, Phase 2 can
train with proprio + action history directly (Option 1). If no, the
locomotion is laundering the disturbance and we need a teacher-student
setup (Option 2 — RMA-style: teacher uses priv_obs, student distills
to a deployable encoder).

Don't pick — *measure*. Run the sweep at a known-safe `(φ, α)` so the
locomotion is operating normally across disturbance levels; log per-step
deployable observations; train an offline regressor.

```bash
for D in 0 15 30 45; do
    ./isaaclab.sh -p ~/Desktop/cbf_rl_mvp/go2/phase1_5_fingerprint_sweep.py \
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/<ts>/model_*.pt \
        --disturbance_force $D \
        --out_npz ~/Desktop/cbf_rl_mvp/go2/phase1_5_d${D}.npz \
        --num_envs 64 --n_episodes 4 \
        --headless
done

python ~/Desktop/cbf_rl_mvp/go2/analyze_fingerprint.py \
    ~/Desktop/cbf_rl_mvp/go2/phase1_5_d*.npz \
    --window 20
```

Decision:

- **R²_test > 0.7** → fingerprint is strong; Phase 2 uses **Option 1** (direct training, proprio + action history obs, no teacher).
- **R²_test < 0.3** → locomotion launders the disturbance; build **Option 2** (teacher on priv_obs, distill to deployable encoder).
- **0.3 ≤ R²_test ≤ 0.7** → borderline; rerun with bigger `--window`. If still mid, lean Option 2.

Each invocation samples ~1,600 step-rows per env × 64 envs ≈ 100k
samples per disturbance level. Total dataset ≈ 400k samples; regressor
runs in seconds on GPU.


### Phase 2 — state-conditional policy (the real RL experiment)

**Where the actual research claim is tested.** Differences from Phase 1:

- Observation: 20-step history of the 48-dim deployable obs (proprio + CBF-filtered cmd + last loco action), 960-dim flattened. This is the window/feature combo that scored R²=0.955 in Phase 1.5.
- Disturbance: per-episode random magnitude in [0, 45] N (DR over the OOD signal the policy must adapt to).
- Larger MLP (256×128) to handle the 960-dim obs.
- Trained on `Isaac-CBF-Adaptive-Go2-Phase2-v0` (new task ID).

```bash
./isaaclab.sh -p ~/Desktop/cbf_rl_mvp/go2/phase2_train.py \
    --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/<ts>/model_*.pt \
    --num_envs 64 \
    --max_iterations 500 \
    --out_dir ~/Desktop/cbf_rl_mvp/go2/phase2_outputs \
    --headless
```

After training, the script evaluates:

- **The learned policy** at each `d ∈ {0, 15, 30, 45} N` (4 cells).
- **Fixed-parameter grid** at each disturbance (7 × 5 × 4 = 140 cells).

Both runs use the same scenario; the learned policy gets to *adapt* (φ, α) per step using the proprio observation, while the grid uses a single fixed (φ, α) for the entire run.

**Pass criterion:** the learned policy is **non-dominated** — at no test disturbance level does any fixed cell achieve *both* `coll ≤ learned_coll` and `reach ≥ learned_reach` and `intervention < learned_intervention`. If the policy is dominated at any disturbance level, **REVIEW**.

Outputs `phase2_outputs/`:

- `rsl_rl/` — Isaac Lab / rsl_rl logging dir (checkpoints, TB events).
- `phase2_learned_eval.csv` — per-disturbance metrics for the trained policy.
- `phase2_grid_eval.csv` — per-(disturbance, φ, α) metrics for the baseline grid.
- `phase2_summary.json` — verdict + summary.

**Throughput note:** 500 PPO iterations × 64 envs × 24 steps ≈ 770k rollout steps. At Phase 1's ~5000 fps this is ~3 min training. The post-training eval is 144 cells × 1250 steps = ~1 hour. Total ~70 min.


## What this project deliberately does NOT carry forward from `safety-go2/`

- No RMA split encoders (yet — that is a Phase 2/3 architecture choice).
- No priv-vs-proprio observation split.
- No domain randomization (Phase 0 is flat ground, default friction).
- No `diagnose_*` or `probe_z_linear` machinery (those answer
  later-phase questions).
- No `phi/alpha/a/b/c` learned parameterization (Phase 0 has no learned
  parameters at all).

Everything starts from the stock Isaac Lab Go2 task. Each phase adds
exactly one mechanism, motivated by a question only that mechanism can
answer.
