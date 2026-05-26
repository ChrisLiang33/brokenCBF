# Dot reach-the-goal MVP

A blue sphere (the "agent") plans a straight line to a red goal in MuJoCo,
then a CBF-QP safety filter projects the planner's desired velocity onto a
safe set, steering the agent around two cylinder obstacles. The planner
itself is intentionally obstacle-blind — all collision avoidance lives in
the filter.

Episode terminates on:

- **success** — agent within 20 cm of the goal
- **failure** — agent contacts a cylinder (means the filter let it through)

## Run

```bash
mjpython mvp.py                       # interactive viewer (macOS)
# python3 mvp.py                      # interactive viewer (Linux/Windows)
python3 mvp.py --record run.mp4       # offline render to mp4
```

## Sync to the GPU lab machine

Edit the defaults at the top of [sync.sh](sync.sh) (`LAB_HOST`, `LAB_PATH`)
or export them in your shell. Then:

```bash
./sync.sh              # push laptop -> lab
./sync.sh --dry-run    # preview
./sync.sh --pull       # pull lab -> laptop (e.g., trained checkpoints)
```

Training logs in `runs/` and `wandb/` are excluded so they live on the lab box.

## Files

- [mvp.py](mvp.py) — orchestrator: sim, planner, pure pursuit, safety filter, collision check, viewer/recorder.
- [planner.py](planner.py) — naive straight-line goal-reaching planner + pure-pursuit tracker. `OBSTACLES` list feeds the safety filter.
- [safety_filter.py](safety_filter.py) — CBF-QP filter with smoothed-SDF barrier; closed-form single-integrator projection.
- [scene_mvp.xml](scene_mvp.xml) — MuJoCo scene: agent sphere, goal/waypoint markers, cylinder obstacles.

## Pipeline

```text
planner.plan(start, goal)              -> straight-line waypoints (obstacle-blind)
PurePursuit(waypoints).command()       -> desired velocity  u_des
CBFSafetyFilter.filter(pos, u_des)     -> safe velocity     u_safe
set agent free-joint qvel = u_safe     -> sphere slides toward the goal, around obstacles
check agent-cylinder contacts          -> terminate on collision (filter failure)
```

The barrier is the smoothed signed distance to the nearest cylinder:

```text
sdf(p) = min_i  ||p - ρ_i|| - (R_robot + R_i)
h(p)   = λ (1 - exp(-γ · sdf(p)))
```

Single-integrator dynamics make L_f h = 0 and L_g h = ∇h, so the CBF-QP

```text
min ||u - u_des||²   s.t.   L_g h · u ≥ -α(h) + φ·||L_g h||²
```

collapses to one scalar inequality and is solved by closed-form projection.
Swap the analytic SDF in `safety_filter.h_and_grad` for a lidar-derived one
when you move toward a real sensor model.
