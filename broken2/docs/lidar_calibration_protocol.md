# Mid-360 ↔ shield_v0c calibration protocol

Goal: measure the actual perception bias of the deploy pipeline (Mid-360 → 2D occupancy → cluster-fit-cylinder) and decide whether sim's `shield_v0c` perception DR needs to be retuned before hardware deploy.

If the measured h_gap distribution falls inside the sim DR range, no retune needed.
If it falls outside, widen `obstacle_radius_perception_error_range` and (optionally) the cluster-fit margin.

## Why this matters

Sim teacher trained on `shield_v0c`:
- 128 rays, 6m range, 2D ray cast
- cluster-fit-cylinder with +0.10m safety margin
- Resulting perception bias: perceived surface ~0.20-0.50m closer than true (h_gap ≈ +0.20 to +0.50m)

Real deploy uses Mid-360:
- Dense 3D points, ~360°×59° FOV, 70m range
- Same downstream cluster-fit-cylinder pipeline
- Unknown empirical h_gap

If the real Mid-360 pipeline produces a much *smaller* h_gap than sim, the teacher's adaptation is over-protective at deploy (steers wider than needed, reduces goal reach).
If it produces a much *larger* h_gap, the teacher under-corrects (collision risk).

## What to measure

Per static-obstacle trial, log:
- `r_true` — measured cylinder radius (caliper, m)
- `d_true` — distance from Go2 body center to cylinder axis (tape, m)
- `r_est(t)` — cluster-fit cylinder radius from Mid-360 pipeline (m)
- `d_est(t)` — distance from Go2 body center to fit-cylinder axis (m)
- `h_true(t) = d_true − r_true`
- `h_perc(t) = d_est(t) − r_est(t) − 0.10` (sim margin convention; subtract whatever margin is in the deploy pipeline)
- `h_gap(t) = h_perc(t) − h_true(t)` ← *this is the bias signal*

For each trial: report mean and std of `h_gap` over a ~30s stationary recording.

## Test matrix

Pick three obstacle radii × five distances × two yaw orientations = 30 trials.

| obstacle radius | distances (m)        | yaw (rad)   |
| --------------- | -------------------- | ----------- |
| 0.10 m          | 1.0, 2.0, 3.0, 4.0, 5.0 | 0, π/4 |
| 0.20 m          | 1.0, 2.0, 3.0, 4.0, 5.0 | 0, π/4 |
| 0.40 m          | 1.0, 2.0, 3.0, 4.0, 5.0 | 0, π/4 |

Materials: cardboard tubes or PVC pipes work fine. Avoid clear/reflective surfaces (Mid-360 dropout).

## Setup

- Open indoor space, ~6×6m clear floor
- Single obstacle in scene (no clutter to confuse cluster-fit)
- Go2 stationary in stand pose, facing the obstacle
- Mid-360 mounted at standard deploy height/orientation
- Record at full sensor rate for ~30s per trial

Repeat the first trial (0.20m radius @ 2.0m distance, yaw=0) at the start, middle, and end of the session to check for drift.

## Analysis

For each (r_true, d_true) cell, compute:
- `mean(h_gap)` and `std(h_gap)` over the 30s window
- `frac(h_gap < 0)` — fraction of time the pipeline *under-estimates* margin (dangerous direction)
- `frac(h_gap > 0.5)` — fraction of time it over-estimates by more than sim's upper DR bound

Aggregate across all 30 trials:
- Histogram of mean(h_gap) values
- Overall mean ± std

## Decision tree

After analysis, compare to sim's `obstacle_radius_perception_error_range` (currently symmetric, see `cbf_go2_env_cfg.py`):

1. **Real h_gap distribution fits inside sim DR range** → no retune needed. Note the calibration in the paper.
2. **Real distribution is *narrower* than sim DR** → no retune needed (sim is conservative, which is fine for transfer).
3. **Real distribution is *wider* than sim DR** → widen the sim DR to cover the real range, re-train one teacher iteration to confirm policy still wins, then deploy.
4. **Real distribution is *shifted* (mean ≠ 0 in the symmetric DR)** → either re-center the DR around the real mean, or add a fixed bias-correction term to the deploy cluster-fit margin.

## Cost & timing

- Measurement session: half a day, single operator
- Analysis: 1-2 hours in a notebook
- Optional sim retune + 1 teacher iteration: ~3 hours wall (matches current iteration cost)

Total: 1 day on hardware-week, doesn't block teacher training in the meantime.

## What this is NOT

- Not a full Mid-360 simulator upgrade (Option 3 in the gap-closing discussion)
- Not a per-frame point-cloud match — only output-statistics match at the cluster-fit level
- Not a moving-obstacle test — separate concern, would need a controlled motion rig

## Deliverables for paper

- One sentence in deploy section: "We calibrated sim perception bias against Mid-360 + cluster-fit output statistics over N=30 trials; measured h_gap distribution: mean=X m, std=Y m, vs sim DR [−0.50, +0.50] m."
- One figure (histogram of mean h_gap per trial overlaid on sim DR range)
