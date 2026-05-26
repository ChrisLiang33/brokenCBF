# Tier 3 hardware trial protocol — CoRL 2026 Fig 5

**Status**: Locked 2026-05-21 (eve of trials). Tomorrow = execution, not design.

Goal: produce the hardware demonstration figure + table for the paper. We compare
**three controllers** driving the Go2 toward a fixed cylindrical obstacle, and
measure (a) how close the robot gets, (b) whether it completes the run, (c)
forward velocity, (d) deflection magnitude.

This is the **MVP angle (cylindrical obstacle, matches sim)**. We are NOT
deploying on a polytope obstacle today — that's a stretch goal at best.

## 0. Setting

Hallway or open lab space. ~6 m clear runway. Hard flat floor (lab tile / wood).
Battery > 60% before the block; swap if it dips below 30% mid-block.

## 1. Obstacle

Single cylindrical object, ~0.25–0.35 m diameter, ≥ 0.6 m tall.
**Preferred**: traffic cone. **Acceptable**: stack 2 cube storage bins vertically
(~0.4 × 0.4 × 0.8 m total — still convex enough that the LiDAR cluster's nearest
point behaves like a cylinder at the distances we deflect at).

Cardboard box from prior plan is OK but less paper-friendly photo.

## 2. Geometry

```
   START                    OBSTACLE                  GOAL
   [X]──────── 3 m ────────[ O ]──────── 2 m ──────[X]
    ^                                                ^
   robot                                          finish line
   faces +x                                        (tape on floor)
```

Mark all three positions with tape. Measure once. Use the same start pose for
every trial across every condition — that's what makes the comparison clean.

## 2.5. Perception sanity check (Tier 2.5) — do this ONCE before any trial block

**Why this exists**: if h(x) is computed wrong from real LiDAR, every trial that
follows will look like the filter works while actually deflecting on garbage.
Reviewers won't catch it. Bag plots won't catch it. Only a static bench test
catches it.

**Setup**: robot standing in its start pose, RecoveryStand done, `tier3_bringup.sh`
running, **`walking_bridge` NOT started**. Robot does not move during this check.

**Check A — h(x) tracks true distance to obstacle**:

Place cone exactly 1.5 m in front of the robot (measure with tape). Then in a
fresh terminal on the Go2:

```bash
source ~/safety-go2/install/setup.bash
ros2 topic echo /cbf/filter_h --field data --once
# Expect: data[0] (h) ≈ 1.5 - 0.70 = 0.80 m  (±0.1 OK)
#         data[1], data[2] (L_g h x, y): ‖(Lgh_x, Lgh_y)‖ ≈ 1.0
```

Slide the cone to 2.0 m. Re-echo. Expect h ≈ 1.30.

Slide the cone to 0.5 m. Re-echo. Expect h ≈ −0.20 (negative — safety violated).
Filter state should show `OK_DEFLECTED` or large negative slack.

**Check B — deflection direction is correct**:

With cone still at ~1.0 m ahead, publish a fake forward teleop command:

```bash
ros2 topic pub --once /u_teleop geometry_msgs/Twist "{linear: {x: 0.5, y: 0.0}}"
ros2 topic echo /u_des --once
# Expect: linear.x significantly reduced (< 0.3) AND/OR linear.y nonzero
# i.e., the filter deflected away from the obstacle.
```

Move the cone to the robot's LEFT (~1.0 m at 90° offset). Re-publish forward
teleop. Expect /u_des linear.y to be NEGATIVE (deflecting away from cone, to
the right). Move cone to the RIGHT, expect linear.y POSITIVE.

**Check C — failsafe behavior**:

Remove the cone entirely (clear field of view). Echo `/cbf/filter_status`.
Expect `state=OK_NO_OBSTACLE`. /u_des should equal /u_teleop verbatim.

**Pass criteria**: A + B + C all behave as above. If any fail, debug before
running trials. Common failure modes:

- TF off: utlidar_lidar ↔ body_link static_transform_publisher args wrong → grid is rotated → deflection direction wrong
- Safety margin off: hardcoded 0.70 m in `cbf_filter_node.py` doesn't match what you want → adjust before trials
- Grid resolution off: `cbf_grid_node` body-frame cell size doesn't match training → h is computed on the wrong scale

This check is ~10–15 minutes. Do it after Tier 3 smoke test, before the first
recorded trial. Skipping it is the most likely way the paper claim falls apart
under review.

## 3. Three conditions

| ID            | Controller stack                          | Bag tag          |
|---------------|-------------------------------------------|------------------|
| `raw_no_cbf`  | Teleop → walking_bridge directly          | `raw_no_cbf`     |
| `fixed_b1`    | Teleop → CBF filter w/ const (α=2, φ=0.5) | `fixed_b1_a2_p05`|
| `ours_v13_1`  | Teleop → CBF filter w/ learned (α,φ)      | `ours_v13_1`     |

### How to switch between conditions (without restarting tier3 every time)

**`raw_no_cbf`**: kill `walking_bridge`, restart with remap:

```bash
ros2 run go2_walking_lidar walking_bridge --ros-args -r /u_des:=/u_teleop
```

(walking_bridge now consumes teleop directly; cbf_filter_node still runs but
its `/u_des` is ignored. Recording still captures /u_teleop + /u_des so we can
verify the filter would have deflected.)

**`fixed_b1`**: kill `cbf_inference_node` (tmux pane 6), then in that window:

```bash
ros2 topic pub -r 50 /cbf/params std_msgs/msg/Float32MultiArray \
  "{data: [2.0, 0.5, 0.0, 0.0, -0.05]}"
# layout: [α, φ, a, b, c]
```

filter_node consumes /cbf/params unchanged. State should be OK / OK_DEFLECTED.

**`ours_v13_1`**: default tier3 stack. inference_node alive, publishing learned
params at 50 Hz.

## 4. Per-trial sequence

1. Robot at start pose. Operator confirms RecoveryStand, robot is standing,
   all 8 tmux windows healthy.
2. Pre-trial check (separate Go2 terminal):
   ```bash
   ros2 topic echo /cbf/inference_status --once   # state: OK
   ros2 topic echo /cbf/filter_status --once      # state: OK or OK_NO_OBSTACLE
   ros2 topic hz /cbf/params                       # ~50 Hz
   ros2 topic hz /poisson_cloud                    # ~10 Hz
   ```
3. Start recording:
   ```bash
   bash ~/safety-go2/deploy/scripts/record_trial.sh <bag_tag> <trial_idx>
   ```
4. Operator holds forward arrow (continuous v_x command). NO yaw/strafe input —
   we want the controller to be the only thing deflecting.
5. Trial ends on one of:
   - **success**: robot crosses goal line (operator visually confirms)
   - **collision**: robot makes contact with obstacle (operator yells STOP)
   - **fall**: robot loses footing / topples (e-stop)
   - **timeout**: 20 s elapsed, robot not at goal line
   - **abort**: operator releases forward, restart (don't count)
6. Operator stops recording (Ctrl-C on the record_trial terminal).
7. Operator logs the outcome verbally / on paper.

## 5. Trial counts and order

- **5 trials per condition minimum**, 10 if battery + time allow.
- **Alternating order**: raw → fixed → ours → raw → fixed → ours → ... This
  controls for battery drift and floor wear within a session. Don't run all
  5 raw trials in a row.

## 6. Metrics (extracted post-hoc via `scripts/plot_trial_bag.py`)

Per trial, the plotter dumps a JSON summary alongside the PNG. Headline metrics
for the paper table:

- `d_min` [m]: closest robot–obstacle distance (from `/cbf/filter_h`, computed
  as `safety_margin + h_min`)
- `outcome` ∈ {success, collision, fall, timeout} (logged manually, merged in)
- `t_complete` [s]: time from first nonzero v_x to crossing goal line (NaN if
  not success)
- `v_mean` [m/s]: mean ‖(v_x, v_y)‖ from /odom over trial duration
- `defl_mean` [m/s]: mean ‖u_teleop − u_des‖ over trial duration

## 7. Headline figure (Fig 5)

After all trials are recorded:

```bash
# laptop side, after rsync of trials/ from Go2
for cond in raw_no_cbf fixed_b1_a2_p05 ours_v13_1; do
  for d in data_from_lab/trials/$cond/trial_*; do
    python3 scripts/plot_trial_bag.py "$d" \
      --output docs/viz/$(basename $cond)_$(basename $d).png \
      --title "$cond / $(basename $d)"
  done
done
```

Then aggregate the JSONs into a bar chart (TBD: `scripts/aggregate_trials.py`,
write tomorrow after trials are done — don't pre-build, structure may shift).

For paper:
- Photo of physical setup (top-down + side, 1 each)
- 3-panel trajectory overlay: top-down x-y trajectories per condition,
  obstacle marked as black circle, goal line dashed
- Bar chart: d_min (mean ± std) per condition; success rate as text annotation

## 8. Pre-flight safety checklist (every block)

- [ ] Battery > 60%
- [ ] E-stop key in operator's hand
- [ ] Floor clear of cables, observers > 2 m from obstacle
- [ ] All 8 tier3 tmux windows show no error tracebacks
- [ ] `inference_status` = OK (NOT INPUT_STALE / TILT_FAIL)
- [ ] `filter_status` ∈ {OK, OK_NO_OBSTACLE} (NOT PASSTHROUGH_LIDAR_STALE)
- [ ] α, φ varying when robot moves (echo `/cbf/params` briefly — they should
      NOT be the safe defaults α=3.0, φ=2.0)

## 9. Stop-the-block conditions (abort the session if any happen)

- Robot falls 2× in 3 trials → check locomotion, battery, floor
- inference_status flips to TILT_FAIL or NAN_INPUT → debug, don't keep trialing
- filter_status pinned at PASSTHROUGH_* → /poisson_cloud or /cbf/params broken,
  fix before continuing
- d_min consistently < 0.2 m on `ours_v13_1` → filter is failing, do NOT
  continue to risk the robot; investigate

## 10. After the session

```bash
# From laptop, pull everything:
rsync -av unitree@192.168.123.18:~/safety-go2/trials/ \
  ~/Desktop/safety-go2/data_from_lab/trials/
```

Then plot, aggregate, log results into `docs/paper_outline.md` Hardware table.

---

## Stretch scenarios — run only if MVP block is in the bag

These add real "show off" weight to the hardware section. They share the
same `tier3_bringup.sh` stack — only the trial procedure and bag tag change.
Each is independently optional; do whichever survives the day.

## 11. Push perturbation — adaptive-parameter showcase

**Why this exists**: the MVP straight-line approach barely exercises adaptive
behavior. A fixed (α=2.0, φ=0.5) controller can handle a static obstacle on a
flat floor just fine — so on the MVP plot, ours looks similar to fixed-param.
The push test forces the controller into a regime where adaptation matters:
sudden disturbance, time pressure, must re-plan in <500 ms.

This is the most direct video-able demo of "adaptive parameters do work."

**Setup**: same geometry as Section 2 (start, obstacle, goal). Operator B
stands ~1.5 m off the robot's path with a padded stick (foam pool noodle,
soft broom, anything that won't bruise the chassis).

**Trial sequence**:

1. Start recording: `bash deploy/scripts/record_trial.sh <cond>_push <idx>`
2. Operator A holds forward teleop, robot starts walking toward obstacle
3. When robot is ~1.5 m from obstacle (visually estimate), **Operator B
   shoves the robot sideways TOWARD the obstacle** with one firm push
   (~1 s contact, sideways component, no downward force on the back)
4. Continue holding forward teleop — let the controller recover
5. Trial ends per normal stop conditions (success / collision / fall / 20 s)
6. Stop recording

**Conditions**: 2 conditions, NOT 3 (raw no-CBF will collide every time —
not informative, and risks the robot).

- `fixed_b1_a2_p05_push` — fixed α=2.0, φ=0.5
- `ours_v13_1_push` — V13.1 student

**Trial count**: 3–5 per condition (lower than MVP because each is harder
to reset cleanly).

**What to look for in post-hoc plots** (don't pre-commit to a finding —
report what you see):

- Trace of α(t) and φ(t) around the push event for `ours_*` — expected
  spike if adaptation works
- d_min around the push event
- Recovery time (push → forward velocity restored)
- Fixed-param failures (collision, fall, freeze) vs ours-condition outcomes

**Safety**: if `ours_v13_1_push` collides 2× in 3 trials, STOP. Either the
push is too hard or the controller doesn't generalize. Don't keep trialing.

## 12. Slippery surface patch — perception/dynamics adaptation

**Why this exists**: lower-stakes alternative to Section 11 (no human-on-robot
contact). Tests whether adaptation responds to the *dynamics* changing under
the robot rather than an external impulse. Fits the paper's "robust-CBF
ISSf actuation margin (φ)" story directly — φ should grow when actuation
is less effective (low friction = slipping = effective actuation reduced).

**Setup**: tape a fleece blanket or thin foam mat to the floor, **starting
~0.8 m before the obstacle and extending to the obstacle base**. Robot walks
onto the patch in the last meter of its approach. Same start/goal as MVP.

```text
   START                BLANKET ZONE    GOAL
   [X]──── 2.2 m ────[~~ 0.8 m ~~][ O ]──── 2 m ────[X]
                       low μ here
```

**Trial sequence**: same as MVP Section 4.

**Conditions**: same 3 as MVP (raw, fixed, ours), bag tags:

- `raw_no_cbf_slip`
- `fixed_b1_a2_p05_slip`
- `ours_v13_1_slip`

**Trial count**: 3 per condition (this is supplementary, MVP and Section 11
are higher priority).

**What to look for**:

- Fixed-param: did the robot overshoot through the low-friction zone and
  end up too close to the obstacle, because the fixed-α controller didn't
  anticipate reduced braking?
- Ours: did φ grow when the robot's feet started slipping (proprio noise
  and IMU signals it)? Is d_min larger than fixed?

**Skip this if MVP + Section 11 ate the day.** It's the lowest-priority
stretch goal of the three.

