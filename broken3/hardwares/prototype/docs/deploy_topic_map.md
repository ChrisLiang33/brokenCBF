# Go2 deploy — topic map + integration plan

Survey of existing ROS 2 topics in `src/go2_walking_lidar/` and the
changes needed to insert the V13.1 CBF inference + filter pipeline.

## Existing topology

```
Go2 SDK ──── sportmodestate ──→ odom_publisher ──→ /odom
                                                   TF: odom → body_link
Go2 SDK ──── /livox/lidar ─────┐
Go2 SDK ──── /utlidar/cloud ───┴─→ cloud_merger ──→ occupancy_grid
                                                     poisson_cloud
teleop ────── /u_des ─────────→ walking_bridge ───→ SportClient.Move
              /estop ────────→ walking_bridge ───→ Damp on E-stop
```

**Key Go2 SDK source**: `unitree_go/msg/SportModeState` (already wrapped
by `odom_publisher`). Contains in one message:
- `position[3]`, `velocity[3]` (linear, body frame)
- `imu_state.rpy[3]`, `gyroscope[3]`, `accelerometer[3]`, `quaternion[4]`
- `foot_force[4]`, `foot_position_body[12]`

## What V13.1 inference needs

Per `deploy/cbf_inference_node.py`:
- **base_height** (1-D): from `/odom` pose.z — already published.
- **tracking_err** (15-D = 5×3): `cmd_vel − measured_vel`. cmd_vel from
  teleop, measured_vel from `/odom` twist (linear) + IMU gyro (yaw).
- **base_ang_vel** (3-D): IMU gyro. NOT in current `/odom` (linear only).
  → Either subscribe to `sportmodestate` directly OR extend
  `odom_publisher` to set `twist.angular` too.
- **occupancy grid** (2 × 64 × 64): NEEDS NEW NODE. Current
  `cloud_merger` outputs 100×100 @ 5cm = 5m×5m, single-channel.
  V13.1 expects 64×64 @ 10cm = 6.4m×6.4m, **2-channel** (current + previous frame).

## Required changes for tomorrow

### 1. Rename teleop output (so we can interpose the filter)
- `teleop.cpp`: publish to `/u_teleop` instead of `/u_des`.
- `walking_bridge.cpp`: no change — still subscribes to `/u_des`.
- New CBF filter node: subscribes to `/u_teleop`, publishes to `/u_des`.

### 2. Extend odom_publisher to include angular velocity
Minimal patch in `odom_publisher.cpp`:
```cpp
// add: subscribe to gyroscope from sportmodestate (already received)
odom_msg.twist.twist.angular.x = msg->imu_state.gyroscope[0];
odom_msg.twist.twist.angular.y = msg->imu_state.gyroscope[1];
odom_msg.twist.twist.angular.z = msg->imu_state.gyroscope[2];
```

### 3. New grid construction node — `cbf_grid_node` (Python)
- Subscribe: `poisson_cloud` (filtered points in TF-aligned frame)
  OR `/livox/lidar` directly + do our own TF lookup.
- Transform points to **body frame** (TF: body_link → world inverse).
- Rasterize into 64×64 grid @ 10cm/cell, robot at center, +x = forward.
- Maintain previous-frame buffer; publish both channels as
  `Float32MultiArray` with shape (2 × 64 × 64) = 8192 dims.
- Topic: `/cbf/grid`.
- Rate: match sim ≈ 50 Hz.

### 4. CBF filter node — `cbf_filter_node` (Python)
- Subscribe: `/u_teleop` (Twist), `/cbf/grid` (Float32MultiArray, 8192),
  `/cbf/params` (Float32MultiArray, 5).
- Solve CBF QP per tick. Use α, φ from `/cbf/params` for the QP terms.
- Publish: `/u_des` (Twist) — the filtered/safe command.
- Match the sim's `_cbf_filter()` logic in
  `cbf_go2_env.py` — straight C++/Python port.

### 5. Wire the existing inference node
- `deploy/cbf_inference_node.py` — already built; just verify topic
  names match. Current subscriptions:
  - `/imu/data` → no such Go2 topic; switch to `sportmodestate` OR
    use the extended `/odom` (after change #2).
  - `/odom` → ✓ matches odom_publisher.
  - `/teleop/cmd_vel` → switch to `/u_teleop` (after change #1).
  - `/cbf/grid` → ✓ matches new grid node (change #3).

## Topic contract after changes

```
sportmodestate ──→ odom_publisher ──→ /odom (linear + angular vel) ──┐
                                       TF: odom → body_link          │
/livox/lidar ─────→ cbf_grid_node ──→ /cbf/grid (2×64×64 flat)  ──── ┼──→ cbf_inference_node ──→ /cbf/params
/utlidar/cloud ────┘                                                  │                         /cbf/inference_status
                                                                       │
teleop ──→ /u_teleop ──┬──────────────────────────────────────────────┤
                       └─→ cbf_filter_node ─→ /u_des ──→ walking_bridge ──→ SportClient.Move
            /estop ───────────────────────────────────→ walking_bridge ──→ Damp
```

## Sequencing for tomorrow

1. Patch `odom_publisher.cpp` (5-line angular-vel addition).
2. Patch `teleop.cpp` rename `u_des` → `u_teleop`.
3. Adjust `cbf_inference_node.py` topic names (`/teleop/cmd_vel` →
   `/u_teleop`; `/imu/data` removal — replaced by the extended `/odom`).
4. Write `cbf_grid_node.py` (~150 lines: TF lookup + rasterize).
5. Write `cbf_filter_node.py` (~200 lines: CBF QP port from sim).
6. Build + bring-up test order:
   1. odom_publisher (verify angular vel populated).
   2. cbf_grid_node alone (verify `/cbf/grid` shape + values look reasonable).
   3. cbf_inference_node (verify `/cbf/params` publishes in safe ranges).
   4. cbf_filter_node with robot OFF (verify `/u_des` deflects when
      grid shows obstacles).
   5. Full stack with robot ON, no obstacles (verify walking).
   6. Full stack with one static obstacle (verify CBF kicks in).
   7. Push tests, then adversarial.

## Frame conventions

Sim uses **body frame**, robot-forward = +x, robot-left = +y, robot-up = +z.
Grid is ego-centric: robot at (0, 0), x_world points along grid columns,
y_world points along grid rows.

The Go2's `body_link` frame in TF matches this convention. Use TF
lookup `body_link → odom` to transform world-frame LiDAR points into
robot-relative grid coordinates.
