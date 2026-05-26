# Go2 CBF Deploy Stack

Adaptive Control Barrier Function (CBF) safety filter for a Unitree Go2.
Teleop velocity → CBF projection using LiDAR perception → safe command to
Go2's frozen locomotion. α and φ are output per step by a trained student
policy (distilled from a two-stream RMA teacher).

## File map

```
deploy/
├── cbf_deploy_model.py     ← PyTorch model loader (no ROS deps)
├── cbf_grid_node.py        ← ROS 2 node: LiDAR cloud → 64×64 grid
├── cbf_inference_node.py   ← ROS 2 node: model → (α, φ)
├── cbf_filter_node.py      ← ROS 2 node: closed-form CBF projection
└── scripts/
    ├── tier3_bringup.sh    ← 8-node tmux bringup
    ├── record_trial.sh     ← rosbag2 capture
    └── test_deploy_load.py ← ckpt smoke test
```

The four `.py` nodes stay at `deploy/` root because `src/go2_walking_lidar/CMakeLists.txt`
hardcodes those paths.

## Run

```bash
# On Go2:
ssh unitree@192.168.123.18
source ~/safety-go2/install/setup.bash
bash ~/safety-go2/deploy/scripts/tier3_bringup.sh
tmux attach -t tier3
```

Verify before enabling motion (separate Go2 terminal):

```bash
source ~/safety-go2/install/setup.bash
ros2 topic echo /cbf/inference_status --once   # state: OK
ros2 topic echo /cbf/filter_status --once      # state: OK or OK_NO_OBSTACLE
ros2 topic hz /cbf/params                      # ~50 Hz
ros2 topic hz /poisson_cloud                   # ~10 Hz
```

Enable motion:

```bash
ros2 run go2_walking_lidar walking_bridge
```

Record a trial:

```bash
bash ~/safety-go2/deploy/scripts/record_trial.sh <condition_name> <trial_idx>
```

## Topics

```
/sportmodestate  →  odom_publisher  →  /odom (300 Hz)
/utlidar/cloud + /livox/lidar  →  cloud_merger  →  /poisson_cloud (10 Hz)
/poisson_cloud  →  cbf_grid_node  →  /cbf/grid (50 Hz)
/cbf/grid + /odom  →  cbf_inference_node  →  /cbf/params (50 Hz)
/u_teleop + /poisson_cloud + /cbf/params  →  cbf_filter_node  →  /u_des (50 Hz)
/u_des  →  walking_bridge  →  Unitree SportClient.Move
```

Status topics: `/cbf/inference_status`, `/cbf/filter_status`, `/cbf/filter_h`.

## CBF inequality

```
   L_g h · u  ≥  -α(h - c)  +  φ ‖L_g h‖²  +  a  +  b ‖u‖
```

V13.1 deploy: `a = 0, b = 0, c = -0.05` (fixed); α and φ are per-step adaptive.
`safety_margin = 0.70 m` in `cbf_filter_node.py` — the distance at which h = 0.

## Tiered validation

| Tier | What | Status |
|---|---|---|
| 1 | Sensors stream (`/sportmodestate`, `/utlidar/cloud`, `/livox/lidar`) | ✓ confirmed |
| 2 | Robot walks under raw teleop, no CBF | ✓ confirmed |
| 2.5 | Perception sanity: cone at known distances, verify h(x) tracks | **pending** |
| 3 | Full stack + motion, multi-condition trials | **pending** |

Tier 2.5 procedure: robot stationary, cone at 1.5 m → expect h ≈ 0.80. Slide to 2.0 m
→ h ≈ 1.30. Slide to 0.5 m → h ≈ −0.20 (negative; filter should deflect).

## Known issues

- **V13.1 outputs aggressive α (~3.4 avg)** — the policy converged to high α
  in sim. Hardware compensation: bump `safety_margin` to 1.0 m in
  `cbf_filter_node.py` to give the filter more geometric buffer.
- **Best-fixed comparison condition** should be α=0.5, φ=2.0 (not α=2.0, φ=0.5
  as originally planned). Multi-seed eval shows the former wins on all 4 dists.
- **Sim video recording broken** on the lab box (RTX 5090 + iray cc 12.0 unsupported in driver 595).
  Use existing April 28 sim videos at `IsaacLab/logs/rsl_rl/cbf_go2_teacher/2026-04-28_20-20-42/videos/play/`
  as paper figure substitute. Don't try to downgrade driver — shared box.
- **Tufts_Secure has client isolation.** Use eduroam from laptop or wire into
  lab box's ethernet IP (130.64.84.163).

## When something breaks

| Symptom | Look at |
|---|---|
| `inference_status` = TILT_FAIL | Robot lifted / tilted; set down level |
| `inference_status` = INPUT_STALE | Check `ros2 topic hz` on `/odom`, `/cbf/grid` |
| `filter_status` = PASSTHROUGH_LIDAR_STALE | `cloud_merger` died or LiDAR driver crashed |
| `filter_status` = PASSTHROUGH_PARAMS_STALE | `cbf_inference_node` died or too slow |
| Robot doesn't deflect | Run Tier 2.5 check; if h(x) wrong, TF for `body_link → utlidar_lidar` is off |
| Build error: `cbf_deploy_model not found` | `install(FILES ...)` line in CMakeLists is wrong |

For node-level errors: `tmux attach -t tier3`, then Ctrl-B + window number.

## What's deployed vs what's in sim

- The **student** policy (BS-A) is deployed. It estimates ẑ_env from a 50-step
  history of (proprio, prev_action). The teacher (BR) requires ground-truth
  privileged env state and CAN'T run on hardware.
- The **CBF filter** is the actual safety mechanism. The policy outputs
  α and φ for the filter, but the filter does the avoidance. If the filter is
  bypassed (`raw_no_cbf` condition), no policy can prevent collision.
- The **locomotion controller** is Unitree's stock policy (frozen, runs on Go2
  firmware). We never train low-level joint control.
- Sim obstacles are kinematic (robot can walk through them); hardware obstacles
  are physical. The sim policy was rewarded for getting close to obstacles
  without contact, which produces aggressive α on hardware.

## Related docs

- Trial protocol: `~/Desktop/safety-go2/docs/hardware_trial_protocol.md`
- Sim training: `~/Desktop/safety-go2/IsaacLab/`
- Paper outline: `~/Desktop/safety-go2/docs/paper_outline.md`
