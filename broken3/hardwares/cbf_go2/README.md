# cbf_go2 — Adaptive CBF deploy stack for Unitree Go2

Sim-to-real bridge for the adaptive CBF parameter policy trained in
[`../../go2/`](../../go2/). One ROS 2 colcon workspace, one package
(`cbf_go2`), bringup scripts at the root.

## Where things came from

| What                                           | Source                                      |
|------------------------------------------------|---------------------------------------------|
| C++ plumbing (`odom_publisher`, `cloud_merger`, `teleop`, `walking_bridge`) and includes (`common/`, `poisson/`, `nlohmann/`) | Copied verbatim from [`../prototype/ros_package/`](../prototype/ros_package/) |
| LiDAR / FAST_LIO / Unitree SDK external workspaces | Referenced (not vendored) from [`../semantic-safety/submodules/`](../semantic-safety/submodules/) — see `setup_env.sh` |
| `cbf_grid_node.py`                             | Copied verbatim from [`../prototype/deploy/`](../prototype/deploy/) (2×64×64 ego-centric occupancy grid) |
| `cbf_inference_node.py`                        | Adapted from prototype: action space trimmed to (α, φ); a/b/c dropped to match MVP |
| `cbf_deploy_model.py`                          | Rewritten as a clean stub for the cbf_rl_mvp policy (V0 returns SAFE_DEFAULTS) |
| `cbf_filter_node.cpp` (NEW)                    | Ported the closed-form 2D CBF QP projection from prototype's Python to C++ |
| `launch/cbf_go2.launch.py` & `scripts/bringup.sh` | Modeled on prototype's `cbf_deploy.launch.py` and `tier3_bringup.sh` |

C++ for the real-time path (50 Hz filter + sensor plumbing). Python for
the ML inference node, matching the prototype and lab mate's stack.

## Layout

```
cbf_go2/
├── README.md                       ← this file
├── setup_env.sh                    ← source first: ROS 2 + unitree_ros2 + livox driver
├── build.sh                        ← colcon build --packages-select cbf_go2
├── scripts/
│   └── bringup.sh                  ← tmux launcher (no walking_bridge — manual opt-in)
├── run/
│   └── checkpoints/                ← drop trained .pt here
└── src/
    └── cbf_go2/                    ← ROS 2 package
        ├── CMakeLists.txt
        ├── package.xml
        ├── config/MID360_config.json
        ├── launch/cbf_go2.launch.py
        ├── include/                ← common, poisson, nlohmann (copied)
        ├── src/                    ← C++ nodes
        │   ├── odom_publisher.cpp
        │   ├── cloud_merger_main.cpp + utils.cpp
        │   ├── teleop.cpp
        │   ├── cbf_filter_node.cpp           ← NEW
        │   ├── walking_bridge.cpp + ros2_sport_client.cpp
        └── deploy/                 ← Python nodes (installed under lib/)
            ├── cbf_grid_node.py
            ├── cbf_inference_node.py
            └── cbf_deploy_model.py
```

## Topic graph

```
/sportmodestate ──► odom_publisher ──► /odom (+ TF odom→body_link)
                                              │
/livox/lidar ──► cloud_merger ──► /poisson_cloud ──┬──► cbf_grid_node ──► /cbf/grid
                                                    │
                                                    └──► cbf_filter_node ◄── /cbf/grid (not used; see note)
                                                                  ▲
                                              /odom ──► cbf_inference_node ──► /cbf/params [α, φ]
                                       /u_teleop ──┘            │
                                                    ▼            ▼
keyboard ──► teleop ──► /u_teleop ──► cbf_filter_node ──► /u_des ──► walking_bridge ──► SportClient.Move
```

`cbf_filter_node` reads `/poisson_cloud` directly to compute `h(x)` /
`∇h` from the nearest in-plane LiDAR point (closed-form 2D projection).
`/cbf/grid` is only consumed by `cbf_inference_node` as policy input.

## What's stubbed vs real

| Component                          | Status                                                                 |
|------------------------------------|------------------------------------------------------------------------|
| All C++ plumbing                   | Real, hardware-tested (lifted from prototype)                          |
| `cbf_filter_node` (C++ QP)         | Real, ports proven Python logic 1:1                                     |
| `cbf_grid_node`                    | Real (verbatim from prototype)                                          |
| `cbf_inference_node` fail-safes    | Real (input watchdog, tilt fail-safe, NaN guards, heartbeat)            |
| `CbfDeployModel.infer()`           | **STUB** — returns SAFE_DEFAULTS (α=2.0, φ=0.5) until a real `.pt` is wired |
| `_build_proprio()` layout          | **PLACEHOLDER** — currently mirrors prototype's 19-D V13 vector; needs to match the final Go2 policy obs order |

The pipeline comes up end-to-end on day 1 with the stub — useful for
validating wiring + perception + filter before training is done.

## First-run checklist

1. **External overlays** (one-time, on the deployment Jetson):
   ```bash
   cd ../semantic-safety/submodules/unitree_ros2  && colcon build
   cd ../semantic-safety/submodules/ws_livox      && colcon build
   ```
2. **Build this workspace**:
   ```bash
   source setup_env.sh
   ./build.sh
   ```
3. **Dry-run (no robot motion)** — launches everything except walking_bridge:
   ```bash
   source install/setup.bash
   ros2 launch cbf_go2 cbf_go2.launch.py
   ```
   Verify in a second terminal:
   ```bash
   ros2 topic echo /cbf/inference_status   # state=OK
   ros2 topic echo /cbf/filter_status      # state=OK / OK_DEFLECTED / PASSTHROUGH_*
   ros2 topic hz   /cbf/params              # ≈ 50 Hz
   ros2 topic hz   /u_des                   # ≈ 50 Hz
   ```
4. **Enable motion** (opt-in):
   ```bash
   ros2 launch cbf_go2 cbf_go2.launch.py enable_walking_bridge:=true \
       checkpoint:=$(pwd)/run/checkpoints/policy.pt
   ```
   or use `scripts/bringup.sh` for a tmux layout and start `walking_bridge` manually.

## Wiring a real policy

Three edits, all in `src/cbf_go2/deploy/cbf_deploy_model.py`:

1. `_load_checkpoint()` — build the encoder + MLP modules and restore
   the trained state_dict (SB3 or rsl_rl, depending on training side).
2. `_forward()` — `torch.no_grad()` forward pass + `tanh` decode into
   `[ALPHA_MIN, ALPHA_MAX]` and `[PHI_MIN, PHI_MAX]`.
3. In `cbf_inference_node.py`, update `_build_proprio()` to match the
   final Go2 obs ordering (cross-check against the training env).

No fail-safe / topic / build changes needed for the swap.

## Defer list

- **FAST_LIO odometry** — prototype runs fine on `/sportmodestate`
  alone; add only if drift becomes an issue.
- **Semantic-safety's YOLO / human tracking / Poisson solver** — not
  needed for the CBF experiment; we route around them.
- **Trial recording / protocol scripts** — port from
  `../prototype/deploy/scripts/record_trial.sh` once we're past
  Tier 2.5 perception sanity.
