## Deploy (MacBook -> Lab Desktop -> Go2)

Three machines, no shared internet. Code moves via git + rsync:

```
MacBook Air ──git push──→ GitHub ──git pull──→ Lab Desktop ──rsync──→ Go2 Jetson
  (Wi-Fi)                                      (wall plug)    (Go2 Ethernet cable)
```

The lab desktop has one Ethernet port and no Wi-Fi, so you swap the cable
between the wall (internet) and the Go2 dock.

```bash
# 1. On lab desktop — plug into wall, pull latest
cd ~/Desktop/safety-go2
git pull

# 2. Swap Ethernet cable to Go2 dock

# 3. Send code to Jetson (only transfers changed files after first run)
rsync -avz ~/Desktop/safety-go2/src/ unitree@192.168.123.18:~/safety-go2/src/

# 4. SSH into Go2
ssh unitree@192.168.123.18

# 5. On the Jetson — build and run
cd ~/safety-go2
colcon build
source install/setup.bash
ros2 launch go2_walking_lidar walking_lidar.launch.py
```

# Go2 Walking + LiDAR — Minimal Extraction

Minimal ROS 2 package for Unitree Go2 walking control with LiDAR-based
occupancy grid mapping. Extracted from the `semantic-safety` project
(lab mate's full CBF/Poisson safety field pipeline).

## What this package does

1. **Keyboard teleop** — Arrow keys publish velocity commands
2. **Walking bridge** — Forwards velocity commands to the Go2 motors
3. **LiDAR processing** — Livox Mid360 + Go2 front UTLidar point clouds
   are merged and converted into a 100x100 occupancy grid (5m x 5m, 5cm cells)
4. **Odometry** — FAST_LIO LiDAR-inertial odometry published as `/odom` + TF

## What was removed from the original

- YOLO human detection (camera-based)
- Camera support (RealSense, ZED)
- Semantic class maps and human tracking
- Poisson safety field solver (CUDA)
- CBF/MPC safety filter
- Social navigation (tangent bias)
- Experiment logging (CSV/BIN)
- OpenMP parallelism, OSQP optimizer

## Workspace layout

`cbf_folder` is a colcon workspace. The actual ROS package lives at
`cbf_folder/src/go2_walking_lidar/`:

```
cbf_folder/                                ← workspace root (you cd here to build)
└── src/                                   ← colcon looks for packages here
    └── go2_walking_lidar/                 ← the package
        ├── CMakeLists.txt
        ├── package.xml
        ├── README.md                       ← you are here
        ├── src/                            ← C++ source files
        │   ├── teleop.cpp
        │   ├── walking_bridge.cpp          ← NEW: u_des -> Go2 motors
        │   ├── odom_publisher.cpp
        │   ├── cloud_merger_main.cpp
        │   ├── ros2_sport_client.cpp
        │   └── utils.cpp
        ├── include/                        ← header files
        │   ├── common/
        │   ├── nlohmann/
        │   └── poisson/
        ├── launch/
        │   └── walking_lidar.launch.py
        └── config/
            └── MID360_config.json
```

After building, colcon creates `build/`, `install/`, and `log/` at the
workspace root. Don't commit those.

## Nodes (4 total)

| Node | Executable | What it does |
|---|---|---|
| `teleop` | `teleOp` | Reads arrow keys, publishes Twist on `u_des` |
| `walking_bridge` | `walking_bridge` | Subscribes to `u_des`, calls `SportClient.Move()` to actually move the Go2 |
| `odom_publisher` | `odom_publisher` | Subscribes to Go2 sportmodestate, publishes `/odom` + TF |
| `cloud_merger` | `cloud_merger` | Merges Livox + UTLidar point clouds, publishes `occupancy_grid` |

## File origins

Each file documents where it came from. Full mapping:

| This file | Original source | Notes |
|---|---|---|
| `src/teleop.cpp` | `robot_ws/src/src/teleop.cpp` | Copied verbatim |
| `src/odom_publisher.cpp` | `robot_ws/src/src/odom_publisher.cpp` | Copied verbatim |
| `src/ros2_sport_client.cpp` | `robot_ws/src/src/common/ros2_sport_client.cpp` | Copied verbatim |
| `src/cloud_merger_main.cpp` | New file | Standalone entry point (originally CloudMergerNode was created inside `semantic_poisson.cpp` main()) |
| `src/walking_bridge.cpp` | New file | Replaces the part of `semantic_poisson.cpp` that consumed `u_des` and called SportClient.Move() |
| `src/utils.cpp` | `robot_ws/src/src/utils.cpp` | Stripped to Timer + ang_diff only |
| `include/common/ros2_sport_client.h` | `robot_ws/src/include/common/ros2_sport_client.h` | Copied verbatim |
| `include/common/patch.hpp` | `robot_ws/src/include/common/patch.hpp` | Copied verbatim |
| `include/common/time_tools.hpp` | `robot_ws/src/include/common/time_tools.hpp` | Copied verbatim |
| `include/nlohmann/` | `robot_ws/src/include/nlohmann/` | Copied verbatim (JSON library) |
| `include/poisson/cloud_merger.h` | `robot_ws/src/include/poisson/cloud_merger.h` | Stripped camera callbacks, kept LiDAR only |
| `include/poisson/grid_params.h` | `robot_ws/src/include/poisson/poisson.h` | Only IMAX, JMAX, DS kept (removed QMAX, DQ, TMAX, etc.) |
| `include/poisson/utils.h` | `robot_ws/src/include/poisson/utils.h` | Stripped to Timer + ang_diff only |
| `config/MID360_config.json` | `robot_ws/src/config/MID360_config.json` | Copied verbatim |
| `launch/walking_lidar.launch.py` | `robot_ws/src/launch/semantic_safety.launch.py` | Stripped to LiDAR + walking nodes only |
| `CMakeLists.txt` | `robot_ws/src/CMakeLists.txt` | Removed CUDA, OSQP, OpenMP, semantic_poisson target |
| `package.xml` | `robot_ws/src/package.xml` | Removed unitree_hg, rosbag2_cpp |

## ROS 2 topics

### Published

- `/odom` (nav_msgs/Odometry) — Robot position
- `occupancy_grid` (nav_msgs/OccupancyGrid) — 100x100 confidence grid
- `poisson_cloud` (sensor_msgs/PointCloud2) — Filtered LiDAR points
- `u_des` (geometry_msgs/Twist) — Velocity command (from teleop)
- `key_press` (std_msgs/Int32) — Raw key code (from teleop)
- `/api/sport/request` (unitree_api/Request) — Motor commands to Go2
- TF: `odom -> body_link`

### Subscribed

- `/livox/lidar` — Livox Mid360 point cloud
- `/utlidar/cloud` — Go2 front UTLidar point cloud
- `/livox/imu` — IMU for FAST_LIO
- `sportmodestate` — Go2 robot state
- `u_des` — Velocity from teleop (consumed by walking_bridge)

## Build

```bash
cd ~/safety-go2
colcon build
source install/setup.bash
```

## Run

```bash
ros2 launch go2_walking_lidar walking_lidar.launch.py
```

## Prerequisites

These must be built and sourced BEFORE building this package:

- ROS 2 (Foxy or Humble)
- Unitree ROS 2 packages (`unitree_go`, `unitree_api`) — from `submodules/`
- Livox ROS 2 driver (`livox_ros_driver2`) — from `submodules/`
- FAST_LIO (`fast_lio`) — from `submodules/`

System packages:

- PCL, OpenCV, Eigen3, ncurses

## Architecture

```
Livox Mid360 ──→ /livox/lidar ──→ FAST_LIO ──→ /Odometry ──→ odom_publisher ──→ /odom + TF
                       │
                       └──→ cloud_merger ←── /utlidar/cloud (Go2 front LiDAR)
                                │
                                └──→ occupancy_grid (100x100 confidence map)

Keyboard ──→ teleOp ──→ u_des ──→ walking_bridge ──→ SportClient.Move() ──→ Go2 motors
```

## Controls (teleop keys)

| Key     | Action            |
| ---     | ---               |
| ↑       | Walk forward      |
| ↓       | Walk backward     |
| ←       | Strafe left       |
| →       | Strafe right      |
| `,`     | Turn left (yaw)   |
| `.`     | Turn right (yaw)  |
| `space` | E-STOP (damp)     |
| `q`     | Quit teleop       |

## Open questions to ask your lab mates

Before you can actually run this on the Go2, you need answers to:

1. What's the Go2's IP and ROS_DOMAIN_ID?
2. What ROS 2 distro are you using?
3. Where do the Livox driver and FAST_LIO run — lab computer or Go2?
4. Are unitree_go/unitree_api already built somewhere on the lab computer?
5. Is there an emergency stop procedure for testing?

See the chat history for the full list.
