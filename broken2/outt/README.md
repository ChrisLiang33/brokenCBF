# outt/ — Go2 CBF deploy bundle

Self-contained snapshot of everything related to deploying the V13.1 student
CBF safety filter on a Unitree Go2. Copy of the live files from the parent
repo, organized for handoff or archival.

## Contents

```
outt/
├── deploy/        ← ROS 2 deploy nodes + scripts (PRIMARY)
│   ├── README.md          ← start here
│   ├── cbf_deploy_model.py
│   ├── cbf_grid_node.py
│   ├── cbf_inference_node.py
│   ├── cbf_filter_node.py
│   └── scripts/           ← tier3_bringup.sh, record_trial.sh, test_deploy_load.py
├── docs/          ← hardware trial protocol + topic survey
└── ros_package/   ← ROS 2 package source (go2_walking_lidar)
                    odom_publisher, teleop, walking_bridge, launch files
```

## Where things go in the actual project

This is a **copy** for handoff. The live working tree is:

| outt/ location | Live repo location |
|---|---|
| `outt/deploy/` | `~/Desktop/safety-go2/deploy/` |
| `outt/docs/` | `~/Desktop/safety-go2/docs/` |
| `outt/ros_package/` | `~/Desktop/safety-go2/src/go2_walking_lidar/` |

To deploy on a fresh Go2:

```bash
# Clone or rsync the full project (not just outt/):
rsync -av ~/Desktop/safety-go2/ unitree@192.168.123.18:~/safety-go2/

# On Go2:
cd ~/safety-go2
colcon build --packages-select go2_walking_lidar --symlink-install
source install/setup.bash
bash ~/safety-go2/deploy/scripts/tier3_bringup.sh
```

See `outt/deploy/README.md` for the full run sequence + topic map + known
issues + when-things-break table.

## Status snapshot (2026-05-22)

- ✓ Tier 1 (sensors) and Tier 2 (raw walk) confirmed
- ✗ Tier 2.5 (perception sanity) pending — gates everything below
- ✗ Tier 3 trials pending (3 conditions × ≥5 trials, alternating)
- See `outt/docs/hardware_trial_protocol.md` for the trial procedure

## Pre-flight before Tier 3

1. Bump `safety_margin` 0.70 → 1.0 m in `cbf_filter_node.py`
2. Fixed-param condition: α=0.5, φ=2.0 (not the originally-planned 2.0/0.5)
3. Procure cone + tape + padded stick + blanket
