# Project context

Minimal ROS 2 package extracted from lab mate's `semantic-safety` CBF/Poisson project.
Only walking (teleop) + LiDAR occupancy grid kept. Safety filter, YOLO, cameras all removed.
Original source: `~/Desktop/SafetyCBF/semantic-safety/robot_ws/src/`

# About the user

- Python background, learning C++ and ROS 2
- Prefer `//` comments, not `/** */` block comments
- Always provide Python equivalents when explaining C++ concepts
- Top-down learner — big picture first, then details
- Use ASCII visualizations for spatial concepts

# Hardware

- Unitree Go2 quadruped with Livox Mid360 LiDAR
- Go2 runs ROS 2 Humble on Jetson Orin
- Lab computer connects via Ethernet, user SSHes into Go2
- Ubuntu 24.04 on local machine (Jazzy if needed locally, but primarily SSH)

# Status

- All source files extracted and in place
- walking_bridge.cpp is NEW (bridges teleop to SportClient.Move)
- cloud_merger_main.cpp is NEW (standalone entry point)
- Not yet built or tested on hardware
- Need answers from lab mates: Go2 IP, existing workspaces, which nodes already run
