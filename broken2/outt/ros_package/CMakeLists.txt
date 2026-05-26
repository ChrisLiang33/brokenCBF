##############################################################################
# CMakeLists.txt — Minimal Go2 Walking + LiDAR
#
# Extracted from: robot_ws/src/CMakeLists.txt
# Removed: CUDA, Poisson solver, OSQP/MPC, OpenMP, semantic_poisson target
# Kept: teleOp, odom_publisher, cloud_merger
##############################################################################

cmake_minimum_required(VERSION 3.14)
project(go2_walking_lidar)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -pthread")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(include include/common include/nlohmann include/poisson)

# Dependencies
set(DEPENDENCY_LIST
  unitree_go
  unitree_api
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  pcl_conversions
  tf2_ros
  tf2_eigen
  OpenCV
)

find_package(ament_cmake REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_api REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Curses REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CURSES_INCLUDE_DIRS})

# ---- Executables ----

# Keyboard teleop controller
add_executable(teleOp src/teleop.cpp)
target_link_libraries(teleOp ${CURSES_LIBRARIES})
ament_target_dependencies(teleOp ${DEPENDENCY_LIST})

# Odometry publisher (Go2 sportmodestate -> /odom + TF)
add_executable(odom_publisher src/odom_publisher.cpp)
ament_target_dependencies(odom_publisher ${DEPENDENCY_LIST})

# LiDAR point cloud -> occupancy grid
add_executable(cloud_merger src/cloud_merger_main.cpp src/utils.cpp)
target_link_libraries(cloud_merger ${OpenCV_LIBS} ${PCL_LIBRARIES})
ament_target_dependencies(cloud_merger ${DEPENDENCY_LIST})

# Walking bridge (u_des -> SportClient.Move)
add_executable(walking_bridge src/walking_bridge.cpp src/ros2_sport_client.cpp)
ament_target_dependencies(walking_bridge ${DEPENDENCY_LIST})

# ---- Install ----

install(TARGETS
  teleOp
  odom_publisher
  cloud_merger
  walking_bridge
  DESTINATION lib/${PROJECT_NAME})

# V14 (2026-05-21): Python ROS 2 nodes for the CBF inference + grid + filter
# pipeline. Lives under <repo>/deploy/, copied to lib/.../<name> (no .py
# suffix) so `ros2 run go2_walking_lidar cbf_inference_node` works.
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/../../deploy/cbf_inference_node.py
        RENAME cbf_inference_node
        DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/../../deploy/cbf_grid_node.py
        RENAME cbf_grid_node
        DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/../../deploy/cbf_filter_node.py
        RENAME cbf_filter_node
        DESTINATION lib/${PROJECT_NAME})

# Also install cbf_deploy_model.py NEXT TO the executables so they can
# `from cbf_deploy_model import CbfDeployModel` via the `sys.path.insert(
# Path(__file__).parent)` shim in each node. NOT a program (not meant to
# be run standalone) — installed via FILES.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../deploy/cbf_deploy_model.py
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

ament_package()
