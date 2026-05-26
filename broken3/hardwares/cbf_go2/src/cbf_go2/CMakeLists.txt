##############################################################################
# CMakeLists.txt — cbf_go2
#
# Single ROS 2 package that deploys the adaptive CBF parameter policy
# (trained in ../../../../go2/) on a Unitree Go2.
#
# C++ targets (real-time critical, ~50Hz):
#   teleOp             — keyboard -> /u_teleop
#   odom_publisher     — Go2 /sportmodestate -> /odom + TF(odom->body_link)
#   cloud_merger       — Livox + utlidar -> /poisson_cloud + /occupancy_grid
#   cbf_filter_node    — /u_teleop + /cbf/params + /poisson_cloud -> /u_des
#   walking_bridge     — /u_des + SportClient -> Go2 motors
#
# Python targets (ML inference; installed via PROGRAMS):
#   cbf_inference_node — /odom + /u_teleop + /cbf/grid -> /cbf/params
#   cbf_grid_node      — /poisson_cloud -> /cbf/grid (2x64x64)
##############################################################################

cmake_minimum_required(VERSION 3.14)
project(cbf_go2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -pthread")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(include include/common include/nlohmann include/poisson)

set(DEPENDENCY_LIST
  unitree_go
  unitree_api
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  pcl_conversions
  tf2
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
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Curses REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CURSES_INCLUDE_DIRS})

# ---- C++ executables ----

add_executable(teleOp src/teleop.cpp)
target_link_libraries(teleOp ${CURSES_LIBRARIES})
ament_target_dependencies(teleOp ${DEPENDENCY_LIST})

add_executable(odom_publisher src/odom_publisher.cpp)
ament_target_dependencies(odom_publisher ${DEPENDENCY_LIST})

add_executable(cloud_merger src/cloud_merger_main.cpp src/utils.cpp)
target_link_libraries(cloud_merger ${OpenCV_LIBS} ${PCL_LIBRARIES})
ament_target_dependencies(cloud_merger ${DEPENDENCY_LIST})

add_executable(cbf_filter_node src/cbf_filter_node.cpp)
ament_target_dependencies(cbf_filter_node ${DEPENDENCY_LIST})

add_executable(walking_bridge src/walking_bridge.cpp src/ros2_sport_client.cpp)
ament_target_dependencies(walking_bridge ${DEPENDENCY_LIST})

install(TARGETS
  teleOp
  odom_publisher
  cloud_merger
  cbf_filter_node
  walking_bridge
  DESTINATION lib/${PROJECT_NAME})

# ---- Python nodes ----
# Installed under lib/${PROJECT_NAME}/ so `ros2 run cbf_go2 cbf_inference_node`
# works. cbf_deploy_model.py is installed as FILES (importable, not
# runnable) next to the node executables, matching the import shim:
#   sys.path.insert(0, str(Path(__file__).resolve().parent))

install(PROGRAMS deploy/cbf_inference_node.py
        RENAME cbf_inference_node
        DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS deploy/cbf_grid_node.py
        RENAME cbf_grid_node
        DESTINATION lib/${PROJECT_NAME})
install(FILES deploy/cbf_deploy_model.py
        DESTINATION lib/${PROJECT_NAME})

# ---- Launch + config ----

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

ament_package()
