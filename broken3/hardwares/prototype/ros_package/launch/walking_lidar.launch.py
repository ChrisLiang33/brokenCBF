#!/usr/bin/env python3
"""
Minimal launch file for Go2 walking + LiDAR occupancy grid.

Extracted from: robot_ws/src/launch/semantic_safety.launch.py
Stripped out: cameras, YOLO, human tracking, semantic_poisson, social navigation
Kept: Livox Mid360 LiDAR, FAST_LIO odometry, TF tree, teleop, occupancy grid

Nodes launched:
  1. livox_ros_driver2   — Livox Mid360 LiDAR driver
  2. fastlio_mapping     — LiDAR-Inertial odometry (FAST_LIO)
  3. odom_publisher      — Republishes FAST_LIO /Odometry as /odom + TF
  4. cloud_merger         — LiDAR point clouds -> occupancy grid
  5. teleOp              — Keyboard teleop (arrow keys + comma/period)
  6. Static TFs          — Frame transforms (odom->camera_init, body->livox, etc.)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    min_z_arg = DeclareLaunchArgument(
        'min_z', default_value='0.1',
        description='Minimum height (Z) for point cloud filtering')

    max_z_arg = DeclareLaunchArgument(
        'max_z', default_value='0.5',
        description='Maximum height (Z) for point cloud filtering')

    vel_max_x_fwd_arg = DeclareLaunchArgument(
        'vel_max_x_fwd', default_value='0.75',
        description='Maximum forward velocity (m/s)')

    vel_max_x_bwd_arg = DeclareLaunchArgument(
        'vel_max_x_bwd', default_value='0.75',
        description='Maximum backward velocity (m/s)')

    vel_max_y_arg = DeclareLaunchArgument(
        'vel_max_y', default_value='0.75',
        description='Maximum lateral velocity (m/s)')

    vel_max_yaw_arg = DeclareLaunchArgument(
        'vel_max_yaw', default_value='0.75',
        description='Maximum yaw rate (rad/s)')

    # --- Livox Mid360 LiDAR ---
    livox_lidar_node = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[
            {'xfer_format': 0},
            {'multi_topic': 0},
            {'data_src': 0},
            {'publish_freq': 15.0},
            {'output_data_type': 0},
            {'frame_id': 'livox_frame'},
            {'lvx_file_path': '/home/livox/livox_test.lvx'},
            {'user_config_path': PathJoinSubstitution([
                FindPackageShare('go2_walking_lidar'),
                'config', 'MID360_config.json'
            ])},
            {'cmdline_input_bd_code': 'livox0000000001'}
        ]
    )

    # --- FAST_LIO (LiDAR-Inertial Odometry) ---
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        name='fastlio_mapping',
        output='screen',
        parameters=[{
            'feature_extract_enable': False,
            'point_filter_num': 3,
            'max_iteration': 3,
            'filter_size_surf': 0.5,
            'filter_size_map': 0.5,
            'cube_side_length': 1000.0,
            'runtime_pos_log_enable': False,
            'common.lid_topic': '/livox/lidar',
            'common.imu_topic': '/livox/imu',
            'common.time_sync_en': False,
            'common.time_offset_lidar_to_imu': 0.0,
            'preprocess.lidar_type': 0,
            'preprocess.scan_line': 4,
            'preprocess.blind': 0.5,
            'preprocess.timestamp_unit': 3,
            'preprocess.scan_rate': 10,
            'mapping.acc_cov': 0.1,
            'mapping.gyr_cov': 0.1,
            'mapping.b_acc_cov': 0.0001,
            'mapping.b_gyr_cov': 0.0001,
            'mapping.fov_degree': 360.0,
            'mapping.det_range': 100.0,
            'mapping.extrinsic_est_en': True,
            'mapping.extrinsic_T': [0.0, 0.0, 0.0],
            'mapping.extrinsic_R': [1., 0., 0., 0., 1., 0., 0., 0., 1.],
            'publish.path_en': False,
            'publish.effect_map_en': False,
            'publish.map_en': False,
            'publish.scan_publish_en': False,
            'publish.dense_publish_en': False,
            'publish.scan_bodyframe_pub_en': False,
            'pcd_save.pcd_save_en': False,
        }],
    )

    # --- Static TFs (frame tree) ---
    # odom -> camera_init (FAST_LIO world frame alias)
    odom_to_camera_init_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='odom_to_camera_init_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'camera_init'],
    )

    # body -> livox_frame (FAST_LIO body frame alias)
    body_to_livox_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='body_to_livox_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'body', 'livox_frame'],
    )

    # livox_frame -> body_link (LiDAR mount offset + 180° flip)
    livox_to_body_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='livox_to_body_tf',
        arguments=['-0.05', '0.0', '0.18', '0', '3.14159', '0',
                   'livox_frame', 'body_link'],
    )

    # body_link -> utlidar_lidar (Go2 front UTLidar mount)
    body_to_utlidar_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='body_to_utlidar_tf',
        arguments=['0.37', '0.0', '0.05', '0', '2.9', '0',
                   'body_link', 'utlidar_lidar'],
    )

    # --- Odometry publisher (FAST_LIO -> /odom + TF) ---
    odom_publisher_node = Node(
        package='go2_walking_lidar',
        executable='odom_publisher',
        name='odom_publisher',
        output='screen',
    )

    # --- Cloud merger (LiDAR -> occupancy grid) ---
    cloud_merger_node = Node(
        package='go2_walking_lidar',
        executable='cloud_merger',
        name='cloud_merger',
        output='screen',
        parameters=[{
            'min_z': LaunchConfiguration('min_z'),
            'max_z': LaunchConfiguration('max_z'),
        }],
    )

    # --- Teleop (keyboard control) ---
    teleop_node = Node(
        package='go2_walking_lidar',
        executable='teleOp',
        name='teleop',
        output='screen',
        parameters=[{
            'vel_max_x_fwd': LaunchConfiguration('vel_max_x_fwd'),
            'vel_max_x_bwd': LaunchConfiguration('vel_max_x_bwd'),
            'vel_max_y': LaunchConfiguration('vel_max_y'),
            'vel_max_yaw': LaunchConfiguration('vel_max_yaw'),
        }],
    )

    # --- Walking bridge (u_des -> Go2 SportClient.Move) ---
    walking_bridge_node = Node(
        package='go2_walking_lidar',
        executable='walking_bridge',
        name='walking_bridge',
        output='screen',
    )

    return LaunchDescription([
        min_z_arg,
        max_z_arg,
        vel_max_x_fwd_arg,
        vel_max_x_bwd_arg,
        vel_max_y_arg,
        vel_max_yaw_arg,
        livox_lidar_node,
        odom_to_camera_init_tf,
        body_to_livox_tf,
        livox_to_body_tf,
        body_to_utlidar_tf,
        fast_lio_node,
        odom_publisher_node,
        cloud_merger_node,
        teleop_node,
        walking_bridge_node,
    ])
