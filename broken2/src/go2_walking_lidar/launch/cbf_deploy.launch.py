#!/usr/bin/env python3
"""V14/V15 CBF deploy launch.

Layers the 3 new Python ROS 2 nodes on top of the existing walking + LiDAR
stack:
  - cbf_grid_node      — /poisson_cloud → /cbf/grid (2×64×64 ego-centric grid)
  - cbf_inference_node — /odom + /u_teleop + /cbf/grid → /cbf/params
  - cbf_filter_node    — /u_teleop + /cbf/params + /poisson_cloud → /u_des

Plus reuses everything walking_lidar.launch.py already sets up:
  - livox_ros_driver2 (Mid360 LiDAR)
  - fast_lio (LiDAR-inertial odometry)
  - 4 static TFs (odom→camera_init, body→livox, livox→body_link, body→utlidar)
  - odom_publisher (FAST_LIO /Odometry → /odom + TF)
  - cloud_merger (LiDAR → /poisson_cloud)
  - teleop (now publishes /u_teleop instead of /u_des)
  - walking_bridge (consumes filtered /u_des from cbf_filter_node)

Usage on Go2:
  ros2 launch go2_walking_lidar cbf_deploy.launch.py \\
    teacher_ckpt:=/home/unitree/safety-go2/checkpoints/teacher_v14.pt \\
    student_ckpt:=/home/unitree/safety-go2/checkpoints/student_v14.pt
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ── Args ────────────────────────────────────────────────────────────
    teacher_ckpt_arg = DeclareLaunchArgument(
        'teacher_ckpt',
        default_value='/home/unitree/safety-go2/checkpoints/teacher.pt',
        description='V14 teacher checkpoint (.pt) — used by cbf_inference_node.',
    )
    student_ckpt_arg = DeclareLaunchArgument(
        'student_ckpt',
        default_value='/home/unitree/safety-go2/checkpoints/student.pt',
        description='V14 student adapter checkpoint (.pt).',
    )
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Inference device. Jetson Orin: try "cuda" first; fall back to "cpu" if GPU is loaded by other nodes.',
    )
    cloud_topic_arg = DeclareLaunchArgument(
        'cloud_topic',
        default_value='/poisson_cloud',
        description='Filtered point cloud (from cloud_merger) consumed by cbf_grid_node + cbf_filter_node.',
    )
    safety_margin_arg = DeclareLaunchArgument(
        'safety_margin',
        default_value='0.70',
        description='r_robot + r_obstacle nominal — defines h(x) = ||p_nearest|| - safety_margin.',
    )
    enable_filter_arg = DeclareLaunchArgument(
        'enable_filter',
        default_value='true',
        description='When false, cbf_filter_node still loads but passes u_teleop through unchanged (testing).',
    )
    enable_walking_bridge_arg = DeclareLaunchArgument(
        'enable_walking_bridge',
        default_value='true',
        description='When false, walking_bridge is NOT launched — robot does not stand up. '
                    'Use for first-time dry runs to verify perception+inference stack '
                    'without any risk of robot motion.',
    )
    # Velocity + grid Z filter args (passthrough to teleop / cloud_merger).
    min_z_arg = DeclareLaunchArgument('min_z', default_value='0.05')
    max_z_arg = DeclareLaunchArgument('max_z', default_value='0.80')
    vel_max_x_fwd_arg = DeclareLaunchArgument('vel_max_x_fwd', default_value='0.75')
    vel_max_x_bwd_arg = DeclareLaunchArgument('vel_max_x_bwd', default_value='0.75')
    vel_max_y_arg     = DeclareLaunchArgument('vel_max_y',     default_value='0.75')
    vel_max_yaw_arg   = DeclareLaunchArgument('vel_max_yaw',   default_value='0.75')

    # ── Livox Mid360 LiDAR ──────────────────────────────────────────────
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
                'config', 'MID360_config.json',
            ])},
            {'cmdline_input_bd_code': 'livox0000000001'},
        ],
    )

    # ── FAST_LIO odometry ───────────────────────────────────────────────
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

    # ── Static TFs ──────────────────────────────────────────────────────
    odom_to_camera_init_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='odom_to_camera_init_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'camera_init'],
    )
    body_to_livox_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='body_to_livox_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'body', 'livox_frame'],
    )
    livox_to_body_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='livox_to_body_tf',
        arguments=['-0.05', '0.0', '0.18', '0', '3.14159', '0',
                   'livox_frame', 'body_link'],
    )
    body_to_utlidar_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='body_to_utlidar_tf',
        arguments=['0.37', '0.0', '0.05', '0', '2.9', '0',
                   'body_link', 'utlidar_lidar'],
    )

    # ── Odometry publisher (sportmodestate-based, patched for ang vel) ──
    odom_publisher_node = Node(
        package='go2_walking_lidar',
        executable='odom_publisher',
        name='odom_publisher',
        output='screen',
    )

    # ── Cloud merger (LiDAR → /poisson_cloud + /occupancy_grid) ─────────
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

    # ── Teleop (publishes /u_teleop after V14 patch) ────────────────────
    teleop_node = Node(
        package='go2_walking_lidar',
        executable='teleOp',
        name='teleop',
        output='screen',
        parameters=[{
            'vel_max_x_fwd': LaunchConfiguration('vel_max_x_fwd'),
            'vel_max_x_bwd': LaunchConfiguration('vel_max_x_bwd'),
            'vel_max_y':     LaunchConfiguration('vel_max_y'),
            'vel_max_yaw':   LaunchConfiguration('vel_max_yaw'),
        }],
    )

    # ── V14 CBF pipeline: 3 new Python ROS 2 nodes ──────────────────────
    cbf_grid_node = Node(
        package='go2_walking_lidar',
        executable='cbf_grid_node',
        name='cbf_grid_node',
        output='screen',
        parameters=[{
            'cloud_topic': LaunchConfiguration('cloud_topic'),
            'body_frame': 'body_link',
            'publish_rate_hz': 50.0,
        }],
    )

    cbf_inference_node = Node(
        package='go2_walking_lidar',
        executable='cbf_inference_node',
        name='cbf_inference_node',
        output='screen',
        parameters=[{
            'teacher_ckpt': LaunchConfiguration('teacher_ckpt'),
            'student_ckpt': LaunchConfiguration('student_ckpt'),
            'device':       LaunchConfiguration('device'),
            'base_height_nominal': 0.30,
            'input_stale_ms': 200.0,
            'history_reset_ms': 500.0,
            'inference_rate_hz': 50.0,
            'tilt_gravity_z_min': 0.5,
        }],
    )

    cbf_filter_node = Node(
        package='go2_walking_lidar',
        executable='cbf_filter_node',
        name='cbf_filter_node',
        output='screen',
        parameters=[{
            'cloud_topic':     LaunchConfiguration('cloud_topic'),
            'body_frame':      'body_link',
            'safety_margin':   LaunchConfiguration('safety_margin'),
            'publish_rate_hz': 50.0,
            'enable_filter':   LaunchConfiguration('enable_filter'),
        }],
    )

    # ── Walking bridge (consumes filtered /u_des; LAST in launch order) ─
    # Gated by `enable_walking_bridge` arg. When false, the robot does NOT
    # stand up — useful for first-time dry runs verifying perception +
    # inference + filter without any risk of motion.
    walking_bridge_node = Node(
        package='go2_walking_lidar',
        executable='walking_bridge',
        name='walking_bridge',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_walking_bridge')),
    )

    return LaunchDescription([
        teacher_ckpt_arg, student_ckpt_arg, device_arg,
        cloud_topic_arg, safety_margin_arg, enable_filter_arg,
        enable_walking_bridge_arg,
        min_z_arg, max_z_arg,
        vel_max_x_fwd_arg, vel_max_x_bwd_arg, vel_max_y_arg, vel_max_yaw_arg,
        # Perception + odometry
        livox_lidar_node,
        odom_to_camera_init_tf, body_to_livox_tf,
        livox_to_body_tf, body_to_utlidar_tf,
        fast_lio_node,
        odom_publisher_node,
        cloud_merger_node,
        # User input
        teleop_node,
        # V14 CBF pipeline
        cbf_grid_node,
        cbf_inference_node,
        cbf_filter_node,
        # Robot motion — last
        walking_bridge_node,
    ])
