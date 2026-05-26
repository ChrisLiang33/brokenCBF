#!/usr/bin/env python3
"""cbf_go2 deploy launch.

Brings up the full stack:
    livox_ros_driver2  -> /livox/lidar
    static TFs         -> body, livox_frame, body_link, utlidar_lidar
    odom_publisher     -> /odom + TF(odom->body_link) from /sportmodestate
    cloud_merger       -> /poisson_cloud + /occupancy_grid
    teleop             -> /u_teleop (raw user command)
    cbf_grid_node      -> /cbf/grid (2x64x64 ego-centric grid)
    cbf_inference_node -> /cbf/params [alpha, phi]
    cbf_filter_node    -> /u_des (CBF-projected safe command)
    walking_bridge     -> SportClient.Move (gated; defaults OFF for dry runs)

Usage on Go2:
    ros2 launch cbf_go2 cbf_go2.launch.py \\
        checkpoint:=/home/unitree/cbf_go2/run/checkpoints/policy.pt \\
        enable_walking_bridge:=true

Dry-run (no robot motion) — verify perception + inference + filter:
    ros2 launch cbf_go2 cbf_go2.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ---- Args ----
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint',
        default_value='',
        description='Path to trained CBF parameter policy (.pt). Empty = stub mode (safe defaults).',
    )
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Inference device: "cpu" or "cuda".',
    )
    cloud_topic_arg = DeclareLaunchArgument(
        'cloud_topic',
        default_value='/poisson_cloud',
        description='Filtered cloud topic consumed by cbf_grid_node + cbf_filter_node.',
    )
    safety_margin_arg = DeclareLaunchArgument(
        'safety_margin',
        default_value='0.70',
        description='r_robot + r_obs_nominal — defines h(x) = ||p_nearest|| - safety_margin.',
    )
    enable_walking_bridge_arg = DeclareLaunchArgument(
        'enable_walking_bridge',
        default_value='false',
        description='When false, walking_bridge is NOT launched -- robot does not stand or move. '
                    'Default false for safety: explicit opt-in required to enable motion.',
    )
    min_z_arg = DeclareLaunchArgument('min_z', default_value='0.05')
    max_z_arg = DeclareLaunchArgument('max_z', default_value='0.80')
    vel_max_x_fwd_arg = DeclareLaunchArgument('vel_max_x_fwd', default_value='0.75')
    vel_max_x_bwd_arg = DeclareLaunchArgument('vel_max_x_bwd', default_value='0.75')
    vel_max_y_arg     = DeclareLaunchArgument('vel_max_y',     default_value='0.75')
    vel_max_yaw_arg   = DeclareLaunchArgument('vel_max_yaw',   default_value='0.75')

    # ---- Livox Mid360 LiDAR ----
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
                FindPackageShare('cbf_go2'),
                'config', 'MID360_config.json',
            ])},
            {'cmdline_input_bd_code': 'livox0000000001'},
        ],
    )

    # ---- Static TFs ----
    # Mirrors prototype/ros_package/launch/cbf_deploy.launch.py (which has
    # been validated on hardware). Livox is mounted upside-down on the
    # Go2 -> 180 deg pitch flip; utlidar is the chin-mounted unit.
    body_to_livox_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='body_to_livox_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'body', 'livox_frame'],
    )
    livox_to_body_link_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='livox_to_body_link_tf',
        arguments=['-0.05', '0.0', '0.18', '0', '3.14159', '0',
                   'livox_frame', 'body_link'],
    )
    body_link_to_utlidar_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='body_link_to_utlidar_tf',
        arguments=['0.37', '0.0', '0.05', '0', '2.9', '0',
                   'body_link', 'utlidar_lidar'],
    )

    # ---- Odometry (from /sportmodestate, includes IMU ang vel) ----
    odom_publisher_node = Node(
        package='cbf_go2',
        executable='odom_publisher',
        name='odom_publisher',
        output='screen',
    )

    # ---- Cloud merger (Livox + utlidar -> /poisson_cloud) ----
    cloud_merger_node = Node(
        package='cbf_go2',
        executable='cloud_merger',
        name='cloud_merger',
        output='screen',
        parameters=[{
            'min_z': LaunchConfiguration('min_z'),
            'max_z': LaunchConfiguration('max_z'),
        }],
    )

    # ---- Teleop (publishes raw /u_teleop) ----
    teleop_node = Node(
        package='cbf_go2',
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

    # ---- CBF pipeline ----
    cbf_grid_node = Node(
        package='cbf_go2',
        executable='cbf_grid_node',
        name='cbf_grid_node',
        output='screen',
        parameters=[{
            'cloud_topic':     LaunchConfiguration('cloud_topic'),
            'body_frame':      'body_link',
            'publish_rate_hz': 50.0,
        }],
    )

    cbf_inference_node = Node(
        package='cbf_go2',
        executable='cbf_inference_node',
        name='cbf_inference_node',
        output='screen',
        parameters=[{
            'checkpoint':          LaunchConfiguration('checkpoint'),
            'device':              LaunchConfiguration('device'),
            'base_height_nominal': 0.30,
            'input_stale_ms':      200.0,
            'history_reset_ms':    500.0,
            'inference_rate_hz':   50.0,
            'tilt_gravity_z_min':  0.5,
        }],
    )

    cbf_filter_node = Node(
        package='cbf_go2',
        executable='cbf_filter_node',
        name='cbf_filter_node',
        output='screen',
        parameters=[{
            'cloud_topic':     LaunchConfiguration('cloud_topic'),
            'body_frame':      'body_link',
            'safety_margin':   LaunchConfiguration('safety_margin'),
            'publish_rate_hz': 50.0,
        }],
    )

    # ---- Walking bridge (LAST; consumes filtered /u_des) ----
    # Defaults OFF: enable_walking_bridge:=true to make the robot stand
    # and accept motion commands.
    walking_bridge_node = Node(
        package='cbf_go2',
        executable='walking_bridge',
        name='walking_bridge',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_walking_bridge')),
    )

    return LaunchDescription([
        checkpoint_arg, device_arg, cloud_topic_arg, safety_margin_arg,
        enable_walking_bridge_arg,
        min_z_arg, max_z_arg,
        vel_max_x_fwd_arg, vel_max_x_bwd_arg, vel_max_y_arg, vel_max_yaw_arg,
        # Perception + odometry
        livox_lidar_node,
        body_to_livox_tf, livox_to_body_link_tf, body_link_to_utlidar_tf,
        odom_publisher_node,
        cloud_merger_node,
        # User input
        teleop_node,
        # CBF pipeline
        cbf_grid_node,
        cbf_inference_node,
        cbf_filter_node,
        # Robot motion — last
        walking_bridge_node,
    ])
