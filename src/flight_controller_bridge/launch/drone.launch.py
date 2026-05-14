from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    vio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('vio_localization'),
                'launch', 'vio.launch.py'
            )
        ])
    )

    return LaunchDescription([

        # ── BRIDGE 1 — LiDARs + IMU + Optical Flow ──────────────────
        # Lightweight, high rate sensors.
        # Dedicated process so LiDAR safety distances are never delayed.
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_bridge_lidars',
            arguments=[
                '/front_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/left_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/right_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/mtf01/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/mtf01/optical_flow@sensor_msgs/msg/Image[gz.msgs.Image',
                '/adafruit/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            ],
            output='screen'
        ),

        # ── BRIDGE 2 — Stereo cameras for VIO ───────────────────────
        # Medium bandwidth. Dedicated process for rtabmap odometry.
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_bridge_stereo',
            arguments=[
                '/oakd/left/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/oakd/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                '/oakd/right/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/oakd/right/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                '/oakd_lite/rgb/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/oakd_lite/rgb/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
            output='screen'
        ),

        # ── BRIDGE 3 — Depth camera only ────────────────────────────
        # Heavy: 320x240 float32 = 300KB per frame.
        # Dedicated process so depth gets its own CPU core.
        # This is the main FPS fix.
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_bridge_depth',
            arguments=[
                '/oakd/depth/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/oakd/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
            output='screen'
        ),

        # ── MicroXRCEAgent — PX4 bridge ─────────────────────────────
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                '/usr/local/bin/MicroXRCEAgent udp4 -p 8888'
            ],
            output='screen'
        ),

        # ── VIO Bridge → PX4 ────────────────────────────────────────
        Node(
            package='flight_controller_bridge',
            executable='vio_bridge',
            name='vio_bridge',
            output='screen'
        ),

        # ── Flight controller bridge ─────────────────────────────────
        Node(
            package='flight_controller_bridge',
            executable='bridge_node',
            name='flight_controller_bridge',
            output='screen'
        ),

        # ── VIO localization (rtabmap) ───────────────────────────────
        vio_launch,

    ])