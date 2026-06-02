#!/usr/bin/env python3
"""
openvins.launch.py — Launch OpenVINS (ov_msckf) for OAK-D stereo + IMU.

Topic remaps:
  /ov_msckf/cam0/image_raw  ← /oakd/sync/left/image
  /ov_msckf/cam1/image_raw  ← /oakd/sync/right/image
  /ov_msckf/imu0            ← /imu

Config: navigation_manager/config/openvins_oakd/estimator_config.yaml
Output: /ov_msckf/poseimu  (geometry_msgs/PoseWithCovarianceStamped)
        consumed by vio_bridge → pose_fusion → /slam/corrected_pose
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    nav_share = get_package_share_directory('navigation_manager')
    config_path = os.path.join(
        nav_share, 'config', 'openvins_oakd', 'estimator_config.yaml')

    ov_node = Node(
        package='ov_msckf',
        executable='run_subscribe_msckf',
        name='ov_msckf',
        namespace='ov_msckf',
        output='screen',
        parameters=[
            {'config_path': config_path},
            {'verbosity': 'INFO'},
            {'use_stereo': True},
            {'max_cameras': 2},
        ],
        remappings=[
            ('cam0/image_raw', '/oakd/sync/left/image'),
            ('cam1/image_raw', '/oakd/sync/right/image'),
            ('imu0',           '/imu'),
        ],
    )

    return LaunchDescription([ov_node])
