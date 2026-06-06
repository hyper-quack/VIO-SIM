import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    config_path = '/home/poorcsky/csky_ws/src/open_vins/config/openvins_oakd/estimator_config.yaml'

    openvins_node = Node(
        package='ov_msckf',
        executable='run_subscribe_msckf',
        name='openvins',
        output='screen',
        arguments=[config_path],
        parameters=[{
            'use_sim_time': True,
            'camera0_qos': 1,    # 1 = BEST_EFFORT
            'camera1_qos': 1,
            'imu_qos': 1,
        }],
        remappings=[
            ('/imu0',           '/adafruit/imu'),
            ('/cam0/image_raw', '/oakd/sync/left/image'),
            ('/cam1/image_raw', '/oakd/sync/right/image'),
            ('/tf', '/tf'),
            ('/poseimu', '/openvins/poseimu'),
            ('/odomimu', '/openvins/odomimu'),
            ('/pathimu', '/openvins/pathimu'),
            ('/points_msckf', '/openvins/points_msckf'),
            ('/points_slam', '/openvins/points_slam'),
            ('/trackhist', '/openvins/trackhist'),
        ],
    )

    return LaunchDescription([
        openvins_node,
    ])
