import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    fc_bridge_dir = get_package_share_directory('flight_controller_bridge')
    vio_dir       = get_package_share_directory('vio_localization')
    nav_dir       = get_package_share_directory('navigation_manager')

    drone_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fc_bridge_dir, 'launch', 'drone.launch.py')))

    vio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(vio_dir, 'launch', 'vio.launch.py')))

    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_dir, 'launch', 'navigation.launch.py')))

    vio_bridge = Node(
        package='flight_controller_bridge',
        executable='vio_bridge',
        name='vio_bridge',
        output='screen',
        respawn=True,
        respawn_delay=3.0,
    )

    mission_node = Node(
        package='mission_manager',
        executable='mission_node',
        name='mission_manager',
        output='screen',
    )

    return LaunchDescription([
        LogInfo(msg='[csky] T=0  drone.launch'),
        drone_launch,

        TimerAction(period=10.0, actions=[
            LogInfo(msg='[csky] T=10 vio.launch'),
            vio_launch,
        ]),

        TimerAction(period=25.0, actions=[
            LogInfo(msg='[csky] T=25 vio_bridge'),
            vio_bridge,
        ]),

        TimerAction(period=30.0, actions=[
            LogInfo(msg='[csky] T=30 navigation.launch'),
            nav_launch,
        ]),

        TimerAction(period=45.0, actions=[
            LogInfo(msg='[csky] T=45 mission_node — READY'),
            mission_node,
        ]),
    ])
