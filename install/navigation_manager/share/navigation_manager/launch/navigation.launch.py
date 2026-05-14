from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    obstacle_detector = Node(
        package='navigation_manager',
        executable='obstacle_detector',
        name='obstacle_detector',
        output='screen'
    )
    safety_layer = Node(
        package='navigation_manager',
        executable='safety_layer',
        name='safety_layer',
        output='screen'
    )
    a_star_planner = Node(
        package='navigation_manager',
        executable='a_star_planner',
        name='a_star_planner',
        output='screen'
    )
    path_follower = Node(
        package='navigation_manager',
        executable='path_follower',
        name='path_follower',
        output='screen'
    )
    waypoint_manager = Node(
        package='navigation_manager',
        executable='waypoint_manager',
        name='waypoint_manager',
        output='screen'
    )
    depth_filter = Node(
        package='navigation_manager',
        executable='depth_filter',
        name='depth_filter',
        output='screen'
    )
    return LaunchDescription([
        obstacle_detector,
        safety_layer,
        a_star_planner,
        path_follower,
        waypoint_manager,
        depth_filter,
    ])