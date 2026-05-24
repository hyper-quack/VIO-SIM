from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo
from launch_ros.actions import Node


def generate_launch_description():

    octomap_manager = Node(
        package='navigation_manager',
        executable='octomap_manager',
        name='octomap_manager',
        output='screen',
    )
    safety_layer = Node(
        package='navigation_manager',
        executable='safety_layer',
        name='safety_layer',
        output='screen',
    )
    a_star_planner = Node(
        package='navigation_manager',
        executable='a_star_planner',
        name='a_star_planner',
        output='screen',
    )
    path_follower = Node(
        package='navigation_manager',
        executable='path_follower',
        name='path_follower',
        output='screen',
    )
    waypoint_manager = Node(
        package='navigation_manager',
        executable='waypoint_manager',
        name='waypoint_manager',
        output='screen',
    )
    slam_frontend = Node(
        package='navigation_manager',
        executable='slam_frontend',
        name='slam_frontend',
        output='screen',
    )
    depth_filter = Node(
        package='navigation_manager',
        executable='depth_filter',
        name='depth_filter',
        output='screen',
    )
    rl_depth_filter = Node(
        package='navigation_manager',
        executable='rl_depth_filter',
        name='rl_depth_filter',
        output='screen',
    )
    pose_graph = Node(
        package='navigation_manager',
        executable='pose_graph',
        name='pose_graph',
        output='screen',
    )
    loop_closure = Node(
        package='navigation_manager',
        executable='loop_closure',
        name='loop_closure',
        output='screen',
    )

    rrt_local_planner = Node(
        package='navigation_manager',
        executable='rrt_local_planner',
        name='rrt_local_planner',
        output='screen',
    )
    return LaunchDescription([
        LogInfo(msg='[nav] T+0  Starting navigation stack'),
        octomap_manager,
        safety_layer,
        a_star_planner,
        rrt_local_planner,
        path_follower,
        waypoint_manager,

        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg='[nav] T+5  Starting slam_frontend, depth_filter, loop_closure'),
                slam_frontend,
                depth_filter,
                loop_closure,
                pose_graph,
                rl_depth_filter,
            ],
        ),

        TimerAction(
            period=7.0,
            actions=[
                LogInfo(msg='[nav] Done — full navigation stack running'),
            ],
        ),
    ])
