from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    """
    Navigation stack launch. Timers are relative to when THIS file is included.
    When called from csky.launch.py at T=58s, global times are shown below.

    Startup sequence (relative / global when called at T=58s):
      T= 0s / T=58s  octomap_manager, safety_layer, a_star_planner,
                     path_follower, waypoint_manager, pose_fusion
      T= 5s / T=63s  depth_filter
    """

    tf_odom_base = Node(
        package='navigation_manager',
        executable='tf_odom_base',
        name='tf_odom_base',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    octomap_manager = Node(
        package='navigation_manager',
        executable='octomap_manager',
        name='octomap_manager',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    safety_layer = Node(
        package='navigation_manager',
        executable='safety_layer',
        name='safety_layer',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    a_star_planner = Node(
        package='navigation_manager',
        executable='a_star_planner',
        name='a_star_planner',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    path_follower = Node(
        package='navigation_manager',
        executable='path_follower',
        name='path_follower',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    waypoint_manager = Node(
        package='navigation_manager',
        executable='waypoint_manager',
        name='waypoint_manager',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    depth_filter = Node(
        package='navigation_manager',
        executable='depth_filter',
        name='depth_filter',
        output='screen',
        parameters=[{'use_sim_time': True}],   # ADD THIS LINE
    )

    pose_fusion = Node(
        package='navigation_manager',
        executable='pose_fusion',
        name='pose_fusion',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        # T=0: core navigation stack
        LogInfo(msg='[nav] T+0  Starting navigation stack'),
        tf_odom_base,
        octomap_manager,
        safety_layer,
        a_star_planner,
        path_follower,
        waypoint_manager,
        pose_fusion,

        # T=5: depth_filter — after stereo bridge is confirmed stable
        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg='[nav] T+5  Starting depth_filter'),
                depth_filter,
            ],
        ),

        TimerAction(
            period=7.0,
            actions=[
                LogInfo(msg='[nav] Done — navigation stack started. '
                            'Verify with: ros2 topic hz /slam/corrected_pose'),
            ],
        ),
    ])
