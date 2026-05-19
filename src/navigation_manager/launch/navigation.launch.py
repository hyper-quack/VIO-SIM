from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    """
    Navigation stack launch. Timers are relative to when THIS file is included.
    When called from csky.launch.py at T=58s, global times are shown below.

    Startup sequence (relative / global when called at T=58s):
      T= 0s / T=58s  obstacle_detector, safety_layer, a_star_planner,
                      path_follower, waypoint_manager
      T=10s / T=68s  depth_filter  — starts after depth bridge has been
                                      running for 63 seconds (well past
                                      any gz-transport discovery backoff)

    depth_filter is the only node that consumes /oakd/depth/image.
    Delaying it prevents its subscriber from appearing on the ROS 2 DDS
    bus before the depth bridge is fully stable.
    """

    obstacle_detector = Node(
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

    # depth_filter subscribes to /oakd/depth/image and does heavy per-pixel
    # processing. It must not start until the depth bridge gz-transport
    # subscription is fully established and delivering stable frames.

    rrt_local_planner = Node(
        package='navigation_manager',
        executable='rrt_local_planner',
        name='rrt_local_planner',
        output='screen',
    )

    depth_filter = Node(
        package='navigation_manager',
        executable='depth_filter',
        name='depth_filter',
        output='screen',
    )

    return LaunchDescription([
        # T=0: all nav nodes except depth_filter
        LogInfo(msg='[nav] T+0  Starting navigation stack (without depth_filter)'),
        obstacle_detector,
        rrt_local_planner,
        safety_layer,
        a_star_planner,
        path_follower,
        waypoint_manager,

        # T=10: depth_filter — after depth bridge is confirmed stable
        TimerAction(
            period=10.0,
            actions=[
                LogInfo(msg='[nav] T+10 Starting depth_filter'),
                depth_filter,
            ],
        ),

        TimerAction(
            period=12.0,
            actions=[
                LogInfo(msg='[nav] Done — navigation stack started. '
                            'Verify with: ros2 topic hz /pointcloud/filtered'),
            ],
        ),
    ])
