from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    """
    Infrastructure launch: gz bridges + MicroXRCEAgent.

    Startup sequence (wall-clock from ros2 launch):
      T= 0s  gz_bridge_lidars  — LiDARs + custom IMU
      T= 0s  MicroXRCEAgent    — PX4 DDS bridge
      T= 5s  gz_bridge_depth   — Depth camera ALONE.  15 s of exclusive IPC
                                  time lets gz-transport fully establish the
                                  ZMQ subscription before stereo traffic starts.
      T=20s  gz_bridge_stereo  — Stereo cameras.  Starts 15 s after depth is
                                  confirmed stable.  Stereo is now 320×240 @ 5 Hz
                                  per camera (~0.75 MB/s total) vs the 6 MB/s
                                  that caused 15-second exponential-back-off
                                  depth gaps at the previous 640×480 @ 10 Hz.

    WHY DEPTH GAPS HAPPENED (historical note):
      gz-transport keeps subscriptions alive via ~5-second ZMQ heartbeats.
      When gz_bridge_stereo sent 2×640×480 frames at 10 Hz (~6 MB/s) over the
      shared Unix-domain IPC socket, Gazebo's single publish thread could not
      deliver gz_bridge_depth's heartbeat ACK during the burst.  Gazebo then
      marked the depth subscription dead and gz_bridge_depth re-ran discovery
      with exponential back-off (1+2+4+8 = 15 s gap).  Three fixes applied:
        1. Stereo resolution reduced 640×480 → 320×240 (4× less data).
        2. Stereo frame rate reduced 10 → 5 Hz (2× less frequent).
        3. gz_bridge_stereo delayed to T=20 s (depth gets 15 s head start).
    """

    # ── BRIDGE 1: LiDARs + custom IMU ──────────────────────────────────────
    gz_bridge_lidars = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge_lidars',
        arguments=[
            '/front_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/left_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/right_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/mtf01/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/adafruit/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0,
    )

    # ── MicroXRCEAgent: PX4 DDS bridge ─────────────────────────────────────
    microxrce = ExecuteProcess(
        cmd=['bash', '-c', '/usr/local/bin/MicroXRCEAgent udp4 -p 8888'],
        output='screen',
    )

    # ── BRIDGE 2: Depth camera ──────────────────────────────────────────────
    # camera_info omitted: Gazebo Harmonic appends /camera_info to the image
    # topic name; DepthFilter uses hardcoded intrinsics so it never subscribes.
    gz_bridge_depth = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge_depth',
        arguments=[
            '/oakd/depth/image@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0,
    )

    # ── BRIDGE 3: Stereo cameras ────────────────────────────────────────────
    # 320×240 @ 5 Hz per camera.  Starts 15 s after depth to avoid IPC
    # contention during depth's gz-transport discovery handshake.
    gz_bridge_stereo = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge_stereo',
        arguments=[
            '/oakd/left/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/oakd/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/oakd/right/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/oakd/right/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0,
    )

    return LaunchDescription([
        # T=0
        LogInfo(msg='[drone] T=0  gz_bridge_lidars + MicroXRCEAgent'),
        gz_bridge_lidars,
        microxrce,

        # T=5: depth alone
        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg='[drone] T=5  gz_bridge_depth (depth-first, 15s before stereo)'),
                gz_bridge_depth,
            ],
        ),

        # T=20: stereo (after depth is stable)
        TimerAction(
            period=20.0,
            actions=[
                LogInfo(msg='[drone] T=20 gz_bridge_stereo (320×240 @ 5Hz)'),
                gz_bridge_stereo,
            ],
        ),

        TimerAction(
            period=23.0,
            actions=[
                LogInfo(msg='[drone] Done — verify: ros2 topic hz /oakd/depth/image '
                            '(~10 Hz) and ros2 topic hz /oakd/left/image (~5 Hz)'),
            ],
        ),
    ])
