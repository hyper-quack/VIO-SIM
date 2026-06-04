from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    """
    VIO stack launch.  Timers are relative to when THIS file is included.
    When called from csky.launch.py at T=30s, global times are shown below.

    Startup sequence (relative / global when called at T=30s):
      T+ 0s / T=30s  imu_filter, TF publishers — no camera dependency
      T+ 5s / T=35s  stereo_sync  — needs gz_bridge_stereo (up since T=20s)
      T+10s / T=40s  rtabmap_odom — needs stereo_sync output
      (rtabmap_slam DISABLED — not needed for VIO; re-enable for loop-closure)

    rtabmap quality=0 root-cause fix (applied in model.sdf):
      Both stereo cameras were on the right side of the drone (y<0), making
      "left" physically righter than "right" → all disparities negative →
      rtabmap cannot compute stereo depth → quality=0.  Cameras repositioned
      to y=+0.0375 (left) and y=-0.0375 (right) for a proper 7.5 cm baseline.
      stereo_sync already injects Tx = -fx*0.075 which now matches reality.

    PX4 EKF2 config (GPS + VIO):
      Load ~/csky_ws/px4_vio_params.params in QGroundControl before arming:
        EKF2_EV_CTRL  = 9   (VIO: h-pos + yaw)
        EKF2_GPS_CTRL = 7   (GPS: horiz + vert + velocity)
        EKF2_HGT_REF  = 1   (GPS altitude reference)
        COM_ARM_WO_GPS = 1  (allow arming without GPS lock)
    """

    tf_oakd = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_oakd',
        arguments=['0.12', '0', '0.06', '0', '0', '0', 'base_link', 'oakd_lite_link'],
        parameters=[{'use_sim_time': True}],
    )

    tf_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_imu',
        arguments=['0', '0', '0.02', '0', '0', '0', 'base_link', 'adafruit_9dof_link'],
        parameters=[{'use_sim_time': True}],
    )

    imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        parameters=[{
            'use_sim_time': True,
            'use_mag':     False,
            'publish_tf':  False,
            'world_frame': 'enu',
            'gain':        0.1,
        }],
        remappings=[
            ('/imu/data_raw', '/adafruit/imu'),
            ('/imu/data',     '/imu/filtered'),
        ],
    )

    # stereo_sync synchronises left+right images+camera_info and injects
    # the 7.5 cm baseline (Tx) into the right camera's projection matrix,
    # since Gazebo publishes each camera independently with Tx=0.
    stereo_sync = Node(
        package='vio_localization',
        executable='stereo_sync',
        name='stereo_sync',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    odom_to_path = Node(
        package='vio_localization',
        executable='odom_to_path',
        name='odom_to_path',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # rtabmap stereo odometry — tuned for simulated corridor environment.
    # Key changes vs defaults:
    #   Vis/FeatureType=6    : GFTT/BRIEF — fast detector+descriptor.
    #   Vis/MaxFeatures=100  : minimal feature budget for 20Hz processing.
    #   Odom/Strategy=1      : Frame-to-Frame — skips map lookup → faster.
    #   Odom/GuessMotion=true: velocity prediction reduces search window.
    #   Odom/ImageDecimation=1: stereo_sync already decimates 60→20Hz.
    #   approx_sync=True     : required because stereo_sync re-stamps images.
    rtabmap_odom = Node(
        package='rtabmap_odom',
        executable='stereo_odometry',
        name='rtabmap_odom',
        parameters=[{
            'use_sim_time':         True,
            'frame_id':             'base_link',
            'odom_frame_id':        'odom',
            'publish_tf':           True,
            'subscribe_imu':        False,
            'qos':                  2,
            'qos_camera_info':      2,
            'Vis/FeatureType':      '6',     # GFTT/BRIEF — fast detector+descriptor
            'Vis/MaxFeatures':      '100',   # minimal features for 20Hz budget
            'Vis/MinInliers':       '5',
            'Vis/EstimationType':   '1',
            'Odom/Strategy':        '1',     # Frame-to-Frame — no map overhead
            'Odom/ImageDecimation': '1',     # sync already decimates to ~20Hz
            'Odom/GuessMotion':     'true',  # velocity prediction → faster convergence
            'Odom/KeyFrameThr':     '0.5',
            'OdomF2M/MaxSize':      '200',
            'approx_sync':          True,
            'sync_queue_size':      10,
            'topic_queue_size':     10,
        }],
        remappings=[
            ('/left/image_rect',   '/oakd/sync/left/image'),
            ('/right/image_rect',  '/oakd/sync/right/image'),
            ('/left/camera_info',  '/oakd/sync/left/camera_info'),
            ('/right/camera_info', '/oakd/sync/right/camera_info'),
        ],
        output='screen',
    )

    # DISABLED — rtabmap SLAM not needed for VIO. Only rtabmap_odom required.
    # Re-enable later for loop closure if needed.
    # rtabmap_slam = Node(
    #     package='rtabmap_slam',
    #     executable='rtabmap',
    #     name='rtabmap',
    #     parameters=[{
    #         'frame_id':           'base_link',
    #         'odom_frame_id':      'odom',
    #         'subscribe_stereo':   True,
    #         'subscribe_depth':    False,
    #         'subscribe_rgb':      False,
    #         'subscribe_imu':      False,
    #         'approx_sync':        True,
    #         'Vis/FeatureType':    '6',
    #         'Vis/MaxFeatures':    '200',
    #         'Mem/STMSize':        '30',
    #         'Optimizer/Strategy': '1',
    #         'Grid/CellSize':      '0.05',
    #         'sync_queue_size':    5,
    #         'topic_queue_size':   5,
    #     }],
    #     remappings=[
    #         ('/left/image_rect',   '/oakd/sync/left/image'),
    #         ('/right/image_rect',  '/oakd/sync/right/image'),
    #         ('/left/camera_info',  '/oakd/sync/left/camera_info'),
    #         ('/right/camera_info', '/oakd/sync/right/camera_info'),
    #         ('/odom',              '/odom'),
    #     ],
    #     output='screen',
    # )

    return LaunchDescription([
        # T+0: TFs and IMU filter — no camera dependency
        LogInfo(msg='[vio] T+0  Starting imu_filter + TF publishers'),
        imu_filter,
        tf_oakd,
        tf_imu,

        # T+5: stereo_sync — gz_bridge_stereo has been up since T=20s globally
        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg='[vio] T+5  Starting stereo_sync'),
                stereo_sync,
            ],
        ),

        # T+15: rtabmap_odom — wait for TF base_link->oakd_lite_link to be available
        TimerAction(
            period=15.0,
            actions=[
                LogInfo(msg='[vio] T+15 Starting rtabmap_odom + odom_to_path'),
                rtabmap_odom,
                odom_to_path,
            ],
        ),

        TimerAction(
            period=27.0,
            actions=[
                LogInfo(msg='[vio] Done — verify: ros2 topic hz /odom  (expect 20-30 Hz)'),
            ],
        ),
    ])
