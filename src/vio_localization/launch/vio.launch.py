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
      T+25s / T=55s  rtabmap_slam — starts after odom CUDA init is complete

    rtabmap quality=0 root-cause fix (applied in model.sdf):
      Both stereo cameras were on the right side of the drone (y<0), making
      "left" physically righter than "right" → all disparities negative →
      rtabmap cannot compute stereo depth → quality=0.  Cameras repositioned
      to y=+0.0375 (left) and y=-0.0375 (right) for a proper 7.5 cm baseline.
      stereo_sync already injects Tx = -fx*0.075 which now matches reality.

    PX4 EKF2 config for GPS-denied flight:
      Load ~/csky_ws/px4_vio_params.params in QGroundControl before arming:
        EKF2_EV_CTRL = 15   (enable all external-vision fusion)
        EKF2_GPS_CTRL = 0   (disable GPS — GPS-denied environment)
        EKF2_HGT_REF  = 0   (baro for altitude — more stable than VIO height)
        COM_ARM_WO_GPS = 1  (allow arming without GPS lock)
    """

    tf_oakd = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_oakd',
        arguments=['0.12', '0', '0.06', '0', '0', '0', 'base_link', 'oakd_lite_link'],
    )

    tf_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_imu',
        arguments=['0', '0', '0.02', '0', '0', '0', 'base_link', 'adafruit_9dof_link'],
    )

    imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        parameters=[{
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
    )

    # rtabmap stereo odometry — tuned for simulated corridor environment.
    # Key changes vs defaults:
    #   Vis/FeatureType=8  : GFTT/ORB — detects corners at wall/floor/ceiling
    #                        intersections (architectural scenes).
    #   Vis/MaxFeatures=800: enough for 320×240 images with corridor edges.
    #   Vis/MinInliers=4   : lower threshold; corridor walls give fewer
    #                        features than cluttered scenes.
    #   Vis/EstimationType=1: PnP (3D-2D) — more robust when stereo depth is
    #                        noisy (near-symmetric textures, few far points).
    #   approx_sync=True   : required because stereo_sync re-stamps both
    #                        images to the same ROS clock tick.
    rtabmap_odom = Node(
        package='rtabmap_odom',
        executable='stereo_odometry',
        name='rtabmap_odom',
        parameters=[{
            'frame_id':           'base_link',
            'odom_frame_id':      'odom',
            'publish_tf':         True,
            'subscribe_imu':      False,
            'Vis/FeatureType':    '8',
            'Vis/MaxFeatures':    '800',
            'Vis/MinInliers':     '4',
            'Vis/EstimationType': '1',
            'OdomF2M/MaxSize':    '2000',
            'approx_sync':        True,
            'sync_queue_size':    20,
            'topic_queue_size':   20,
        }],
        remappings=[
            ('/left/image_rect',   '/oakd/sync/left/image'),
            ('/right/image_rect',  '/oakd/sync/right/image'),
            ('/left/camera_info',  '/oakd/sync/left/camera_info'),
            ('/right/camera_info', '/oakd/sync/right/camera_info'),
        ],
        output='screen',
    )

    # rtabmap SLAM — builds a persistent 3D map and provides loop-closure
    # correction to rtabmap_odom.  Starts after odom's CUDA init completes.
    rtabmap_slam = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        parameters=[{
            'frame_id':           'base_link',
            'odom_frame_id':      'odom',
            'subscribe_stereo':   True,
            'subscribe_depth':    False,
            'subscribe_rgb':      False,
            'subscribe_imu':      False,
            'approx_sync':        True,
            'Vis/FeatureType':    '8',
            'Vis/MaxFeatures':    '800',
            'Mem/STMSize':        '30',
            'Optimizer/Strategy': '1',
            'Grid/CellSize':      '0.05',
            'sync_queue_size':    20,
            'topic_queue_size':   20,
        }],
        remappings=[
            ('/left/image_rect',   '/oakd/sync/left/image'),
            ('/right/image_rect',  '/oakd/sync/right/image'),
            ('/left/camera_info',  '/oakd/sync/left/camera_info'),
            ('/right/camera_info', '/oakd/sync/right/camera_info'),
            ('/odom',              '/odom'),
        ],
        output='screen',
    )

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

        # T+10: rtabmap_odom — stereo_sync has had 5 s to produce sync'd frames
        TimerAction(
            period=10.0,
            actions=[
                LogInfo(msg='[vio] T+10 Starting rtabmap_odom (CUDA init spikes GPU)'),
                rtabmap_odom,
            ],
        ),

        # T+25: rtabmap_slam — after odom CUDA init is complete
        TimerAction(
            period=25.0,
            actions=[
                LogInfo(msg='[vio] T+25 Starting rtabmap_slam'),
                rtabmap_slam,
            ],
        ),

        TimerAction(
            period=27.0,
            actions=[
                LogInfo(msg='[vio] Done — verify: ros2 topic hz /odom  (expect >4 Hz)'),
            ],
        ),
    ])
