from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # Filtre IMU Madgwick
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
        ]
    )

    # TF statique base_link → oakd_lite_link
    tf_oakd = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_oakd',
        arguments=[
            '0.12', '0', '0.06',
            '0', '0', '0',
            'base_link', 'oakd_lite_link'
        ]
    )

    # TF statique base_link → adafruit_9dof_link
    tf_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_imu',
        arguments=[
            '0', '0', '0.02',
            '0', '0', '0',
            'base_link', 'adafruit_9dof_link'
        ]
    )

    # Synchronisation stéréo
    stereo_sync = Node(
        package='vio_localization',
        executable='stereo_sync',
        name='stereo_sync',
        output='screen'
    )

    # RTAB-Map odométrie stéréo
    rtabmap_odom = Node(
        package='rtabmap_odom',
        executable='stereo_odometry',
        name='rtabmap_odom',
        parameters=[{
            'frame_id':         'base_link',
            'odom_frame_id':    'odom',
            'publish_tf':       True,
            'subscribe_imu':    False,
            'Vis/FeatureType':  '6',
            'Vis/MaxFeatures':  '500',
            'Vis/MinInliers':   '5',
            'OdomF2M/MaxSize':  '1000',
            'Imu/Strategy':     '1',
            'approx_sync':      True,
            'sync_queue_size':  50,
            'topic_queue_size': 50,
        }],
        remappings=[
            ('/left/image_rect',   '/oakd/sync/left/image'),
            ('/right/image_rect',  '/oakd/sync/right/image'),
            ('/left/camera_info',  '/oakd/sync/left/camera_info'),
            ('/right/camera_info', '/oakd/sync/right/camera_info'),
            ('/imu',               '/imu/filtered'),
        ]
    )

    # RTAB-Map SLAM
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
            'subscribe_imu':      True,
            'approx_sync':        True,
            'Vis/FeatureType':    '6',
            'Vis/MaxFeatures':    '500',
            'Mem/STMSize':        '30',
            'Optimizer/Strategy': '1',
            'Grid/CellSize':      '0.05',
            'sync_queue_size':    50,
            'topic_queue_size':   50,
        }],
        remappings=[
            ('/left/image_rect',   '/oakd/sync/left/image'),
            ('/right/image_rect',  '/oakd/sync/right/image'),
            ('/left/camera_info',  '/oakd/sync/left/camera_info'),
            ('/right/camera_info', '/oakd/sync/right/camera_info'),
            ('/imu',               '/imu/filtered'),
            ('/odom',              '/odom'),
        ]
    )

    return LaunchDescription([
        imu_filter,
        tf_oakd,
        tf_imu,
        stereo_sync,
        rtabmap_odom,
        rtabmap_slam,
    ])