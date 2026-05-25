#!/usr/bin/env python3
"""
depth_filter.py — Depth image filter, publishes in CAMERA FRAME.
 
octomap_manager handles full 3D projection using IMU+LiDAR+VIO.
 
Publishes:
  /pointcloud/camera  → valid depth pixels in camera frame (x_cam, y_cam, z_cam)
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField, Imu
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
 
# ── Camera intrinsics (320×240) ───────────────────────────────────
FX = 161.4
FY = 161.4
CX = 160.0
CY = 120.0
 
# ── Depth limits ──────────────────────────────────────────────────
MIN_DEPTH = 0.6
MAX_DEPTH = 6.0
 
# ── Subsampling ───────────────────────────────────────────────────
SUBSAMPLE = 4
 
# ── IMU gate ──────────────────────────────────────────────────────
MAX_YAW_RATE   = 0.20   # rad/s
MAX_PITCH_RATE = 0.20
MAX_ROLL_RATE  = 0.20
 
 
def _make_pc2(pts, stamp, frame_id='oakd_lite_link'):
    msg = PointCloud2()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height    = 1
    msg.width     = len(pts)
    msg.fields    = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = 12 * len(pts)
    msg.is_dense     = True
    msg.data         = pts.tobytes()
    return msg
 
 
class DepthFilter(Node):
 
    def __init__(self):
        super().__init__('depth_filter')
 
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
 
        qos_best = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
 
        self.create_subscription(Image, '/oakd/depth/image', self._depth_cb, qos_best)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._odom_cb, qos_px4)
        self.create_subscription(Imu, '/imu/filtered', self._imu_cb, qos_best)
        self.create_subscription(PoseStamped, '/slam/corrected_pose', self._slam_cb, qos_best)

        self.pub = self.create_publisher(PointCloud2, '/pointcloud/camera', 10)

        self.bridge     = CvBridge()
        self.yaw_rate   = 0.0
        self.pitch_rate = 0.0
        self.roll_rate  = 0.0

        # Loop-closure jump detection
        self.skip_frames  = 0
        self.last_slam_x  = None
        self.last_slam_y  = None

        self.get_logger().info('DepthFilter started ✓ — camera frame output')
 
    def _odom_cb(self, msg):
        # Use PX4 angular velocity directly — no filter lag
        if hasattr(msg, 'angular_velocity') and len(msg.angular_velocity) >= 3:
            self.roll_rate  = abs(float(msg.angular_velocity[0]))
            self.pitch_rate = abs(float(msg.angular_velocity[1]))
            self.yaw_rate   = abs(float(msg.angular_velocity[2]))
 
    def _imu_cb(self, msg):
        # Fallback if PX4 angular velocity not available
        if self.yaw_rate == 0.0:
            self.yaw_rate   = abs(float(msg.angular_velocity.z))
            self.pitch_rate = abs(float(msg.angular_velocity.y))
            self.roll_rate  = abs(float(msg.angular_velocity.x))

    def _slam_cb(self, msg):
        """Detect loop-closure pose jumps and suppress depth frames."""
        nx = float(msg.pose.position.x)
        ny = float(msg.pose.position.y)
        if self.last_slam_x is not None:
            jump = math.sqrt((nx - self.last_slam_x)**2 + (ny - self.last_slam_y)**2)
            if jump > 0.3:
                self.skip_frames = 30
                self.get_logger().warn(
                    f'SLAM jump {jump:.2f} m detected — suppressing {self.skip_frames} depth frames')
        self.last_slam_x = nx
        self.last_slam_y = ny

    def _depth_cb(self, msg):
        # IMU gate — skip during fast rotation
        if (self.yaw_rate   > MAX_YAW_RATE or
            self.pitch_rate > MAX_PITCH_RATE or
            self.roll_rate  > MAX_ROLL_RATE):
            self.get_logger().debug(
                f'IMU gate: y={self.yaw_rate:.2f} p={self.pitch_rate:.2f} r={self.roll_rate:.2f}')
            return

        # Loop-closure gate — skip frames after a SLAM pose jump
        if self.skip_frames > 0:
            self.skip_frames -= 1
            self.get_logger().debug(f'SLAM jump gate: {self.skip_frames} frames remaining')
            return
 
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f'depth error: {e}')
            return
 
        h, w = depth.shape
 
        # Subsample
        rows = np.arange(0, h, SUBSAMPLE)
        cols = np.arange(0, w, SUBSAMPLE)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        rr = rr.ravel(); cc = cc.ravel()
        d  = depth[rr, cc]
 
        # Depth validity
        valid = np.isfinite(d) & (d >= MIN_DEPTH) & (d <= MAX_DEPTH)
        rr = rr[valid]; cc = cc[valid]; d = d[valid]
 
        if len(d) == 0:
            return
 
        # Project to camera frame
        # Camera frame: x=right, y=down, z=forward
        x_cam = (cc - CX) * d / FX
        y_cam = (rr - CY) * d / FY
        z_cam = d
 
        pts = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
        self.pub.publish(_make_pc2(pts, msg.header.stamp))
 
        self.get_logger().info(
            f'Published {len(pts)} camera-frame points',
            throttle_duration_sec=3.0)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = DepthFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
 