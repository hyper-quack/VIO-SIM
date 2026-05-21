#!/usr/bin/env python3
"""
depth_filter.py — Simple depth image to point cloud

Subscribes:
  /oakd/depth/image            → raw depth (32FC1, metres)
  /fmu/out/vehicle_odometry    → drone pose (PX4 NED)
  /mtf01/lidar                 → altitude (Z source)
  /imu/filtered                → yaw rate (IMU gate)

Publishes:
  /pointcloud/filtered         → PointCloud2 in world frame (odom)

Coordinate convention (same as all other nodes):
  world_x = px4_position[1] + SPAWN_X   (PX4 East  → World X)
  world_y = px4_position[0] + SPAWN_Y   (PX4 North → World Y)
  world_z = lidar_z                      (MTF-01 altitude)
  yaw     = atan2(siny,cosy) - pi/2     (GPS mode)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField, Imu, LaserScan
from px4_msgs.msg import VehicleOdometry
from cv_bridge import CvBridge

# ── Camera intrinsics (OAK-D depth at 640×480, but we subsample) ─────────────
# CORRECT — 320×240
FX = 161.4
FY = 161.4
CX = 160.0
CY = 120.0

# ── Depth limits ──────────────────────────────────────────────────────────────
MIN_DEPTH = 0.6       # metres — exclude drone body
MAX_DEPTH = 6.0       # metres — exclude open-space noise

# ── Altitude filter (world Z) ─────────────────────────────────────────────────
Z_MIN = 1.0           # metres — floor filter
Z_MAX = 2.8           # metres — ceiling filter

# ── Self exclusion ────────────────────────────────────────────────────────────
SELF_EXCLUSION_RADIUS = 0.8   # metres — ignore points near drone center

# ── Subsampling ───────────────────────────────────────────────────────────────
SUBSAMPLE = 4         # process every Nth pixel row and column

# ── IMU gate ──────────────────────────────────────────────────────────────────
MAX_YAW_RATE = 0.5    # rad/s — skip frame if rotating faster

# ── World frame spawn offsets ─────────────────────────────────────────────────
SPAWN_X = 1.0
SPAWN_Y = 3.0


def _wrap_angle(a: float) -> float:
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


class DepthFilter(Node):

    def __init__(self):
        super().__init__('depth_filter')

        # ── QoS profiles ──────────────────────────────────────────────────────
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        qos_best = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Image, '/oakd/depth/image',
            self._depth_cb, qos_best)

        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self._odom_cb, qos_px4)

        self.create_subscription(
            LaserScan, '/mtf01/lidar',
            self._mtf01_cb, 10)

        self.create_subscription(
            Imu, '/imu/filtered',
            self._imu_cb, qos_best)

        # ── Publisher ─────────────────────────────────────────────────────────
        self.pub = self.create_publisher(PointCloud2, '/pointcloud/filtered', 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.drone_x   = None
        self.drone_y   = None
        self.drone_z   = 0.0
        self.drone_yaw = 0.0
        self.lidar_z   = 0.0
        self.yaw_rate  = 0.0
        self.bridge    = CvBridge()

        self.get_logger().info('DepthFilter started ✓')

    # ═════════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═════════════════════════════════════════════════════════════════════════

    def _odom_cb(self, msg: VehicleOdometry):
        """PX4 NED → world frame."""
        self.drone_x = float(msg.position[1]) + SPAWN_X
        self.drone_y = float(msg.position[0]) + SPAWN_Y
        self.drone_z = -float(msg.position[2])

        q = msg.q  # [w, x, y, z]
        siny = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy = 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
        self.drone_yaw = _wrap_angle(math.atan2(siny, cosy) - math.pi / 2.0)

    def _mtf01_cb(self, msg: LaserScan):
        """MTF-01 downward LiDAR — altitude above ground."""
        if msg.ranges and math.isfinite(msg.ranges[0]) and msg.ranges[0] > 0.01:
            self.lidar_z = float(msg.ranges[0])

    def _imu_cb(self, msg: Imu):
        """Extract yaw rate for IMU gate."""
        self.yaw_rate = abs(float(msg.angular_velocity.z))

    def _depth_cb(self, msg: Image):
        """
        Main processing:
        1. IMU gate — skip if rotating too fast
        2. Convert depth image to numpy
        3. Subsample pixels
        4. Filter by depth range
        5. Project to world frame
        6. Filter by altitude band
        7. Remove points too close to drone
        8. Publish PointCloud2
        """
        if self.drone_x is None:
            return

        # IMU gate
        if self.yaw_rate > MAX_YAW_RATE:
            self.get_logger().debug(
                f'IMU gate: yaw_rate={self.yaw_rate:.2f} rad/s — skipping')
            return

        # Convert depth image
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f'depth conversion error: {e}')
            return

        h, w = depth.shape

        # Subsample pixel grid
        rows = np.arange(0, h, SUBSAMPLE)
        cols = np.arange(0, w, SUBSAMPLE)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        rr = rr.ravel()
        cc = cc.ravel()
        d  = depth[rr, cc]

        # Depth validity filter
        valid = np.isfinite(d) & (d >= MIN_DEPTH) & (d <= MAX_DEPTH)
        rr = rr[valid]
        cc = cc[valid]
        d  = d[valid]

        if len(d) == 0:
            return

        # ── Camera → body → world projection ─────────────────────────────────
        # Camera frame
        x_cam =  (cc - CX) * d / FX
        y_cam =  (rr - CY) * d / FY
        z_cam =  d

        # Body frame (OAK-D faces forward along drone nose)
        body_x =  z_cam    # forward
        body_y = -x_cam    # left
        body_z = -y_cam    # up

        # World frame (rotate by drone yaw, translate by drone position)
        cos_y = math.cos(self.drone_yaw)
        sin_y = math.sin(self.drone_yaw)

        world_x = self.drone_x + cos_y * body_x - sin_y * body_y
        world_y = self.drone_y + sin_y * body_x + cos_y * body_y
        world_z = self.lidar_z + body_z   # use LiDAR altitude as Z base

        # Altitude filter
        alt_ok  = (world_z >= Z_MIN) & (world_z <= Z_MAX)
        world_x = world_x[alt_ok]
        world_y = world_y[alt_ok]
        world_z = world_z[alt_ok]

        if len(world_x) == 0:
            return

        # Self exclusion — remove points too close to drone center
        dist_2d = np.sqrt(
            (world_x - self.drone_x) ** 2 +
            (world_y - self.drone_y) ** 2)
        far_enough = dist_2d > SELF_EXCLUSION_RADIUS
        world_x = world_x[far_enough]
        world_y = world_y[far_enough]
        world_z = world_z[far_enough]

        if len(world_x) == 0:
            return

        # ── Build and publish PointCloud2 ─────────────────────────────────────
        pts = np.stack(
            [world_x, world_y, world_z], axis=1).astype(np.float32)

        pc_msg = PointCloud2()
        pc_msg.header.stamp    = msg.header.stamp
        pc_msg.header.frame_id = 'odom'
        pc_msg.height    = 1
        pc_msg.width     = len(pts)
        pc_msg.fields    = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg.is_bigendian = False
        pc_msg.point_step   = 12
        pc_msg.row_step     = 12 * len(pts)
        pc_msg.is_dense     = True
        pc_msg.data         = pts.tobytes()
        self.pub.publish(pc_msg)

        self.get_logger().debug(f'Published {len(pts)} points')
        self.get_logger().info(
            f'depth→cloud: {len(pts)} pts  '
            f'drone=({self.drone_x:.2f},{self.drone_y:.2f},{self.lidar_z:.2f})',
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