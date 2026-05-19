#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField, Imu
from px4_msgs.msg import VehicleOdometry
from cv_bridge import CvBridge

SPAWN_X   = 1.0
SPAWN_Y   = 3.0
MIN_DEPTH = 0.6
MAX_DEPTH = 6.0
Z_MIN     = 0.5
Z_MAX     = 2.7
SUBSAMPLE = 4
FX        = 161.4
FY        = 161.4
CX        = 160.0
CY        = 120.0

# IMU gate
MAX_YAW_RATE         = 0.8
MAX_VIBRATION        = 5.0
STABLE_FRAMES_NEEDED = 2

# Point/frame thresholds
MIN_POINTS_PER_FRAME = 50    # skip frame if fewer valid points
OBSTACLE_MIN_FRAMES  = 3
MIN_NEIGHBORS        = 5     # min neighbors within 0.3m radius     # point must appear in N consecutive frames to publish


class DepthFilter(Node):
    def __init__(self):
        super().__init__('depth_filter')
        self.bridge         = CvBridge()
        self.drone_x        = None
        self.drone_y        = None
        self.drone_z        = 0.0
        self.drone_yaw      = 0.0
        self.yaw_rate       = 0.0
        self.vibration      = 0.0
        self.stable_count   = 0
        self.mapping_active = True

        # Frame consistency buffer: voxel -> consecutive hit count
        self.voxel_hits     = {}
        self.VOXEL_RES      = 0.2   # metres — voxel size for consistency check

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self.create_subscription(Image,           '/oakd/depth/image',         self.depth_cb, qos)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb,  qos_px4)
        # IMU subscription removed — using PX4 odometry angular velocity instead
        self.pub = self.create_publisher(PointCloud2, '/pointcloud/filtered', 10)
        self.get_logger().info('DepthFilter + IMU gate + frame consistency started')

    def _wrap_angle(self, a):
        while a >  math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a


    def odom_cb(self, msg):
        q = msg.q
        siny = 2.0*(q[0]*q[3] + q[1]*q[2])
        cosy = 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3])
        self.drone_yaw = self._wrap_angle(math.atan2(siny, cosy) - math.pi / 2.0)
        self.drone_x   = float(msg.position[1]) + SPAWN_X
        self.drone_y   = float(msg.position[0]) + SPAWN_Y
        self.drone_z   = -float(msg.position[2])
        self.yaw_rate  = abs(float(msg.angular_velocity[2]))
        ax = float(msg.angular_velocity[0])
        ay = float(msg.angular_velocity[1])
        self.vibration = math.sqrt(ax*ax + ay*ay)

    def _pt_to_voxel(self, x, y, z):
        """Snap world point to voxel key."""
        return (
            int(x / self.VOXEL_RES),
            int(y / self.VOXEL_RES),
            int(z / self.VOXEL_RES)
        )

    def depth_cb(self, msg):
        if self.drone_x is None:
            return

        # ── IMU gate ──────────────────────────────────────────────────────────
        if self.yaw_rate > MAX_YAW_RATE or self.vibration > MAX_VIBRATION:
            self.stable_count   = 0
            self.mapping_active = False
            # Decay all voxel hits when skipping
            self.voxel_hits = {k: max(0, v-1) for k, v in self.voxel_hits.items()}
            self.voxel_hits = {k: v for k, v in self.voxel_hits.items() if v > 0}
            return

        if not self.mapping_active:
            self.stable_count += 1
            if self.stable_count < STABLE_FRAMES_NEEDED:
                return
            self.mapping_active = True
            self.get_logger().info('Mapping resumed')

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f'depth error: {e}')
            return

        h, w = depth.shape
        rows  = np.arange(0, h, SUBSAMPLE)
        cols  = np.arange(0, w, SUBSAMPLE)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        rr = rr.ravel()
        cc = cc.ravel()
        d  = depth[rr, cc]

        valid = np.isfinite(d) & (d >= MIN_DEPTH) & (d < MAX_DEPTH)
        rr, cc, d = rr[valid], cc[valid], d[valid]

        # ── Min points threshold ──────────────────────────────────────────────
        if len(d) < MIN_POINTS_PER_FRAME:
            self.get_logger().debug(f'Frame skipped: only {len(d)} points')
            return

        # ── Project to world frame ────────────────────────────────────────────
        x_cam  = (cc - CX) * d / FX
        y_cam  = (rr - CY) * d / FY
        body_x =  d
        body_y = -x_cam
        body_z = -y_cam

        cos_y   = math.cos(self.drone_yaw)
        sin_y   = math.sin(self.drone_yaw)
        world_x = self.drone_x + cos_y * body_x - sin_y * body_y
        world_y = self.drone_y + sin_y * body_x + cos_y * body_y
        world_z = self.drone_z + body_z

        alt_valid = (world_z >= Z_MIN) & (world_z <= Z_MAX)
        world_x = world_x[alt_valid]
        world_y = world_y[alt_valid]
        world_z = world_z[alt_valid]
        if len(world_x) == 0:
            return

            return

        # ── Frame consistency filter ──────────────────────────────────────────
        # Increment hit count for voxels seen this frame
        seen_this_frame = set()
        for i in range(len(world_x)):
            vk = self._pt_to_voxel(world_x[i], world_y[i], world_z[i])
            seen_this_frame.add(vk)
            self.voxel_hits[vk] = self.voxel_hits.get(vk, 0) + 1

        # Decay voxels NOT seen this frame
        for vk in list(self.voxel_hits.keys()):
            if vk not in seen_this_frame:
                self.voxel_hits[vk] = max(0, self.voxel_hits[vk] - 1)
                if self.voxel_hits[vk] == 0:
                    del self.voxel_hits[vk]

        # Only keep points whose voxel has been seen >= OBSTACLE_MIN_FRAMES
        confirmed_mask = np.array([
            self.voxel_hits.get(
                self._pt_to_voxel(world_x[i], world_y[i], world_z[i]), 0
            ) >= OBSTACLE_MIN_FRAMES
            for i in range(len(world_x))
        ])

        world_x = world_x[confirmed_mask]
        world_y = world_y[confirmed_mask]
        world_z = world_z[confirmed_mask]

        if len(world_x) == 0:
            return

        pts = np.stack([world_x, world_y, world_z], axis=1).astype(np.float32)

        pc = PointCloud2()
        pc.header.stamp    = msg.header.stamp
        pc.header.frame_id = 'odom'
        pc.height    = 1
        pc.width     = len(pts)
        pc.fields    = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        pc.is_bigendian = False
        pc.point_step   = 12
        pc.row_step     = 12 * len(pts)
        pc.is_dense     = True
        pc.data         = pts.tobytes()
        self.pub.publish(pc)
        self.get_logger().debug(f'Published {len(pts)} confirmed pts')


def main(args=None):
    rclpy.init(args=args)
    node = DepthFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
