#!/usr/bin/env python3
"""
pose_fusion.py — Adaptive GPS/VIO sensor fusion.

Fuses PX4/EKF2 pose (GPS-based) with OpenVINS VIO pose.
Outdoor / GPS good  → alpha ≈ 1.0 → pure GPS
Indoor  / GPS weak  → alpha ≈ 0.0 → pure VIO

Output: /slam/corrected_pose (PoseStamped) consumed by octomap_manager.
        /pose_fusion/{alpha,gps_quality,vio_quality} for debugging.

Coordinate transforms (NEVER CHANGE):
  GPS:  gps_world_x = odom.position[1] + 1.0
        gps_world_y = odom.position[0] + 3.0
        gps_world_z = -odom.position[2]
  VIO:  vio_world_x = vio.pose.pose.position.x + 1.0
        vio_world_y = vio.pose.pose.position.y + 3.0
        vio_world_z = vio.pose.pose.position.z
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from px4_msgs.msg import VehicleOdometry, SensorGps
from rtabmap_msgs.msg import OdomInfo

try:
    from navigation_manager.config import SPAWN_X, SPAWN_Y
except ImportError:
    SPAWN_X, SPAWN_Y = 1.0, 3.0

VIO_TIMEOUT_S = 2.0   # seconds without VIO → quality = 0


class PoseFusion(Node):

    def __init__(self):
        super().__init__('pose_fusion')

        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # --- GPS state ---
        self._gps_x = None
        self._gps_y = None
        self._gps_z = 0.0
        self._gps_qw = 1.0
        self._gps_qx = 0.0
        self._gps_qy = 0.0
        self._gps_qz = 0.0
        self._gps_quality = 0.0

        # --- VIO state (RTAB-Map /odom + /odom_info) ---
        self._vio_x = None
        self._vio_y = None
        self._vio_z = 0.0
        self._vio_quality = 0.0
        self._vio_last_stamp = None    # time of last /odom message
        self._odom_info_last_stamp = None  # time of last /odom_info message

        # --- Fusion state ---
        self._alpha = 1.0          # 1.0 = pure GPS, 0.0 = pure VIO
        self._last_pub_x = None
        self._last_pub_y = None
        self._frame_count = 0

        self.create_subscription(VehicleOdometry,
                                 '/fmu/out/vehicle_odometry',
                                 self._odom_cb, qos_px4)
        self.create_subscription(SensorGps,
                                 '/fmu/out/vehicle_gps_position',
                                 self._gps_cb, qos_px4)
        self.create_subscription(Odometry,
                                 '/odom',
                                 self._rtab_odom_cb, qos_be)
        self.create_subscription(OdomInfo,
                                 '/odom_info',
                                 self._rtab_info_cb, qos_be)

        self._pose_pub  = self.create_publisher(PoseStamped,  '/slam/corrected_pose', 10)
        self._alpha_pub = self.create_publisher(Float32, '/pose_fusion/alpha', 10)
        self._gps_q_pub = self.create_publisher(Float32, '/pose_fusion/gps_quality', 10)
        self._vio_q_pub = self.create_publisher(Float32, '/pose_fusion/vio_quality', 10)

        self.create_timer(0.05, self._fuse)   # 20 Hz

        self.get_logger().info('PoseFusion started — GPS/VIO adaptive fusion → /slam/corrected_pose')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: VehicleOdometry):
        self._gps_x = float(msg.position[1]) + SPAWN_X
        self._gps_y = float(msg.position[0]) + SPAWN_Y
        self._gps_z = -float(msg.position[2])
        self._gps_qw = float(msg.q[0])
        self._gps_qx = float(msg.q[1])
        self._gps_qy = float(msg.q[2])
        self._gps_qz = float(msg.q[3])

    def _gps_cb(self, msg: SensorGps):
        self._gps_quality = self._compute_gps_quality(msg)

    def _rtab_odom_cb(self, msg: Odometry):
        self._vio_x = msg.pose.pose.position.x + SPAWN_X
        self._vio_y = msg.pose.pose.position.y + SPAWN_Y
        self._vio_z = msg.pose.pose.position.z
        self._vio_last_stamp = self.get_clock().now()

    def _rtab_info_cb(self, msg: OdomInfo):
        self._odom_info_last_stamp = self.get_clock().now()
        inliers = msg.inliers
        if inliers >= 50:
            self._vio_quality = 1.0
        elif inliers >= 30:
            self._vio_quality = 0.7
        elif inliers >= 15:
            self._vio_quality = 0.4
        elif inliers >= 5:
            self._vio_quality = 0.1
        else:
            self._vio_quality = 0.0

    # ── Quality helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_gps_quality(msg: SensorGps) -> float:
        sat = msg.satellites_used
        fix = msg.fix_type
        eph = msg.eph
        if sat >= 8 and fix >= 3 and eph < 1.5:
            return 1.0
        elif sat >= 6 and fix >= 3 and eph < 3.0:
            return 0.6
        elif sat >= 4 and fix >= 2:
            return 0.2
        return 0.0

    def _current_vio_quality(self) -> float:
        now = self.get_clock().now()
        if self._vio_last_stamp is None:
            return 0.0
        if (now - self._vio_last_stamp).nanoseconds * 1e-9 > VIO_TIMEOUT_S:
            return 0.0
        if self._odom_info_last_stamp is None:
            return 0.0
        if (now - self._odom_info_last_stamp).nanoseconds * 1e-9 > 3.0:
            return 0.0
        return self._vio_quality

    # ── Fusion timer (20 Hz) ─────────────────────────────────────────────────

    def _fuse(self):
        if self._gps_x is None:
            return

        gps_q = self._gps_quality
        vio_q = self._current_vio_quality()

        # Alpha: 1.0 = pure GPS, 0.0 = pure VIO
        alpha_raw = gps_q / (gps_q + vio_q + 1e-6)
        self._alpha = 0.95 * self._alpha + 0.05 * alpha_raw

        alpha = self._alpha

        # GPS-only if VIO never started
        if self._vio_x is None:
            alpha = 1.0

        fused_x = alpha * self._gps_x + (1.0 - alpha) * (self._vio_x or self._gps_x)
        fused_y = alpha * self._gps_y + (1.0 - alpha) * (self._vio_y or self._gps_y)
        fused_z = alpha * self._gps_z + (1.0 - alpha) * (self._vio_z or self._gps_z)

        # Both sources failed — hold last pose
        if gps_q == 0.0 and vio_q == 0.0:
            if self._last_pub_x is None:
                return
            fused_x = self._last_pub_x
            fused_y = self._last_pub_y

        # Jump guard
        if self._last_pub_x is not None:
            jump = math.sqrt((fused_x - self._last_pub_x)**2 +
                             (fused_y - self._last_pub_y)**2)
            if jump > 1.0:
                self.get_logger().warn(
                    f'[FUSION] position jump {jump:.2f}m — skipping',
                    throttle_duration_sec=1.0)
                return

        self._last_pub_x = fused_x
        self._last_pub_y = fused_y

        # Orientation: always from GPS (PX4 EKF2 yaw is reliable)
        siny = 2.0 * (self._gps_qw * self._gps_qz + self._gps_qx * self._gps_qy)
        cosy = 1.0 - 2.0 * (self._gps_qy * self._gps_qy + self._gps_qz * self._gps_qz)
        yaw  = math.atan2(siny, cosy) - math.pi / 2.0

        pose_msg = PoseStamped()
        pose_msg.header.stamp    = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.position.x = float(fused_x)
        pose_msg.pose.position.y = float(fused_y)
        pose_msg.pose.position.z = float(fused_z)
        pose_msg.pose.orientation.z = float(math.sin(yaw / 2.0))
        pose_msg.pose.orientation.w = float(math.cos(yaw / 2.0))
        self._pose_pub.publish(pose_msg)

        # Publish diagnostics
        self._alpha_pub.publish(Float32(data=float(alpha)))
        self._gps_q_pub.publish(Float32(data=float(gps_q)))
        self._vio_q_pub.publish(Float32(data=float(vio_q)))

        # Periodic log
        self._frame_count += 1
        if self._frame_count % 50 == 0:
            if alpha > 0.8:
                source = 'GPS'
            elif alpha < 0.2:
                source = 'VIO'
            else:
                source = 'BLEND'
            self.get_logger().info(
                f'[FUSION] GPS_Q={gps_q:.2f} VIO_Q={vio_q:.2f} '
                f'alpha={alpha:.2f} source={source}')


def main():
    rclpy.init()
    node = PoseFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
