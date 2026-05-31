#!/usr/bin/env python3
"""
tf_odom_base.py — Publishes odom→base_link TF from PX4 VehicleOdometry.

Publishes at 100 Hz by replaying the last known transform on a timer so RViz
never sees a TF gap even when odometry messages are delayed or dropped.
An identity transform is broadcast immediately at startup so the frame exists
before the first odometry message arrives (avoids RViz startup race).
"""

import math
import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf2_ros import TransformBroadcaster
from px4_msgs.msg import VehicleOdometry

try:
    from navigation_manager.config import SPAWN_X, SPAWN_Y
except ImportError:
    SPAWN_X, SPAWN_Y = 1.0, 3.0

# PX4 NED → ROS ENU yaw offset: PX4 yaw=0 is North, ROS ENU yaw=0 is East.
# Pre-multiply by a -90° rotation around Z to convert NED→ENU.
# q_enu = q_corr * q_ned  with  q_corr = (w=1/√2, x=0, y=0, z=-1/√2)
_INV_SQRT2 = math.sqrt(0.5)


class TfOdomBase(Node):
    def __init__(self):
        super().__init__('tf_odom_base')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        self._tf_broadcaster = TransformBroadcaster(self)
        # 5-second local cache so lookups within this node reach back further
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=5)
        )

        # Publish identity immediately so 'odom'/'base_link' frames exist
        # before the first odometry message arrives (prevents RViz startup race).
        identity = TransformStamped()
        identity.header.stamp = self.get_clock().now().to_msg()
        identity.header.frame_id = 'odom'
        identity.child_frame_id = 'base_link'
        identity.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(identity)
        self.last_transform: TransformStamped = identity

        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self._odom_cb,
            qos,
        )

        # 100 Hz keep-alive: republish last known pose so TF is never stale
        self.create_timer(0.01, self._timer_cb)

        self.get_logger().info('tf_odom_base ready — publishing odom→base_link at 100 Hz')

    # ── Callbacks ──────────────────────────────────────────────────

    def _odom_cb(self, msg: VehicleOdometry) -> None:
        t = TransformStamped()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = float(msg.position[1]) + SPAWN_X
        t.transform.translation.y = float(msg.position[0]) + SPAWN_Y
        t.transform.translation.z = -float(msg.position[2])

        # Apply NED→ENU -90° Z rotation: q_enu = (1/√2, 0, 0, -1/√2) * q_ned
        qw, qx, qy, qz = float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])
        t.transform.rotation.w = (qw + qz) * _INV_SQRT2
        t.transform.rotation.x = (qx + qy) * _INV_SQRT2
        t.transform.rotation.y = (qy - qx) * _INV_SQRT2
        t.transform.rotation.z = (qz - qw) * _INV_SQRT2

        self.last_transform = t

    def _timer_cb(self) -> None:
        self.last_transform.header.stamp = self.get_clock().now().to_msg()
        self._tf_broadcaster.sendTransform(self.last_transform)


def main(args=None):
    rclpy.init(args=args)
    node = TfOdomBase()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
