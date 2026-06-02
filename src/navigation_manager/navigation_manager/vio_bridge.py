#!/usr/bin/env python3
"""
vio_bridge.py — Thin adapter: OpenVINS output → our world frame.

Subscribes to /ov_msckf/poseimu (geometry_msgs/PoseWithCovarianceStamped)
and re-publishes as /vio/pose with the SPAWN offset applied to the position.
Covariance and orientation are passed through unchanged.

pose_fusion.py consumes /vio/pose together with GPS to produce
/slam/corrected_pose for octomap_manager.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped

try:
    from navigation_manager.config import SPAWN_X, SPAWN_Y
except ImportError:
    SPAWN_X, SPAWN_Y = 1.0, 3.0


class VioBridge(Node):

    def __init__(self):
        super().__init__('vio_bridge')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/ov_msckf/poseimu',
            self._cb,
            qos)

        self.pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/vio/pose',
            10)

        self.get_logger().info(
            f'VioBridge started — /ov_msckf/poseimu → /vio/pose  '
            f'offset=({SPAWN_X},{SPAWN_Y})')

    def _cb(self, msg: PoseWithCovarianceStamped):
        out = PoseWithCovarianceStamped()
        out.header = msg.header
        out.pose.pose.position.x = msg.pose.pose.position.x + SPAWN_X
        out.pose.pose.position.y = msg.pose.pose.position.y + SPAWN_Y
        out.pose.pose.position.z = msg.pose.pose.position.z
        out.pose.pose.orientation = msg.pose.pose.orientation
        out.pose.covariance = msg.pose.covariance
        self.pub.publish(out)


def main():
    rclpy.init()
    node = VioBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
