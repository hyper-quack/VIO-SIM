import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan


class OdomToPath(Node):
    def __init__(self):
        super().__init__('odom_to_path')
        self.path = Path()
        self.path.header.frame_id = 'odom'
        self._lidar_z = 0.0

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(Path, '/odom_path', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_be)
        self.create_subscription(LaserScan, '/mtf01/lidar', self.lidar_cb, qos_be)
        self.get_logger().info('odom_to_path started — publishing /odom_path with LiDAR Z')

    def lidar_cb(self, msg: LaserScan):
        if msg.ranges and math.isfinite(msg.ranges[0]):
            self._lidar_z = float(msg.ranges[0])

    def odom_cb(self, msg: Odometry):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        pose.pose.position.z = self._lidar_z
        self.path.poses.append(pose)
        self.path.header.stamp = msg.header.stamp
        self.pub.publish(self.path)


def main():
    rclpy.init()
    rclpy.spin(OdomToPath())


if __name__ == '__main__':
    main()
