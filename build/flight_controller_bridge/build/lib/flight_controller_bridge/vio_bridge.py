import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from px4_msgs.msg import VehicleOdometry, DistanceSensor
import math


class VIOBridge(Node):
    def __init__(self):
        super().__init__('vio_bridge')

        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_sub)

        self.lidar_sub = self.create_subscription(
            LaserScan, '/mtf01/lidar', self.lidar_callback, qos_sub)

        # Publishers vers PX4
        self.vio_pub = self.create_publisher(
            VehicleOdometry,
            '/fmu/in/vehicle_odometry',
            qos_px4
        )

        self.distance_pub = self.create_publisher(
            DistanceSensor,
            '/fmu/in/distance_sensor',
            qos_px4
        )

        self.get_logger().info('VIO Bridge démarré ✓')

    def odom_callback(self, msg):
        vio = VehicleOdometry()

        t = self.get_clock().now().nanoseconds // 1000
        vio.timestamp = t
        vio.timestamp_sample = t

        # Position ROS (ENU) → PX4 (NED)
        vio.position[0] = msg.pose.pose.position.y   # North
        vio.position[1] = msg.pose.pose.position.x   # East
        vio.position[2] = -msg.pose.pose.position.z  # Down

        # Orientation ROS (ENU) → PX4 (NED)
        q = msg.pose.pose.orientation
        vio.q[0] = q.w
        vio.q[1] = q.y
        vio.q[2] = q.x
        vio.q[3] = -q.z

        # Vitesse linéaire
        vio.velocity[0] = msg.twist.twist.linear.y
        vio.velocity[1] = msg.twist.twist.linear.x
        vio.velocity[2] = -msg.twist.twist.linear.z

        vio.angular_velocity[0] = msg.twist.twist.angular.x
        vio.angular_velocity[1] = msg.twist.twist.angular.y
        vio.angular_velocity[2] = msg.twist.twist.angular.z
        # Frame IDs
        vio.pose_frame = VehicleOdometry.POSE_FRAME_NED
        vio.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED

        # Covariance position
        vio.position_variance[0] = 0.01
        vio.position_variance[1] = 0.01
        vio.position_variance[2] = 0.01

        # Covariance vitesse
        vio.velocity_variance[0] = 0.1
        vio.velocity_variance[1] = 0.1
        vio.velocity_variance[2] = 0.1

        vio.reset_counter = 0

        self.vio_pub.publish(vio)
        self.get_logger().info(
            f'VIO → PX4: N={vio.position[0]:.2f} E={vio.position[1]:.2f} D={vio.position[2]:.2f}',
            throttle_duration_sec=2.0
        )

    def lidar_callback(self, msg):
        if not msg.ranges:
            return

        distance = msg.ranges[0]
        if math.isnan(distance) or math.isinf(distance):
            return

        ds = DistanceSensor()
        t = self.get_clock().now().nanoseconds // 1000
        ds.timestamp = t
        ds.min_distance = 0.1
        ds.max_distance = 8.0
        ds.current_distance = distance
        ds.variance = 0.0004
        ds.signal_quality = 100
        ds.type = 1
        ds.orientation = 0

        self.distance_pub.publish(ds)
        self.get_logger().info(
            f'LiDAR altitude: {distance:.2f}m',
            throttle_duration_sec=2.0
        )


def main():
    rclpy.init()
    node = VIOBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()