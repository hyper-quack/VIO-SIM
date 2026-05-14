import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry, VehicleStatus


class FlightControllerBridge(Node):
    def __init__(self):
        super().__init__('flight_controller_bridge')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
           durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v4',
            self.status_callback,
            qos
        )

        self.get_logger().info('Bridge PX4 ↔ ROS 2 démarré ✓')

    def odom_callback(self, msg):
        x = msg.position[0]
        y = msg.position[1]
        z = msg.position[2]
        self.get_logger().info(
            f'Position → x:{x:.2f}m  y:{y:.2f}m  z:{z:.2f}m',
            throttle_duration_sec=1.0
        )

    def status_callback(self, msg):
        armed = msg.arming_state == 2
        self.get_logger().info(
            f'Drone armé: {armed}',
            throttle_duration_sec=2.0
        )


def main():
    rclpy.init()
    node = FlightControllerBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()