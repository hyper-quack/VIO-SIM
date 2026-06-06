import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from px4_msgs.msg import VehicleOdometry


SPAWN_X = 1.0
SPAWN_Y = 3.0


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

        # Second path for OpenVINS odometry.
        self.openvins_path = Path()
        self.openvins_path.header.frame_id = 'odom'
        self._openvins_origin = None

        self.openvins_pub = self.create_publisher(Path, '/openvins_path', 10)

        # Velocity integration state (PX4 velocity → world position).
        self._integrated_x = 1.0
        self._integrated_y = 3.0
        self._last_vel_time = None

        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.vel_cb, qos_be)

        self.create_subscription(
            Odometry, '/openvins/odomimu',
            self.openvins_cb, qos_be)

        self.get_logger().info('odom_to_path started — publishing /odom_path with LiDAR Z')

    def lidar_cb(self, msg: LaserScan):
        if msg.ranges and math.isfinite(msg.ranges[0]):
            self._lidar_z = float(msg.ranges[0])

    def vel_cb(self, msg: VehicleOdometry):
        if self._lidar_z <= 0.3:
            self._integrated_x = 1.0
            self._integrated_y = 3.0
            self._last_vel_time = None
            return

        now = msg.timestamp * 1e-6
        if self._last_vel_time is None:
            self._last_vel_time = now
            return
        dt = now - self._last_vel_time
        self._last_vel_time = now
        if dt <= 0 or dt > 0.1:
            return

        # NED → World
        vx_world =  msg.velocity[1]
        vy_world =  msg.velocity[0]   # removed negation
        self._integrated_x += vx_world * dt
        self._integrated_y += vy_world * dt

    def odom_cb(self, msg: Odometry):
        # Airborne guard: only record while above 0.3 m, reset path when grounded.
        if self._lidar_z <= 0.3:
            self.path.poses = []
            self.path.header.stamp = msg.header.stamp
            self.pub.publish(self.path)
            return

        # Extract yaw from the /odom quaternion (visual frame).
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        raw_yaw = math.atan2(siny, cosy)
        world_yaw = raw_yaw - math.pi / 2.0

        # Transform RTAB-Map /odom -> world frame.
        world_x = -msg.pose.pose.position.y + SPAWN_X
        world_y = msg.pose.pose.position.x + SPAWN_Y
        world_z = self._lidar_z

        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.header.stamp = msg.header.stamp
        pose.pose.position.x = world_x
        pose.pose.position.y = world_y
        pose.pose.position.z = world_z
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(world_yaw / 2.0)
        pose.pose.orientation.w = math.cos(world_yaw / 2.0)

        self.path.poses.append(pose)
        self.path.header.stamp = msg.header.stamp
        self.pub.publish(self.path)

    def openvins_cb(self, msg: Odometry):
        if self._lidar_z <= 0.3:
            self._openvins_origin = None
            self.openvins_path.poses = []
            self.openvins_pub.publish(self.openvins_path)
            return

        pos = msg.pose.pose.position

        if self._openvins_origin is None:
            self._openvins_origin = (pos.x, pos.y, pos.z)
            return

        # OpenVINS relative position (y=forward, x=lateral in this setup)
        dy = pos.y - self._openvins_origin[1]  # forward
        dx = pos.x - self._openvins_origin[0]  # lateral

        ov_x = dy + 1.0    # OpenVINS forward → World X
        ov_y = -dx + 3.0   # OpenVINS lateral → World Y

        # Complementary filter: 70% velocity integration, 30% OpenVINS
        # Use OpenVINS only if jump < 2m from integrated position
        if abs(ov_x - self._integrated_x) < 2.0 and abs(ov_y - self._integrated_y) < 2.0:
            fused_x = 0.7 * self._integrated_x + 0.3 * ov_x
            fused_y = 0.7 * self._integrated_y + 0.3 * ov_y
        else:
            fused_x = self._integrated_x
            fused_y = self._integrated_y

        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = fused_x
        pose.pose.position.y = fused_y
        pose.pose.position.z = self._lidar_z
        pose.pose.orientation.w = 1.0

        self.openvins_path.poses.append(pose)
        self.openvins_path.header.stamp = pose.header.stamp
        self.openvins_pub.publish(self.openvins_path)


def main():
    rclpy.init()
    rclpy.spin(OdomToPath())


if __name__ == '__main__':
    main()
