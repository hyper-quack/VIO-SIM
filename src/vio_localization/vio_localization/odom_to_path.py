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

        self.rtab_raw_pub = self.create_publisher(Path, '/rtab_raw_path', 10)
        self.rtab_raw_path = Path()
        self.rtab_raw_path.header.frame_id = 'odom'

        self._rtab_raw_x0 = None
        self._rtab_raw_y0 = None
        self._rtab_raw_z0 = None

        # Velocity integration state (PX4 velocity → world position).
        self._integrated_x = 1.0
        self._integrated_y = 3.0
        self._last_vel_time = None

        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.vel_cb, qos_be)

        self.create_subscription(
            Odometry, '/ov_msckf/odomimu',
            self.openvins_cb, qos_be)

        # SE(2) alignment state: RTAB initial pose, world initial pose, locked transform.
        self._rtab_x0 = None
        self._rtab_y0 = None
        self._rtab_yaw0 = None
        self._world_x0 = None
        self._world_y0 = None
        self._world_yaw0 = None
        self._t_world_rtab_ready = False
        self._last_world_x = None
        self._last_world_y = None

        self.create_subscription(PoseStamped, '/current_pose', self._world_pose_cb, 10)

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

    def _world_pose_cb(self, msg):
        if self._world_x0 is None and msg.pose.position.z > 1.5:
            self._world_x0 = msg.pose.position.x
            self._world_y0 = msg.pose.position.y
            q = msg.pose.orientation
            siny = 2*(q.w*q.z + q.x*q.y)
            cosy = 1 - 2*(q.y*q.y + q.z*q.z)
            self._world_yaw0 = math.atan2(siny, cosy)
            self.get_logger().info(
                f'[WORLD_INIT] x={self._world_x0:.3f} y={self._world_y0:.3f} '
                f'yaw={math.degrees(self._world_yaw0):.1f}deg'
            )

    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        if pos.x == 0.0 and pos.y == 0.0 and pos.z == 0.0:
            return
        # Capture RTAB initial pose
        if self._rtab_x0 is None and self._lidar_z > 1.5:
            self._rtab_x0 = msg.pose.pose.position.z
            self._rtab_y0 = msg.pose.pose.position.x
            q = msg.pose.pose.orientation
            siny = 2*(q.w*q.z + q.x*q.y)
            cosy = 1 - 2*(q.y*q.y + q.z*q.z)
            self._rtab_yaw0 = math.atan2(siny, cosy)
            self.get_logger().info(
                f'[RTAB_INIT] x={self._rtab_x0:.3f} y={self._rtab_y0:.3f} '
                f'yaw={math.degrees(self._rtab_yaw0):.1f}deg'
            )

        raw_pose = PoseStamped()
        raw_pose.header.stamp = self.get_clock().now().to_msg()
        raw_pose.header.frame_id = 'odom'
        if self._rtab_raw_x0 is None:
            self._rtab_raw_x0 = msg.pose.pose.position.y
            self._rtab_raw_y0 = msg.pose.pose.position.x
            self._rtab_raw_z0 = msg.pose.pose.position.z
        raw_pose.pose.position.x = msg.pose.pose.position.z - self._rtab_raw_z0 + SPAWN_X
        raw_pose.pose.position.y = msg.pose.pose.position.y - self._rtab_raw_y0 + SPAWN_Y
        raw_pose.pose.position.z = -(msg.pose.pose.position.x - self._rtab_raw_x0)
        raw_pose.pose.orientation = msg.pose.pose.orientation
        self.rtab_raw_path.poses.append(raw_pose)
        self.rtab_raw_path.header.stamp = raw_pose.header.stamp
        self.rtab_raw_pub.publish(self.rtab_raw_path)

        # Wait until world pose is also captured
        if self._world_x0 is None:
            return

        # Compute T_world_rtab once
        if not self._t_world_rtab_ready:
            self._yaw_offset = self._world_yaw0 - self._rtab_yaw0
            self._t_world_rtab_ready = True
            self.get_logger().info(
                f'T_world_rtab locked: yaw_offset={math.degrees(self._yaw_offset):.1f} deg')

        # SE(2) alignment
        dx = msg.pose.pose.position.z - self._rtab_x0
        dy = -(msg.pose.pose.position.x - self._rtab_y0)
        cos_y = math.cos(self._yaw_offset)
        sin_y = math.sin(self._yaw_offset)
        world_x = self._world_x0 + cos_y*dx - sin_y*dy
        world_y = self._world_y0 + sin_y*dx + cos_y*dy
        world_z = self._lidar_z

        # Detect RTAB reset — jump > 3m means re-anchor
        if self._last_world_x is not None:
            jump = math.sqrt((world_x - self._last_world_x)**2 + (world_y - self._last_world_y)**2)
            if jump > 3.0:
                self.get_logger().warn(f'[RTAB] Jump detected {jump:.2f}m — re-anchoring')
                self._rtab_x0 = msg.pose.pose.position.z
                self._rtab_y0 = msg.pose.pose.position.x
                q = msg.pose.pose.orientation
                siny = 2*(q.w*q.z + q.x*q.y)
                cosy = 1 - 2*(q.y*q.y + q.z*q.z)
                self._rtab_yaw0 = math.atan2(siny, cosy)
                self._yaw_offset = self._world_yaw0 - self._rtab_yaw0
                world_x = self._last_world_x
                world_y = self._last_world_y
                self.get_logger().warn(f'[RTAB] Re-anchored at world=({world_x:.2f},{world_y:.2f})')

        self._last_world_x = world_x
        self._last_world_y = world_y

        self.get_logger().info(
            f'[RTAB] raw=({msg.pose.pose.position.x:.3f},{msg.pose.pose.position.y:.3f},{msg.pose.pose.position.z:.3f}) '
            f'world=({world_x:.3f},{world_y:.3f},{world_z:.3f}) '
            f'yaw_offset={math.degrees(self._yaw_offset):.1f}deg'
        )

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'odom'
        pose.pose.position.x = world_x
        pose.pose.position.y = world_y
        pose.pose.position.z = world_z
        self.path.poses.append(pose)
        self.path.header.stamp = pose.header.stamp
        self.path.header.frame_id = 'odom'

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
