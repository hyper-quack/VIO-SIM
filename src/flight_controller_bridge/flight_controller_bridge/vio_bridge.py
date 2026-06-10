import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rtabmap_msgs.msg import OdomInfo
from px4_msgs.msg import VehicleOdometry, DistanceSensor


class VIOBridge(Node):
    def __init__(self):
        super().__init__('vio_bridge')

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # rtabmap quality: 0=lost, >0=tracking.
        # Start at 1 so we publish immediately; odom_info sets it to 0 when lost.
        self._odom_quality = 1
        self._odom_info_received = False  # True once /odom_info arrives at least once
        self._reset_counter = 0           # incremented on each tracking recovery
        self._odom_lost = True            # rtabmap lost flag; block publishing until tracking
        self._prev_pos = None             # previous (x, y) for jump rejection
        self._raw_x0 = None               # RTAB origin captured on first valid frame
        self._raw_y0 = None
        self._raw_z0 = None

        odom_info_topic = self.declare_parameter('odom_info_topic', '/odom_info').value
        odom_topic      = self.declare_parameter('odom_topic',      '/odom').value

        self.create_subscription(OdomInfo, odom_info_topic,
                                 self._odom_info_cb, qos_be)
        self.create_subscription(Odometry, odom_topic,
                                 self._odom_cb, qos_be)
        self.get_logger().info(
            f'Subscribing: odom_info={odom_info_topic}  odom={odom_topic}')
        self.create_subscription(LaserScan, '/mtf01/lidar',
                                 self._lidar_cb, qos_be)

        self._vio_pub = self.create_publisher(
            VehicleOdometry, '/fmu/in/vehicle_visual_odometry', qos_px4)
        self._dist_pub = self.create_publisher(
            DistanceSensor, '/fmu/in/distance_sensor', qos_px4)

        self.get_logger().info('VIO Bridge started ✓')

    # ── rtabmap quality gate ─────────────────────────────────────────────────
    def _odom_info_cb(self, msg: OdomInfo):
        prev = self._odom_quality
        self._odom_quality = msg.inliers
        self._odom_lost = msg.lost

        if not self._odom_info_received:
            self._odom_info_received = True
            self.get_logger().info(
                f'/odom_info received — quality gate active (inliers={msg.inliers})')

        self.get_logger().debug(
            f'odom_info: inliers={msg.inliers} lost={msg.lost} type={msg.type}',
            throttle_duration_sec=2.0)

        if prev == 0 and msg.inliers > 0:
            self._reset_counter = (self._reset_counter + 1) % 256
            self.get_logger().info(
                f'VIO tracking recovered (inliers={msg.inliers}), '
                f'reset_counter={self._reset_counter}')
        elif msg.inliers == 0 and prev > 0:
            self.get_logger().warn('VIO tracking lost — suspending odometry to PX4')

    # ── odometry: rtabmap ENU → PX4 NED ─────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        # Gate: block publishing when rtabmap reports lost tracking.
        if self._odom_lost:
            return
        # Gate: require a minimum inlier count (was checking == 0 only).
        if self._odom_quality < 15:
            return

        p = msg.pose.pose.position
        # Capture RTAB origin on first valid frame
        if self._raw_x0 is None:
            self._raw_x0 = float(p.x)
            self._raw_y0 = float(p.y)
            self._raw_z0 = float(p.z)
            self.get_logger().info(
                f'RTAB origin captured: x0={self._raw_x0:.3f} y0={self._raw_y0:.3f} z0={self._raw_z0:.3f}')

        # RTAB → NED (relative to captured origin):
        #   North = RTAB y, East = RTAB z, Down = RTAB x
        north = float(p.y) - self._raw_y0
        east  = float(p.z) - self._raw_z0
        down  = float(p.x) - self._raw_x0
        x = east   # for jump gate check
        y = north

        # Position jump gate: a teleport > 2.0 m between consecutive samples
        # signals a VIO discontinuity (e.g. loop closure). Re-anchor the origin
        # and bump reset_counter so EKF2 re-aligns instead of treating it as motion.
        if self._prev_pos is not None:
            dx = abs(x - self._prev_pos[0])
            dy = abs(y - self._prev_pos[1])
            if dx > 2.0 or dy > 2.0:
                self.get_logger().warn(f'VIO jump {dx:.2f},{dy:.2f}m — re-anchoring origin')
                self._raw_x0 = float(p.x)
                self._raw_y0 = float(p.y)
                self._raw_z0 = float(p.z)
                self._reset_counter = (self._reset_counter + 1) % 256
                self._prev_pos = (0.0, 0.0)
                return
        self._prev_pos = (x, y)

        vio = VehicleOdometry()
        # timestamp_sample = when the sensor measurement was taken (from rtabmap header)
        # timestamp        = when this message is sent (now)
        stamp = msg.header.stamp
        vio.timestamp_sample = stamp.sec * 1_000_000 + stamp.nanosec // 1000
        vio.timestamp = self.get_clock().now().nanoseconds // 1000

        # Position: RTAB → NED (relative to captured origin)
        vio.position[0] = north
        vio.position[1] = east
        vio.position[2] = down

        # Orientation: not fused with EKF2_EV_CTRL=1 (no yaw) — send NaN.
        vio.q[0] = float('nan')
        vio.q[1] = float('nan')
        vio.q[2] = float('nan')
        vio.q[3] = float('nan')

        # Velocity / angular velocity: not fused with EKF2_EV_CTRL=1 — send NaN.
        vio.velocity[0] = float('nan')
        vio.velocity[1] = float('nan')
        vio.velocity[2] = float('nan')
        vio.angular_velocity[0] = float('nan')
        vio.angular_velocity[1] = float('nan')
        vio.angular_velocity[2] = float('nan')

        vio.pose_frame     = VehicleOdometry.POSE_FRAME_NED
        vio.velocity_frame = VehicleOdometry.VELOCITY_FRAME_BODY_FRD

        vio.position_variance[0] = 0.01
        vio.position_variance[1] = 0.01
        vio.position_variance[2] = 0.02

        vio.orientation_variance[0] = 0.01  # roll variance  (rad²)
        vio.orientation_variance[1] = 0.01  # pitch variance (rad²)
        vio.orientation_variance[2] = 0.01  # yaw variance   (rad²)

        vio.velocity_variance[0] = 0.1
        vio.velocity_variance[1] = 0.1
        vio.velocity_variance[2] = 0.1

        vio.reset_counter = self._reset_counter
        # quality=0 tells EKF2 the measurement is invalid; use at least 1
        vio.quality = max(1, min(255, self._odom_quality))
        self._vio_pub.publish(vio)
        self.get_logger().info(
            f'VIO→PX4 q={vio.quality} rst={self._reset_counter} '
            f'N={vio.position[0]:.3f} E={vio.position[1]:.3f} D={vio.position[2]:.3f}',
            throttle_duration_sec=2.0,
        )

    # ── rangefinder: MTF-01 LiDAR → PX4 DistanceSensor ─────────────────────
    def _lidar_cb(self, msg: LaserScan):
        if not msg.ranges:
            return
        distance = msg.ranges[0]
        if not math.isfinite(distance):
            return

        ds = DistanceSensor()
        t = self.get_clock().now().nanoseconds // 1000
        ds.timestamp       = t
        ds.min_distance    = 0.1
        ds.max_distance    = 8.0
        ds.current_distance = float(distance)
        ds.variance        = 0.0004
        ds.signal_quality  = 100
        ds.type            = 1
        ds.orientation     = 0
        self._dist_pub.publish(ds)
        self.get_logger().info(
            f'LiDAR altitude: {distance:.2f} m',
            throttle_duration_sec=2.0,
        )


def main():
    rclpy.init()
    node = VIOBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
