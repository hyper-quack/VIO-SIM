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
        # Gate: only block when odom_info explicitly reports lost (inliers=0).
        # Before odom_info arrives, _odom_quality=1 so we publish immediately.
        if self._odom_info_received and self._odom_quality == 0:
            return

        vio = VehicleOdometry()
        # timestamp_sample = when the sensor measurement was taken (from rtabmap header)
        # timestamp        = when this message is sent (now)
        stamp = msg.header.stamp
        vio.timestamp_sample = stamp.sec * 1_000_000 + stamp.nanosec // 1000
        vio.timestamp = self.get_clock().now().nanoseconds // 1000

        # Position: ENU (x=East, y=North, z=Up) → NED (x=North, y=East, z=Down)
        p = msg.pose.pose.position
        vio.position[0] = float(p.y)   # North
        vio.position[1] = float(p.x)   # East
        vio.position[2] = float(-p.z)  # Down

        # Orientation quaternion: ENU body (FLU) → NED body (FRD).
        # The frame change swaps the North/East axes and flips Down:
        #   q_ned.w =  q_enu.w
        #   q_ned.x =  q_enu.y   (NED North ← ENU North = enu.y)
        #   q_ned.y =  q_enu.x   (NED East  ← ENU East  = enu.x)
        #   q_ned.z = -q_enu.z   (NED Down  ← ENU Up negated)
        q = msg.pose.pose.orientation
        vio.q[0] = float(q.w)
        vio.q[1] = float(q.y)
        vio.q[2] = float(q.x)
        vio.q[3] = float(-q.z)

        # Linear velocity: nav_msgs/Odometry twist is in child frame (body FLU).
        # Convert FLU → FRD: forward=same, right=-left, down=-up
        v = msg.twist.twist.linear
        vio.velocity[0] = float(v.x)
        vio.velocity[1] = float(-v.y)
        vio.velocity[2] = float(-v.z)

        # Angular velocity: FLU body → FRD body
        #   roll  rate (about fwd)  : same sign
        #   pitch rate (about right): negated (FLU left = FRD right)
        #   yaw   rate (about down) : negated (FLU up CCW = FRD down CW)
        w = msg.twist.twist.angular
        vio.angular_velocity[0] = float(w.x)
        vio.angular_velocity[1] = float(-w.y)
        vio.angular_velocity[2] = float(-w.z)

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
