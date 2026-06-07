import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleOdometry


class SafetyLayer(Node):
    """
    Collision-prevention with AGPF-inspired velocity redirection.

    Computes a repulsive force from LiDAR distances in body frame, converts
    to world frame, strips the parallel component (so it does not cancel the
    attractive force), then adds the perpendicular component to the desired
    velocity.  Hard stop only when critically close (EMERGENCY_DIST).
    """

    # ── Repulsion parameters ──────────────────────────────────────────────
    REPULSE_RANGE  = 0.5   # m — repulsion starts at this distance
    REPULSE_GAIN   = 0.3   # force magnitude gain
    EMERGENCY_DIST = 0.4   # m — hard stop (front only)
    LATERAL_STOP   = 0.4   # m — cancel lateral velocity toward that wall

    def __init__(self):
        super().__init__('safety_layer')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )

        # ── Sensor state ──────────────────────────────────────────────────
        self.front_dist = float('inf')
        self.left_dist  = float('inf')
        self.right_dist = float('inf')

        # ── Internal state ────────────────────────────────────────────────
        self.desired_velocity = None
        self.drone_yaw        = 0.0

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray, '/obstacle_distances',
            self.distances_callback, qos)

        self.create_subscription(
            TwistStamped, '/desired_velocity',
            self.velocity_callback, qos)

        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self._odom_cb, qos_px4)

        # ── Publishers ────────────────────────────────────────────────────
        self.safe_vel_pub = self.create_publisher(
            TwistStamped, '/safe_velocity', 10)

        # 20 Hz watchdog
        self.create_timer(0.05, self.watchdog)

        self.get_logger().info(
            f'Safety Layer started ✓  '
            f'repulse_range={self.REPULSE_RANGE}m  '
            f'emergency={self.EMERGENCY_DIST}m')

    # ═══════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════════════

    def distances_callback(self, msg):
        if len(msg.data) >= 3:
            self.front_dist = msg.data[0]
            self.left_dist  = msg.data[1]
            self.right_dist = msg.data[2]

    def velocity_callback(self, msg):
        self.desired_velocity = msg

    def _odom_cb(self, msg):
        qw = float(msg.q[0])
        qx = float(msg.q[1])
        qy = float(msg.q[2])
        qz = float(msg.q[3])
        siny = 2.0 * (qw*qz + qx*qy)
        cosy = 1.0 - 2.0 * (qz*qz + qy*qy)
        self.drone_yaw = math.atan2(siny, cosy) - math.pi / 2.0

    # ═══════════════════════════════════════════════════════════════════════
    # Watchdog — 20 Hz
    # ═══════════════════════════════════════════════════════════════════════

    def watchdog(self):
        self._compute_safe_velocity()

    def _compute_safe_velocity(self):
        front_dist = self.front_dist
        left_dist  = self.left_dist
        right_dist = self.right_dist

        safe_vel = TwistStamped()
        safe_vel.header.stamp    = self.get_clock().now().to_msg()
        safe_vel.header.frame_id = 'odom'

        if self.desired_velocity is None:
            self.safe_vel_pub.publish(safe_vel)
            return

        des_vx = self.desired_velocity.twist.linear.x
        des_vy = self.desired_velocity.twist.linear.y
        des_vz = self.desired_velocity.twist.linear.z

        # A/B) Body → world frame helpers
        cos_y = math.cos(self.drone_yaw)
        sin_y = math.sin(self.drone_yaw)
        att_norm = math.sqrt(des_vx**2 + des_vy**2)

        # C) For SIDE forces only, strip parallel component.
        #    Front repulsion keeps its full backward push.
        side_rep_x = 0.0
        side_rep_y = 0.0
        if left_dist < self.REPULSE_RANGE:
            side_rep_y -= self.REPULSE_GAIN * (1.0/left_dist - 1.0/self.REPULSE_RANGE)
        if right_dist < self.REPULSE_RANGE:
            side_rep_y += self.REPULSE_GAIN * (1.0/right_dist - 1.0/self.REPULSE_RANGE)

        # Body → world for side forces
        side_world_x = cos_y * side_rep_x - sin_y * side_rep_y
        side_world_y = sin_y * side_rep_x + cos_y * side_rep_y

        # Strip parallel from side forces only
        if att_norm > 0.01:
            dot = side_world_x*(des_vx/att_norm) + side_world_y*(des_vy/att_norm)
            side_world_x -= dot * (des_vx/att_norm)
            side_world_y -= dot * (des_vy/att_norm)

        # Front repulsion: full backward force (no parallel stripping)
        front_rep_x = 0.0
        if front_dist < self.REPULSE_RANGE:
            front_rep_x -= self.REPULSE_GAIN * (1.0/front_dist - 1.0/self.REPULSE_RANGE)
        front_world_x = cos_y * front_rep_x
        front_world_y = sin_y * front_rep_x

        # D) Blend both into desired velocity
        safe_vx = des_vx + front_world_x + side_world_x
        safe_vy = des_vy + front_world_y + side_world_y
        safe_vz = des_vz

        # E) Front emergency hard stop
        if front_dist < self.EMERGENCY_DIST:
            safe_vx = 0.0
            safe_vy = 0.0
            safe_vz = 0.0
            self.get_logger().warn(
                f'EMERGENCY STOP: front={front_dist:.2f}m',
                throttle_duration_sec=0.5)

        # F) Lateral: cancel velocity component only when moving toward that wall
        # body_vy = -sin_y*vx + cos_y*vy  (positive = leftward)
        body_vy = -sin_y*safe_vx + cos_y*safe_vy
        if left_dist < self.LATERAL_STOP and body_vy > 0:
            # Zero out leftward body component, keep forward
            body_vx = cos_y*safe_vx + sin_y*safe_vy
            safe_vx = cos_y*body_vx
            safe_vy = sin_y*body_vx
        elif right_dist < self.LATERAL_STOP and body_vy < 0:
            # Zero out rightward body component, keep forward
            body_vx = cos_y*safe_vx + sin_y*safe_vy
            safe_vx = cos_y*body_vx
            safe_vy = sin_y*body_vx

        # Log when repulsion is active
        rep_world_x = front_world_x + side_world_x
        rep_world_y = front_world_y + side_world_y
        if abs(rep_world_x) > 0.1 or abs(rep_world_y) > 0.1:
            self.get_logger().info(
                f'APF redirect: rep=({rep_world_x:.2f},{rep_world_y:.2f}) '
                f'F={front_dist:.2f} L={left_dist:.2f} R={right_dist:.2f}',
                throttle_duration_sec=0.5)

        safe_vel.twist.linear.x  = float(safe_vx)
        safe_vel.twist.linear.y  = float(safe_vy)
        safe_vel.twist.linear.z  = float(safe_vz)
        safe_vel.twist.angular   = self.desired_velocity.twist.angular
        self.safe_vel_pub.publish(safe_vel)


def main():
    rclpy.init()
    node = SafetyLayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
