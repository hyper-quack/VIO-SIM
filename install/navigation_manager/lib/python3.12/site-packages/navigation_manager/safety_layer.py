import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool, Float32MultiArray


class SafetyLayer(Node):
    """
    Thin collision-prevention watchdog.
    AGPF handles smooth obstacle avoidance; this layer only intervenes
    when something is genuinely about to hit the drone.
    """

    # ── Hard collision thresholds ─────────────────────────────────────────
    COLLISION_DIST_ON  = 0.20    # m — emergency triggers below this
    COLLISION_DIST_OFF = 0.35    # m — emergency clears above this (hysteresis)
    LATERAL_CRITICAL   = 0.15    # m — side wall genuinely too close

    def __init__(self):
        super().__init__('safety_layer')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Distances from obstacle_detector (fused LiDAR + depth)
        self.front_dist = float('inf')
        self.left_dist  = float('inf')
        self.right_dist = float('inf')

        self.emergency        = False
        self.desired_velocity = None

        # Subscribers
        self.distances_sub = self.create_subscription(
            Float32MultiArray, '/obstacle_distances',
            self.distances_callback, qos)

        self.vel_sub = self.create_subscription(
            TwistStamped, '/desired_velocity',
            self.velocity_callback, qos)

        # Publishers
        self.safe_vel_pub = self.create_publisher(
            TwistStamped, '/safe_velocity', 10)

        self.emergency_pub = self.create_publisher(
            Bool, '/emergency_stop', 10)

        # Watchdog at 20 Hz
        self.create_timer(0.05, self.watchdog)

        self.get_logger().info('Safety Layer (watchdog) démarré ✓')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def distances_callback(self, msg):
        if len(msg.data) >= 3:
            self.front_dist = msg.data[0]
            self.left_dist  = msg.data[1]
            self.right_dist = msg.data[2]

    def velocity_callback(self, msg):
        self.desired_velocity = msg

    # ── Watchdog ───────────────────────────────────────────────────────────

    def watchdog(self):
        self._update_emergency()
        self._publish_safe_velocity()

    def _update_emergency(self):
        """Hysteresis-based collision detection."""
        if self.emergency:
            # Already triggered: clear only when really safe
            front_clear   = self.front_dist > self.COLLISION_DIST_OFF
            lateral_clear = (self.left_dist  > self.LATERAL_CRITICAL and
                             self.right_dist > self.LATERAL_CRITICAL)
            if front_clear and lateral_clear:
                self.emergency = False
                self._publish_emergency(False)
                self.get_logger().info(
                    f'Collision risk dégagé — front={self.front_dist:.2f}m')
        else:
            # Trigger only on imminent collision
            front_critical = self.front_dist < self.COLLISION_DIST_ON
            side_critical  = (self.left_dist  < self.LATERAL_CRITICAL or
                              self.right_dist < self.LATERAL_CRITICAL)
            if front_critical or side_critical:
                self.emergency = True
                self._publish_emergency(True)
                self.get_logger().error(
                    f'⚠️  COLLISION IMMINENTE — '
                    f'front={self.front_dist:.2f}m  '
                    f'L={self.left_dist:.2f}m  R={self.right_dist:.2f}m')

    def _publish_emergency(self, state: bool):
        msg = Bool()
        msg.data = state
        self.emergency_pub.publish(msg)

    def _publish_safe_velocity(self):
        safe_vel = TwistStamped()
        safe_vel.header.stamp = self.get_clock().now().to_msg()
        safe_vel.header.frame_id = 'odom'

        if self.emergency or self.desired_velocity is None:
            # Zero velocity — collision imminent or no command yet
            self.safe_vel_pub.publish(safe_vel)
            return

        # Pass-through: AGPF already handled smooth avoidance
        safe_vel.twist = self.desired_velocity.twist
        self.safe_vel_pub.publish(safe_vel)


def main():
    rclpy.init()
    node = SafetyLayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
