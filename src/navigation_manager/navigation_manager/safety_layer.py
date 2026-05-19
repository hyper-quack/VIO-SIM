import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool, Float32MultiArray
 
 
class SafetyLayer(Node):
    """
    Collision-prevention watchdog.
 
    Behaviour:
      - Normal:    pass AGPF /desired_velocity through to /safe_velocity unchanged.
      - Obstacle:  full stop (zero velocity) when anything is too close.
                   AGPF keeps computing forces in the background.
      - Resume:    when obstacle clears hysteresis threshold, pass-through resumes
                   automatically — AGPF force already points around the obstacle.
 
    NO /emergency_stop is published — that would put mission_node into
    STATE_EMERGENCY_STOP and trigger a 100-tick countdown before landing.
    Safety layer handles everything locally by zeroing the velocity output.
 
    Thresholds match AGPF safe distance (0.8 m) so the drone stops
    well before AGPF's repulsion zone is exceeded.
    """
 
    # ── Thresholds ────────────────────────────────────────────────────────
    # Front: stop when obstacle closer than ON, resume when farther than OFF
    FRONT_STOP_ON  = 0.5    # m  ← trigger full stop
    FRONT_STOP_OFF = 0.7    # m  ← resume (hysteresis)
 
    # Lateral: corridor walls — stop if side is genuinely too close
    LATERAL_STOP_ON  = 0.4  # m  ← trigger
    LATERAL_STOP_OFF = 0.6  # m  ← resume
 
    # How many consecutive clear readings before resuming (debounce)
    CLEAR_DEBOUNCE = 5      # ticks at 20 Hz = 0.25 s
 
    def __init__(self):
        super().__init__('safety_layer')
 
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
 
        # ── Sensor state ──────────────────────────────────────────────────
        self.front_dist = float('inf')
        self.left_dist  = float('inf')
        self.right_dist = float('inf')
 
        # ── Internal state ────────────────────────────────────────────────
        self.stopped       = False   # True = currently holding drone stopped
        self.clear_counter = 0       # debounce counter for resume
        self.desired_velocity = None
 
        # ── Subscribers ───────────────────────────────────────────────────
        self.distances_sub = self.create_subscription(
            Float32MultiArray, '/obstacle_distances',
            self.distances_callback, qos)
 
        self.vel_sub = self.create_subscription(
            TwistStamped, '/desired_velocity',
            self.velocity_callback, qos)
 
        # ── Publishers ────────────────────────────────────────────────────
        self.safe_vel_pub = self.create_publisher(
            TwistStamped, '/safe_velocity', 10)
 
        # Watchdog at 20 Hz
        self.create_timer(0.05, self.watchdog)
 
        self.get_logger().info(
            f'Safety Layer démarré ✓  '
            f'front stop={self.FRONT_STOP_ON}m  '
            f'lateral stop={self.LATERAL_STOP_ON}m')
 
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
 
    # ═══════════════════════════════════════════════════════════════════════
    # Watchdog — 20 Hz
    # ═══════════════════════════════════════════════════════════════════════
 
    def watchdog(self):
        self._update_stop_state()
        self._publish_safe_velocity()
 
    def _update_stop_state(self):
        """
        Hysteresis state machine:
          RUNNING → STOPPED when obstacle enters danger zone
          STOPPED → RUNNING when obstacle clears for CLEAR_DEBOUNCE ticks
        """
        front_danger   = self.front_dist  < self.FRONT_STOP_ON
        lateral_danger = (self.left_dist  < self.LATERAL_STOP_ON or
                          self.right_dist < self.LATERAL_STOP_ON)
 
        front_clear   = self.front_dist  > self.FRONT_STOP_OFF
        lateral_clear = (self.left_dist  > self.LATERAL_STOP_OFF and
                         self.right_dist > self.LATERAL_STOP_OFF)
 
        if not self.stopped:
            # ── Running → check if we need to stop ────────────────────
            if front_danger or lateral_danger:
                self.stopped       = True
                self.clear_counter = 0
                self.get_logger().warn(
                    f'🛑 STOP — obstacle proche: '
                    f'front={self.front_dist:.2f}m  '
                    f'L={self.left_dist:.2f}m  '
                    f'R={self.right_dist:.2f}m  '
                    f'— AGPF calcule contournement...')
        else:
            # ── Stopped → check if obstacle has cleared ────────────────
            if front_clear and lateral_clear:
                self.clear_counter += 1
                if self.clear_counter >= self.CLEAR_DEBOUNCE:
                    self.stopped = False
                    self.get_logger().info(
                        f'✅ Voie libre — reprise navigation  '
                        f'front={self.front_dist:.2f}m  '
                        f'L={self.left_dist:.2f}m  '
                        f'R={self.right_dist:.2f}m')
            else:
                # Not clear yet — reset debounce counter
                self.clear_counter = 0
                self.get_logger().info(
                    f'⏸  En attente... '
                    f'front={self.front_dist:.2f}m  '
                    f'L={self.left_dist:.2f}m  '
                    f'R={self.right_dist:.2f}m',
                    throttle_duration_sec=1.0)
 
    def _publish_safe_velocity(self):
        """
        Stopped  → publish zero velocity (drone holds position).
        Running  → pass AGPF velocity through unchanged.
        """
        safe_vel = TwistStamped()
        safe_vel.header.stamp    = self.get_clock().now().to_msg()
        safe_vel.header.frame_id = 'odom'
 
        if self.stopped or self.desired_velocity is None:
            # Zero velocity — hold position while AGPF computes new direction
            # linear.x/y/z default to 0.0
            self.safe_vel_pub.publish(safe_vel)
            return
 
        # Pass-through — AGPF handles all smooth avoidance
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
