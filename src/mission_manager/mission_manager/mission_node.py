import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleOdometry,
)
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
import math


class MissionManager(Node):

    STATE_IDLE           = 0
    STATE_WAIT_EKF       = 1
    STATE_ARMING         = 2
    STATE_TAKEOFF        = 3
    STATE_BUILD_MAP      = 4
    STATE_PLAN_PATH      = 5
    STATE_FOLLOW_PATH    = 6
    STATE_EMERGENCY_STOP = 7
    STATE_LAND           = 8

    # PX4 local frame: (0,0) = spawn point
    # World frame: spawn is at (1.0, 3.0)
    SPAWN_X = 1.0
    SPAWN_Y = 3.0

    # Default goal in WORLD frame
    GOAL_X = 18.0
    GOAL_Y = 3.0

    TARGET_ALTITUDE = -2.0    # NED: negative = up

    def __init__(self):
        super().__init__('mission_manager')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_default = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)

        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)

        self.cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.pose_pub = self.create_publisher(
            PoseStamped, '/current_pose', 10)

        self.nav_start_pub = self.create_publisher(
            Bool, '/start_navigation', 10)
        self.map_reset_pub = self.create_publisher(
            Bool, '/map_reset', 10)

        self.goal_pub = self.create_publisher(
            PoseStamped, '/goal_pose', 10)

        # Subscribers
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4',
            self.status_callback, qos)

        self.odom_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.odom_callback, qos)

        self.safe_vel_sub = self.create_subscription(
            TwistStamped, '/safe_velocity',
            self.safe_velocity_callback, qos_default)

        self.emergency_sub = self.create_subscription(
            Bool, '/emergency_stop',
            self.emergency_callback, qos_default)

        self.nav_active_sub = self.create_subscription(
            Bool, '/navigation_active',
            self.navigation_active_callback, 10)

        self.mtf01_sub = self.create_subscription(
            LaserScan, '/mtf01/lidar',
            self.mtf01_callback, 10)

        self.goal_raw_sub = self.create_subscription(
            PoseStamped, '/goal_raw',
            self.goal_raw_callback, 10)

        # Mission state
        self.state = self.STATE_IDLE
        self.armed    = False
        self.nav_state = 0    # PX4 nav_state; 14 = OFFBOARD

        # PX4 local position
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0

        self.offboard_counter = 0
        self.wait_counter = 0
        self.target_altitude = self.TARGET_ALTITUDE

        self.safe_velocity = None

        self.map_build_counter = 0
        self.plan_counter = 0
        self.arm_counter  = 0    # ticks inside STATE_ARMING
        self.navigation_started = False
        self.emergency_counter = 0
        self.nav_inactive_counter = 0
        self.stuck_counter = 0
        self.stuck_threshold = 200  # 4s at 50Hz
        self.recovery_counter = 0
        self.in_recovery = False
        self.nav_goal_received = False
        self.altitude_m = 0.0

        self.raw_goal_x = self.GOAL_X
        self.raw_goal_y = self.GOAL_Y

        # --------------------------------------------------------------
        # Smooth yaw control
        # --------------------------------------------------------------
        # Initial yaw faces world +X direction, which corresponds to PX4 East.
        self.last_cmd_yaw = math.pi / 2.0
        self.last_yaw_time = None

        # If speed is below this, keep previous yaw.
        self.YAW_SPEED_THRESHOLD = 0.12

        # Ignore tiny yaw changes to avoid oscillation.
        self.YAW_DEADBAND = 0.12          # rad, about 7 degrees

        # Maximum yaw speed.
        # 0.30 rad/s = about 17 deg/s.
        # Reduce to 0.20 if still too fast.
        self.MAX_YAW_RATE = 0.30

        self.timer = self.create_timer(0.02, self.mission_loop)

        self.get_logger().info('Mission Manager démarré ✓')

    # ──────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────

    def status_callback(self, msg):
        self.armed     = (msg.arming_state == 2)
        self.nav_state = msg.nav_state

    def odom_callback(self, msg):
        # PX4 local frame
        self.current_x = msg.position[0]  # North
        self.current_y = msg.position[1]  # East
        self.current_z = msg.position[2]  # Down

        # Publish WORLD frame for A*, path_follower, costmap
        # World X = PX4 East
        # World Y = PX4 North
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'odom'
        pose.pose.position.x = float(self.current_y) + self.SPAWN_X
        pose.pose.position.y = float(self.current_x) + self.SPAWN_Y
        pose.pose.position.z =  float(-self.current_z)
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

    def safe_velocity_callback(self, msg):
        self.safe_velocity = msg

    def emergency_callback(self, msg):
        if msg.data and self.state not in (
                self.STATE_EMERGENCY_STOP, self.STATE_LAND):
            self.get_logger().error('⚠️ EMERGENCY STOP déclenché !')
            self.emergency_counter = 0
            self.state = self.STATE_EMERGENCY_STOP

    def navigation_active_callback(self, msg):
        if not msg.data and self.state == self.STATE_FOLLOW_PATH:
            self.nav_inactive_counter += 1
            if self.nav_inactive_counter >= 50:   # 50 × 0.02 s = 1 s
                self.get_logger().info('🎯 Mission complète — atterrissage')
                self.state = self.STATE_LAND
        else:
            self.nav_inactive_counter = 0

    def mtf01_callback(self, msg):
        if msg.ranges:
            self.altitude_m = float(min(r for r in msg.ranges if r > 0.01))

    def goal_raw_callback(self, msg):
        self.raw_goal_x = float(msg.pose.position.x)
        self.raw_goal_y = float(msg.pose.position.y)
        if self.state == self.STATE_PLAN_PATH:
            self.nav_goal_received = True

        self.get_logger().info(
            f'Raw goal reçu: ({self.raw_goal_x:.2f}, {self.raw_goal_y:.2f})',
            throttle_duration_sec=1.0)

        self._publish_goal()

    # ──────────────────────────────────────────────────────────────────
    # PX4 commands
    # ──────────────────────────────────────────────────────────────────

    def publish_offboard_mode(self, velocity=False):
        msg = OffboardControlMode()
        msg.position = not velocity
        msg.velocity = velocity
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_setpoint(self, x=0.0, y=0.0, z=0.0):
        """Position setpoint in PX4 local frame."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.jerk = [float('nan'), float('nan'), float('nan')]
        msg.yaw = float('nan')
        msg.yawspeed = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def publish_velocity(self, vx=0.0, vy=0.0, vz=0.0, yaw=float('nan')):
        """Velocity setpoint in PX4 local frame."""
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [vx, vy, vz]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.jerk = [float('nan'), float('nan'), float('nan')]
        msg.yaw = yaw
        msg.yawspeed = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def send_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)

    def _publish_goal(self):
        """Publish corrected goal in RViz/A*/PathFollower frame."""
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'odom'
        goal.pose.position.x = float(self.raw_goal_x)
        goal.pose.position.y = float(self.raw_goal_y)
        goal.pose.position.z = 2.0
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)

        self.get_logger().info(
            f'Goal publié /goal_pose: ({goal.pose.position.x:.2f}, '
            f'{goal.pose.position.y:.2f}, {goal.pose.position.z:.2f})',
            throttle_duration_sec=1.0)

    # ──────────────────────────────────────────────────────────────────
    # Smooth yaw helpers
    # ──────────────────────────────────────────────────────────────────

    def wrap_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def compute_smooth_px4_yaw(self, world_vx, world_vy):
        """
        Compute PX4 yaw that faces movement direction,
        but with yaw deadband and yaw-rate limiting.

        Coordinate convention:
            World X = PX4 East
            World Y = PX4 North

        PX4 yaw convention:
            yaw = 0      -> face North
            yaw = pi/2   -> face East

        Therefore:
            target_yaw = atan2(East_velocity, North_velocity)
                       = atan2(world_vx, world_vy)
        """

        now = self.get_clock().now().nanoseconds * 1e-9

        if self.last_yaw_time is None:
            dt = 0.05
        else:
            dt = max(0.01, now - self.last_yaw_time)

        self.last_yaw_time = now

        speed = math.sqrt(world_vx * world_vx + world_vy * world_vy)

        # If the drone is barely moving, do not rotate the camera.
        if speed < self.YAW_SPEED_THRESHOLD:
            return self.last_cmd_yaw

        # Desired yaw in PX4 NED frame
        target_yaw = math.atan2(world_vx, world_vy)

        yaw_error = self.wrap_angle(target_yaw - self.last_cmd_yaw)

        # Ignore tiny direction changes.
        if abs(yaw_error) < self.YAW_DEADBAND:
            return self.last_cmd_yaw

        # Limit yaw rate.
        max_step = self.MAX_YAW_RATE * dt

        if yaw_error > max_step:
            yaw_step = max_step
        elif yaw_error < -max_step:
            yaw_step = -max_step
        else:
            yaw_step = yaw_error

        self.last_cmd_yaw = self.wrap_angle(self.last_cmd_yaw + yaw_step)

        return self.last_cmd_yaw

    # ──────────────────────────────────────────────────────────────────
    # FSM
    # ──────────────────────────────────────────────────────────────────

    def mission_loop(self):
        self.offboard_counter += 1

        # Correct world position for logging
        wx = self.current_y + self.SPAWN_X
        wy = self.current_x + self.SPAWN_Y

        if self.state == self.STATE_IDLE:
            # Stream at 50 Hz so PX4 sees fresh offboard setpoints from the start.
            self.publish_offboard_mode()
            self.publish_setpoint(self.current_x, self.current_y, self.current_z)

            # 100 ticks × 0.02 s = 2 s minimum streaming before any commands.
            if self.offboard_counter >= 100:
                self.state = self.STATE_WAIT_EKF
                self.wait_counter = 0
                self.get_logger().info('Attente stabilisation EKF2...')

        elif self.state == self.STATE_WAIT_EKF:
            self.publish_offboard_mode()
            self.publish_setpoint(self.current_x, self.current_y, self.current_z)

            self.wait_counter += 1

            self.get_logger().info(
                f'Attente EKF2... {self.wait_counter}/200',
                throttle_duration_sec=1.0)

            # 200 ticks × 0.02 s = 4 s — EKF/VIO should be fully converged.
            if self.wait_counter >= 200:
                self.get_logger().info('Envoi commande mode Offboard...')
                self.send_command(176, 1.0, 6.0)   # MAV_CMD_DO_SET_MODE, offboard
                self.arm_counter = 0
                self.state = self.STATE_ARMING

        elif self.state == self.STATE_ARMING:
            # Must keep streaming — PX4 drops offboard if setpoints go stale.
            self.publish_offboard_mode()
            self.publish_setpoint(self.current_x, self.current_y, self.current_z)

            self.arm_counter += 1

            if self.nav_state != 14:
                # Not in offboard yet — retry mode switch every 50 ticks (1 s).
                if self.arm_counter % 50 == 1:
                    self.get_logger().info(
                        f'Retry mode offboard (nav_state={self.nav_state})...')
                    self.send_command(176, 1.0, 6.0)
                self.get_logger().info(
                    f'Attente mode offboard — nav_state={self.nav_state} (cible=14)',
                    throttle_duration_sec=1.0)

            elif not self.armed:
                # Offboard confirmed — send arm command every 25 ticks (0.5 s).
                if self.arm_counter % 25 == 1:
                    self.get_logger().info('Armement...')
                    self.send_command(400, 1.0)
                self.get_logger().info(
                    'Offboard OK — attente armement...',
                    throttle_duration_sec=1.0)

            else:
                # nav_state==14 AND armed — safe to take off.
                self.get_logger().info('Offboard + Armé ✓ — décollage vers 2m...')
                self.state = self.STATE_TAKEOFF

        elif self.state == self.STATE_TAKEOFF:
            self.publish_offboard_mode(velocity=True)
            if self.altitude_m < 1.8:
                self.publish_velocity(0.0, 0.0, -0.3)
            else:
                self.publish_velocity(0.0, 0.0, 0.0)
            self.get_logger().info(
                f'Takeoff alt={self.altitude_m:.2f}m cible=2.0m',
                throttle_duration_sec=1.0)
            if self.altitude_m >= 1.8:
                self.get_logger().info('Takeoff complete — construction carte...')
                self.state = self.STATE_BUILD_MAP
                self.map_build_counter = 0

        elif self.state == self.STATE_BUILD_MAP:
            # Hold current position — use actual NED z not hardcoded
            self.publish_offboard_mode(velocity=False)
            self.publish_setpoint(self.current_x, self.current_y, self.current_z)
            self.map_build_counter += 1
            self.get_logger().info(
                f'Construction carte... {self.map_build_counter}/250 alt={self.altitude_m:.2f}m',
                throttle_duration_sec=2.0)
            if self.map_build_counter >= 250:
                self.get_logger().info('Carte construite — envoi goal + planification...')
                self._publish_goal()
                self.nav_goal_received = False
                self.state = self.STATE_PLAN_PATH
                self.plan_counter = 0
        elif self.state == self.STATE_PLAN_PATH:
            self.publish_offboard_mode()
            self.publish_setpoint(self.current_x, self.current_y, self.current_z)

            self._publish_goal()

            nav_msg = Bool()
            nav_msg.data = True
            self.nav_start_pub.publish(nav_msg)

            self.plan_counter += 1

            self.get_logger().info(
                f'Planification... {self.plan_counter}/150 '
                f'goal_ok={self.nav_goal_received}',
                throttle_duration_sec=1.0)

            if self.plan_counter >= 150 and self.nav_goal_received:  # 150 × 0.02 s = 3 s
                self.plan_counter = 0
                self.get_logger().info('Navigation démarrée — suivi PathFollower...')
                self.state = self.STATE_FOLLOW_PATH

        elif self.state == self.STATE_FOLLOW_PATH:
            self.publish_offboard_mode(velocity=True)
            self._publish_goal()
            nav_msg = Bool()
            nav_msg.data = True
            self.nav_start_pub.publish(nav_msg)

            if self.safe_velocity is not None:
                world_vx = self.safe_velocity.twist.linear.x
                world_vy = self.safe_velocity.twist.linear.y
                vz = -self.safe_velocity.twist.linear.z
                px4_vx = world_vy
                px4_vy = world_vx
                speed = math.sqrt(world_vx**2 + world_vy**2)

                # Stuck detection — only when path_follower active
                if speed < 0.02:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
                    self.in_recovery = False

                # Recovery — back up then resume
                if self.stuck_counter > self.stuck_threshold and not self.in_recovery:
                    self.stuck_counter = 0
                    self.in_recovery = True
                    self.recovery_counter = 0
                    self.get_logger().warn('STUCK — backing up for recovery')

                if self.in_recovery:
                    self.recovery_counter += 1
                    if self.recovery_counter == 1:
                        # Trigger force map update
                        fu = Bool()
                        fu.data = True
                        self.force_update_pub.publish(fu)
                        self.get_logger().warn('STUCK — forcing map update')
                    # Hover in place while map updates
                    self.publish_velocity(0.0, 0.0, 0.0, self.last_cmd_yaw)
                    if self.recovery_counter >= 100:
                        # Stop force update
                        fu = Bool()
                        fu.data = False
                        self.force_update_pub.publish(fu)
                        self.in_recovery = False
                        self.stuck_counter = 0
                        self.get_logger().info('Recovery complete — resuming')
                    self.get_logger().warn(
                        f'Recovery {self.recovery_counter}/100',
                        throttle_duration_sec=1.0)
                else:
                    px4_yaw = self.compute_smooth_px4_yaw(world_vx, world_vy)
                    self.publish_velocity(px4_vx, px4_vy, vz, px4_yaw)
            else:
                # No velocity yet — hover in place, reset stuck counter
                self.stuck_counter = 0
                self.in_recovery = False
                self.publish_velocity(0.0, 0.0, 0.0, self.last_cmd_yaw)

            self.get_logger().info(
                f'FOLLOW_PATH world:({wx:.2f},{wy:.2f}) yaw:{self.last_cmd_yaw:.2f}',
                throttle_duration_sec=2.0)
        elif self.state == self.STATE_EMERGENCY_STOP:
            self.publish_offboard_mode(velocity=True)
            self.publish_velocity(0.0, 0.0, 0.0, self.last_cmd_yaw)

            self.emergency_counter += 1

            self.get_logger().error(
                f'EMERGENCY STOP — hovering ({self.emergency_counter}/250)',
                throttle_duration_sec=1.0)

            if self.emergency_counter >= 250:   # 250 × 0.02 s = 5 s
                self.get_logger().error('Emergency timeout → atterrissage')
                self.state = self.STATE_LAND

        elif self.state == self.STATE_LAND:
            self.publish_offboard_mode()
            self.send_command(21)
            self.publish_setpoint(0.0, 0.0, 0.0)

            self.get_logger().info(
                f'Atterrissage — altitude: {-self.current_z:.2f}m',
                throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = MissionManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()