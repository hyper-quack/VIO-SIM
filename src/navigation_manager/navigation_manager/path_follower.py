import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Bool
import math

class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Waypoint acceptance radii
        self.waypoint_radius    = 0.22   # XY
        self.waypoint_radius_z  = 0.15   # Z
        self.goal_radius        = 0.35   # XY final goal
        self.goal_radius_z      = 0.20   # Z final goal

        # Speeds
        self.max_speed    = 0.22
        self.cruise_speed = 0.18
        self.min_speed    = 0.05
        self.max_vz       = 0.15   # max vertical speed m/s
        self.min_vz       = 0.03

        # Smoothing
        self.vel_alpha = 0.25
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.filtered_vz = 0.0

        # State
        self.path              = []
        self.current_idx       = 0
        self.current_pose      = None
        self.emergency         = False
        self.active            = False
        self.goal_reached_sent = False

        # Subscribers
        self.create_subscription(Path, '/planned_path',
            self.path_callback, 10)
        self.create_subscription(PoseStamped, '/current_pose',
            self.pose_callback, qos)
        self.create_subscription(Bool, '/emergency_stop',
            self.emergency_callback, 10)
        self.create_subscription(Bool, '/navigation_active',
            self.active_callback, 10)

        # Publishers
        self.vel_pub     = self.create_publisher(TwistStamped, '/desired_velocity', 10)
        self.reached_pub = self.create_publisher(Bool, '/goal_reached', 10)

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('PathFollower 3D started ✓')

    # ==================================================================
    # Callbacks
    # ==================================================================
    def path_callback(self, msg):
        if not msg.poses:
            return
        self.path = msg.poses
        self.active = True
        self.goal_reached_sent = False

        if self.current_pose is not None:
            best_idx  = 0
            best_dist = float('inf')
            for i, wp in enumerate(self.path):
                dx = wp.pose.position.x - self.current_pose.pose.position.x
                dy = wp.pose.position.y - self.current_pose.pose.position.y
                dz = wp.pose.position.z - self.current_pose.pose.position.z
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i
            self.current_idx = best_idx
            if self.current_idx < len(self.path)-1 and best_dist < self.waypoint_radius:
                self.current_idx += 1
        else:
            self.current_idx = 0

        self.get_logger().info(
            f'New path: {len(self.path)} waypoints, start idx={self.current_idx}'
        )

    def pose_callback(self, msg):
        self.current_pose = msg

    def emergency_callback(self, msg):
        self.emergency = msg.data
        if self.emergency:
            self.stop()

    def active_callback(self, msg):
        self.active = msg.data
        if not self.active:
            self.stop()

    # ==================================================================
    # Helpers
    # ==================================================================
    def stop(self):
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.filtered_vz = 0.0
        vel = TwistStamped()
        vel.header.stamp = self.get_clock().now().to_msg()
        vel.header.frame_id = 'odom'
        vel.twist.linear.x = 0.0
        vel.twist.linear.y = 0.0
        vel.twist.linear.z = 0.0
        self.vel_pub.publish(vel)

    def distance_3d(self, wp):
        if self.current_pose is None:
            return float('inf')
        dx = wp.pose.position.x - self.current_pose.pose.position.x
        dy = wp.pose.position.y - self.current_pose.pose.position.y
        dz = wp.pose.position.z - self.current_pose.pose.position.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def distance_xy(self, wp):
        if self.current_pose is None:
            return float('inf')
        dx = wp.pose.position.x - self.current_pose.pose.position.x
        dy = wp.pose.position.y - self.current_pose.pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    def distance_z(self, wp):
        if self.current_pose is None:
            return float('inf')
        return abs(wp.pose.position.z - self.current_pose.pose.position.z)

    def publish_goal_reached_once(self):
        if self.goal_reached_sent:
            return
        msg = Bool()
        msg.data = True
        self.reached_pub.publish(msg)
        self.goal_reached_sent = True
        self.get_logger().info('Goal reached ✓')

    def publish_velocity(self, vx, vy, vz):
        self.filtered_vx = (1.0-self.vel_alpha)*self.filtered_vx + self.vel_alpha*vx
        self.filtered_vy = (1.0-self.vel_alpha)*self.filtered_vy + self.vel_alpha*vy
        self.filtered_vz = (1.0-self.vel_alpha)*self.filtered_vz + self.vel_alpha*vz

        vel = TwistStamped()
        vel.header.stamp = self.get_clock().now().to_msg()
        vel.header.frame_id = 'odom'
        vel.twist.linear.x = float(self.filtered_vx)
        vel.twist.linear.y = float(self.filtered_vy)
        vel.twist.linear.z = float(self.filtered_vz)
        self.vel_pub.publish(vel)

    # ==================================================================
    # Control loop
    # ==================================================================
    def control_loop(self):
        if not self.active or self.emergency or self.current_pose is None or not self.path:
            self.stop()
            return

        if self.current_idx >= len(self.path):
            self.stop()
            return

        final_goal = self.path[-1]

        # Advance waypoint index
        target = self.path[self.current_idx]
        while (self.distance_xy(target) < self.waypoint_radius and
               self.distance_z(target) < self.waypoint_radius_z and
               self.current_idx < len(self.path)-1):
            self.current_idx += 1
            target = self.path[self.current_idx]
            self.get_logger().info(
                f'Waypoint {self.current_idx}/{len(self.path)} reached'
            )
        
        # Compute XY direction
        dx = target.pose.position.x - self.current_pose.pose.position.x
        dy = target.pose.position.y - self.current_pose.pose.position.y
        dz = target.pose.position.z - self.current_pose.pose.position.z

        norm_xy = math.sqrt(dx*dx + dy*dy)
        dist_to_goal = self.distance_xy(final_goal)

        if norm_xy < 0.01 and abs(dz) < 0.05:
            self.stop()
            return

        # XY speed
        speed = self.cruise_speed
        if dist_to_goal < 1.0:
            speed = max(self.min_speed, self.cruise_speed * dist_to_goal)
        speed = min(speed, self.max_speed)

        if norm_xy > 0.01:
            vx = (dx / norm_xy) * speed
            vy = (dy / norm_xy) * speed
        else:
            vx = 0.0
            vy = 0.0

        # Z speed — proportional to altitude error
        if abs(dz) > 0.05:
            vz = max(self.min_vz, min(self.max_vz, abs(dz) * 0.5))
            vz = math.copysign(vz, dz)
        else:
            vz = 0.0
     

        self.publish_velocity(vx, vy, vz)

        self.get_logger().info(
            f'PF3D [{self.current_idx}/{len(self.path)}] '
            f'pos:({self.current_pose.pose.position.x:.2f},'
            f'{self.current_pose.pose.position.y:.2f},'
            f'{self.current_pose.pose.position.z:.2f}) '
            f'target:({target.pose.position.x:.2f},'
            f'{target.pose.position.y:.2f},'
            f'{target.pose.position.z:.2f}) '
            f'vel:({self.filtered_vx:.2f},{self.filtered_vy:.2f},{self.filtered_vz:.2f})',
            throttle_duration_sec=1.0
        )

def main():
    rclpy.init()
    node = PathFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()