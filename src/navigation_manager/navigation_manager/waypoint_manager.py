import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import yaml
import os
import math
from ament_index_python.packages import get_package_share_directory


WAYPOINT_RADIUS = 1.0   # metres — how close drone must be to advance waypoint


class WaypointManager(Node):
    def __init__(self):
        super().__init__('waypoint_manager')

        waypoints_file = os.path.join(
            get_package_share_directory('navigation_manager'),
            'waypoints', 'corridor.yaml'
        )

        self.waypoints   = []
        self.current_idx = 0
        self.active      = False
        self.current_pose = None

        try:
            with open(waypoints_file, 'r') as f:
                data = yaml.safe_load(f)
                self.waypoints = data['waypoints']
            self.get_logger().info(
                f'Waypoints chargés : {len(self.waypoints)} points')
        except Exception as e:
            self.get_logger().error(f'Erreur chargement waypoints: {e}')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_raw', 10)
        self.nav_active_pub = self.create_publisher(Bool, '/navigation_active', 10)

        # Subscribers
        self.create_subscription(Bool, '/start_navigation',
            self.start_callback, 10)
        self.create_subscription(Bool, '/emergency_stop',
            self.emergency_callback, 10)
        self.create_subscription(PoseStamped, '/current_pose',
            self.pose_callback, qos)

        # Timer — check if waypoint reached at 5Hz
        self.create_timer(0.2, self.check_progress)

        self.get_logger().info('Waypoint Manager démarré ✓')

    def pose_callback(self, msg):
        self.current_pose = msg

    def start_callback(self, msg):
        if not msg.data or not self.waypoints:
            return
        if self.active:
            return
        self.active      = True
        self.current_idx = 0
        self.get_logger().info('Navigation démarrée ✓')
        self.send_next_goal()

    def emergency_callback(self, msg):
        if msg.data:
            self.active = False
            nav_msg = Bool()
            nav_msg.data = False
            self.nav_active_pub.publish(nav_msg)
            self.get_logger().error('Navigation arrêtée — emergency stop')

    def check_progress(self):
        """Check if drone reached current waypoint by position."""
        if not self.active or self.current_pose is None:
            return
        if self.current_idx >= len(self.waypoints):
            return

        wp = self.waypoints[self.current_idx]
        dx = self.current_pose.pose.position.x - float(wp['x'])
        dy = self.current_pose.pose.position.y - float(wp['y'])
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < WAYPOINT_RADIUS:
            self.get_logger().info(
                f'Waypoint {self.current_idx+1} atteint (dist={dist:.2f}m)')
            self.current_idx += 1

            if self.current_idx >= len(self.waypoints):
                self.get_logger().info('Mission complète ✓')
                self.active = False
                nav_msg = Bool()
                nav_msg.data = False
                self.nav_active_pub.publish(nav_msg)
                return

            self.send_next_goal()

    def send_next_goal(self):
        if self.current_idx >= len(self.waypoints):
            return

        wp = self.waypoints[self.current_idx]
        goal = PoseStamped()
        goal.header.stamp    = self.get_clock().now().to_msg()
        goal.header.frame_id = 'odom'
        goal.pose.position.x = float(wp['x'])
        goal.pose.position.y = float(wp['y'])
        goal.pose.position.z = float(wp['z'])
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)
        self.get_logger().info(
            f'Goal {self.current_idx + 1}/{len(self.waypoints)}: '
            f"x={wp['x']} y={wp['y']} — {wp['description']}"
        )

        nav_msg = Bool()
        nav_msg.data = True
        self.nav_active_pub.publish(nav_msg)


def main():
    rclpy.init()
    node = WaypointManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
