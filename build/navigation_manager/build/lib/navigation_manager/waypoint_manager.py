import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import yaml
import os
from ament_index_python.packages import get_package_share_directory


class WaypointManager(Node):
    def __init__(self):
        super().__init__('waypoint_manager')

        # Charger waypoints
        waypoints_file = os.path.join(
            get_package_share_directory('navigation_manager'),
            'waypoints', 'corridor.yaml'
        )

        self.waypoints   = []
        self.current_idx = 0
        self.active      = False

        try:
            with open(waypoints_file, 'r') as f:
                data = yaml.safe_load(f)
                self.waypoints = data['waypoints']
            self.get_logger().info(
                f'Waypoints chargés : {len(self.waypoints)} points')
        except Exception as e:
            self.get_logger().error(f'Erreur chargement waypoints: {e}')

        # Publishers
        self.goal_pub = self.create_publisher(
            PoseStamped,'/goal_raw', 10)
        self.nav_active_pub = self.create_publisher(
            Bool, '/navigation_active', 10)

        # Subscribers
        self.reached_sub = self.create_subscription(
            Bool, '/goal_reached',
            self.goal_reached_callback, 10)
        self.start_sub = self.create_subscription(
            Bool, '/start_navigation',
            self.start_callback, 10)
        self.emergency_sub = self.create_subscription(
            Bool, '/emergency_stop',
            self.emergency_callback, 10)

        self.get_logger().info('Waypoint Manager démarré ✓')
        
    def start_callback(self, msg):
        if not msg.data or not self.waypoints:
            return
        if self.active:          # ← prevent reset on repeated /start_navigation
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

    def goal_reached_callback(self, msg):
        if not msg.data or not self.active:
            return

        self.current_idx += 1

        if self.current_idx >= len(self.waypoints):
            self.get_logger().info('Mission complète — tous les waypoints atteints ✓')
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