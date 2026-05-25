#!/usr/bin/env python3
"""
path_follower.py — 3D path follower

Only resets waypoint tracking when path destination actually changes.
Prevents oscillation from repeated identical path messages.
"""
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
            depth=10)

        # Waypoint acceptance radii
        self.waypoint_radius   = 0.22
        self.waypoint_radius_z = 0.15
        self.goal_radius       = 0.35
        self.goal_radius_z     = 0.20

        # Speeds
        self.max_speed    = 0.22
        self.cruise_speed = 0.18
        self.min_speed    = 0.05
        self.max_vz       = 0.15
        self.min_vz       = 0.03

        # Smoothing
        self.vel_alpha   = 0.25
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

        # Last path fingerprint — to detect real path changes
        self.last_path_end_x   = None
        self.last_path_end_y   = None
        self.last_path_len     = 0

        # Subscribers
        self.create_subscription(Path,        '/planned_path',      self.path_cb,      10)
        self.create_subscription(PoseStamped, '/current_pose',      self.pose_cb,      qos)
        self.create_subscription(Bool,        '/emergency_stop',    self.emergency_cb, 10)
        self.create_subscription(Bool,        '/navigation_active', self.active_cb,    10)

        # Publishers
        self.vel_pub     = self.create_publisher(TwistStamped, '/desired_velocity', 10)
        self.reached_pub = self.create_publisher(Bool,         '/goal_reached',     10)

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('PathFollower 3D started ✓')

    # ── Callbacks ─────────────────────────────────────────────────

    def path_cb(self, msg):
        if not msg.poses:
            return

        new_end_x = msg.poses[-1].pose.position.x
        new_end_y = msg.poses[-1].pose.position.y
        new_len   = len(msg.poses)

        # Check if path actually changed
        path_changed = (
            self.last_path_end_x is None or
            abs(new_end_x - self.last_path_end_x) > 0.15 or
            abs(new_end_y - self.last_path_end_y) > 0.15 or
            abs(new_len   - self.last_path_len)    > 2
        )

        if not path_changed:
            # Same path — just update poses without resetting index
            self.path   = msg.poses
            self.active = True
            return

        # New path — reset tracking
        self.path              = msg.poses
        self.active            = True
        self.goal_reached_sent = False
        self.last_path_end_x   = new_end_x
        self.last_path_end_y   = new_end_y
        self.last_path_len     = new_len

        # Find closest waypoint ahead of drone
        if self.current_pose is not None:
            sx = self.current_pose.pose.position.x
            sy = self.current_pose.pose.position.y
            best_idx, best_dist = 0, float('inf')
            for i, wp in enumerate(self.path):
                dx = wp.pose.position.x - sx
                dy = wp.pose.position.y - sy
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i
            self.current_idx = best_idx
            if self.current_idx < len(self.path)-1 and best_dist < self.waypoint_radius:
                self.current_idx += 1
        else:
            self.current_idx = 0

        self.get_logger().info(
            f'New path: {len(self.path)} waypoints  idx={self.current_idx}  '
            f'end=({new_end_x:.1f},{new_end_y:.1f})')

    def pose_cb(self, msg):
        self.current_pose = msg

    def emergency_cb(self, msg):
        self.emergency = msg.data
        if self.emergency:
            self.stop()

    def active_cb(self, msg):
        self.active = msg.data
        if not self.active:
            self.stop()

    # ── Helpers ───────────────────────────────────────────────────

    def stop(self):
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.filtered_vz = 0.0
        vel = TwistStamped()
        vel.header.stamp    = self.get_clock().now().to_msg()
        vel.header.frame_id = 'odom'
        vel.twist.linear.x  = 0.0
        vel.twist.linear.y  = 0.0
        vel.twist.linear.z  = 0.0
        self.vel_pub.publish(vel)

    def dist_xy(self, wp):
        if self.current_pose is None: return float('inf')
        dx = wp.pose.position.x - self.current_pose.pose.position.x
        dy = wp.pose.position.y - self.current_pose.pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    def dist_z(self, wp):
        if self.current_pose is None: return float('inf')
        return abs(wp.pose.position.z - self.current_pose.pose.position.z)

    def publish_velocity(self, vx, vy, vz):
        self.filtered_vx = (1.0-self.vel_alpha)*self.filtered_vx + self.vel_alpha*vx
        self.filtered_vy = (1.0-self.vel_alpha)*self.filtered_vy + self.vel_alpha*vy
        self.filtered_vz = (1.0-self.vel_alpha)*self.filtered_vz + self.vel_alpha*vz
        vel = TwistStamped()
        vel.header.stamp    = self.get_clock().now().to_msg()
        vel.header.frame_id = 'odom'
        vel.twist.linear.x  = float(self.filtered_vx)
        vel.twist.linear.y  = float(self.filtered_vy)
        vel.twist.linear.z  = float(self.filtered_vz)
        self.vel_pub.publish(vel)

    # ── Control loop ──────────────────────────────────────────────

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
        while (self.dist_xy(target)  < self.waypoint_radius and
               self.dist_z(target)   < self.waypoint_radius_z and
               self.current_idx < len(self.path)-1):
            self.current_idx += 1
            target = self.path[self.current_idx]
            self.get_logger().info(
                f'Waypoint {self.current_idx}/{len(self.path)} reached')

        # Direction to target
        dx = target.pose.position.x - self.current_pose.pose.position.x
        dy = target.pose.position.y - self.current_pose.pose.position.y
        dz = target.pose.position.z - self.current_pose.pose.position.z
        norm_xy = math.sqrt(dx*dx + dy*dy)

        dist_to_final = self.dist_xy(final_goal)

        if norm_xy < 0.01 and abs(dz) < 0.05:
            self.stop()
            return

        # XY speed — slow down near final goal
        speed = self.cruise_speed
        if dist_to_final < 1.0:
            speed = max(self.min_speed, self.cruise_speed * dist_to_final)
        speed = min(speed, self.max_speed)

        vx = (dx / norm_xy) * speed if norm_xy > 0.01 else 0.0
        vy = (dy / norm_xy) * speed if norm_xy > 0.01 else 0.0

        # Z speed
        if abs(dz) > 0.05:
            vz = math.copysign(max(self.min_vz, min(self.max_vz, abs(dz)*0.5)), dz)
        else:
            vz = 0.0

        self.publish_velocity(vx, vy, vz)

        self.get_logger().info(
            f'PF [{self.current_idx}/{len(self.path)}] '
            f'pos:({self.current_pose.pose.position.x:.2f},{self.current_pose.pose.position.y:.2f}) '
            f'tgt:({target.pose.position.x:.2f},{target.pose.position.y:.2f}) '
            f'vel:({self.filtered_vx:.2f},{self.filtered_vy:.2f})',
            throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = PathFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()