#!/usr/bin/env python3
"""
rrt_local_planner.py — RRT* local planner for dynamic obstacle avoidance

Role in architecture:
  Global planner (A*) → /global_path (coarse waypoints, empty-space)
  THIS NODE (RRT*)    → /planned_path (smooth, obstacle-aware, 3D)
  Path follower       → /desired_velocity

What it does:
  1. Takes next global waypoint as local goal (within LOCAL_GOAL_DIST)
  2. Queries octomap_manager confirmed voxels for obstacles
  3. Samples paths in continuous 3D space using RRT*
  4. Smooths the path (shortcutting)
  5. Publishes smooth obstacle-free trajectory at REPLAN_RATE Hz
  6. If stuck → publishes /local_planner_stuck → global replans

Reads:
  /global_path        ← from global planner (sparse waypoints)
  /current_pose       ← drone position
  /voxel_map          ← confirmed obstacles from octomap_manager (PointCloud2)
  /navigation_active  ← only plan when active

Publishes:
  /planned_path       ← smooth local trajectory for path_follower
  /local_planner_stuck ← signals global planner to replan
  /rrt_debug          ← RRT tree visualization for RViz (PointCloud2)
"""

import math
import random
import numpy as np
import struct
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool
from scipy.spatial import KDTree


# ═══════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════

# RRT* core
MAX_ITERATIONS      = 300       # samples per planning cycle
STEP_SIZE           = 0.4       # metres per tree extension
REWIRE_RADIUS       = 1.2      # metres — rewiring neighborhood
GOAL_BIAS           = 0.15     # probability of sampling the goal directly
GOAL_TOLERANCE      = 0.4      # metres — close enough to local goal

# Safety
MIN_OBSTACLE_DIST   = 0.5      # metres — minimum clearance from obstacles
EDGE_CHECK_STEP     = 0.1      # metres — collision check resolution along edges

# Local planning window
LOCAL_GOAL_DIST     = 4.0      # metres — pick global waypoint within this range
LOCAL_SAMPLE_RADIUS = 5.0      # metres — sample space around drone

# Altitude
Z_MIN               = 0.5      # metres — minimum planning altitude
Z_MAX               = 3.0      # metres — maximum planning altitude
Z_SAMPLE_RANGE      = 0.5      # metres — sample ±this around target altitude

# Replanning
REPLAN_RATE         = 5.0      # Hz
STUCK_TIMEOUT       = 5.0      # seconds — report stuck after this
STUCK_PROGRESS_DIST = 0.3      # metres — must progress this much to not be stuck

# Smoothing
SMOOTH_ITERATIONS   = 50       # shortcutting attempts
SMOOTH_WAYPOINT_SPACING = 0.3  # metres between output waypoints


class RRTNode:
    """Single node in the RRT* tree."""
    __slots__ = ['pos', 'parent', 'cost', 'children']

    def __init__(self, pos, parent=None, cost=0.0):
        self.pos = np.array(pos, dtype=np.float64)
        self.parent = parent
        self.cost = cost
        self.children = []


class RRTLocalPlanner(Node):

    def __init__(self):
        super().__init__('rrt_local_planner')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # ── State ────────────────────────────────────────────────
        self.current_pose     = None
        self.global_path      = None
        self.nav_active       = False
        self.emergency        = False

        # ── Obstacle data ────────────────────────────────────────
        self.obstacle_points  = None    # Nx3 numpy array
        self.obstacle_tree    = None    # KDTree for fast distance queries

        # ── Stuck detection ──────────────────────────────────────
        self.last_progress_pos  = None
        self.last_progress_time = 0.0
        self.stuck_reported     = False

        # ── Subscribers ──────────────────────────────────────────
        self.create_subscription(
            Path, '/global_path',
            self.global_path_cb, 10)

        self.create_subscription(
            PoseStamped, '/current_pose',
            self.pose_cb, qos)

        self.create_subscription(
            PointCloud2, '/voxel_map',
            self.voxel_cb, 10)

        self.create_subscription(
            Bool, '/navigation_active',
            self.nav_active_cb, 10)

        self.create_subscription(
            Bool, '/emergency_stop',
            self.emergency_cb, 10)

        # ── Publishers ───────────────────────────────────────────
        self.path_pub  = self.create_publisher(Path, '/planned_path', 10)
        self.stuck_pub = self.create_publisher(Bool, '/local_planner_stuck', 10)
        self.debug_pub = self.create_publisher(PointCloud2, '/rrt_debug', 10)

        # ── Timer ────────────────────────────────────────────────
        self.create_timer(1.0 / REPLAN_RATE, self.replan)

        self.get_logger().info(
            f'RRT* Local Planner started ✓  '
            f'iterations={MAX_ITERATIONS}  step={STEP_SIZE}m  '
            f'clearance={MIN_OBSTACLE_DIST}m')

    # ═══════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════

    def pose_cb(self, msg):
        self.current_pose = msg

    def global_path_cb(self, msg):
        if msg.poses:
            self.global_path = msg
            self.stuck_reported = False
            self.get_logger().info(
                f'Received global path: {len(msg.poses)} waypoints')

    def nav_active_cb(self, msg):
        self.nav_active = msg.data

    def emergency_cb(self, msg):
        self.emergency = msg.data

    def voxel_cb(self, msg):
        """Parse confirmed obstacle voxels from octomap_manager."""
        points = self._parse_pointcloud2(msg)
        if points is not None and len(points) > 0:
            self.obstacle_points = points
            self.obstacle_tree = KDTree(points)
        else:
            self.obstacle_points = None
            self.obstacle_tree = None

    # ═══════════════════════════════════════════════════════════════
    # Obstacle queries
    # ═══════════════════════════════════════════════════════════════

    def is_point_safe(self, point):
        """Check if a 3D point is far enough from all obstacles."""
        if self.obstacle_tree is None:
            return True     # no obstacles known → assume free

        dist, _ = self.obstacle_tree.query(point)
        return dist > MIN_OBSTACLE_DIST

    def is_edge_safe(self, p1, p2):
        """Check all points along edge for collision."""
        diff = p2 - p1
        length = np.linalg.norm(diff)
        if length < 0.001:
            return True

        n_checks = max(2, int(length / EDGE_CHECK_STEP))
        for i in range(n_checks + 1):
            t = i / n_checks
            point = p1 + diff * t
            if not self.is_point_safe(point):
                return False
        return True

    def get_obstacle_distance(self, point):
        """Get distance to nearest obstacle."""
        if self.obstacle_tree is None:
            return float('inf')
        dist, _ = self.obstacle_tree.query(point)
        return dist

    # ═══════════════════════════════════════════════════════════════
    # Local goal selection
    # ═══════════════════════════════════════════════════════════════

    def get_local_goal(self):
        """Pick the farthest reachable global waypoint within LOCAL_GOAL_DIST."""
        if self.global_path is None or self.current_pose is None:
            return None

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        cz = self.current_pose.pose.position.z

        best_wp = None
        best_dist = float('inf')

        for pose in self.global_path.poses:
            wx = pose.pose.position.x
            wy = pose.pose.position.y
            wz = pose.pose.position.z
            dist = math.sqrt((wx - cx)**2 + (wy - cy)**2 + (wz - cz)**2)

            # Pick farthest waypoint still within LOCAL_GOAL_DIST
            if dist <= LOCAL_GOAL_DIST:
                best_wp = np.array([wx, wy, wz])
                # Don't break — keep looking for farther ones within range

        # If no waypoint within range, pick the closest one
        if best_wp is None:
            closest_dist = float('inf')
            for pose in self.global_path.poses:
                wx = pose.pose.position.x
                wy = pose.pose.position.y
                wz = pose.pose.position.z
                dist = math.sqrt((wx - cx)**2 + (wy - cy)**2 + (wz - cz)**2)
                if dist < closest_dist:
                    closest_dist = dist
                    best_wp = np.array([wx, wy, wz])

        return best_wp

    # ═══════════════════════════════════════════════════════════════
    # RRT* core
    # ═══════════════════════════════════════════════════════════════

    def sample_point(self, start, goal):
        """Sample a random point in the local planning space."""
        if random.random() < GOAL_BIAS:
            return goal.copy()

        # Sample in sphere around drone
        angle_xy = random.uniform(0, 2 * math.pi)
        radius   = random.uniform(0, LOCAL_SAMPLE_RADIUS)
        x = start[0] + radius * math.cos(angle_xy)
        y = start[1] + radius * math.sin(angle_xy)

        # Z: sample around target altitude
        target_z = goal[2]
        z = target_z + random.uniform(-Z_SAMPLE_RANGE, Z_SAMPLE_RANGE)
        z = max(Z_MIN, min(Z_MAX, z))

        return np.array([x, y, z])

    def nearest_node(self, tree, point):
        """Find nearest node in tree to point."""
        best_node = None
        best_dist = float('inf')
        for node in tree:
            dist = np.linalg.norm(node.pos - point)
            if dist < best_dist:
                best_dist = dist
                best_node = node
        return best_node

    def steer(self, from_pos, to_pos):
        """Extend from from_pos toward to_pos by STEP_SIZE."""
        diff = to_pos - from_pos
        dist = np.linalg.norm(diff)
        if dist <= STEP_SIZE:
            return to_pos.copy()
        return from_pos + (diff / dist) * STEP_SIZE

    def near_nodes(self, tree, point):
        """Find all nodes within REWIRE_RADIUS."""
        return [n for n in tree
                if np.linalg.norm(n.pos - point) <= REWIRE_RADIUS]

    def rrt_star(self, start_pos, goal_pos):
        """Run RRT* and return path if found."""
        root = RRTNode(start_pos)
        tree = [root]

        best_goal_node = None
        best_goal_cost = float('inf')

        for iteration in range(MAX_ITERATIONS):
            # 1. Sample
            rand_point = self.sample_point(start_pos, goal_pos)

            # 2. Find nearest
            nearest = self.nearest_node(tree, rand_point)

            # 3. Steer
            new_pos = self.steer(nearest.pos, rand_point)

            # 4. Collision check
            if not self.is_point_safe(new_pos):
                continue
            if not self.is_edge_safe(nearest.pos, new_pos):
                continue

            # 5. Find best parent (RRT* rewiring)
            new_cost = nearest.cost + np.linalg.norm(new_pos - nearest.pos)
            best_parent = nearest

            neighbors = self.near_nodes(tree, new_pos)
            for neighbor in neighbors:
                candidate_cost = neighbor.cost + np.linalg.norm(new_pos - neighbor.pos)
                if candidate_cost < new_cost:
                    if self.is_edge_safe(neighbor.pos, new_pos):
                        best_parent = neighbor
                        new_cost = candidate_cost

            # 6. Add node
            new_node = RRTNode(new_pos, parent=best_parent, cost=new_cost)
            best_parent.children.append(new_node)
            tree.append(new_node)

            # 7. Rewire neighbors
            for neighbor in neighbors:
                if neighbor == best_parent:
                    continue
                rewire_cost = new_cost + np.linalg.norm(neighbor.pos - new_pos)
                if rewire_cost < neighbor.cost:
                    if self.is_edge_safe(new_pos, neighbor.pos):
                        # Remove from old parent
                        if neighbor.parent is not None:
                            try:
                                neighbor.parent.children.remove(neighbor)
                            except ValueError:
                                pass
                        neighbor.parent = new_node
                        neighbor.cost = rewire_cost
                        new_node.children.append(neighbor)

            # 8. Check if goal reached
            dist_to_goal = np.linalg.norm(new_pos - goal_pos)
            if dist_to_goal < GOAL_TOLERANCE and new_cost < best_goal_cost:
                best_goal_node = new_node
                best_goal_cost = new_cost

        # Extract path
        if best_goal_node is not None:
            path = self._extract_path(best_goal_node)
            return path, tree

        return None, tree

    def _extract_path(self, goal_node):
        """Trace back from goal node to root."""
        path = []
        node = goal_node
        while node is not None:
            path.append(node.pos.copy())
            node = node.parent
        path.reverse()
        return path

    # ═══════════════════════════════════════════════════════════════
    # Path smoothing
    # ═══════════════════════════════════════════════════════════════

    def smooth_path(self, path):
        """Shortcutting: try random shortcuts and keep if collision-free."""
        if len(path) <= 2:
            return path

        smoothed = [p.copy() for p in path]

        for _ in range(SMOOTH_ITERATIONS):
            if len(smoothed) <= 2:
                break

            # Pick two random indices
            i = random.randint(0, len(smoothed) - 2)
            j = random.randint(i + 1, len(smoothed) - 1)

            if j - i <= 1:
                continue

            # Check if direct connection is collision-free
            if self.is_edge_safe(smoothed[i], smoothed[j]):
                # Remove intermediate points
                smoothed = smoothed[:i+1] + smoothed[j:]

        return smoothed

    def resample_path(self, path):
        """Resample path at regular intervals."""
        if len(path) <= 1:
            return path

        resampled = [path[0].copy()]
        accum = 0.0

        for i in range(1, len(path)):
            diff = path[i] - path[i-1]
            segment_len = np.linalg.norm(diff)

            if segment_len < 0.001:
                continue

            direction = diff / segment_len
            remaining = segment_len

            while remaining > 0:
                step = min(remaining, SMOOTH_WAYPOINT_SPACING - accum)
                accum += step
                remaining -= step

                if accum >= SMOOTH_WAYPOINT_SPACING:
                    point = path[i] - direction * remaining
                    resampled.append(point.copy())
                    accum = 0.0

        # Always include final point
        if np.linalg.norm(resampled[-1] - path[-1]) > 0.05:
            resampled.append(path[-1].copy())

        return resampled

    # ═══════════════════════════════════════════════════════════════
    # Stuck detection
    # ═══════════════════════════════════════════════════════════════

    def check_stuck(self):
        """Detect if drone is not making progress."""
        if self.current_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        current_pos = np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z])

        if self.last_progress_pos is None:
            self.last_progress_pos = current_pos.copy()
            self.last_progress_time = now
            return

        progress = np.linalg.norm(current_pos - self.last_progress_pos)

        if progress > STUCK_PROGRESS_DIST:
            self.last_progress_pos = current_pos.copy()
            self.last_progress_time = now
            self.stuck_reported = False
            return

        # Check timeout
        if (now - self.last_progress_time) > STUCK_TIMEOUT:
            if not self.stuck_reported:
                msg = Bool()
                msg.data = True
                self.stuck_pub.publish(msg)
                self.stuck_reported = True
                self.get_logger().warn(
                    f'STUCK for {STUCK_TIMEOUT}s — signaling global planner')

    # ═══════════════════════════════════════════════════════════════
    # Main replan loop
    # ═══════════════════════════════════════════════════════════════

    def replan(self):
        """Called at REPLAN_RATE Hz."""
        if self.emergency or not self.nav_active:
            return
        if self.current_pose is None or self.global_path is None:
            return

        # Check stuck
        self.check_stuck()

        # Get local goal
        local_goal = self.get_local_goal()
        if local_goal is None:
            return

        # Start position
        start = np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z])

        # Check if already at goal
        if np.linalg.norm(start - local_goal) < GOAL_TOLERANCE:
            return

        # Run RRT*
        path, tree = self.rrt_star(start, local_goal)

        if path is None:
            # RRT* failed — try straight line as fallback
            if self.is_edge_safe(start, local_goal):
                path = [start, local_goal]
                self.get_logger().debug('RRT* failed — using direct line')
            else:
                self.get_logger().debug(
                    f'RRT* failed ({len(tree)} nodes explored)')
                return

        # Smooth
        smoothed = self.smooth_path(path)
        resampled = self.resample_path(smoothed)

        # Publish path
        self._publish_path(resampled)

        # Publish debug tree
        self._publish_debug_tree(tree)

        self.get_logger().debug(
            f'RRT* plan: {len(tree)} nodes → '
            f'{len(path)} raw → {len(resampled)} smooth waypoints')

    # ═══════════════════════════════════════════════════════════════
    # Publishers
    # ═══════════════════════════════════════════════════════════════

    def _publish_path(self, waypoints):
        """Publish local trajectory as /planned_path."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'

        for wp in waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = float(wp[2])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def _publish_debug_tree(self, tree):
        """Publish RRT tree nodes as PointCloud2 for RViz."""
        if len(tree) == 0:
            return

        pts = np.zeros((len(tree), 3), dtype=np.float32)
        for i, node in enumerate(tree):
            pts[i] = node.pos.astype(np.float32)

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.height = 1
        msg.width  = len(pts)
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 12
        msg.row_step     = 12 * len(pts)
        msg.is_dense     = True
        msg.data         = pts.tobytes()
        self.debug_pub.publish(msg)

    # ═══════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════

    def _parse_pointcloud2(self, msg):
        """Parse PointCloud2 to Nx3 numpy array."""
        field_offsets = {f.name: f.offset for f in msg.fields}
        if 'x' not in field_offsets:
            return None

        x_off = field_offsets['x']
        y_off = field_offsets['y']
        z_off = field_offsets['z']
        step  = msg.point_step
        data  = bytes(msg.data)
        n     = msg.width * msg.height

        if n == 0:
            return None

        points = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            base = i * step
            points[i, 0] = struct.unpack_from('f', data, base + x_off)[0]
            points[i, 1] = struct.unpack_from('f', data, base + y_off)[0]
            points[i, 2] = struct.unpack_from('f', data, base + z_off)[0]

        valid = np.all(np.isfinite(points), axis=1)
        points = points[valid]

        return points if len(points) > 0 else None


def main():
    rclpy.init()
    node = RRTLocalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()