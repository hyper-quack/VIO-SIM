#!/usr/bin/env python3
"""
global_planner.py — Combined global route planner
 
Two modes:
  MODE 1 — No map file: creates empty map from waypoint boundaries, uses A*
  MODE 2 — Map file provided: loads it, uses A* on known obstacles
 
In both modes:
  - Does NOT read live /costmap (that's for local planner)
  - Plans on a COARSE grid (0.5m cells)
  - Outputs sparse waypoints for local planner
  - Replans only when goal changes or local planner reports stuck
 
Map file format:
  PNG or PGM image where:
    black (0)   = wall/obstacle
    white (255) = free space
  Accompanied by parameters: resolution, origin_x, origin_y
 
Reads:
  /current_pose       <- drone position
  /goal_pose          <- mission goal
  /navigation_active  <- only plan when active
  /local_planner_stuck <- local planner cannot reach waypoint (future)
 
Publishes:
  /planned_path       <- sparse waypoints
  /global_costmap     <- visualization of the global map for RViz
 
Parameters (ROS):
  map_file            <- path to PNG/PGM map image (empty = no map)
  map_resolution      <- metres per pixel of the map image
  map_origin_x        <- world X of map image bottom-left corner
  map_origin_y        <- world Y of map image bottom-left corner
"""
 
import math
import numpy as np
import heapq
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Bool
import cv2
 
 
# Parameters
WAYPOINT_SPACING    = 3.0
SAME_GOAL_EPS       = 0.3
REPLAN_INTERVAL     = 2.0
DEFAULT_ALTITUDE    = 2.0
GOAL_REACHED_DIST   = 0.5
PUBLISH_RATE        = 1.0
 
COARSE_RESOLUTION   = 0.5
INFLATION_RADIUS    = 2
DIAGONAL_COST       = 1.414
CARDINAL_COST       = 1.0
 
MAP_PADDING         = 3.0
 
 
class GlobalPlanner(Node):
 
    def __init__(self):
        super().__init__('a_star_planner')
 
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
 
        self.declare_parameter('map_file', '')
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_origin_x', 0.0)
        self.declare_parameter('map_origin_y', 0.0)
 
        self.current_pose   = None
        self.goal           = None
        self.nav_active     = False
        self.emergency      = False
        self.last_path      = None
        self.last_plan_time = 0.0
        self.local_stuck    = False
 
        self.map_loaded       = False
        self.global_map       = None
        self.map_resolution   = COARSE_RESOLUTION
        self.map_origin_x     = 0.0
        self.map_origin_y     = 0.0
        self.map_width        = 0
        self.map_height       = 0
        self.inflated_map     = None
 
        self.all_goals = []
 
        self._try_load_map()
 
        self.create_subscription(PoseStamped, '/current_pose', self.pose_cb, qos)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.create_subscription(Bool, '/navigation_active', self.nav_active_cb, 10)
        self.create_subscription(Bool, '/emergency_stop', self.emergency_cb, 10)
        self.create_subscription(Bool, '/local_planner_stuck', self.stuck_cb, 10)
 
        self.create_subscription(OccupancyGrid, '/costmap', self.costmap_cb, 10)

        self.path_pub    = self.create_publisher(Path, '/planned_path', 10)
        self.map_viz_pub = self.create_publisher(OccupancyGrid, '/global_costmap', 10)
 
        self.create_timer(1.0 / PUBLISH_RATE, self.periodic_check)
 
        mode = "A* on loaded map" if self.map_loaded else "auto-map from waypoints"
        self.get_logger().info(
            f'Global Planner started  mode={mode}  '
            f'coarse_res={COARSE_RESOLUTION}m  spacing={WAYPOINT_SPACING}m')
 
    # === Map loading =============================================
 
    def _try_load_map(self):
        map_file = self.get_parameter('map_file').get_parameter_value().string_value
        if not map_file or map_file == '':
            self.get_logger().info('No map file — will auto-generate from waypoints')
            return
 
        try:
            img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.get_logger().error(f'Cannot read map file: {map_file}')
                return
 
            img_resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
            self.map_origin_x = self.get_parameter('map_origin_x').get_parameter_value().double_value
            self.map_origin_y = self.get_parameter('map_origin_y').get_parameter_value().double_value
 
            scale = img_resolution / COARSE_RESOLUTION
            if abs(scale - 1.0) > 0.01:
                new_w = max(1, int(img.shape[1] * scale))
                new_h = max(1, int(img.shape[0] * scale))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
 
            obstacle_map = np.zeros_like(img, dtype=np.uint8)
            obstacle_map[img < 128] = 1
 
            self.global_map = obstacle_map
            self.map_width  = obstacle_map.shape[1]
            self.map_height = obstacle_map.shape[0]
            self.map_resolution = COARSE_RESOLUTION
 
            self._inflate_map()
            self.map_loaded = True
 
            self.get_logger().info(
                f'Map loaded: {map_file}  '
                f'size={self.map_width}x{self.map_height} '
                f'origin=({self.map_origin_x},{self.map_origin_y})')
 
        except Exception as e:
            self.get_logger().error(f'Map load error: {e}')
 
    def _create_auto_map(self):
        if self.current_pose is None:
            return False
 
        all_points = list(self.all_goals)
        all_points.append((
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y))
 
        if len(all_points) < 2:
            return False
 
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        min_x = min(xs) - MAP_PADDING
        max_x = max(xs) + MAP_PADDING
        min_y = min(ys) - MAP_PADDING
        max_y = max(ys) + MAP_PADDING
 
        self.map_origin_x = min_x
        self.map_origin_y = min_y
        self.map_resolution = COARSE_RESOLUTION
        self.map_width  = max(1, int((max_x - min_x) / COARSE_RESOLUTION))
        self.map_height = max(1, int((max_y - min_y) / COARSE_RESOLUTION))
 
        self.global_map = np.zeros(
            (self.map_height, self.map_width), dtype=np.uint8)
 
        self.global_map[0, :]  = 1
        self.global_map[-1, :] = 1
        self.global_map[:, 0]  = 1
        self.global_map[:, -1] = 1
 
        self._inflate_map()
        self.map_loaded = True
 
        self.get_logger().info(
            f'Auto-map created: {self.map_width}x{self.map_height} '
            f'({(max_x-min_x):.1f}m x {(max_y-min_y):.1f}m) '
            f'origin=({min_x:.1f},{min_y:.1f})')
        return True
 
    def _inflate_map(self):
        if self.global_map is None:
            return
        if INFLATION_RADIUS > 0:
            kernel_size = 2 * INFLATION_RADIUS + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            self.inflated_map = cv2.dilate(self.global_map, kernel, iterations=1)
        else:
            self.inflated_map = self.global_map.copy()
 
    # === Callbacks ===============================================
 
    def pose_cb(self, msg):
        self.current_pose = msg
 
    def goal_cb(self, msg):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.all_goals.append((gx, gy))
 
        if self.goal is not None:
            dx = gx - self.goal.pose.position.x
            dy = gy - self.goal.pose.position.y
            if math.sqrt(dx * dx + dy * dy) < SAME_GOAL_EPS:
                return
 
        self.goal = msg
        self.last_path = None
        self.get_logger().info(f'New goal: ({gx:.2f}, {gy:.2f}, {msg.pose.position.z:.2f})')
 
    def nav_active_cb(self, msg):
        self.nav_active = msg.data
 
    def emergency_cb(self, msg):
        self.emergency = msg.data
 
    def stuck_cb(self, msg):
        if msg.data:
            self.local_stuck = True
            self.get_logger().warn('Local planner stuck — will replan globally')
 
    def costmap_cb(self, msg):
        """Receive live costmap from octomap_manager."""
        h = msg.info.height
        w = msg.info.width
        data = np.array(msg.data, dtype=np.int8).reshape(h, w)
        # Mark occupied cells (value > 50)
        occupied = (data > 50).astype(np.uint8)
        self.live_costmap = {
            'grid':     occupied,
            'origin_x': msg.info.origin.position.x,
            'origin_y': msg.info.origin.position.y,
            'res':      msg.info.resolution,
            'width':    w,
            'height':   h,
        }
        self.costmap_updated = True  # throttled in periodic_check

    # === Coordinate transforms ===================================
 
    def _world_to_grid(self, wx, wy):
        gx = int((wx - self.map_origin_x) / self.map_resolution)
        gy = int((wy - self.map_origin_y) / self.map_resolution)
        if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
            return gx, gy
        return None, None
 
    def _grid_to_world(self, gx, gy):
        wx = gx * self.map_resolution + self.map_origin_x + self.map_resolution / 2.0
        wy = gy * self.map_resolution + self.map_origin_y + self.map_resolution / 2.0
        return wx, wy
 
    # === A* search ===============================================
 
    def is_free(self, gx, gy):
        if not (0 <= gx < self.map_width and 0 <= gy < self.map_height):
            return False
        if self.inflated_map[gy, gx] != 0:
            return False
        # Also check live costmap from octomap
        if self.live_costmap is not None:
            wx, wy = self._grid_to_world(gx, gy)
            cm = self.live_costmap
            # Check a small region around the world point (inflation)
            check_radius = 2  # cells in costmap resolution
            blocked = False
            for ddx in range(-check_radius, check_radius+1):
                for ddy in range(-check_radius, check_radius+1):
                    cx = int((wx - cm['origin_x']) / cm['res']) + ddx
                    cy = int((wy - cm['origin_y']) / cm['res']) + ddy
                    if 0 <= cx < cm['width'] and 0 <= cy < cm['height']:
                        if cm['grid'][cy, cx] != 0:
                            blocked = True
                            break
                if blocked:
                    break
            if blocked:
                return False
        return True
 
    @staticmethod
    def heuristic(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return CARDINAL_COST * max(dx, dy) + (DIAGONAL_COST - CARDINAL_COST) * min(dx, dy)
 
    def a_star(self, start, goal):
        if start == goal:
            return [start]
 
        if not self.is_free(*start):
            start = self._find_nearest_free(start)
            if start is None:
                self.get_logger().error('No free cell near start')
                return None
 
        if not self.is_free(*goal):
            goal = self._find_nearest_free(goal)
            if goal is None:
                self.get_logger().warn('No free cell near goal')
                return None
 
        moves = [
            (-1, 0, CARDINAL_COST),  (1, 0, CARDINAL_COST),
            (0, -1, CARDINAL_COST),  (0, 1, CARDINAL_COST),
            (-1, -1, DIAGONAL_COST), (-1, 1, DIAGONAL_COST),
            (1, -1, DIAGONAL_COST),  (1, 1, DIAGONAL_COST),
        ]
 
        open_set  = []
        came_from = {}
        g_score   = {start: 0.0}
        closed    = set()
        heapq.heappush(open_set, (0.0, start))
 
        max_iter = self.map_width * self.map_height
        iterations = 0
 
        while open_set:
            iterations += 1
            if iterations > max_iter:
                self.get_logger().warn('A* exceeded max iterations')
                return None
 
            _, current = heapq.heappop(open_set)
            if current in closed:
                continue
            closed.add(current)
 
            if current == goal:
                return self._reconstruct(came_from, current)
 
            for dx, dy, cost in moves:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in closed:
                    continue
                if not self.is_free(*neighbor):
                    continue
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
 
        return None
 
    def _reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
 
    def _find_nearest_free(self, cell, max_radius=20):
        if self.is_free(*cell):
            return cell
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    nx = cell[0] + dx
                    ny = cell[1] + dy
                    if self.is_free(nx, ny):
                        return (nx, ny)
        return None
 
    # === Path simplification =====================================
 
    def simplify_path(self, path_cells):
        if path_cells is None or len(path_cells) <= 2:
            return path_cells
 
        turning = [path_cells[0]]
        for i in range(1, len(path_cells) - 1):
            prev, curr, nxt = path_cells[i - 1], path_cells[i], path_cells[i + 1]
            d1 = (_sign(curr[0] - prev[0]), _sign(curr[1] - prev[1]))
            d2 = (_sign(nxt[0] - curr[0]),  _sign(nxt[1] - curr[1]))
            if d1 != d2:
                turning.append(curr)
        turning.append(path_cells[-1])
 
        if len(turning) <= 2:
            return turning
 
        spaced = [turning[0]]
        accum = 0.0
        for i in range(1, len(turning)):
            wx0, wy0 = self._grid_to_world(*turning[i - 1])
            wx1, wy1 = self._grid_to_world(*turning[i])
            dist = math.sqrt((wx1 - wx0) ** 2 + (wy1 - wy0) ** 2)
            accum += dist
            if accum >= WAYPOINT_SPACING:
                spaced.append(turning[i])
                accum = 0.0
 
        if spaced[-1] != turning[-1]:
            spaced.append(turning[-1])
        return spaced
 
    # === Straight line fallback ===================================
 
    def straight_line_waypoints(self, start, goal):
        sx, sy, sz = start
        gx, gy, gz = goal
        dx = gx - sx
        dy = gy - sy
        dz = gz - sz
        total_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
 
        if total_dist < GOAL_REACHED_DIST:
            return [(gx, gy, gz)]
 
        n = max(1, int(total_dist / WAYPOINT_SPACING))
        waypoints = []
        for i in range(1, n):
            t = i / n
            waypoints.append((sx + dx * t, sy + dy * t, sz + dz * t))
        waypoints.append((gx, gy, gz))
        return waypoints
 
    # === Main planning ============================================
 
    def plan(self):
        if self.current_pose is None or self.goal is None:
            return False
 
        sx = self.current_pose.pose.position.x
        sy = self.current_pose.pose.position.y
        sz = self.current_pose.pose.position.z
 
        gx = self.goal.pose.position.x
        gy = self.goal.pose.position.y
        gz = self.goal.pose.position.z
        if gz < 0.5 or gz > 4.0:
            gz = DEFAULT_ALTITUDE
 
        if not self.map_loaded:
            self._create_auto_map()
 
        waypoints_3d = None
        if self.map_loaded and self.inflated_map is not None:
            sg = self._world_to_grid(sx, sy)
            gg = self._world_to_grid(gx, gy)
 
            if sg[0] is not None and gg[0] is not None:
                path_cells = self.a_star(sg, gg)
                if path_cells is not None:
                    sparse = self.simplify_path(path_cells)
                    waypoints_3d = []
                    for idx, cell in enumerate(sparse):
                        wx, wy = self._grid_to_world(*cell)
                        t = idx / max(1, len(sparse) - 1) if len(sparse) > 1 else 1.0
                        wz = sz + (gz - sz) * t
                        waypoints_3d.append((wx, wy, wz))
                    self.get_logger().info(
                        f'A* global: {len(path_cells)} cells -> {len(waypoints_3d)} waypoints')
                else:
                    self.get_logger().warn('A* failed — falling back to straight line')
 
        if waypoints_3d is None:
            waypoints_3d = self.straight_line_waypoints((sx, sy, sz), (gx, gy, gz))
            self.get_logger().info(f'Straight line: {len(waypoints_3d)} waypoints')
 
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'
 
        for wx, wy, wz in waypoints_3d:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wx)
            pose.pose.position.y = float(wy)
            pose.pose.position.z = float(wz)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
 
        self.path_pub.publish(path_msg)
        self.last_path = path_msg
        self.local_stuck = False
 
        self._publish_global_map()
        return True
 
    # === Visualization ============================================
 
    def _publish_global_map(self):
        if self.inflated_map is None:
            return
 
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info.resolution = self.map_resolution
        msg.info.width  = self.map_width
        msg.info.height = self.map_height
        msg.info.origin = Pose()
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
 
        data = (self.inflated_map * 100).astype(np.int8)
        msg.data = data.flatten().tolist()
        self.map_viz_pub.publish(msg)
 
    # === Periodic check ===========================================
 
    def periodic_check(self):
        if self.emergency:
            return
        if not self.nav_active:
            return
        if self.current_pose is None or self.goal is None:
            return
 
        now = self.get_clock().now().nanoseconds * 1e-9
 
        if self.last_path is None:
            self.plan()
            self.last_plan_time = now
            return
 
        if self.local_stuck:
            self.get_logger().info('Replanning due to local planner stuck')
            self.plan()
            self.last_plan_time = now
            return
 
        if self.costmap_updated and (now - self.last_plan_time) > REPLAN_INTERVAL:
            self.costmap_updated = False
            self.plan()
            self.last_plan_time = now
            return

        if (now - self.last_plan_time) > REPLAN_INTERVAL:
            self.plan()
            self.last_plan_time = now
 
 
def _sign(x):
    return (x > 0) - (x < 0)
 
 
def main():
    rclpy.init()
    node = GlobalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()