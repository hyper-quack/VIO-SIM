import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import numpy as np
import heapq
import math
import cv2

class AStarPlanner(Node):
    OBSTACLE_THRESHOLD    = 70
    INFLATION_RADIUS      = 6       # cells (XY plane)
    INFLATION_RADIUS_Z    = 3       # cells (Z axis)
    PATH_CLEARANCE_RADIUS = 6
    DIAGONAL_COST         = 1.414
    CARDINAL_COST         = 1.0
    TRIAGONAL_COST        = 1.732   # sqrt(3) for 3D diagonal
    REPLAN_INTERVAL       = 0.35
    PATH_VALIDATION_INTERVAL = 0.50
    BLOCK_ZONE_RADIUS     = 6
    PATH_CHECK_STEP       = 0.10
    DOWNSAMPLE_STEP_CELLS = 4
    SAME_GOAL_EPS         = 0.05

    # 3D grid parameters
    CORRIDOR_LENGTH = 20.0
    CORRIDOR_WIDTH  = 6.0
    CORRIDOR_HEIGHT = 3.0
    RESOLUTION      = 0.10
    GRID_W = int(20.0 / 0.10)   # 200
    GRID_H = int(6.0  / 0.10)   # 60
    GRID_Z = int(3.0  / 0.10)   # 30
    ORIGIN_X = 0.0
    ORIGIN_Y = 0.0
    ORIGIN_Z = 0.0
    def __init__(self):
        super().__init__('a_star_planner')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 3D obstacle map
        self.obstacle_3d = np.zeros(
            (self.GRID_Z, self.GRID_H, self.GRID_W), dtype=np.uint8
        )
        self.inflated_3d = np.zeros(
            (self.GRID_Z, self.GRID_H, self.GRID_W), dtype=np.uint8
        )

        # 2D costmap (from obstacle_detector projection)
        self.costmap_2d = None

        # State
        self.current_pose     = None
        self.goal             = None
        self.emergency        = False
        self.blocked_cells    = set()
        self.last_path        = None
        self.replan_requested = False
        self.costmap_dirty    = False
        self.last_plan_time   = 0.0
        self.last_validation_time = 0.0

        self.grid_width  = self.GRID_W
        self.grid_height = self.GRID_H
        self.resolution  = self.RESOLUTION
        self.origin_x    = self.ORIGIN_X
        self.origin_y    = self.ORIGIN_Y

        # Subscribers
        self.create_subscription(
            OccupancyGrid, '/costmap',
            self.costmap_callback, 10
        )
        self.create_subscription(
            PoseStamped, '/current_pose',
            self.pose_callback, qos
        )
        self.create_subscription(
            PoseStamped, '/goal_pose',
            self.goal_callback, 10
        )
        self.create_subscription(
            Bool, '/emergency_stop',
            self.emergency_callback, 10
        )
        self.create_subscription(
            PoseStamped, '/block_zone',
            self.block_zone_callback, 10
        )

        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.create_timer(0.1, self.periodic_check)

        self.get_logger().info(
            f'3D A* started ✓  '
            f'Grid={self.GRID_W}x{self.GRID_H}x{self.GRID_Z}  '
            f'inflation_xy={self.INFLATION_RADIUS} inflation_z={self.INFLATION_RADIUS_Z}'
        )

    # ==================================================================
    # Callbacks
    # ==================================================================
    def costmap_callback(self, msg):
        """Receive 2D projected costmap and expand to 3D."""
        self.costmap_2d = np.array(
            msg.data, dtype=np.int8
        ).reshape(msg.info.height, msg.info.width)

        self.grid_width  = msg.info.width
        self.grid_height = msg.info.height
        self.resolution  = msg.info.resolution
        self.origin_x    = msg.info.origin.position.x
        self.origin_y    = msg.info.origin.position.y

        self._build_3d_obstacle_map()
        self.costmap_dirty = True

    def pose_callback(self, msg):
        self.current_pose = msg

    def goal_callback(self, msg):
        if self.goal is not None:
            dx = msg.pose.position.x - self.goal.pose.position.x
            dy = msg.pose.position.y - self.goal.pose.position.y
            dz = msg.pose.position.z - self.goal.pose.position.z
            if math.sqrt(dx*dx + dy*dy + dz*dz) < self.SAME_GOAL_EPS:
                return
        self.goal = msg
        self.last_path = None
        self.replan_requested = True
        self.get_logger().info(
            f'New goal: ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f}, '
            f'{msg.pose.position.z:.2f})'
        )

    def emergency_callback(self, msg):
        self.emergency = msg.data

    def block_zone_callback(self, msg):
        gx, gy = self._world_to_grid_2d(
            msg.pose.position.x, msg.pose.position.y
        )
        if gx is None:
            return
        for dx in range(-self.BLOCK_ZONE_RADIUS, self.BLOCK_ZONE_RADIUS+1):
            for dy in range(-self.BLOCK_ZONE_RADIUS, self.BLOCK_ZONE_RADIUS+1):
                nx, ny = gx+dx, gy+dy
                if 0<=nx<self.GRID_W and 0<=ny<self.GRID_H:
                    self.blocked_cells.add((nx, ny))
        self._build_3d_obstacle_map()
        self.replan_requested = True
        self.costmap_dirty = True

    # ==================================================================
    # 3D obstacle map construction
    # ==================================================================
    def _build_3d_obstacle_map(self):
        if self.costmap_2d is None:
            return

        # Apply blocked cells to 2D map
        costmap = self.costmap_2d.copy()
        for gx, gy in self.blocked_cells:
            if 0<=gx<self.GRID_W and 0<=gy<self.GRID_H:
                costmap[gy, gx] = 100

        # Binary obstacle mask (2D)
        obstacle_2d = (costmap >= self.OBSTACLE_THRESHOLD).astype(np.uint8)

        # Inflate 2D with cv2
        r = self.INFLATION_RADIUS
        kernel = np.ones((2*r+1, 2*r+1), dtype=np.uint8)
        inflated_2d = cv2.dilate(obstacle_2d, kernel, iterations=1)
        inflated_2d[:r, :] = 1
        inflated_2d[-r:, :] = 1
        inflated_2d[:, :r] = 1
        inflated_2d[:, -r:] = 1

        # Expand to 3D — same obstacle footprint at all Z levels
        # Then apply Z inflation
        self.obstacle_3d = np.zeros(
            (self.GRID_Z, self.GRID_H, self.GRID_W), dtype=np.uint8
        )
        for z in range(self.GRID_Z):
            self.obstacle_3d[z] = inflated_2d

        # Z inflation — if a voxel is obstacle, mark neighbors in Z
        rz = self.INFLATION_RADIUS_Z
        inflated_3d = self.obstacle_3d.copy()
        for dz in range(1, rz+1):
            inflated_3d[dz:,  :, :] |= self.obstacle_3d[:-dz, :, :]
            inflated_3d[:-dz, :, :] |= self.obstacle_3d[dz:,  :, :]

        # Floor and ceiling margins
        inflated_3d[:rz, :, :] = 1
        inflated_3d[-rz:, :, :] = 1

        self.inflated_3d = inflated_3d

    # ==================================================================
    # Coordinate helpers
    # ==================================================================
    def _world_to_grid_3d(self, wx, wy, wz):
        gx = int((wx - self.ORIGIN_X) / self.RESOLUTION)
        gy = int((wy - self.ORIGIN_Y) / self.RESOLUTION)
        gz = int((wz - self.ORIGIN_Z) / self.RESOLUTION)
        if (0<=gx<self.GRID_W and
            0<=gy<self.GRID_H and
            0<=gz<self.GRID_Z):
            return gx, gy, gz
        return None, None, None

    def _world_to_grid_2d(self, wx, wy):
        gx = int((wx - self.ORIGIN_X) / self.RESOLUTION)
        gy = int((wy - self.ORIGIN_Y) / self.RESOLUTION)
        if 0<=gx<self.GRID_W and 0<=gy<self.GRID_H:
            return gx, gy
        return None, None

    def _grid_to_world_3d(self, gx, gy, gz):
        wx = gx*self.RESOLUTION + self.ORIGIN_X + self.RESOLUTION/2.0
        wy = gy*self.RESOLUTION + self.ORIGIN_Y + self.RESOLUTION/2.0
        wz = gz*self.RESOLUTION + self.ORIGIN_Z + self.RESOLUTION/2.0
        return wx, wy, wz

    # ==================================================================
    # A* helpers
    # ==================================================================
    def is_free_3d(self, gx, gy, gz):
        if not (0<=gx<self.GRID_W and 0<=gy<self.GRID_H and 0<=gz<self.GRID_Z):
            return False
        return self.inflated_3d[gz, gy, gx] == 0

    @staticmethod
    def heuristic_3d(a, b):
        dx = abs(a[0]-b[0])
        dy = abs(a[1]-b[1])
        dz = abs(a[2]-b[2])
        # Octile distance in 3D
        dmax = max(dx, dy, dz)
        dmid = sorted([dx, dy, dz])[1]
        dmin = min(dx, dy, dz)
        return (
            dmax * AStarPlanner.CARDINAL_COST +
            dmid * (AStarPlanner.DIAGONAL_COST - AStarPlanner.CARDINAL_COST) +
            dmin * (AStarPlanner.TRIAGONAL_COST - AStarPlanner.DIAGONAL_COST)
        )

    # ==================================================================
    # 3D A*
    # ==================================================================
    def a_star_3d(self, start, goal):
        if start == goal:
            return [start]

        if not self.is_free_3d(*start):
            start = self._find_nearest_free_3d(start)
            if start is None:
                self.get_logger().error('No free cell near start')
                return None

        if not self.is_free_3d(*goal):
            self.get_logger().warn('Goal blocked')
            return None

        # 26-directional moves
        moves = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx==0 and dy==0 and dz==0:
                        continue
                    nonzero = sum([dx!=0, dy!=0, dz!=0])
                    if nonzero == 1:
                        cost = self.CARDINAL_COST
                    elif nonzero == 2:
                        cost = self.DIAGONAL_COST
                    else:
                        cost = self.TRIAGONAL_COST
                    moves.append((dx, dy, dz, cost))

        open_set  = []
        came_from = {}
        g_score   = {start: 0.0}
        closed    = set()

        heapq.heappush(open_set, (0.0, start))

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                return self._reconstruct(came_from, current)

            for dx, dy, dz, move_cost in moves:
                nx = current[0]+dx
                ny = current[1]+dy
                nz = current[2]+dz
                neighbor = (nx, ny, nz)

                if neighbor in closed:
                    continue
                if not self.is_free_3d(nx, ny, nz):
                    continue

                tentative_g = g_score[current] + move_cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor]   = tentative_g
                    f = tentative_g + self.heuristic_3d(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return None

    def _reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _find_nearest_free_3d(self, cell, max_radius=20):
        if self.is_free_3d(*cell):
            return cell
        for r in range(1, max_radius+1):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    for dz in range(-r, r+1):
                        if max(abs(dx),abs(dy),abs(dz)) != r:
                            continue
                        nx = cell[0]+dx
                        ny = cell[1]+dy
                        nz = cell[2]+dz
                        if self.is_free_3d(nx, ny, nz):
                            return (nx, ny, nz)
        return None

    # ==================================================================
    # Path downsampling
    # ==================================================================
    @staticmethod
    def _sign(x):
        return (x>0)-(x<0)

    def downsample_path(self, path, step_cells=4):
        if path is None or len(path) <= 2:
            return path
        result = [path[0]]
        for i in range(1, len(path)-1):
            prev, curr, nxt = path[i-1], path[i], path[i+1]
            d1 = tuple(self._sign(curr[j]-prev[j]) for j in range(3))
            d2 = tuple(self._sign(nxt[j]-curr[j])  for j in range(3))
            if d1 != d2 or (i % step_cells) == 0:
                result.append(curr)
        result.append(path[-1])
        return result

    # ==================================================================
    # Path validity
    # ==================================================================
    def is_path_still_valid(self):
        if self.last_path is None:
            return False
        for pose in self.last_path.poses:
            wx = pose.pose.position.x
            wy = pose.pose.position.y
            wz = pose.pose.position.z
            gx, gy, gz = self._world_to_grid_3d(wx, wy, wz)
            if gx is None:
                return False
            if not self.is_free_3d(gx, gy, gz):
                return False
        return True

    # ==================================================================
    # Periodic check
    # ==================================================================
    def periodic_check(self):
        if self.emergency:
            return
        if self.current_pose is None or self.goal is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if self.last_path is None:
            if (now - self.last_plan_time) > self.REPLAN_INTERVAL:
                self.plan()
                self.last_plan_time = now
            return

        if self.replan_requested:
            if (now - self.last_plan_time) > self.REPLAN_INTERVAL:
                self.plan()
                self.last_plan_time = now
            return

        if self.costmap_dirty and (now - self.last_plan_time) > self.REPLAN_INTERVAL:
            self.costmap_dirty = False
            self.last_validation_time = now
            if not self.is_path_still_valid():
                self.get_logger().warn('Path blocked — replanning')
                self.plan()
                self.last_plan_time = now
            return

        if (now - self.last_validation_time) > self.PATH_VALIDATION_INTERVAL:
            self.last_validation_time = now
            if not self.is_path_still_valid():
                self.get_logger().warn('Backup check — replanning')
                self.plan()
                self.last_plan_time = now

    # ==================================================================
    # Plan
    # ==================================================================
    def plan(self):
        if self.current_pose is None or self.goal is None:
            return False

        # Current pose in 3D grid
        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        cz = self.current_pose.pose.position.z
        sx, sy, sz = self._world_to_grid_3d(cx, cy, cz)

        # Goal in 3D grid
        gx_w = self.goal.pose.position.x
        gy_w = self.goal.pose.position.y
        gz_w = self.goal.pose.position.z
        gx, gy, gz = self._world_to_grid_3d(gx_w, gy_w, gz_w)

        if sx is None or gx is None:
            self.get_logger().error('Position outside 3D grid')
            return False

        start = (sx, sy, sz)
        goal  = (gx, gy, gz)

        self.get_logger().info(f'3D A*: {start} -> {goal}')

        path_cells = self.a_star_3d(start, goal)
        if path_cells is None:
            self.get_logger().warn('No 3D path found')
            return False

        dense = self.downsample_path(path_cells, self.DOWNSAMPLE_STEP_CELLS)
        self.get_logger().info(
            f'Path found: {len(path_cells)} cells -> {len(dense)} waypoints'
        )

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'

        for cell in dense:
            wx, wy, wz = self._grid_to_world_3d(*cell)
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = wz
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.last_path = path_msg
        self.replan_requested = False
        return True

def main():
    rclpy.init()
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()