#!/usr/bin/env python3
"""
a_star_planner.py — Single global planner (3D, voxel-driven)

A* is the ONLY planner and it plans in 3D so the drone can fly over/under
obstacles. The planning map is built directly from /voxel_map (the confirmed
obstacle voxels published by octomap_manager). /costmap is ignored here.

  - /voxel_map (PointCloud2) -> 3D occupancy set (with 3D inflation)
  - path_follower is purely a velocity publisher; it does no planning.

State machine:
  - self.goal         : latest goal requested (queue head, from /goal_pose)
  - self.active_goal  : goal the drone is currently flying to (locked at plan)

Flow:
  1. goal_cb stores self.goal silently. NEVER triggers a plan.
  2. voxel_map_cb rebuilds the 3D occupancy set. May clear last_path if the
     live path is blocked (rate-limited by a cooldown).
  3. periodic_check (1Hz) is the single decision-maker:
       a. active_goal reached (3D dist < GOAL_REACHED_DIST) -> clear, replan
       b. last_path is None -> plan()
       c. local_stuck -> plan()
  4. plan() runs 3D A*.
       - success -> publish /global_path
       - failure -> publish empty path (drone hovers), retry next tick.
"""

import math
import numpy as np
import heapq
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool

try:
    from scipy.interpolate import splprep, splev
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# === Parameters =================================================
VOXEL_SIZE          = 0.2
INFLATION_RADIUS    = 4       # voxels, inflated along the 6 axis directions
WAYPOINT_SPACING    = 1.0     # m between simplified waypoints (3D)
SAME_GOAL_EPS       = 0.3
DEFAULT_ALTITUDE    = 2.0
GOAL_REACHED_DIST   = 0.5     # m, measured in 3D
PUBLISH_RATE        = 1.0
BLOCK_COOLDOWN      = 3.0     # s — min interval between path-blocked replans

# Cubic B-spline smoothing (post-processing of the sparse A* waypoints)
SPLINE_S           = 10.0    # smoothing factor (Pollock λ): 0=interpolate, >0=smoother
SPLINE_SAMPLE_STEP = 0.3    # m between sampled points on the spline
KAPPA_MAX          = 2.0    # rad/m max curvature (IJISA Eq. 2); else retry/fallback
SPLINE_S_RETRY     = 2.0    # multiplier applied to s on the one curvature retry

# 26-connected move costs
FACE_COST   = 1.0
EDGE_COST   = 1.41421356
CORNER_COST = 1.73205081

MAX_ITER = 200000

# Search bounds in voxel indices (keeps A* tractable + inside the corridor).
# World corridor ~ x[-1,44]  y[-1,7]  z[0.2,4.0]  at 0.2m voxels.
IX_MIN, IX_MAX = -6, 222
IY_MIN, IY_MAX = -6,  36
IZ_MIN, IZ_MAX =  1,  20


class GlobalPlanner(Node):

    def __init__(self):
        super().__init__('a_star_planner')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # === State ===
        self.current_pose = None
        self.goal         = None   # latest queued goal
        self.active_goal  = None   # goal currently being flown to
        self.nav_active   = False
        self.emergency    = False
        self.last_path    = None
        self.local_stuck  = False
        self.last_blocked_time = 0.0   # cooldown gate for path-blocked replans

        # === 3D map (from /voxel_map) ===
        # Set of occupied (ix, iy, iz) voxels (already inflated). Empty = free.
        self.voxel_grid    = set()
        self.raw_obstacles = set()   # non-inflated obstacle voxels

        # Precompute 26-connected moves with costs
        self.moves = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nnz = abs(dx) + abs(dy) + abs(dz)
                    cost = (FACE_COST if nnz == 1 else
                            EDGE_COST if nnz == 2 else CORNER_COST)
                    self.moves.append((dx, dy, dz, cost))

        # === ROS I/O ===
        self.create_subscription(PoseStamped, '/current_pose',   self.pose_cb, qos)
        self.create_subscription(PoseStamped, '/goal_pose',      self.goal_cb, 10)
        self.create_subscription(Bool, '/navigation_active',     self.nav_active_cb, 10)
        self.create_subscription(Bool, '/emergency_stop',        self.emergency_cb, 10)
        self.create_subscription(Bool, '/local_planner_stuck',   self.stuck_cb, 10)
        self.create_subscription(PointCloud2, '/voxel_map',      self.voxel_map_cb, 10)

        self.path_pub    = self.create_publisher(Path, '/global_path', 10)
        self.infl_pub    = self.create_publisher(PointCloud2, '/global_inflated', 10)

        self.create_timer(1.0 / PUBLISH_RATE, self.periodic_check)

        self.get_logger().info(
            f'GlobalPlanner started  mode=3D voxel-driven  '
            f'voxel={VOXEL_SIZE}m  inflation={INFLATION_RADIUS}cells  '
            f'spacing={WAYPOINT_SPACING}m')

    # === Helpers ====================================================

    @staticmethod
    def _xy(msg):
        return (msg.pose.position.x, msg.pose.position.y)

    @staticmethod
    def _dist_3d(a_msg, b_msg):
        ax = a_msg.pose.position.x; ay = a_msg.pose.position.y; az = a_msg.pose.position.z
        bx = b_msg.pose.position.x; by = b_msg.pose.position.y; bz = b_msg.pose.position.z
        return math.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2)

    # === Coordinate transforms (3D voxel) ===========================

    @staticmethod
    def _world_to_voxel(wx, wy, wz):
        return (int(wx / VOXEL_SIZE), int(wy / VOXEL_SIZE), int(wz / VOXEL_SIZE))

    @staticmethod
    def _voxel_to_world(ix, iy, iz):
        return (ix * VOXEL_SIZE + VOXEL_SIZE / 2.0,
                iy * VOXEL_SIZE + VOXEL_SIZE / 2.0,
                iz * VOXEL_SIZE + VOXEL_SIZE / 2.0)

    # === Callbacks ==================================================

    def pose_cb(self, msg):
        self.current_pose = msg

    def goal_cb(self, msg):
        """Store the latest goal silently. NEVER trigger planning here."""
        gx, gy = self._xy(msg)

        if self.goal is not None:
            ox, oy = self._xy(self.goal)
            if abs(gx - ox) < SAME_GOAL_EPS and abs(gy - oy) < SAME_GOAL_EPS:
                return
            self.get_logger().info(
                f'[GOAL_CB] queued new goal ({gx:.2f},{gy:.2f}) '
                f'— was ({ox:.2f},{oy:.2f}) '
                f'active={"(%.2f,%.2f)" % self._xy(self.active_goal) if self.active_goal else "None"}')
        else:
            self.get_logger().info(f'[GOAL_CB] first goal queued ({gx:.2f},{gy:.2f})')

        self.goal = msg

    def nav_active_cb(self, msg):
        self.nav_active = msg.data

    def emergency_cb(self, msg):
        self.emergency = msg.data

    def stuck_cb(self, msg):
        if msg.data:
            self.local_stuck = True
            self.get_logger().warn('[STUCK] local planner stuck — will replan')

    def voxel_map_cb(self, msg):
        """Rebuild the inflated 3D occupancy set from /voxel_map."""
        pts = self._parse_cloud(msg)
        grid = set()
        if pts is not None and len(pts) > 0:
            ix = (pts[:, 0] / VOXEL_SIZE).astype(int)
            iy = (pts[:, 1] / VOXEL_SIZE).astype(int)
            iz = (pts[:, 2] / VOXEL_SIZE).astype(int)
            r = INFLATION_RADIUS
            # Store raw obstacles (no inflation) for spline collision check
            raw = set()
            for a, b, c in zip(ix.tolist(), iy.tolist(), iz.tolist()):
                raw.add((a, b, c))
            self.raw_obstacles = raw
            for a, b, c in zip(ix.tolist(), iy.tolist(), iz.tolist()):
                # Cubic inflation: fill the full (2r+1)^3 cube around each
                # obstacle voxel so 26-connected A* can't slip through a
                # diagonal gap.
                for ddx in range(-r, r + 1):
                    for ddy in range(-r, r + 1):
                        for ddz in range(-r, r + 1):
                            grid.add((a + ddx, b + ddy, c + ddz))
        self.voxel_grid = grid

        # If the live path now passes through occupancy, force a replan on the
        # next periodic tick. A cooldown prevents an infinite clear→replan loop.
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_blocked_time < BLOCK_COOLDOWN:
            return
        if (self.nav_active and not self.emergency and
                self.current_pose is not None and self.goal is not None and
                self._check_path_blocked()):
            self.get_logger().info('[VOXEL] path blocked — clearing last_path')
            self.last_blocked_time = now
            self.last_path = None

    @staticmethod
    def _parse_cloud(msg):
        offs = {f.name: f.offset for f in msg.fields}
        if not all(k in offs for k in ('x', 'y', 'z')):
            return None
        n = msg.width * msg.height
        if n == 0:
            return None
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)

        def col(off):
            return raw[:, off:off + 4].copy().view(np.float32).reshape(-1)

        pts = np.stack([col(offs['x']), col(offs['y']), col(offs['z'])], axis=1)
        finite = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite]
        return pts if len(pts) > 0 else None

    # === A* search (3D) =============================================

    @staticmethod
    def _in_bounds(c):
        return (IX_MIN <= c[0] <= IX_MAX and
                IY_MIN <= c[1] <= IY_MAX and
                IZ_MIN <= c[2] <= IZ_MAX)

    def is_free(self, gx, gy, gz):
        return (gx, gy, gz) not in self.voxel_grid

    def _usable(self, c):
        return self._in_bounds(c) and (c not in self.voxel_grid)

    @staticmethod
    def heuristic(a, b):
        d = sorted((abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])))
        amin, amid, amax = d
        return ((amax - amid) * FACE_COST +
                (amid - amin) * EDGE_COST +
                amin * CORNER_COST)

    def a_star(self, start, goal):
        if start == goal:
            return [start]

        if not self._usable(start):
            start = self._find_nearest_free(start)
            if start is None:
                self.get_logger().error('A* no free voxel near start')
                return None

        if not self._usable(goal):
            goal = self._find_nearest_free(goal)
            if goal is None:
                self.get_logger().warn('A* no free voxel near goal')
                return None

        open_set  = []
        came_from = {}
        g_score   = {start: 0.0}
        closed    = set()
        h0 = self.heuristic(start, goal)
        heapq.heappush(open_set, (h0, h0, start))

        it = 0
        while open_set:
            it += 1
            if it > MAX_ITER:
                self.get_logger().warn('A* exceeded max iterations')
                return None

            _, _, current = heapq.heappop(open_set)
            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                return self._reconstruct(came_from, current)

            cx, cy, cz = current
            base_g = g_score[current]
            for dx, dy, dz, cost in self.moves:
                neighbor = (cx + dx, cy + dy, cz + dz)
                if neighbor in closed:
                    continue
                if not self._usable(neighbor):
                    continue
                tentative_g = base_g + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor]   = tentative_g
                    h = self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (tentative_g + h, h, neighbor))

        return None

    def _reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _find_nearest_free(self, cell, max_radius=10):
        if self._usable(cell):
            return cell
        cx, cy, cz = cell
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != r:
                            continue
                        nb = (cx + dx, cy + dy, cz + dz)
                        if self._usable(nb):
                            return nb
        return None

    # === Path simplification (3D) ===================================

    def simplify_path(self, path_cells):
        if path_cells is None or len(path_cells) <= 2:
            return path_cells

        turning = [path_cells[0]]
        for i in range(1, len(path_cells) - 1):
            prev, curr, nxt = path_cells[i - 1], path_cells[i], path_cells[i + 1]
            d1 = (_sign(curr[0] - prev[0]), _sign(curr[1] - prev[1]), _sign(curr[2] - prev[2]))
            d2 = (_sign(nxt[0]  - curr[0]), _sign(nxt[1]  - curr[1]), _sign(nxt[2]  - curr[2]))
            if d1 != d2:
                turning.append(curr)
        turning.append(path_cells[-1])

        if len(turning) <= 2:
            return turning

        spaced = [turning[0]]
        accum = 0.0
        for i in range(1, len(turning)):
            w0 = self._voxel_to_world(*turning[i - 1])
            w1 = self._voxel_to_world(*turning[i])
            d = math.sqrt((w1[0]-w0[0])**2 + (w1[1]-w0[1])**2 + (w1[2]-w0[2])**2)
            accum += d
            if accum >= WAYPOINT_SPACING:
                spaced.append(turning[i])
                accum = 0.0

        if spaced[-1] != turning[-1]:
            spaced.append(turning[-1])
        return spaced

    # === Cubic B-spline smoothing ===================================

    def smooth_path_bspline(self, waypoints_3d, s=0.5, kappa_max=2.0):
        """
        Smooth 3D waypoints using cubic B-spline with chord-length
        parameterization. Returns smoothed waypoint list.
        Falls back to original if < 4 points, collision, or high curvature.

        Based on: Nikolaiev & Novotarskyi (2025) "Enhanced Adaptive B-Spline
        Smoothing Approach for UAV Path Planning", IJISA V17 N4.

        Smoothing parameter s controls fidelity vs smoothness tradeoff
        (Pollock, "Smoothing with Cubic Splines"):
          s=0 -> interpolating spline (passes through all points)
          s>0 -> allows deviation for smoother curves
        """
        # --- 1. Edge cases ---------------------------------------------------
        if not _SCIPY_OK or waypoints_3d is None or len(waypoints_3d) < 3:
            return waypoints_3d

        # Drop consecutive duplicates (zero-length segments break chord-length
        # normalization and splprep).
        pts = [waypoints_3d[0]]
        for p in waypoints_3d[1:]:
            q = pts[-1]
            if (abs(p[0]-q[0]) > 1e-6 or abs(p[1]-q[1]) > 1e-6 or
                    abs(p[2]-q[2]) > 1e-6):
                pts.append(p)
        if len(pts) < 3:
            return waypoints_3d

        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        zs = np.array([p[2] for p in pts], dtype=float)

        # --- 2. Chord-length parameterization (IJISA Sec. 3.3) ---------------
        seg = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2 + np.diff(zs)**2)
        total_len = float(seg.sum())
        if total_len < 1e-6:
            return waypoints_3d
        u = np.concatenate(([0.0], np.cumsum(seg))) / total_len

        raw_start = waypoints_3d[0]
        raw_end   = waypoints_3d[-1]
        num = max(10, int(total_len / SPLINE_SAMPLE_STEP))

        def _fit_sample_check(s_val):
            """Fit + sample + verify (obstacles, curvature). Returns smoothed
            list on success, or None if it must fall back to the raw path."""
            try:
                # k=3 (cubic, C2) needs >= 4 points
                # k=2 (quadratic, C1) needs >= 3 points
                # Use highest degree the data supports
                degree = min(3, len(xs) - 1)
                tck, _ = splprep([xs, ys, zs], u=u, k=degree, s=s_val)
            except Exception as e:  # noqa: BLE001 — any splprep failure -> raw
                self.get_logger().warn(f'[SPLINE] splprep failed ({e}) — using raw A* path')
                return None

            uu = np.linspace(0.0, 1.0, num)
            sx, sy, sz = splev(uu, tck)

            smoothed = [(float(sx[i]), float(sy[i]), float(sz[i])) for i in range(num)]
            # 5. Preserve start/end exactly.
            smoothed[0]  = raw_start
            smoothed[-1] = raw_end

            # 6. Obstacle safety check (IJISA Sec. 2.3).
            for (wx, wy, wz) in smoothed:
                if self._world_to_voxel(wx, wy, wz) in self.raw_obstacles:
                    self.get_logger().warn('[SPLINE] collision detected, using raw A* path')
                    return None

            # 7. Curvature check (IJISA Eq. 2): kappa = ||C' x C''|| / ||C'||^3
            d1 = np.array(splev(uu, tck, der=1))   # shape (3, num)
            d2 = np.array(splev(uu, tck, der=2))
            cross = np.cross(d1.T, d2.T)            # (num, 3)
            num_k = np.linalg.norm(cross, axis=1)
            den_k = np.linalg.norm(d1.T, axis=1)**3
            den_k = np.where(den_k < 1e-9, np.inf, den_k)  # guard ||C'||~0
            kappa = num_k / den_k
            max_k = float(np.max(kappa)) if len(kappa) else 0.0
            if max_k > kappa_max:
                return ('CURV', max_k)

            return smoothed

        # --- 3-7. Fit, with one curvature retry at higher smoothing ----------
        result = _fit_sample_check(s)
        if isinstance(result, tuple) and result[0] == 'CURV':
            self.get_logger().info(
                f'[SPLINE] curvature {result[1]:.2f} > {kappa_max} — retry s*{SPLINE_S_RETRY}')
            result = _fit_sample_check(s * SPLINE_S_RETRY)

        if result is None or (isinstance(result, tuple) and result[0] == 'CURV'):
            if isinstance(result, tuple):
                self.get_logger().warn(
                    f'[SPLINE] curvature still {result[1]:.2f} > {kappa_max} — using raw A* path')
            return waypoints_3d

        return result

    # === Main planning ==============================================

    def plan(self):
        """Plan a 3D A* path from current_pose to self.goal. Publish on success;
        on failure publish an empty path so path_follower holds position."""
        if self.current_pose is None or self.goal is None:
            return False

        sx, sy, sz = (self.current_pose.pose.position.x,
                      self.current_pose.pose.position.y,
                      self.current_pose.pose.position.z)
        gx, gy, gz = (self.goal.pose.position.x,
                      self.goal.pose.position.y,
                      self.goal.pose.position.z)
        if gz < 0.5 or gz > 4.0:
            gz = DEFAULT_ALTITUDE

        # Detect if active goal actually changes — used to force publish
        active_changed = True
        if self.active_goal is not None:
            ax, ay = self._xy(self.active_goal)
            if abs(ax - gx) < SAME_GOAL_EPS and abs(ay - gy) < SAME_GOAL_EPS:
                active_changed = False

        self.active_goal = self.goal

        self.get_logger().info(
            f'[PLAN] from ({sx:.2f},{sy:.2f},{sz:.2f}) to '
            f'({gx:.2f},{gy:.2f},{gz:.2f}) active_changed={active_changed}')

        start = self._world_to_voxel(sx, sy, sz)
        goal  = self._world_to_voxel(gx, gy, gz)

        path_cells = self.a_star(start, goal)
        if path_cells is None:
            self.get_logger().warn(
                '[A*] FAILED — no 3D path to goal. Publishing empty path → hold.')
            self._publish_empty_path()
            self.last_path = None
            self._publish_inflated_viz()
            return False

        sparse = self.simplify_path(path_cells)
        waypoints_3d = [self._voxel_to_world(*cell) for cell in sparse]
        self.get_logger().info(
            f'[A*] {len(path_cells)} voxels -> {len(waypoints_3d)} waypoints')

        raw_count = len(waypoints_3d)
        waypoints_3d = self.smooth_path_bspline(waypoints_3d, s=SPLINE_S, kappa_max=KAPPA_MAX)
        self.get_logger().info(
            f'[SPLINE] {raw_count} raw -> {len(waypoints_3d)} smoothed')

        # --- Build path message ---
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'
        for wx, wy, wz in waypoints_3d:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wx)
            pose.pose.position.y = float(wy)
            pose.pose.position.z = float(wz)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.get_logger().info(
            f'[PLAN_END] built path {len(path_msg.poses)} poses '
            f'endpoint=({path_msg.poses[-1].pose.position.x:.2f},'
            f'{path_msg.poses[-1].pose.position.y:.2f},'
            f'{path_msg.poses[-1].pose.position.z:.2f}) '
            f'for goal=({gx:.2f},{gy:.2f},{gz:.2f})')

        # --- Publish decision (suppress identical replans) ---
        should_publish = True
        if not active_changed and self.last_path is not None and self.last_path.poses:
            op  = self.last_path.poses
            np_ = path_msg.poses
            if (abs(len(np_) - len(op)) <= 2 and
                    abs(np_[-1].pose.position.x - op[-1].pose.position.x) < 0.10 and
                    abs(np_[-1].pose.position.y - op[-1].pose.position.y) < 0.10 and
                    abs(np_[-1].pose.position.z - op[-1].pose.position.z) < 0.10 and
                    abs(np_[ 0].pose.position.x - op[ 0].pose.position.x) < 0.30 and
                    abs(np_[ 0].pose.position.y - op[ 0].pose.position.y) < 0.30):
                should_publish = False

        if should_publish:
            self.get_logger().info(
                f'[PUBLISH] /global_path {len(path_msg.poses)} waypoints '
                f'to ({path_msg.poses[-1].pose.position.x:.2f},'
                f'{path_msg.poses[-1].pose.position.y:.2f},'
                f'{path_msg.poses[-1].pose.position.z:.2f})')
            self.path_pub.publish(path_msg)
        else:
            self.get_logger().debug('[PUBLISH] suppressed — path unchanged')

        self.last_path   = path_msg
        self.local_stuck = False

        self._publish_inflated_viz()
        return True

    # === Periodic check =============================================

    def periodic_check(self):
        """Single decision-maker for planning."""
        goal_str   = (f'{self.goal.pose.position.x:.1f},{self.goal.pose.position.y:.1f}'
                      if self.goal is not None else 'None')
        active_str = (f'{self.active_goal.pose.position.x:.1f},'
                      f'{self.active_goal.pose.position.y:.1f}'
                      if self.active_goal is not None else 'None')
        pose_str   = (f'{self.current_pose.pose.position.x:.2f},'
                      f'{self.current_pose.pose.position.y:.2f},'
                      f'{self.current_pose.pose.position.z:.2f}'
                      if self.current_pose is not None else 'None')
        self.get_logger().info(
            f'[PERIODIC] nav={self.nav_active} goal={goal_str} '
            f'active={active_str} path={self.last_path is not None} '
            f'occ={len(self.voxel_grid)} drone=({pose_str})')

        if self.emergency or not self.nav_active:
            return
        if self.current_pose is None or self.goal is None:
            return

        # --- Step 1: Goal-reached check (3D, uses active_goal) ---
        if self.active_goal is not None:
            dist = self._dist_3d(self.current_pose, self.active_goal)
            self.get_logger().info(
                f'[PERIODIC] dist_to_active={dist:.2f} thr={GOAL_REACHED_DIST}')
            if dist < GOAL_REACHED_DIST:
                self.get_logger().info(
                    f'[GOAL_REACHED] active ({active_str}) reached, '
                    f'queued goal=({goal_str})')
                self.last_path   = None
                self.active_goal = None

        # --- Step 2: Plan if no active path ---
        if self.last_path is None:
            self.plan()
            return

        # --- Step 3: Replan if local planner reports stuck ---
        if self.local_stuck:
            self.plan()

    def _publish_empty_path(self):
        """Tell path_follower to hold position (no valid path)."""
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'
        self.path_pub.publish(path_msg)

    # === Path blocked check =========================================

    def _check_path_blocked(self):
        """A path pose lies in an occupied (inflated) voxel -> blocked."""
        if self.last_path is None:
            return False
        for pose in self.last_path.poses:
            v = self._world_to_voxel(pose.pose.position.x,
                                     pose.pose.position.y,
                                     pose.pose.position.z)
            if v in self.voxel_grid:
                return True
        return False

    # === Visualization (debug) ======================================

    def _publish_inflated_viz(self):
        if not self.voxel_grid:
            return
        pts = np.array([self._voxel_to_world(*v) for v in self.voxel_grid],
                       dtype=np.float32)
        msg = PointCloud2()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.height = 1
        msg.width  = len(pts)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 12
        msg.row_step     = 12 * len(pts)
        msg.is_dense     = True
        msg.data         = pts.tobytes()
        self.infl_pub.publish(msg)


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
