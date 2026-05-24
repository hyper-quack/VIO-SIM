#!/usr/bin/env python3
"""
rrt_local_planner.py — Bidirectional RRT* Local Planner

Architecture:
  /global_path  (A*, coarse, slow)      → local goal selection
  /costmap      (rolling, 10Hz)         → obstacle checking
        ↓
  Bidirectional RRT* (5Hz)
        ↓
  /planned_path → path_follower → PX4

Behavior:
  1. Pick next local goal from global path (within LOCAL_GOAL_DIST)
  2. If straight line clear → follow global path directly
  3. If obstacle detected → Bi-RRT* computes detour
  4. When obstacle cleared → snap back to global path
  5. If stuck > STUCK_TIMEOUT → signal A* to replan globally
"""

import math
import random
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

# ── Planning parameters ───────────────────────────────────────────
MAX_ITERATIONS    = 600
STEP_SIZE         = 0.4
REWIRE_RADIUS     = 1.2
GOAL_BIAS         = 0.25
CONNECT_DIST      = 0.8
MIN_OBSTACLE_DIST = 0.8

# ── Local goal ────────────────────────────────────────────────────
LOCAL_GOAL_DIST   = 4.0
GOAL_TOLERANCE    = 0.4
PATH_CLEAR_CHECK  = 0.3

# ── Timing ────────────────────────────────────────────────────────
REPLAN_RATE       = 5.0
STUCK_TIMEOUT     = 4.0

# ── Costmap ───────────────────────────────────────────────────────
COSTMAP_INFLATE   = 2

# ── Z ─────────────────────────────────────────────────────────────
Z_MIN = 0.5
Z_MAX = 3.0


class RRTNode:
    __slots__ = ['x', 'y', 'z', 'parent', 'cost']
    def __init__(self, x, y, z=2.0):
        self.x = x; self.y = y; self.z = z
        self.parent = None; self.cost = 0.0


class RRTLocalPlanner(Node):

    def __init__(self):
        super().__init__('rrt_local_planner')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.current_pose  = None
        self.global_path   = []
        self.costmap       = None
        self.nav_active    = False
        self.emergency     = False
        self.global_idx    = 0
        self.stuck_start   = None

        self.create_subscription(PoseStamped,   '/current_pose',       self.pose_cb,     qos)
        self.create_subscription(Path,          '/global_path',        self.gpath_cb,    10)
        self.create_subscription(OccupancyGrid, '/costmap',            self.costmap_cb,  10)
        self.create_subscription(Bool,          '/navigation_active',  self.nav_cb,      10)
        self.create_subscription(Bool,          '/emergency_stop',     self.emergency_cb, 10)

        self.path_pub    = self.create_publisher(Path, '/planned_path', 10)
        self.stuck_pub   = self.create_publisher(Bool, '/local_planner_stuck', 10)
        self.reached_pub = self.create_publisher(Bool, '/goal_reached', 10)
        self.map_reset_pub  = self.create_publisher(Bool, '/map_reset', 10)
        self.force_update_pub = self.create_publisher(Bool, '/force_update', 10)
        self.recovery_active = False
        self.recovery_start  = None
        self.RECOVERY_WAIT   = 3.0
        self.current_planned_path = None  # last published path
        self.last_rrt_time = 0.0          # last time RRT* was triggered
        self.RRT_COOLDOWN  = 2.0          # minimum seconds between RRT* runs

        self.create_timer(1.0 / REPLAN_RATE, self.plan_loop)
        self.get_logger().info('Bi-RRT* Local Planner started ✓')

    # ── Callbacks ─────────────────────────────────────────────────

    def pose_cb(self, msg):      self.current_pose = msg
    def nav_cb(self, msg):       self.nav_active = msg.data
    def emergency_cb(self, msg): self.emergency = msg.data

    def gpath_cb(self, msg):
        if not msg.poses: return
        self.global_path = msg.poses
        self.global_idx  = 0
        self.get_logger().info(f'Global path: {len(self.global_path)} waypoints')

    def costmap_cb(self, msg):
        h = msg.info.height; w = msg.info.width
        data = np.array(msg.data, dtype=np.int8).reshape(h, w)
        self.costmap = {
            'grid': (data > 50).astype(np.uint8),
            'ox': msg.info.origin.position.x,
            'oy': msg.info.origin.position.y,
            'res': msg.info.resolution,
            'w': w, 'h': h,
        }

    # ── Obstacle checking ─────────────────────────────────────────

    def _is_free(self, x, y):
        if self.costmap is None: return True
        cm = self.costmap
        cx = int((x - cm['ox']) / cm['res'])
        cy = int((y - cm['oy']) / cm['res'])
        inf = COSTMAP_INFLATE
        for dx in range(-inf, inf + 1):
            for dy in range(-inf, inf + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cm['w'] and 0 <= ny < cm['h']:
                    if cm['grid'][ny, nx] != 0:
                        return False
        return True

    def _path_clear(self, x0, y0, x1, y1):
        dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
        if dist < 0.01: return True
        steps = max(2, int(dist / PATH_CLEAR_CHECK))
        for i in range(steps + 1):
            t = i / steps
            if not self._is_free(x0 + t*(x1-x0), y0 + t*(y1-y0)):
                return False
        return True

    def _obstacle_nearby(self, x, y):
        if self.costmap is None: return False
        cm = self.costmap
        cells = max(1, int(MIN_OBSTACLE_DIST / cm['res']))
        cx = int((x - cm['ox']) / cm['res'])
        cy = int((y - cm['oy']) / cm['res'])
        for dx in range(-cells, cells + 1):
            for dy in range(-cells, cells + 1):
                if dx*dx + dy*dy > cells*cells: continue
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cm['w'] and 0 <= ny < cm['h']:
                    if cm['grid'][ny, nx] != 0: return True
        return False

    # ── Local goal selection ──────────────────────────────────────

    def _select_local_goal(self):
        if not self.global_path or self.current_pose is None: return None
        sx = self.current_pose.pose.position.x
        sy = self.current_pose.pose.position.y

        # Advance past reached waypoints
        while self.global_idx < len(self.global_path) - 1:
            wp = self.global_path[self.global_idx]
            dx = wp.pose.position.x - sx
            dy = wp.pose.position.y - sy
            if math.sqrt(dx*dx + dy*dy) < GOAL_TOLERANCE:
                self.global_idx += 1
            else:
                break

        # Pick farthest within LOCAL_GOAL_DIST
        best = None
        for i in range(self.global_idx, len(self.global_path)):
            wp = self.global_path[i]
            dx = wp.pose.position.x - sx
            dy = wp.pose.position.y - sy
            if math.sqrt(dx*dx + dy*dy) <= LOCAL_GOAL_DIST:
                best = wp
            else:
                if best is None: best = wp
                break
        return best or self.global_path[-1]

    # ── Bi-RRT* ───────────────────────────────────────────────────

    def _dist(self, a, b):
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

    def _nearest(self, tree, pt):
        return min(tree, key=lambda n: (n.x-pt.x)**2 + (n.y-pt.y)**2)

    def _steer(self, frm, to):
        d = self._dist(frm, to)
        if d < 0.01: return None
        r = min(STEP_SIZE, d) / d
        nd = RRTNode(frm.x + r*(to.x-frm.x), frm.y + r*(to.y-frm.y),
                     frm.z + r*(to.z-frm.z))
        nd.parent = frm
        nd.cost   = frm.cost + min(STEP_SIZE, d)
        return nd

    def _rewire(self, tree, new_node):
        for node in tree:
            if node is new_node or node is new_node.parent: continue
            d = self._dist(node, new_node)
            if d > REWIRE_RADIUS: continue
            if new_node.cost + d < node.cost:
                if self._path_clear(new_node.x, new_node.y, node.x, node.y):
                    node.parent = new_node
                    node.cost   = new_node.cost + d

    def _extract(self, na, nb):
        path_a = []
        n = na
        while n: path_a.append(n); n = n.parent
        path_a.reverse()
        path_b = []
        n = nb
        while n: path_b.append(n); n = n.parent
        return path_a + path_b

    def _sample(self, sx, sy, gx, gy):
        if random.random() < GOAL_BIAS:
            return RRTNode(gx, gy)
        t  = random.random()
        mx = sx + t*(gx-sx); my = sy + t*(gy-sy)
        spread = max(1.5, abs(gy-sy)*0.5)
        return RRTNode(mx + random.gauss(0, spread*0.3),
                       my + random.gauss(0, spread))

    def _bi_rrt_star(self, sx, sy, sz, gx, gy, gz):
        start = RRTNode(sx, sy, sz)
        goal  = RRTNode(gx, gy, gz)
        tree_a = [start]
        tree_b = [goal]

        for i in range(MAX_ITERATIONS):
            active, other = (tree_a, tree_b) if i % 2 == 0 else (tree_b, tree_a)
            rand = self._sample(sx, sy, gx, gy)
            near = self._nearest(active, rand)
            new  = self._steer(near, rand)
            if new is None: continue
            if not self._is_free(new.x, new.y): continue
            if not self._path_clear(near.x, near.y, new.x, new.y): continue
            active.append(new)
            self._rewire(active, new)

            near_other = self._nearest(other, new)
            if self._dist(new, near_other) < CONNECT_DIST:
                if self._path_clear(new.x, new.y, near_other.x, near_other.y):
                    nodes = self._extract(new, near_other) if active is tree_a \
                            else self._extract(near_other, new)
                    result = []
                    for j, nd in enumerate(nodes):
                        t  = j / max(1, len(nodes)-1)
                        wz = max(Z_MIN, min(Z_MAX, sz + t*(gz-sz)))
                        result.append((nd.x, nd.y, wz))
                    return result
        return None

    # ── Main loop ─────────────────────────────────────────────────

    def plan_loop(self):
        if self.emergency or not self.nav_active: return
        if self.current_pose is None or not self.global_path: return

        now = self.get_clock().now().nanoseconds * 1e-9
        sx  = self.current_pose.pose.position.x
        sy  = self.current_pose.pose.position.y
        sz  = self.current_pose.pose.position.z

        # Check final goal
        final = self.global_path[-1]
        dx = final.pose.position.x - sx
        dy = final.pose.position.y - sy
        if math.sqrt(dx*dx + dy*dy) < GOAL_TOLERANCE:
            msg = Bool(); msg.data = True
            self.reached_pub.publish(msg)
            self.get_logger().info('Final goal reached ✓')
            return

        local_wp = self._select_local_goal()
        if local_wp is None: return

        gx = local_wp.pose.position.x
        gy = local_wp.pose.position.y
        gz = local_wp.pose.position.z
        if gz < 0.5 or gz > 4.0: gz = 2.0

        # If path clear → follow global path directly
        if self._path_clear(sx, sy, gx, gy) and not self._obstacle_nearby(sx, sy):
            self._publish_global_segment(sx, sy, sz)
            self.stuck_start = None
            return

        # Obstacle → run Bi-RRT*
        self.get_logger().info(
            f'Obstacle — RRT* to ({gx:.1f},{gy:.1f})',
            throttle_duration_sec=1.0)

        path = self._bi_rrt_star(sx, sy, sz, gx, gy, gz)

        if path is not None:
            self._publish_rrt_path(path)
            self.last_rrt_time   = now
            self.stuck_start     = None
            self.recovery_active = False
            self.recovery_start  = None
        else:
            self.get_logger().warn('RRT* failed', throttle_duration_sec=1.0)

            # If in recovery — wait for map to rebuild then replan
            if self.recovery_active:
                elapsed = now - self.recovery_start
                self.get_logger().warn(
                    f'Recovery: waiting for map rebuild {elapsed:.1f}/{self.RECOVERY_WAIT}s',
                    throttle_duration_sec=1.0)
                if elapsed > self.RECOVERY_WAIT:
                    # Map rebuilt — trigger global replan and resume
                    self.get_logger().warn('Recovery complete — triggering global replan')
                    msg = Bool(); msg.data = False
                    self.force_update_pub.publish(msg)
                    msg2 = Bool(); msg2.data = True
                    self.stuck_pub.publish(msg2)
                    self.recovery_active = False
                    self.recovery_start  = None
                    self.stuck_start     = None
                return

            # Start stuck timer
            if self.stuck_start is None:
                self.stuck_start = now
            elif now - self.stuck_start > STUCK_TIMEOUT:
                # Stuck confirmed — start recovery
                self.get_logger().warn('STUCK confirmed — resetting map and rebuilding')
                # Step 1: reset voxel map
                msg = Bool(); msg.data = True
                self.map_reset_pub.publish(msg)
                # Step 2: force octomap to update aggressively
                msg2 = Bool(); msg2.data = True
                self.force_update_pub.publish(msg2)
                # Step 3: enter recovery mode — hover and wait
                self.recovery_active = True
                self.recovery_start  = now
                self.stuck_start     = None
                self.get_logger().warn(
                    f'Recovery started — hovering {self.RECOVERY_WAIT}s for map rebuild')

    def _initial_plan(self, sx, sy, sz, now):
        """Plan initial path — try direct first, then RRT*."""
        local_wp = self._select_local_goal()
        if local_wp is None: return
        gx = local_wp.pose.position.x
        gy = local_wp.pose.position.y
        gz = local_wp.pose.position.z
        if gz < 0.5 or gz > 4.0: gz = 2.0

        if self._path_clear(sx, sy, gx, gy):
            self._publish_global_segment(sx, sy, sz)
        else:
            path = self._bi_rrt_star(sx, sy, sz, gx, gy, gz)
            if path is not None:
                self._publish_rrt_path(path)
                self.last_rrt_time = now

    def _publish_global_segment(self, sx, sy, sz):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        for i in range(self.global_idx, len(self.global_path)):
            wp = self.global_path[i]
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose   = wp.pose
            msg.poses.append(ps)
            dx = wp.pose.position.x - sx
            dy = wp.pose.position.y - sy
            if math.sqrt(dx*dx + dy*dy) > LOCAL_GOAL_DIST * 1.5: break
        if msg.poses:
            self.path_pub.publish(msg)

    def _planned_path_clear(self, path_msg, sx, sy):
        """Check if a previously planned path is still obstacle-free."""
        for pose in path_msg.poses:
            px = pose.pose.position.x
            py = pose.pose.position.y
            if not self._is_free(px, py):
                return False
        return True

    def _publish_rrt_path(self, waypoints):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        for wx, wy, wz in waypoints:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.position.z = float(wz)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.current_planned_path = msg
        self.path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RRTLocalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()