#!/usr/bin/env python3
"""
rrt_local_planner.py — Bidirectional RRT* Local Planner

Clean event-driven architecture:
- Follows global path directly when clear
- Only runs RRT* when obstacle persistently blocks path
- Once RRT* detour computed, follows it completely without rechecking
- Resumes global path after detour complete
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

# Planning
MAX_ITERATIONS   = 600
STEP_SIZE        = 0.4
REWIRE_RADIUS    = 1.2
GOAL_BIAS        = 0.25
CONNECT_DIST     = 0.8

# Local goal
LOCAL_GOAL_DIST  = 4.0
GOAL_TOLERANCE   = 0.4
PATH_CLEAR_CHECK = 0.3

# Timing
REPLAN_RATE      = 5.0
RRT_COOLDOWN     = 8.0
STUCK_TIMEOUT    = 6.0

# Obstacle persistence — must block N checks before replan
OBSTACLE_CONFIRM = 10    # 10 × 0.2s = 2 seconds

# Costmap
COSTMAP_INFLATE  = 2
REPLAN_LOOKAHEAD = 3.0
BLOCK_MIN_CELLS  = 7
MIN_CHECK_DIST   = 0.8   # skip waypoints closer than this

# Z
Z_MIN = 0.5
Z_MAX = 3.0

# Path post-processing
SMOOTH_RRT_SPACING    = 0.5   # resample spacing for RRT* detour paths (m)
SMOOTH_GLOBAL_SPACING = 0.8   # resample spacing for global segment paths (m)
SMOOTH_PASSES         = 3     # shortcutting iterations
SAFETY_MARGIN         = 0.80  # minimum clearance from obstacles after smoothing (m)

# Obstacle memory — guards against rolling-costmap blind spots after detour
OBSTACLE_MEMORY_DURATION = 10.0   # seconds before a memory entry expires
OBSTACLE_MEMORY_MAX      = 20     # maximum concurrent memory entries
OBSTACLE_MEMORY_RADIUS   = 1.0    # path-clearance check radius for memory (m)


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

        # Drone state
        self.current_pose = None
        self.global_path  = []
        self.costmap      = None
        self.nav_active   = False
        self.emergency    = False
        self.global_idx   = 0

        # Path state
        self.current_planned_path = None
        self.path_is_rrt          = False  # True = following RRT* detour
        self.last_rrt_time        = 0.0
        self.obstacle_counter     = 0
        self.goal_reached_sent    = False  # one-shot flag — prevents 5 Hz /goal_reached flood
        self.obstacle_memory      = []     # [(x, y, timestamp), …] — where RRT* was triggered

        # Recovery
        self.stuck_start     = None
        self.recovery_active = False
        self.recovery_start  = None
        self.RECOVERY_WAIT   = 3.0

        # Subscribers
        self.create_subscription(PoseStamped,   '/current_pose',      self.pose_cb,      qos)
        self.create_subscription(Path,          '/global_path',       self.gpath_cb,     10)
        self.create_subscription(OccupancyGrid, '/costmap',           self.costmap_cb,   10)
        self.create_subscription(Bool,          '/navigation_active', self.nav_cb,       10)
        self.create_subscription(Bool,          '/emergency_stop',    self.emergency_cb, 10)

        # Publishers
        self.path_pub         = self.create_publisher(Path, '/planned_path', 10)
        self.stuck_pub        = self.create_publisher(Bool, '/local_planner_stuck', 10)
        self.reached_pub      = self.create_publisher(Bool, '/goal_reached', 10)
        self.map_reset_pub    = self.create_publisher(Bool, '/map_reset', 10)
        self.force_update_pub = self.create_publisher(Bool, '/force_update', 10)

        self.create_timer(1.0 / REPLAN_RATE, self.plan_loop)
        self.get_logger().info('Bi-RRT* Local Planner started ✓')

    # ── Callbacks ─────────────────────────────────────────────────

    def pose_cb(self, msg):      self.current_pose = msg
    def nav_cb(self, msg):       self.nav_active = msg.data
    def emergency_cb(self, msg): self.emergency = msg.data

    def gpath_cb(self, msg):
        if not msg.poses:
            return

        new_len   = len(msg.poses)
        new_first = msg.poses[0].pose.position
        new_last  = msg.poses[-1].pose.position

        path_changed = False
        if not self.global_path:
            path_changed = True
            self.get_logger().info(
                f'[gpath_cb] First global path received: {new_len} waypoints')
        else:
            old_len   = len(self.global_path)
            old_first = self.global_path[0].pose.position
            old_last  = self.global_path[-1].pose.position

            len_diff = abs(new_len - old_len)
            first_dx = abs(new_first.x - old_first.x)
            first_dy = abs(new_first.y - old_first.y)
            last_dx  = abs(new_last.x  - old_last.x)
            last_dy  = abs(new_last.y  - old_last.y)

            if len_diff > 2 or first_dx > 0.15 or first_dy > 0.15 \
                             or last_dx  > 0.10 or last_dy  > 0.10:
                path_changed = True
                self.get_logger().info(
                    f'[gpath_cb] Path CHANGED: len {old_len}→{new_len}  '
                    f'first_Δ=({first_dx:.3f},{first_dy:.3f})  '
                    f'last_Δ=({last_dx:.3f},{last_dy:.3f}) → resetting state')
            else:
                self.get_logger().debug(
                    f'[gpath_cb] Path unchanged (len={new_len}, '
                    f'first_Δ=({first_dx:.3f},{first_dy:.3f}), '
                    f'last_Δ=({last_dx:.3f},{last_dy:.3f})) — skipping reset')

        # Always refresh waypoint array so _select_local_goal() stays current
        self.global_path = msg.poses

        if path_changed:
            self.current_planned_path = None
            self.path_is_rrt          = False
            self.obstacle_counter     = 0
            self.goal_reached_sent    = False  # reset one-shot flag on new path
            if self.current_pose is not None:
                sx = self.current_pose.pose.position.x
                sy = self.current_pose.pose.position.y
                best_idx, best_dist = 0, float('inf')
                for i, wp in enumerate(self.global_path):
                    d = math.sqrt((wp.pose.position.x - sx)**2 +
                                  (wp.pose.position.y - sy)**2)
                    if d < best_dist:
                        best_dist = d; best_idx = i
                self.global_idx = best_idx
            else:
                self.global_idx = 0

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
        if not (0 <= cx < cm['w'] and 0 <= cy < cm['h']):
            return True  # outside window = unknown = free
        for dx in range(-COSTMAP_INFLATE, COSTMAP_INFLATE+1):
            for dy in range(-COSTMAP_INFLATE, COSTMAP_INFLATE+1):
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < cm['w'] and 0 <= ny < cm['h']:
                    if cm['grid'][ny, nx] != 0:
                        return False
        return True

    def _path_clear(self, x0, y0, x1, y1):
        dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
        if dist < 0.01: return True
        steps = max(2, int(dist / PATH_CLEAR_CHECK))
        for i in range(steps+1):
            t = i / steps
            if not self._is_free(x0+t*(x1-x0), y0+t*(y1-y0)):
                return False
        return True

    def _stored_path_blocked(self, sx, sy):
        """Check if global path has persistent obstacle ahead (not RRT* path)."""
        if self.current_planned_path is None or self.costmap is None:
            return False
        cm = self.costmap
        for pose in self.current_planned_path.poses:
            px = pose.pose.position.x
            py = pose.pose.position.y
            dist = math.sqrt((px-sx)**2 + (py-sy)**2)
            if dist < MIN_CHECK_DIST: continue  # skip near-drone waypoints
            if dist > REPLAN_LOOKAHEAD: break
            cx = int((px - cm['ox']) / cm['res'])
            cy = int((py - cm['oy']) / cm['res'])
            if not (0 <= cx < cm['w'] and 0 <= cy < cm['h']): continue
            blocked = 0
            for ddx in range(-1, 2):
                for ddy in range(-1, 2):
                    nx, ny = cx+ddx, cy+ddy
                    if 0 <= nx < cm['w'] and 0 <= ny < cm['h']:
                        if cm['grid'][ny, nx] != 0:
                            blocked += 1
            if blocked >= BLOCK_MIN_CELLS:
                return True

        # ── Memory check: catch obstacles that rolled out of costmap window ──
        if self.obstacle_memory:
            for pose in self.current_planned_path.poses:
                px   = pose.pose.position.x
                py   = pose.pose.position.y
                dist = math.sqrt((px - sx)**2 + (py - sy)**2)
                if dist < MIN_CHECK_DIST:  continue
                if dist > REPLAN_LOOKAHEAD: break
                for (ox, oy, _) in self.obstacle_memory:
                    if math.sqrt((px - ox)**2 + (py - oy)**2) < OBSTACLE_MEMORY_RADIUS:
                        self.get_logger().debug(
                            f'[_stored_path_blocked] Memory hit: waypoint '
                            f'({px:.2f},{py:.2f}) within {OBSTACLE_MEMORY_RADIUS:.1f} m '
                            f'of remembered trigger ({ox:.2f},{oy:.2f})')
                        return True
        return False

    # ── Local goal selection ──────────────────────────────────────

    def _select_local_goal(self):
        if not self.global_path or self.current_pose is None: return None
        sx = self.current_pose.pose.position.x
        sy = self.current_pose.pose.position.y

        while self.global_idx < len(self.global_path)-1:
            wp = self.global_path[self.global_idx]
            if math.sqrt((wp.pose.position.x-sx)**2 + (wp.pose.position.y-sy)**2) < GOAL_TOLERANCE:
                self.global_idx += 1
            else:
                break

        best = None
        for i in range(self.global_idx, len(self.global_path)):
            wp = self.global_path[i]
            dist = math.sqrt((wp.pose.position.x-sx)**2 + (wp.pose.position.y-sy)**2)
            if self._is_free(wp.pose.position.x, wp.pose.position.y):
                best = wp
                if dist > LOCAL_GOAL_DIST: break
            if dist > LOCAL_GOAL_DIST * 2.0: break
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
        nd = RRTNode(frm.x+r*(to.x-frm.x), frm.y+r*(to.y-frm.y), frm.z+r*(to.z-frm.z))
        nd.parent = frm; nd.cost = frm.cost + min(STEP_SIZE, d)
        return nd

    def _rewire(self, tree, new_node):
        for node in tree:
            if node is new_node or node is new_node.parent: continue
            d = self._dist(node, new_node)
            if d > REWIRE_RADIUS: continue
            if new_node.cost+d < node.cost:
                if self._path_clear(new_node.x, new_node.y, node.x, node.y):
                    node.parent = new_node; node.cost = new_node.cost+d

    def _extract(self, na, nb):
        a = []; n = na
        while n: a.append(n); n = n.parent
        a.reverse()
        b = []; n = nb
        while n: b.append(n); n = n.parent
        return a + b

    def _sample(self, sx, sy, gx, gy):
        if random.random() < GOAL_BIAS: return RRTNode(gx, gy)
        t = random.random()
        mx = sx+t*(gx-sx); my = sy+t*(gy-sy)
        spread = max(1.5, abs(gy-sy)*0.5)
        return RRTNode(mx+random.gauss(0, spread*0.3), my+random.gauss(0, spread))

    def _bi_rrt_star(self, sx, sy, sz, gx, gy, gz):
        start = RRTNode(sx, sy, sz); goal = RRTNode(gx, gy, gz)
        ta = [start]; tb = [goal]
        for i in range(MAX_ITERATIONS):
            act, oth = (ta, tb) if i%2==0 else (tb, ta)
            rand = self._sample(sx, sy, gx, gy)
            near = self._nearest(act, rand)
            new  = self._steer(near, rand)
            if new is None: continue
            if not self._is_free(new.x, new.y): continue
            if not self._path_clear(near.x, near.y, new.x, new.y): continue
            act.append(new); self._rewire(act, new)
            no = self._nearest(oth, new)
            if self._dist(new, no) < CONNECT_DIST:
                if self._path_clear(new.x, new.y, no.x, no.y):
                    nodes = self._extract(new, no) if act is ta else self._extract(no, new)
                    result = []
                    for j, nd in enumerate(nodes):
                        t = j/max(1, len(nodes)-1)
                        wz = max(Z_MIN, min(Z_MAX, sz+t*(gz-sz)))
                        result.append((nd.x, nd.y, wz))
                    return result
        return None

    # ── Path post-processing ─────────────────────────────────────

    def _smooth_path(self, waypoints, spacing=SMOOTH_RRT_SPACING):
        """
        Three-stage path smoother.
        Input/output: list of (x, y, z) tuples.

        Stage 1 — Shortcutting (SMOOTH_PASSES iterations):
            Greedily skip intermediate waypoints when a direct segment is
            collision-free, reducing unnecessary detour kinks.

        Stage 2 — Uniform resample at `spacing` metres:
            Arc-length parameterisation guarantees even waypoint distribution
            so path_follower switches waypoints at predictable intervals.

        Stage 3 — Moving-average (window=3) on x,y,z:
            Rounds remaining corners. First and last waypoints are fixed to
            preserve exact start/end positions.
        """
        pts = list(waypoints)
        if len(pts) < 3:
            return pts

        # ── Stage 1: Shortcutting ────────────────────────────────
        for _ in range(SMOOTH_PASSES):
            i = 0
            new_pts = [pts[0]]
            while i < len(pts) - 1:
                # Try to jump to the farthest directly reachable waypoint
                j = len(pts) - 1
                while j > i + 1:
                    if self._path_clear(pts[i][0], pts[i][1],
                                        pts[j][0], pts[j][1]):
                        break
                    j -= 1
                new_pts.append(pts[j])
                i = j
            pts = new_pts
            if len(pts) < 3:
                break   # fully shortcut — nothing left to smooth

        # ── Stage 2: Uniform resample ────────────────────────────
        arc = [0.0]
        for k in range(1, len(pts)):
            dx = pts[k][0] - pts[k-1][0]
            dy = pts[k][1] - pts[k-1][1]
            arc.append(arc[-1] + math.sqrt(dx*dx + dy*dy))
        total = arc[-1]

        if total >= spacing:
            n_steps = max(1, int(total / spacing))
            resampled = [pts[0]]
            for s in range(1, n_steps):
                target = s * total / n_steps
                for k in range(1, len(arc)):
                    if arc[k] >= target:
                        seg = arc[k] - arc[k-1]
                        t   = (target - arc[k-1]) / seg if seg > 1e-9 else 0.0
                        rx  = pts[k-1][0] + t * (pts[k][0] - pts[k-1][0])
                        ry  = pts[k-1][1] + t * (pts[k][1] - pts[k-1][1])
                        rz  = pts[k-1][2] + t * (pts[k][2] - pts[k-1][2])
                        resampled.append((rx, ry, rz))
                        break
            resampled.append(pts[-1])
            pts = resampled

        # ── Stage 3: Moving average (window=3) ───────────────────
        if len(pts) >= 3:
            smoothed = [pts[0]]
            for k in range(1, len(pts) - 1):
                mx = (pts[k-1][0] + pts[k][0] + pts[k+1][0]) / 3.0
                my = (pts[k-1][1] + pts[k][1] + pts[k+1][1]) / 3.0
                mz = (pts[k-1][2] + pts[k][2] + pts[k+1][2]) / 3.0
                smoothed.append((mx, my, mz))
            smoothed.append(pts[-1])
            pts = smoothed

        return pts

    def _enforce_safety_margin(self, waypoints, margin=SAFETY_MARGIN):
        """
        Push intermediate waypoints away from obstacles until each one is at
        least `margin` metres clear.

        Algorithm — for each non-endpoint waypoint:
          1. Sample 8 equally-spaced directions in 0.1 m steps up to `margin`.
          2. For each direction record the first blocked distance (if any).
          3. Identify the direction with the MINIMUM blocked distance — this is
             the most threatening obstacle side. Push directly opposite to it
             by (margin - min_dist + 0.05 m).
          4. Verify the pushed position with _is_free(); drop the waypoint if
             it is still blocked.

        Using the single closest-threat direction (rather than a vector sum)
        avoids cancellation when inflation makes multiple directions appear
        equally blocked at the same small radius.

        Falls back to the original list if fewer than 2 waypoints survive.
        """
        if self.costmap is None:
            return list(waypoints)

        result = []
        DIRS  = 8
        STEP  = 0.10   # radial step size (m)
        EXTRA = 0.05   # extra clearance beyond margin (m)

        for i, (x, y, z) in enumerate(waypoints):
            # Always keep start and end exactly as planned
            if i == 0 or i == len(waypoints) - 1:
                result.append((x, y, z))
                continue

            # Find the single closest-threat direction across all 8 samples
            closest_dist = margin + 1.0   # sentinel — no obstacle found yet
            closest_cos  = 0.0
            closest_sin  = 0.0

            for k in range(DIRS):
                angle = 2.0 * math.pi * k / DIRS
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                r = STEP
                while r <= margin + 1e-6:
                    if not self._is_free(x + cos_a * r, y + sin_a * r):
                        if r < closest_dist:
                            closest_dist = r
                            closest_cos  = cos_a
                            closest_sin  = sin_a
                        break   # found closest in this direction; try next dir
                    r += STEP

            if closest_dist > margin:
                # No obstacle within margin in any sampled direction — keep
                result.append((x, y, z))
                continue

            # Push directly opposite the closest-threat direction
            push_amount = margin - closest_dist + EXTRA
            new_x = x - closest_cos * push_amount
            new_y = y - closest_sin * push_amount

            if self._is_free(new_x, new_y):
                result.append((new_x, new_y, z))
            else:
                self.get_logger().debug(
                    f'[_enforce_safety_margin] Dropping waypoint '
                    f'({x:.2f},{y:.2f}) — pushed ({new_x:.2f},{new_y:.2f}) '
                    f'still blocked (closest={closest_dist:.2f} m)')
                # waypoint not appended → removed from path

        # Safety: if post-processing removed too many points, keep original
        if len(result) < 2:
            self.get_logger().warn(
                '[_enforce_safety_margin] All intermediate waypoints removed — '
                'falling back to raw path')
            return list(waypoints)

        return result

    # ── Main planning loop ────────────────────────────────────────

    def plan_loop(self):
        if self.emergency or not self.nav_active: return
        if self.current_pose is None or not self.global_path: return

        now = self.get_clock().now().nanoseconds * 1e-9

        # Expire obstacle memory entries older than OBSTACLE_MEMORY_DURATION
        self.obstacle_memory = [
            e for e in self.obstacle_memory
            if now - e[2] < OBSTACLE_MEMORY_DURATION
        ]

        sx  = self.current_pose.pose.position.x
        sy  = self.current_pose.pose.position.y
        sz  = self.current_pose.pose.position.z

        # Check final goal reached
        final = self.global_path[-1]
        if math.sqrt((final.pose.position.x-sx)**2 + (final.pose.position.y-sy)**2) < GOAL_TOLERANCE:
            if not self.goal_reached_sent:   # one-shot: don't flood at 5 Hz
                msg = Bool(); msg.data = True
                self.reached_pub.publish(msg)
                self.get_logger().info('Final goal reached ✓')
                self.goal_reached_sent = True
            return

        # ── CASE 1: Following RRT* detour ────────────────────────
        if self.path_is_rrt and self.current_planned_path is not None:
            final_wp = self.current_planned_path.poses[-1]
            dist_to_end = math.sqrt(
                (final_wp.pose.position.x-sx)**2 +
                (final_wp.pose.position.y-sy)**2)
            if dist_to_end > GOAL_TOLERANCE:
                # Still following detour — do nothing, let path_follower drive
                return
            else:
                # Detour complete — advance global_idx past the obstacle area
                rrt_ex = final_wp.pose.position.x
                rrt_ey = final_wp.pose.position.y
                best_i, best_d = self.global_idx, float('inf')
                for gi in range(self.global_idx, len(self.global_path)):
                    wp = self.global_path[gi]
                    d  = math.sqrt((wp.pose.position.x - rrt_ex)**2 +
                                   (wp.pose.position.y - rrt_ey)**2)
                    if d < best_d:
                        best_d = d; best_i = gi
                self.global_idx = min(best_i + 1, len(self.global_path) - 1)
                self.get_logger().info(
                    f'[plan_loop] CASE 1: RRT* detour complete — '
                    f'advancing global_idx → {self.global_idx} '
                    f'(nearest to detour end ({rrt_ex:.2f},{rrt_ey:.2f}), '
                    f'dist={best_d:.2f} m)')
                self.path_is_rrt          = False
                self.current_planned_path = None
                self.obstacle_counter     = 0
                return

        # ── CASE 2: No path yet — plan initial ───────────────────
        if self.current_planned_path is None:
            self.get_logger().info(
                '[plan_loop] CASE 2: current_planned_path is None — planning initial segment')
            local_wp = self._select_local_goal()
            if local_wp is None: return
            gx = local_wp.pose.position.x
            gy = local_wp.pose.position.y
            gz = local_wp.pose.position.z if 0.5 <= local_wp.pose.position.z <= 4.0 else 2.0

            if self._path_clear(sx, sy, gx, gy):
                self._publish_global_segment(sx, sy, sz)
            else:
                path = self._bi_rrt_star(sx, sy, sz, gx, gy, gz)
                if path is not None:
                    # Record trigger position in obstacle memory
                    self.obstacle_memory.append((sx, sy, now))
                    if len(self.obstacle_memory) > OBSTACLE_MEMORY_MAX:
                        self.obstacle_memory.pop(0)
                    self._publish_rrt_path(path)
                    self.last_rrt_time = now
                    self.path_is_rrt   = True
            self.obstacle_counter = 0
            return

        # ── CASE 3: Following global path — check for obstacles ──
        blocked = self._stored_path_blocked(sx, sy)

        if not blocked:
            self.obstacle_counter = 0
            self.stuck_start = None
            return  # path clear — do nothing

        # Obstacle detected
        self.obstacle_counter += 1
        self.get_logger().info(
            f'Obstacle on path {self.obstacle_counter}/{OBSTACLE_CONFIRM}',
            throttle_duration_sec=1.0)

        if self.obstacle_counter < OBSTACLE_CONFIRM:
            return  # wait for persistence

        # Confirmed obstacle — check cooldown
        if now - self.last_rrt_time < RRT_COOLDOWN:
            return

        # Run RRT*
        self.get_logger().warn('Confirmed obstacle — running RRT*')
        self.obstacle_counter = 0
        # Record trigger position so post-detour path avoids this area
        self.obstacle_memory.append((sx, sy, now))
        if len(self.obstacle_memory) > OBSTACLE_MEMORY_MAX:
            self.obstacle_memory.pop(0)

        local_wp = self._select_local_goal()
        if local_wp is None: return
        gx = local_wp.pose.position.x
        gy = local_wp.pose.position.y
        gz = local_wp.pose.position.z if 0.5 <= local_wp.pose.position.z <= 4.0 else 2.0

        path = self._bi_rrt_star(sx, sy, sz, gx, gy, gz)

        if path is not None:
            self._publish_rrt_path(path)
            self.last_rrt_time   = now
            self.path_is_rrt     = True
            self.stuck_start     = None
            self.recovery_active = False
        else:
            self.get_logger().warn('RRT* failed')
            if self.stuck_start is None:
                self.stuck_start = now
            elif now - self.stuck_start > STUCK_TIMEOUT:
                if not self.recovery_active:
                    msg = Bool(); msg.data = True
                    self.map_reset_pub.publish(msg)
                    msg2 = Bool(); msg2.data = True
                    self.force_update_pub.publish(msg2)
                    self.recovery_active = True
                    self.recovery_start  = now
                    self.stuck_start     = None
                else:
                    if now - self.recovery_start > self.RECOVERY_WAIT:
                        msg = Bool(); msg.data = True
                        self.stuck_pub.publish(msg)
                        self.recovery_active      = False
                        self.current_planned_path = None

    # ── Publishers ────────────────────────────────────────────────

    def _publish_global_segment(self, sx, sy, sz):
        # ── Collect raw waypoints as (x,y,z) tuples ──────────────
        raw_wps = []
        for i in range(self.global_idx, len(self.global_path)):
            wp = self.global_path[i]
            p  = wp.pose.position
            raw_wps.append((p.x, p.y, p.z if p.z != 0.0 else sz))
            dx = p.x - sx
            dy = p.y - sy
            if math.sqrt(dx*dx + dy*dy) > LOCAL_GOAL_DIST * 1.5:
                break

        if not raw_wps:
            self.get_logger().warn('[_publish_global_segment] Empty segment — skipping')
            return

        # ── Smooth the global segment (corner rounding, 0.8 m spacing) ──
        smoothed = self._smooth_path(raw_wps, spacing=SMOOTH_GLOBAL_SPACING)

        # ── Build Path message from smoothed tuples ───────────────
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        for (wx, wy, wz) in smoothed:
            ps = PoseStamped()
            ps.header              = msg.header
            ps.pose.position.x     = float(wx)
            ps.pose.position.y     = float(wy)
            ps.pose.position.z     = float(max(Z_MIN, min(Z_MAX, wz)))
            ps.pose.orientation.w  = 1.0
            msg.poses.append(ps)

        # ── Deduplication: suppress if endpoint + length unchanged ─
        if self.current_planned_path is not None and not self.path_is_rrt:
            old_end  = self.current_planned_path.poses[-1].pose.position
            new_end  = msg.poses[-1].pose.position
            end_dx   = abs(new_end.x - old_end.x)
            end_dy   = abs(new_end.y - old_end.y)
            len_diff = abs(len(msg.poses) - len(self.current_planned_path.poses))
            if end_dx < 0.10 and end_dy < 0.10 and len_diff <= 1:
                self.get_logger().debug(
                    f'[_publish_global_segment] Segment unchanged — suppressed '
                    f'(len={len(msg.poses)} '
                    f'end=({new_end.x:.2f},{new_end.y:.2f}))')
                self.current_planned_path = msg  # refresh poses, no publish
                return

        self.get_logger().info(
            f'[_publish_global_segment] PUBLISHING {len(msg.poses)}-wp segment '
            f'(raw={len(raw_wps)}) '
            f'end=({msg.poses[-1].pose.position.x:.2f},'
            f'{msg.poses[-1].pose.position.y:.2f})')
        self.current_planned_path = msg
        self.path_pub.publish(msg)

    def _publish_rrt_path(self, waypoints):
        # Pipeline: smooth → enforce safety margin → publish
        n_raw = len(waypoints)
        waypoints = self._smooth_path(waypoints, spacing=SMOOTH_RRT_SPACING)
        waypoints = self._enforce_safety_margin(waypoints, margin=SAFETY_MARGIN)
        if len(waypoints) < 2:
            self.get_logger().warn(
                f'[_publish_rrt_path] Post-processing reduced {n_raw}-wp path '
                f'to {len(waypoints)} — skipping publish')
            return
        self.get_logger().info(
            f'[_publish_rrt_path] raw={n_raw} → smoothed={len(waypoints)} waypoints')

        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        for wx, wy, wz in waypoints:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.position.z = float(max(Z_MIN, min(Z_MAX, wz)))
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.get_logger().info(
            f'[_publish_rrt_path] PUBLISHING RRT* path: {len(msg.poses)} waypoints '
            f'end=({msg.poses[-1].pose.position.x:.2f},'
            f'{msg.poses[-1].pose.position.y:.2f})')
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