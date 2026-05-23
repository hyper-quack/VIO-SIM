#!/usr/bin/env python3
"""
pose_graph.py — GTSAM iSAM2 Pose Graph Optimizer

Pipeline:
  /slam/pose       → odometry edges (6 Hz from slam_frontend)
  /slam/loop_edge  → loop closure constraints (from loop_closure node)
        ↓
  GTSAM iSAM2 incremental optimizer
        ↓
  /slam/corrected_pose  → octomap_manager (drift-corrected pose)

Coordinate convention (same as all other nodes):
  world_x = px4_position[1] + SPAWN_X   (PX4 East  → World X)
  world_y = px4_position[0] + SPAWN_Y   (PX4 North → World Y)
  world_z = lidar range                  (MTF-01 altitude)
  yaw     = atan2(siny,cosy) - pi/2     (GPS mode)

Notes:
  - Only XY + Yaw are optimized (Z always from LiDAR, trusted)
  - Loop edge encoding: position = query pose, orientation.xyzw = match pose
    (compact encoding from loop_closure.py — NOT a unit quaternion)
  - iSAM2 re-linearizes only changed variables → efficient for long trajectories
  - GPS navigation is UNTOUCHED — this only feeds octomap_manager
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

try:
    import gtsam
    from gtsam import (
        ISAM2, ISAM2Params,
        NonlinearFactorGraph, Values,
        Pose2, PriorFactorPose2, BetweenFactorPose2,
        noiseModel
    )
    GTSAM_AVAILABLE = False
except ImportError:
    GTSAM_AVAILABLE = False

# ── iSAM2 parameters ──────────────────────────────────────────────────────────
RELINEARIZE_THRESHOLD  = 0.1   # re-linearize variable if delta > this
RELINEARIZE_SKIP       = 10    # check every N updates

# ── Noise models (tuned for corridor navigation) ───────────────────────────────
# Odometry noise: [x, y, theta] standard deviations
ODOM_NOISE_X     = 0.05   # metres per step
ODOM_NOISE_Y     = 0.05
ODOM_NOISE_THETA = 0.02   # radians per step

# Loop closure noise: tighter than odometry (verified by RANSAC)
LOOP_NOISE_X     = 0.10
LOOP_NOISE_Y     = 0.10
LOOP_NOISE_THETA = 0.05

# Prior noise on first pose (very tight — we know spawn point)
PRIOR_NOISE_X     = 0.001
PRIOR_NOISE_Y     = 0.001
PRIOR_NOISE_THETA = 0.001

# ── Graph parameters ──────────────────────────────────────────────────────────
MIN_POSE_DIST    = 0.20   # metres — minimum distance to add new node
MAX_GRAPH_SIZE   = 2000   # maximum nodes before pruning oldest
OPTIMIZE_EVERY   = 5      # optimize every N new nodes

# ── Spawn offset ──────────────────────────────────────────────────────────────
SPAWN_X = 1.0
SPAWN_Y = 3.0


def _wrap_angle(a: float) -> float:
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def _pose2_between(p1, p2):
    """
    Compute relative Pose2 between two absolute poses.
    p1, p2: (x, y, theta)
    Returns Pose2 representing p2 in frame of p1.
    """
    dx    = p2[0] - p1[0]
    dy    = p2[1] - p1[1]
    dtheta = _wrap_angle(p2[2] - p1[2])

    cos_t = math.cos(p1[2])
    sin_t = math.sin(p1[2])

    # Rotate dx, dy into p1 frame
    local_x =  cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy

    return local_x, local_y, dtheta


class PoseGraph(Node):

    def __init__(self):
        super().__init__('pose_graph')

        if not GTSAM_AVAILABLE:
            self.get_logger().error(
                'GTSAM not available! Install with: pip install gtsam --break-system-packages')
            self.get_logger().error(
                'Running in PASSTHROUGH mode — /slam/corrected_pose = /slam/pose')

        # ── QoS ───────────────────────────────────────────────────────────────
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, '/slam/pose',
            self._pose_cb, qos_reliable)

        self.create_subscription(
            PoseStamped, '/slam/loop_edge',
            self._loop_edge_cb, qos_reliable)

        # ── Publishers ────────────────────────────────────────────────────────
        self.corrected_pub = self.create_publisher(
            PoseStamped, '/slam/corrected_pose', qos_reliable)
        self.debug_pub = self.create_publisher(
            String, '/slam/graph_debug', qos_reliable)

        # ── GTSAM iSAM2 ───────────────────────────────────────────────────────
        if GTSAM_AVAILABLE:
            params = ISAM2Params()
            params.setRelinearizeThreshold(RELINEARIZE_THRESHOLD)
            params.relinearizeSkip      = RELINEARIZE_SKIP
            self.isam2 = ISAM2(params)

            self.odom_noise = noiseModel.Diagonal.Sigmas(
                np.array([ODOM_NOISE_X, ODOM_NOISE_Y, ODOM_NOISE_THETA]))
            self.loop_noise = noiseModel.Diagonal.Sigmas(
                np.array([LOOP_NOISE_X, LOOP_NOISE_Y, LOOP_NOISE_THETA]))
            self.prior_noise = noiseModel.Diagonal.Sigmas(
                np.array([PRIOR_NOISE_X, PRIOR_NOISE_Y, PRIOR_NOISE_THETA]))

        # ── Graph state ───────────────────────────────────────────────────────
        self.node_id        = 0          # current node index
        self.nodes          = []         # list of (x, y, z, yaw) raw poses
        self.corrected      = []         # list of (x, y, z, yaw) optimized poses
        self.last_pose      = None       # last added raw pose
        self.initialized    = False
        self.new_nodes      = 0          # counter for optimize_every

        # Z history (Z not optimized — kept as-is from LiDAR)
        self.z_history      = []

        self.loop_count     = 0
        self.optimize_count = 0

        self.get_logger().info('Pose Graph node started ✓')
        if GTSAM_AVAILABLE:
            self.get_logger().info('GTSAM iSAM2 ready')
        else:
            self.get_logger().warn('PASSTHROUGH mode — no optimization')

    # ═════════════════════════════════════════════════════════════════════════
    # Pose callback — main odometry stream
    # ═════════════════════════════════════════════════════════════════════════

    def _pose_cb(self, msg: PoseStamped):
        """
        Receive raw SLAM pose from slam_frontend.
        Add as new graph node if moved enough.
        Optimize every OPTIMIZE_EVERY nodes.
        Publish corrected pose.
        """
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)

        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        yaw = _wrap_angle(2.0 * math.atan2(qz, qw))

        raw_pose = (x, y, z, yaw)

        # ── First pose: initialize graph ──────────────────────────────────────
        if not self.initialized:
            self._initialize(raw_pose)
            return

        # ── Check minimum distance to add new node ────────────────────────────
        dist = math.sqrt(
            (x - self.last_pose[0]) ** 2 +
            (y - self.last_pose[1]) ** 2)

        if dist < MIN_POSE_DIST:
            # Not far enough — just publish last corrected pose
            self._publish_corrected(msg.header.stamp)
            return

        # ── Add new node ──────────────────────────────────────────────────────
        self._add_odometry_node(raw_pose)

        # ── Optimize periodically ─────────────────────────────────────────────
        self.new_nodes += 1
        if self.new_nodes >= OPTIMIZE_EVERY:
            self._optimize()
            self.new_nodes = 0

        # ── Publish ───────────────────────────────────────────────────────────
        self._publish_corrected(msg.header.stamp)

    # ═════════════════════════════════════════════════════════════════════════
    # Loop edge callback
    # ═════════════════════════════════════════════════════════════════════════

    def _loop_edge_cb(self, msg: PoseStamped):
        """
        Receive confirmed loop closure from loop_closure node.

        Encoding (from loop_closure.py _publish_loop_edge):
          position    = query keyframe pose (x, y, z)
          orientation = match keyframe pose encoded as:
            .x = match_x
            .y = match_y
            .z = match_z
            .w = match_yaw
        This is NOT a unit quaternion — it's a compact 4-field encoding.
        """
        if not GTSAM_AVAILABLE:
            return
        if len(self.nodes) < 2:
            return

        # Unpack query pose
        query_x   = float(msg.pose.position.x)
        query_y   = float(msg.pose.position.y)
        query_yaw = _wrap_angle(2.0 * math.atan2(
            msg.pose.orientation.z, msg.pose.orientation.w))

        # Unpack match pose (compact encoding)
        match_x   = float(msg.pose.orientation.x)
        match_y   = float(msg.pose.orientation.y)
        match_yaw = float(msg.pose.orientation.w)

        # Find closest graph nodes to query and match poses
        query_node = self._find_closest_node(query_x, query_y)
        match_node = self._find_closest_node(match_x, match_y)

        if query_node is None or match_node is None:
            return
        if query_node == match_node:
            return

        query_id = query_node
        match_id = match_node

        # Compute relative pose between the two nodes
        q_pose = self.nodes[query_id]
        m_pose = self.nodes[match_id]

        lx, ly, ltheta = _pose2_between(
            (m_pose[0], m_pose[1], m_pose[3]),
            (q_pose[0], q_pose[1], q_pose[3]))

        # Add loop constraint to graph
        graph = NonlinearFactorGraph()
        graph.add(BetweenFactorPose2(
            query_id, match_id,
            Pose2(lx, ly, ltheta),
            self.loop_noise))

        try:
            self.isam2.update(graph, Values())
            self.loop_count += 1
            self.get_logger().info(
                f'Loop edge added: node {query_id} → node {match_id} '
                f'total_loops={self.loop_count}')

            # Force optimization after loop closure
            self._optimize()
            self._update_corrected_from_isam2()

        except Exception as e:
            self.get_logger().warn(f'Loop edge failed: {e}')

    # ═════════════════════════════════════════════════════════════════════════
    # Graph operations
    # ═════════════════════════════════════════════════════════════════════════

    def _initialize(self, raw_pose):
        """Add first node with prior factor."""
        x, y, z, yaw = raw_pose

        self.nodes.append(raw_pose)
        self.corrected.append(raw_pose)
        self.z_history.append(z)
        self.last_pose = raw_pose
        self.initialized = True

        if GTSAM_AVAILABLE:
            graph  = NonlinearFactorGraph()
            values = Values()

            prior = Pose2(x, y, yaw)
            graph.add(PriorFactorPose2(0, prior, self.prior_noise))
            values.insert(0, prior)

            try:
                self.isam2.update(graph, values)
                self.get_logger().info(
                    f'Graph initialized at ({x:.2f}, {y:.2f}, yaw={math.degrees(yaw):.1f}°)')
            except Exception as e:
                self.get_logger().warn(f'iSAM2 init failed: {e}')

        self.node_id = 1

    def _add_odometry_node(self, raw_pose):
        """Add new node connected to previous by odometry edge."""
        x, y, z, yaw = raw_pose
        prev_pose = self.nodes[-1]

        self.nodes.append(raw_pose)
        self.corrected.append(raw_pose)   # will be updated after optimization
        self.z_history.append(z)
        self.last_pose = raw_pose

        # Prune if too large
        if len(self.nodes) > MAX_GRAPH_SIZE:
            self.nodes.pop(0)
            self.corrected.pop(0)
            self.z_history.pop(0)

        if not GTSAM_AVAILABLE:
            self.node_id += 1
            return

        # Compute relative odometry
        lx, ly, ltheta = _pose2_between(
            (prev_pose[0], prev_pose[1], prev_pose[3]),
            (x, y, yaw))

        prev_id = self.node_id - 1
        curr_id = self.node_id

        graph  = NonlinearFactorGraph()
        values = Values()

        graph.add(BetweenFactorPose2(
            prev_id, curr_id,
            Pose2(lx, ly, ltheta),
            self.odom_noise))

        values.insert(curr_id, Pose2(x, y, yaw))

        try:
            self.isam2.update(graph, values)
        except Exception as e:
            self.get_logger().warn(
                f'iSAM2 update failed at node {curr_id}: {e}',
                throttle_duration_sec=2.0)

        self.node_id += 1

    def _optimize(self):
        """Run iSAM2 optimization and update corrected poses."""
        if not GTSAM_AVAILABLE:
            return

        try:
            # iSAM2 optimizes incrementally — just call update with empty graph
            self.isam2.update()
            self._update_corrected_from_isam2()
            self.optimize_count += 1

            self.get_logger().info(
                f'iSAM2 optimized: nodes={len(self.nodes)} '
                f'loops={self.loop_count} '
                f'optimizations={self.optimize_count}',
                throttle_duration_sec=5.0)

        except Exception as e:
            self.get_logger().warn(f'iSAM2 optimize failed: {e}')

    def _update_corrected_from_isam2(self):
        """Extract optimized poses from iSAM2 and update corrected list."""
        if not GTSAM_AVAILABLE:
            return

        try:
            result = self.isam2.calculateEstimate()
            n = min(len(self.nodes), self.node_id)
            start_id = self.node_id - len(self.nodes)

            for i in range(len(self.nodes)):
                node_id = start_id + i
                if node_id < 0:
                    continue
                try:
                    p2 = result.atPose2(node_id)
                    z  = self.z_history[i]
                    self.corrected[i] = (p2.x(), p2.y(), z,
                                         _wrap_angle(p2.theta()))
                except Exception:
                    pass  # node not yet in result

        except Exception as e:
            self.get_logger().warn(
                f'Extract corrected poses failed: {e}',
                throttle_duration_sec=5.0)

    def _find_closest_node(self, x, y):
        """Find index of graph node closest to (x, y)."""
        if not self.nodes:
            return None

        best_idx  = 0
        best_dist = float('inf')

        for i, node in enumerate(self.nodes):
            d = math.sqrt((node[0] - x) ** 2 + (node[1] - y) ** 2)
            if d < best_dist:
                best_dist = d
                best_idx  = i

        return best_idx if best_dist < 2.0 else None

    # ═════════════════════════════════════════════════════════════════════════
    # Publisher
    # ═════════════════════════════════════════════════════════════════════════

    def _publish_corrected(self, stamp):
        """Publish latest corrected pose."""
        if not self.corrected:
            return

        cx, cy, cz, cyaw = self.corrected[-1]

        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'odom'
        msg.pose.position.x = float(cx)
        msg.pose.position.y = float(cy)
        msg.pose.position.z = float(cz)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(cyaw / 2.0)
        msg.pose.orientation.w = math.cos(cyaw / 2.0)

        self.corrected_pub.publish(msg)

        # Debug
        raw_x, raw_y = self.last_pose[0], self.last_pose[1]
        drift_x = cx - raw_x
        drift_y = cy - raw_y

        debug = String()
        debug.data = (
            f'nodes={len(self.nodes)} '
            f'raw=({raw_x:.2f},{raw_y:.2f}) '
            f'corrected=({cx:.2f},{cy:.2f}) '
            f'drift=({drift_x:.3f},{drift_y:.3f}) '
            f'loops={self.loop_count}')
        self.debug_pub.publish(debug)

        self.get_logger().info(
            debug.data,
            throttle_duration_sec=3.0)


def main(args=None):
    rclpy.init(args=args)
    node = PoseGraph()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()