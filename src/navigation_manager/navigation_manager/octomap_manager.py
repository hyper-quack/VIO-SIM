#!/usr/bin/env python3
"""
octomap_manager.py — 3D volumetric mapping using Open3D

Replaces obstacle_detector.py with:
  - Open3D voxel-based 3D map (no fixed grid size)
  - Rolling window: only keeps voxels within WINDOW_RADIUS of drone
  - Evidence accumulation per voxel (like SLAM map)
  - Distance field via KDTree on occupied voxels
  - Trust weight input for RL depth filter integration
  - Backward-compatible /costmap, /voxel_map, /obstacle_distances

Architecture:
  depth_filter → /pointcloud/filtered → THIS NODE → /costmap (A*)
                                                   → /voxel_map (RViz)
                                                   → /obstacle_distances (safety)
                                                   → /distance_field (future RRT*)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField, LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray, Float32, Bool
from px4_msgs.msg import VehicleOdometry
import struct

try:
    from navigation_manager.config import SPAWN_X, SPAWN_Y
except ImportError:
    SPAWN_X, SPAWN_Y = 1.0, 3.0


# ═══════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════

# Rolling window
WINDOW_RADIUS       = 6.0       # metres — keep voxels within this radius
PRUNE_INTERVAL      = 2.0       # seconds — how often to prune old voxels

# Voxel resolution
VOXEL_SIZE          = 0.10      # metres per voxel (same as old grid)

# Evidence accumulation
MARK_INCREMENT      = 20.0      # evidence added per point hit
FREE_DECREMENT      = 8.0       # evidence removed per free observation
CONFIRM_THRESHOLD   = 60.0      # evidence needed to confirm obstacle
FREE_THRESHOLD      = 10.0      # evidence below this → remove voxel
MAX_EVIDENCE        = 200.0     # cap evidence accumulation

# Safety filtering
SELF_EXCLUSION_RADIUS = 1.5     # metres — ignore points near drone
MIN_POINTS_PER_FRAME  = 50      # skip frames with too few points
Z_MIN               = 0.8       # metres — minimum obstacle altitude
Z_MAX               = 3.0       # metres — maximum obstacle altitude

# Costmap output (backward compatibility with A*)
COSTMAP_RESOLUTION  = 0.10      # metres per cell
COSTMAP_WIDTH       = 200       # cells (covers WINDOW_RADIUS * 2)
COSTMAP_HEIGHT      = 120       # cells
COSTMAP_Z_MIN       = 0.5       # only project obstacles in this Z band
COSTMAP_Z_MAX       = 2.8       # into the 2D costmap

# Update rate
UPDATE_RATE          = 10.0     # Hz — publish rate


class VoxelData:
    """Stores evidence and state for each voxel."""
    __slots__ = ['evidence', 'confirmed']

    def __init__(self):
        self.evidence = 0.0
        self.confirmed = False


class OctomapManager(Node):

    def __init__(self):
        super().__init__('octomap_manager')

        # ── QoS profiles ─────────────────────────────────────────────
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)

        qos_besteffort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        # ── Voxel map ────────────────────────────────────────────────
        # Key: (gx, gy, gz) tuple — grid coordinates
        # Value: VoxelData with evidence accumulation
        self.voxels = {}

        # ── Drone state ──────────────────────────────────────────────
        self.drone_x   = None
        self.drone_y   = None
        self.drone_z   = 0.0
        self.drone_yaw = 0.0

        # ── Trust weight (RL filter will set this later) ─────────────
        self.trust_weight = 1.0     # 1.0 = full trust, 0.0 = skip

        # ── Safety distances ─────────────────────────────────────────
        self.front_dist       = float('inf')
        self.left_dist        = float('inf')
        self.right_dist       = float('inf')
        self.depth_front_dist = float('inf')
        self.depth_left_dist  = float('inf')
        self.depth_right_dist = float('inf')

        # ── Last prune time ──────────────────────────────────────────
        self.last_prune_time = 0.0

        # ── Force update state ───────────────────────────────────────
        self.force_update_active  = False
        self.force_update_counter = 0

        # ── Subscribers ──────────────────────────────────────────────
        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.odom_cb, qos_px4)

        self.create_subscription(
            Bool, '/force_update',
            self.force_update_cb, 10)

        self.create_subscription(
            PointCloud2, '/pointcloud/filtered',
            self.cloud_cb, qos_besteffort)

        self.create_subscription(
            LaserScan, '/front_lidar/scan',
            self.front_cb, qos_besteffort)

        self.create_subscription(
            LaserScan, '/left_lidar/scan',
            self.left_cb, qos_besteffort)

        self.create_subscription(
            LaserScan, '/right_lidar/scan',
            self.right_cb, qos_besteffort)

        # Trust weight from RL filter (future)
        self.create_subscription(
            Float32, '/depth_trust_weight',
            self.trust_cb, 10)

        # ── Publishers ───────────────────────────────────────────────
        self.costmap_pub   = self.create_publisher(OccupancyGrid, '/costmap', 10)
        self.voxel_pub     = self.create_publisher(PointCloud2, '/voxel_map', 10)
        self.distances_pub = self.create_publisher(Float32MultiArray, '/obstacle_distances', 10)

        # ── Timer ────────────────────────────────────────────────────
        self.create_timer(1.0 / UPDATE_RATE, self.update_and_publish)

        self.get_logger().info(
            f'OctomapManager started ✓  '
            f'voxel={VOXEL_SIZE}m, window={WINDOW_RADIUS}m, '
            f'confirm={CONFIRM_THRESHOLD}, '
            f'Open3D backed')

    # ═══════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════

    def _wrap_angle(self, a):
        while a > math.pi:  a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a

    def _world_to_voxel(self, wx, wy, wz):
        """Convert world coordinate to voxel grid key."""
        gx = int(math.floor(wx / VOXEL_SIZE))
        gy = int(math.floor(wy / VOXEL_SIZE))
        gz = int(math.floor(wz / VOXEL_SIZE))
        return (gx, gy, gz)

    def _voxel_to_world(self, gx, gy, gz):
        """Convert voxel key to world coordinate (center of voxel)."""
        wx = (gx + 0.5) * VOXEL_SIZE
        wy = (gy + 0.5) * VOXEL_SIZE
        wz = (gz + 0.5) * VOXEL_SIZE
        return wx, wy, wz

    def _parse_pointcloud2(self, msg):
        """Fast PointCloud2 parsing to numpy array."""
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

        # Fast numpy parsing
        points = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            base = i * step
            points[i, 0] = struct.unpack_from('f', data, base + x_off)[0]
            points[i, 1] = struct.unpack_from('f', data, base + y_off)[0]
            points[i, 2] = struct.unpack_from('f', data, base + z_off)[0]

        # Filter NaN/inf
        valid = np.all(np.isfinite(points), axis=1)
        points = points[valid]

        return points if len(points) > 0 else None

    # ═══════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════

    def odom_cb(self, msg):
        """PX4 NED → World frame."""
        self.drone_x = float(msg.position[1]) + SPAWN_X
        self.drone_y = float(msg.position[0]) + SPAWN_Y
        self.drone_z = -float(msg.position[2])

        q = msg.q
        siny = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy = 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
        self.drone_yaw = self._wrap_angle(
            math.atan2(siny, cosy) - math.pi / 2.0)

    def trust_cb(self, msg):
        """Receive trust weight from RL depth filter."""
        self.trust_weight = max(0.0, min(1.0, float(msg.data)))

    def front_cb(self, msg):
        r = [x for x in msg.ranges if math.isfinite(x) and x > 0.05]
        self.front_dist = min(r) if r else float('inf')

    def left_cb(self, msg):
        r = [x for x in msg.ranges if math.isfinite(x) and x > 0.05]
        self.left_dist = min(r) if r else float('inf')

    def right_cb(self, msg):
        r = [x for x in msg.ranges if math.isfinite(x) and x > 0.05]
        self.right_dist = min(r) if r else float('inf')

    # ═══════════════════════════════════════════════════════════════
    # Point cloud integration
    # ═══════════════════════════════════════════════════════════════

    def force_update_cb(self, msg):
        """Temporarily lower confirm threshold to update map faster when stuck."""
        if msg.data:
            self.force_update_active = True
            self.force_update_counter = 100  # ~10s at 10Hz
            self.get_logger().warn('Force map update active')
        else:
            self.force_update_active = False

    def cloud_cb(self, msg):
        """Integrate filtered point cloud into voxel map."""
        # Decrement force update counter once per frame
        if self.force_update_active and self.force_update_counter > 0:
            self.force_update_counter -= 1
            if self.force_update_counter == 0:
                self.force_update_active = False
                self.get_logger().info('Force update complete — normal threshold restored')

        if self.drone_x is None:
            return

        # ── Apply trust weight ───────────────────────────────────
        if self.trust_weight < 0.05:
            # RL filter says: skip this frame entirely
            return

        points = self._parse_pointcloud2(msg)
        if points is None or len(points) < MIN_POINTS_PER_FRAME:
            return

        drone_pos = np.array([self.drone_x, self.drone_y, self.drone_z])

        # ── Scale evidence by trust weight ───────────────────────
        mark_inc = MARK_INCREMENT * self.trust_weight
        free_dec = FREE_DECREMENT * self.trust_weight

        # ── Drone voxel for ray casting origin ───────────────────
        drone_voxel = self._world_to_voxel(
            self.drone_x, self.drone_y, self.drone_z)

        # ── Distance tracking for safety ─────────────────────────
        front_min = float('inf')
        left_min  = float('inf')
        right_min = float('inf')

        # ── Process each point ───────────────────────────────────
        for i in range(len(points)):
            wx, wy, wz = points[i]

            # Self exclusion
            dist_2d = math.sqrt(
                (wx - self.drone_x) ** 2 + (wy - self.drone_y) ** 2)
            if dist_2d < SELF_EXCLUSION_RADIUS:
                continue

            # Rolling window — ignore points far from drone
            dist_3d = math.sqrt(
                (wx - self.drone_x) ** 2 +
                (wy - self.drone_y) ** 2 +
                (wz - self.drone_z) ** 2)
            if dist_3d > WINDOW_RADIUS:
                continue

            # Altitude filter
            if wz < Z_MIN or wz > Z_MAX:
                continue

            # ── Mark obstacle voxel ──────────────────────────────
            vkey = self._world_to_voxel(wx, wy, wz)
            if vkey not in self.voxels:
                self.voxels[vkey] = VoxelData()
            vd = self.voxels[vkey]
            vd.evidence = min(MAX_EVIDENCE, vd.evidence + mark_inc)
            # Lower threshold when force update active
            threshold = CONFIRM_THRESHOLD * 0.4 if self.force_update_active else CONFIRM_THRESHOLD
            if vd.evidence >= threshold:
                vd.confirmed = True

            # ── Ray casting: free-space clearing ─────────────────
            # Simple fast version: sample points along ray
            ray_len = dist_3d
            n_samples = max(1, int(ray_len / VOXEL_SIZE))
            for s in range(n_samples):
                t = s / n_samples
                rx = self.drone_x + t * (wx - self.drone_x)
                ry = self.drone_y + t * (wy - self.drone_y)
                rz = self.drone_z + t * (wz - self.drone_z)
                fkey = self._world_to_voxel(rx, ry, rz)
                if fkey == vkey:
                    break   # don't clear the obstacle voxel itself
                if fkey in self.voxels:
                    fd = self.voxels[fkey]
                    fd.evidence = max(0.0, fd.evidence - free_dec)
                    if fd.evidence < FREE_THRESHOLD:
                        fd.confirmed = False

            # ── Zone distances for safety ────────────────────────
            body_x = wx - self.drone_x
            body_y = wy - self.drone_y
            cos_y = math.cos(-self.drone_yaw)
            sin_y = math.sin(-self.drone_yaw)
            fwd  =  cos_y * body_x + sin_y * body_y
            side = -sin_y * body_x + cos_y * body_y

            if fwd > 0:
                if side < -0.15:
                    left_min = min(left_min, dist_2d)
                elif side > 0.15:
                    right_min = min(right_min, dist_2d)
                else:
                    front_min = min(front_min, dist_2d)

        self.depth_front_dist = front_min
        self.depth_left_dist  = left_min
        self.depth_right_dist = right_min

    # ═══════════════════════════════════════════════════════════════
    # Rolling window pruning
    # ═══════════════════════════════════════════════════════════════

    def _prune_voxels(self):
        """Remove voxels outside rolling window."""
        if self.drone_x is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_prune_time < PRUNE_INTERVAL:
            return
        self.last_prune_time = now

        drone_pos = np.array([self.drone_x, self.drone_y, self.drone_z])
        radius_sq = WINDOW_RADIUS ** 2

        keys_to_remove = []
        for vkey in self.voxels:
            wx, wy, wz = self._voxel_to_world(*vkey)
            dx = wx - drone_pos[0]
            dy = wy - drone_pos[1]
            dz = wz - drone_pos[2]
            if dx * dx + dy * dy + dz * dz > radius_sq:
                keys_to_remove.append(vkey)

        for k in keys_to_remove:
            del self.voxels[k]

        if len(keys_to_remove) > 0:
            self.get_logger().debug(
                f'Pruned {len(keys_to_remove)} voxels, '
                f'{len(self.voxels)} remaining')

    # ═══════════════════════════════════════════════════════════════
    # Also remove low-evidence unconfirmed voxels periodically
    # ═══════════════════════════════════════════════════════════════

    def _cleanup_weak_voxels(self):
        """Remove voxels with very low evidence (noise cleanup)."""
        keys_to_remove = [
            k for k, v in self.voxels.items()
            if v.evidence < FREE_THRESHOLD and not v.confirmed
        ]
        for k in keys_to_remove:
            del self.voxels[k]

    # ═══════════════════════════════════════════════════════════════
    # Publishers
    # ═══════════════════════════════════════════════════════════════

    def update_and_publish(self):
        """Main publish loop at UPDATE_RATE Hz."""
        self._prune_voxels()
        self._cleanup_weak_voxels()
        self._publish_costmap()
        self._publish_voxel_map()
        self._publish_distances()

    def _publish_costmap(self):
        """
        Project confirmed 3D voxels into a 2D OccupancyGrid.
        The grid is centered on the drone and moves with it.
        Backward compatible with A* planner.
        """
        if self.drone_x is None:
            return

        # Costmap centered on drone
        origin_x = self.drone_x - (COSTMAP_WIDTH * COSTMAP_RESOLUTION) / 2.0
        origin_y = self.drone_y - (COSTMAP_HEIGHT * COSTMAP_RESOLUTION) / 2.0

        grid = np.zeros((COSTMAP_HEIGHT, COSTMAP_WIDTH), dtype=np.int8)

        for vkey, vd in self.voxels.items():
            if not vd.confirmed:
                continue

            wx, wy, wz = self._voxel_to_world(*vkey)

            # Only project obstacles in flight altitude band
            if wz < COSTMAP_Z_MIN or wz > COSTMAP_Z_MAX:
                continue

            # Convert to local costmap cell
            cx = int((wx - origin_x) / COSTMAP_RESOLUTION)
            cy = int((wy - origin_y) / COSTMAP_RESOLUTION)

            if 0 <= cx < COSTMAP_WIDTH and 0 <= cy < COSTMAP_HEIGHT:
                grid[cy, cx] = 100

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info.resolution = COSTMAP_RESOLUTION
        msg.info.width  = COSTMAP_WIDTH
        msg.info.height = COSTMAP_HEIGHT
        msg.info.origin = Pose()
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self.costmap_pub.publish(msg)

    def _publish_voxel_map(self):
        """Publish confirmed voxels as PointCloud2 for RViz."""
        confirmed_voxels = [
            k for k, v in self.voxels.items() if v.confirmed
        ]

        if len(confirmed_voxels) == 0:
            return

        pts = np.zeros((len(confirmed_voxels), 3), dtype=np.float32)
        for i, vkey in enumerate(confirmed_voxels):
            pts[i, 0], pts[i, 1], pts[i, 2] = self._voxel_to_world(*vkey)

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
        self.voxel_pub.publish(msg)

    def _publish_distances(self):
        """Fuse LiDAR + depth distances for safety layer."""
        fused_front = min(self.front_dist, self.depth_front_dist)
        fused_left  = min(self.left_dist,  self.depth_left_dist)
        fused_right = min(self.right_dist, self.depth_right_dist)

        msg = Float32MultiArray()
        msg.data = [
            float(fused_front), float(fused_left), float(fused_right),
            float(self.depth_front_dist),
            float(self.depth_left_dist),
            float(self.depth_right_dist),
        ]
        self.distances_pub.publish(msg)
        self.get_logger().info(
            f'Voxels={len(self.voxels)} confirmed='
            f'{sum(1 for v in self.voxels.values() if v.confirmed)} '
            f'F={fused_front:.2f} L={fused_left:.2f} R={fused_right:.2f}',
            throttle_duration_sec=3.0)


def main(args=None):
    rclpy.init(args=args)
    node = OctomapManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()