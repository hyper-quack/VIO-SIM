#!/usr/bin/env python3
"""
octomap_manager.py — 3D volumetric mapping with sliding window.

Session 15 v2 — "sliding map" approach:
  - New voxels: accumulate evidence with speed scaling
  - Confirmed voxels: position FROZEN, new observations only refresh hits
  - No ray casting (no false free-space correction)
  - No proximity unlock (drone doesn't delete obstacles by proximity)
  - Only removal: voxel beyond 12m window → prune
  - Window centered on drone, slides as drone moves
  - Dual evidence: raw (safety) + stable (planning/lock)

Coordinate convention:
  World X = PX4 East  = corridor length
  World Y = PX4 North = corridor width
  World Z = up (LiDAR altitude)

Camera mount offset (model.sdf):
  x=+0.01233m forward, y=+0.0375m left, z=+0.01878m up
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField, LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float32MultiArray, Float32, Bool
from px4_msgs.msg import VehicleOdometry
from collections import deque
import struct

try:
    from navigation_manager.config import SPAWN_X, SPAWN_Y
except ImportError:
    SPAWN_X, SPAWN_Y = 1.0, 3.0

# ── Camera mount offset (body frame) ─────────────────────────────
CAM_X =  0.01233
CAM_Y =  0.0375
CAM_Z =  0.01878

# ── Sliding window ────────────────────────────────────────────────
WINDOW_RADIUS      = 12.0

# ── Voxel resolution ─────────────────────────────────────────────
VOXEL_SIZE         = 0.20

# ── Temporal consistency ──────────────────────────────────────────
CONSISTENCY_FRAMES = 60

# ── Evidence ──────────────────────────────────────────────────────
MARK_INCREMENT      = 8.0
CONFIRM_THRESHOLD   = 300.0
HIGH_CONF_THRESHOLD = 150.0
MAX_EVIDENCE        = 600.0

# ── Speed scale thresholds ────────────────────────────────────────
SPEED_FULL   = 0.10
SPEED_HALF   = 0.20
SPEED_LOW    = 0.40

# ── Safety filtering ──────────────────────────────────────────────
SELF_EXCLUSION_RADIUS = 0.8
MIN_POINTS_PER_FRAME  = 20
Z_MIN                 = 0.0
Z_MAX                 = 4.0

# ── Costmap ───────────────────────────────────────────────────────
COSTMAP_RESOLUTION = 0.10
COSTMAP_WIDTH      = 200
COSTMAP_HEIGHT     = 120
COSTMAP_Z_MIN      = 0.0
COSTMAP_Z_MAX      = 4.0

# ── Pose history ──────────────────────────────────────────────────
POSE_HISTORY_SIZE  = 100

# ── Update rate ───────────────────────────────────────────────────
UPDATE_RATE        = 10.0

# ── Pruning ───────────────────────────────────────────────────────
PRUNE_INTERVAL     = 1.0


class VoxelData:
    __slots__ = [
        'hits',
        'raw_evidence',
        'stable_evidence',
        'confirmed',
        'high_confidence',
        'world_x',
        'world_y',
        'world_z',
    ]

    def __init__(self):
        self.hits            = 0
        self.raw_evidence    = 0.0
        self.stable_evidence = 0.0
        self.confirmed       = False
        self.high_confidence = False
        self.world_x         = 0.0
        self.world_y         = 0.0
        self.world_z         = 0.0


def _wrap_angle(a):
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a


def _quat_to_rotation_matrix(w, x, y, z):
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)
    return R


def _speed_scale(speed):
    if speed < SPEED_FULL:
        return 1.0
    elif speed < SPEED_HALF:
        return 0.5
    elif speed < SPEED_LOW:
        return 0.2
    else:
        return 0.05
    


class OctomapManager(Node):

    def __init__(self):
        super().__init__('octomap_manager')

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        qos_best = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.voxels = {}
        self.pose_history = deque(maxlen=POSE_HISTORY_SIZE)

        self.drone_x     = None
        self.drone_y     = None
        self.drone_z     = 0.0
        self.drone_yaw   = 0.0
        self.drone_speed = 0.0
        self.lidar_z     = 0.0

        self._prev_x  = None
        self._prev_y  = None
        self._prev_ts = None

        self.slam_pose_available = False

        self.slam_jump_active    = False
        self.slam_jump_counter   = 0
        self.last_slam_x_jump    = None
        self.last_slam_y_jump    = None

        self.trust_weight = 1.0

        self.front_dist  = float('inf')
        self.left_dist   = float('inf')
        self.right_dist  = float('inf')
        self.depth_front = float('inf')
        self.depth_left  = float('inf')
        self.depth_right = float('inf')
        self.raw_pub = self.create_publisher(
            PointCloud2,
            '/debug/raw',
            10
        )

        self.body_pub = self.create_publisher(
            PointCloud2,
            '/debug/body',
            10
        )

        self.world_pub = self.create_publisher(
            PointCloud2,
            '/debug/world',
            10
        )
        self.last_prune_time = 0.0

        self.force_update_active  = False
        self.force_update_counter = 0

        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos_px4)
        self.create_subscription(
            LaserScan, '/mtf01/lidar', self.lidar_cb, 10)
        self.create_subscription(
            PointCloud2, '/pointcloud/camera', self.cloud_cb, qos_best)
        self.create_subscription(
            LaserScan, '/front_lidar/scan', self.front_cb, qos_best)
        self.create_subscription(
            LaserScan, '/left_lidar/scan',  self.left_cb,  qos_best)
        self.create_subscription(
            LaserScan, '/right_lidar/scan', self.right_cb, qos_best)
        self.create_subscription(
            Float32, '/depth_trust_weight', self.trust_cb, 10)
        self.create_subscription(
            PoseStamped, '/slam/corrected_pose', self.slam_pose_cb, qos_reliable)
        self.create_subscription(
            Bool, '/force_update', self.force_update_cb, 10)
        self.create_subscription(
            Bool, '/map_reset', self.map_reset_cb, 10)

        self.costmap_pub   = self.create_publisher(OccupancyGrid, '/costmap', 10)
        self.voxel_pub     = self.create_publisher(PointCloud2, '/voxel_map', 10)
        self.distances_pub = self.create_publisher(Float32MultiArray, '/obstacle_distances', 10)

        self.create_timer(1.0 / UPDATE_RATE, self.update_and_publish)

        self.get_logger().info(
            f'OctomapManager started ✓ (sliding window mode) '
            f'voxel={VOXEL_SIZE}m window={WINDOW_RADIUS}m')

    # ─────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────

    def odom_cb(self, msg):
        x = float(msg.position[1]) + SPAWN_X
        y = float(msg.position[0]) + SPAWN_Y
        z = -float(msg.position[2])

        qw = float(msg.q[0])
        qx = float(msg.q[1])
        qy = float(msg.q[2])
        qz = float(msg.q[3])

        siny = 2.0*(qw*qz + qx*qy)
        cosy = 1.0 - 2.0*(qy*qy + qz*qz)
        yaw  = _wrap_angle(math.atan2(siny, cosy) - math.pi/2.0)

        ts_ns = msg.timestamp * 1000
        ts_s  = ts_ns * 1e-9

        if self._prev_x is not None and self._prev_ts is not None:
            dt = ts_s - self._prev_ts
            if dt > 0.01:
                dx = x - self._prev_x
                dy = y - self._prev_y
                self.drone_speed = math.sqrt(dx*dx + dy*dy) / dt

        self._prev_x  = x
        self._prev_y  = y
        self._prev_ts = ts_s

        self.pose_history.append((ts_ns, x, y, z, qw, qx, qy, qz, yaw))

        if not self.slam_pose_available:
            self.drone_x   = x
            self.drone_y   = y
            self.drone_z   = z
            self.drone_yaw = yaw

    def lidar_cb(self, msg):
        if msg.ranges and math.isfinite(msg.ranges[0]) and msg.ranges[0] > 0.01:
            self.lidar_z = float(msg.ranges[0])
            if self.pose_history:
                entry    = list(self.pose_history[-1])
                entry[3] = self.lidar_z
                self.pose_history[-1] = tuple(entry)

    def slam_pose_cb(self, msg):
        nx = float(msg.pose.position.x)
        ny = float(msg.pose.position.y)
        nz = float(msg.pose.position.z)

        if self.drone_x is not None:
            jump = math.sqrt((nx-self.drone_x)**2 + (ny-self.drone_y)**2)
            if jump > 1.0:
                self.get_logger().warn(
                    f'VIO jump {jump:.2f}m rejected', throttle_duration_sec=2.0)
                return

        self.drone_x = nx
        self.drone_y = ny
        self.drone_z = nz

        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        self.drone_yaw = _wrap_angle(2.0*math.atan2(qz, qw))

        if self.last_slam_x_jump is not None:
            jump = math.sqrt((nx - self.last_slam_x_jump)**2 +
                             (ny - self.last_slam_y_jump)**2)
            if jump > 0.3:
                self.slam_jump_active  = True
                self.slam_jump_counter = 20
                self.get_logger().warn(
                    f'Loop-closure jump {jump:.2f}m — freezing integration')

        self.last_slam_x_jump = nx
        self.last_slam_y_jump = ny
        self.slam_pose_available = True

    def trust_cb(self, msg):
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

    def map_reset_cb(self, msg):
        if msg.data:
            count = len(self.voxels)
            self.voxels.clear()
            self.get_logger().warn(f'Map reset — cleared {count} voxels')

    def force_update_cb(self, msg):
        if msg.data:
            self.force_update_active  = True
            self.force_update_counter = 100
        else:
            self.force_update_active = False

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _get_pose_at(self, ts_ns):
        if not self.pose_history:
            return None
        return min(self.pose_history, key=lambda p: abs(p[0] - ts_ns))

    def _world_to_voxel(self, wx, wy, wz):
        return (int(math.floor(wx / VOXEL_SIZE)),
                int(math.floor(wy / VOXEL_SIZE)),
                int(math.floor(wz / VOXEL_SIZE)))

    def _parse_pointcloud2(self, msg):
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
        pts = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            base      = i * step
            pts[i, 0] = struct.unpack_from('f', data, base + x_off)[0]
            pts[i, 1] = struct.unpack_from('f', data, base + y_off)[0]
            pts[i, 2] = struct.unpack_from('f', data, base + z_off)[0]
        valid = np.all(np.isfinite(pts), axis=1)
        pts   = pts[valid]
        return pts if len(pts) > 0 else None
    def publish_debug_cloud(
        self,
        pub,
        pts,
        frame='odom'
    ):

        if pts is None or len(pts) == 0:
            return

        pts = np.asarray(
            pts,
            dtype=np.float32
        )

        msg = PointCloud2()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame

        msg.height = 1
        msg.width = len(pts)

        msg.fields = [
            PointField(
                name='x',
                offset=0,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name='y',
                offset=4,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name='z',
                offset=8,
                datatype=PointField.FLOAT32,
                count=1
            ),
        ]

        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(pts)
        msg.is_dense = True

        msg.data = pts.tobytes()

        pub.publish(msg)
    # ─────────────────────────────────────────────────────────────
    # Main cloud callback
    # ─────────────────────────────────────────────────────────────

    def cloud_cb(self, msg):
        """
        Sliding window logic:
          - New voxels: accumulate evidence (speed-scaled)
          - Confirmed voxels: position FROZEN, only hits++ on re-observation
          - Removal: ONLY by window prune (beyond 12m)
        """
        if self.force_update_active and self.force_update_counter > 0:
            self.force_update_counter -= 1
            if self.force_update_counter == 0:
                self.force_update_active = False

        if self.trust_weight < 0.05:
            return

        if self.slam_jump_active:
            if self.slam_jump_counter > 0:
                self.slam_jump_counter -= 1
            if self.slam_jump_counter == 0:
                self.slam_jump_active = False
            return

        ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        pose  = self._get_pose_at(ts_ns)
        if pose is None:
            return

        _, px, py, pz, qw, qx, qy, qz, yaw = pose
        alt_z = self.lidar_z if self.lidar_z > 0.1 else pz

        cam_pts = self._parse_pointcloud2(msg)
        if cam_pts is None or len(cam_pts) < MIN_POINTS_PER_FRAME:
            return

        # ─────────────────────────────────────────────────────────
        # DEBUG RAW CLOUD
        # ─────────────────────────────────────────────────────────

        self.publish_debug_cloud(
            self.raw_pub,
            cam_pts,
            frame='base_link'
        )

       # ─────────────────────────────────────────────────────────
        # CAMERA → BODY FRAME
        # ─────────────────────────────────────────────────────────

        body_x =  cam_pts[:, 2]
        body_y = -cam_pts[:, 0]
        body_z = -cam_pts[:, 1]

        # DO NOT APPLY CAMERA OFFSETS YET
        # Gazebo already accounts for sensor placement visually

        body_pts = np.stack(
            [body_x, body_y, body_z],
            axis=1
        )

        # DEBUG BODY CLOUD
        self.publish_debug_cloud(
            self.body_pub,
            body_pts,
        frame='base_link'
        )

        # ─────────────────────────────────────────────────────────
        # YAW DEBUG
        # ─────────────────────────────────────────────────────────

        self.get_logger().warn(
            f'YAW={math.degrees(yaw):.1f}',
            throttle_duration_sec=1.0
        )

        # ─────────────────────────────────────────────────────────
        # ROTATION MATRIX
        # ─────────────────────────────────────────────────────────

        R = _quat_to_rotation_matrix(
            qw,
            qx,
            qy,
            qz
        )

        # TEST VERSION
        ned_pts = (R.T @ body_pts.T).T

        # ─────────────────────────────────────────────────────────
        # NED → WORLD
        # ─────────────────────────────────────────────────────────
        world_x = -ned_pts[:, 1] + px
        world_y =  ned_pts[:, 0] + py
        world_z =  ned_pts[:, 2] + alt_z

        world_pts = np.stack(
            [world_x, world_y, world_z],
            axis=1
        )

        # ─────────────────────────────────────────────────────────
        # DEBUG WORLD CLOUD
        # ─────────────────────────────────────────────────────────

        self.publish_debug_cloud(
            self.world_pub,
            world_pts,
            frame='odom'
        )

        # ─────────────────────────────────────────────────────────
        # TIMESTAMP DEBUG
        # ─────────────────────────────────────────────────────────

        best = min(
            self.pose_history,
            key=lambda p: abs(p[0] - ts_ns)
        )

        dt_ms = abs(best[0] - ts_ns) / 1e6

        self.get_logger().warn(
            f'SYNC ERROR = {dt_ms:.1f} ms',
            throttle_duration_sec=1.0
        )
        # ── Altitude filter ───────────────────────────────────────
        alt_ok  = (world_z >= Z_MIN) & (world_z <= Z_MAX)
        world_x = world_x[alt_ok]
        world_y = world_y[alt_ok]
        world_z = world_z[alt_ok]
        if len(world_x) == 0:
            return

        # ── Self exclusion ────────────────────────────────────────
        dist_2d = np.sqrt((world_x - px)**2 + (world_y - py)**2)
        far     = dist_2d > SELF_EXCLUSION_RADIUS
        world_x = world_x[far]
        world_y = world_y[far]
        world_z = world_z[far]
        dist_2d = dist_2d[far]
        if len(world_x) == 0:
            return

        # ── Rolling window ────────────────────────────────────────
        dist_3d = np.sqrt((world_x-px)**2 + (world_y-py)**2 + (world_z-alt_z)**2)
        window  = dist_3d <= WINDOW_RADIUS
        world_x = world_x[window]
        world_y = world_y[window]
        world_z = world_z[window]
        dist_2d = dist_2d[window]
        if len(world_x) == 0:
            return

        # ── Speed scale ───────────────────────────────────────────
        scale     = _speed_scale(self.drone_speed)
        mark_raw  = MARK_INCREMENT * self.trust_weight
        mark_stab = MARK_INCREMENT * self.trust_weight * scale

        threshold = CONFIRM_THRESHOLD * 0.4 if self.force_update_active else CONFIRM_THRESHOLD

        # ── Process observed voxels ──────────────────────────────
        seen_voxels = set()
        for i in range(len(world_x)):
            vkey = self._world_to_voxel(world_x[i], world_y[i], world_z[i])

            if vkey in seen_voxels:
                continue
            seen_voxels.add(vkey)

            if vkey not in self.voxels:
                # New voxel — create
                vd = VoxelData()
                vd.world_x = float(world_x[i])
                vd.world_y = float(world_y[i])
                vd.world_z = float(world_z[i])
                self.voxels[vkey] = vd
            else:
                vd = self.voxels[vkey]

            # ALWAYS refresh hits — keeps voxel "alive"
            vd.hits += 1

            if vd.confirmed:
                # FROZEN — do not change position or evidence
                continue

            # NEW voxel — accumulate evidence
            vd.raw_evidence    = min(MAX_EVIDENCE, vd.raw_evidence    + mark_raw)
            vd.stable_evidence = min(MAX_EVIDENCE, vd.stable_evidence + mark_stab)

            # Update position only while NOT confirmed
            vd.world_x = float(world_x[i])
            vd.world_y = float(world_y[i])
            vd.world_z = float(world_z[i])

            # Confirmation
            if vd.hits >= CONSISTENCY_FRAMES and vd.raw_evidence >= threshold:
                vd.confirmed = True

            # High confidence
            if vd.stable_evidence >= HIGH_CONF_THRESHOLD:
                vd.high_confidence = True

        # ── Safety distances ──────────────────────────────────────
        cos_y     = math.cos(-yaw)
        sin_y     = math.sin(-yaw)
        front_min = float('inf')
        left_min  = float('inf')
        right_min = float('inf')

        for i in range(len(world_x)):
            vkey = self._world_to_voxel(world_x[i], world_y[i], world_z[i])
            vd   = self.voxels.get(vkey)
            if vd is None or vd.raw_evidence < CONFIRM_THRESHOLD:
                continue
            bx = cos_y*(world_x[i]-px) + sin_y*(world_y[i]-py)
            by = -sin_y*(world_x[i]-px) + cos_y*(world_y[i]-py)
            d  = dist_2d[i] if i < len(dist_2d) else float('inf')
            if bx > 0:
                if by < -0.15:  left_min  = min(left_min,  d)
                elif by > 0.15: right_min = min(right_min, d)
                else:           front_min = min(front_min, d)

        self.depth_front = front_min
        self.depth_left  = left_min
        self.depth_right = right_min

        confirmed_count = sum(1 for v in self.voxels.values() if v.confirmed)
        self.get_logger().info(
            f'Voxels={len(self.voxels)} confirmed={confirmed_count} '
            f'speed={self.drone_speed:.2f}m/s scale={scale:.2f} '
            f'F={front_min:.2f} L={left_min:.2f} R={right_min:.2f}',
            throttle_duration_sec=3.0)

    # ─────────────────────────────────────────────────────────────
    # Sliding window prune
    # ─────────────────────────────────────────────────────────────

    def _prune_voxels(self):
        """Remove voxels beyond WINDOW_RADIUS from drone."""
        if self.drone_x is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_prune_time < PRUNE_INTERVAL:
            return
        self.last_prune_time = now

        keys_to_remove = []
        for vkey, vd in self.voxels.items():
            dx = vd.world_x - self.drone_x
            dy = vd.world_y - self.drone_y
            dist_2d = math.sqrt(dx*dx + dy*dy)

            if dist_2d > WINDOW_RADIUS:
                keys_to_remove.append(vkey)

        for k in keys_to_remove:
            del self.voxels[k]

    # ─────────────────────────────────────────────────────────────
    # Publishers
    # ─────────────────────────────────────────────────────────────

    def update_and_publish(self):
        self._prune_voxels()
        self._publish_costmap()
        self._publish_voxel_map()
        self._publish_distances()

    def _publish_costmap(self):
        """Costmap uses confirmed voxels — frozen positions."""
        if self.drone_x is None:
            return
        origin_x = self.drone_x - (COSTMAP_WIDTH  * COSTMAP_RESOLUTION) / 2.0
        origin_y = self.drone_y - (COSTMAP_HEIGHT * COSTMAP_RESOLUTION) / 2.0
        grid = np.zeros((COSTMAP_HEIGHT, COSTMAP_WIDTH), dtype=np.int8)

        for vkey, vd in self.voxels.items():
            if not vd.confirmed:
                continue
            if vd.world_z < COSTMAP_Z_MIN or vd.world_z > COSTMAP_Z_MAX:
                continue
            cx = int((vd.world_x - origin_x) / COSTMAP_RESOLUTION)
            cy = int((vd.world_y - origin_y) / COSTMAP_RESOLUTION)
            if 0 <= cx < COSTMAP_WIDTH and 0 <= cy < COSTMAP_HEIGHT:
                grid[cy, cx] = 100

        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info.resolution = COSTMAP_RESOLUTION
        msg.info.width      = COSTMAP_WIDTH
        msg.info.height     = COSTMAP_HEIGHT
        msg.info.origin     = Pose()
        msg.info.origin.position.x  = origin_x
        msg.info.origin.position.y  = origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self.costmap_pub.publish(msg)

    def _publish_voxel_map(self):
        confirmed = [v for v in self.voxels.values() if v.confirmed]
        if not confirmed:
            return
        pts = np.array([(v.world_x, v.world_y, v.world_z) for v in confirmed],
                       dtype=np.float32)
        msg = PointCloud2()
        msg.header.stamp    = self.get_clock().now().to_msg()
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
        front = min(self.front_dist, self.depth_front)
        left  = min(self.left_dist,  self.depth_left)
        right = min(self.right_dist, self.depth_right)
        msg   = Float32MultiArray()
        msg.data = [
            float(front), float(left), float(right),
            float(self.depth_front),
            float(self.depth_left),
            float(self.depth_right),
        ]
        self.distances_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OctomapManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()