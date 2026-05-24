#!/usr/bin/env python3
"""
octomap_manager.py — 3D volumetric mapping with full sensor fusion.

Receives camera-frame points from depth_filter and projects them
to world frame using:
  - PX4 full attitude (pitch, roll, yaw) via quaternion
  - MTF-01 LiDAR altitude
  - VIO/GPS position (x, y)

Camera mount offset (from model.sdf):
  x=+0.01233m forward, y=+0.0375m left, z=+0.01878m up

Coordinate convention:
  World X = PX4 East  = corridor length (0→20m)
  World Y = PX4 North = corridor width  (0→6m)
  World Z = up (from LiDAR altitude)
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
CAM_X =  0.01233   # forward
CAM_Y =  0.0375    # left
CAM_Z =  0.01878   # up

# ── Rolling window ────────────────────────────────────────────────
WINDOW_RADIUS    = 12.0
PRUNE_INTERVAL   = 2.0

# ── Voxel resolution ─────────────────────────────────────────────
VOXEL_SIZE       = 0.30

# ── Temporal consistency ──────────────────────────────────────────
CONSISTENCY_FRAMES = 3      # frames voxel must be seen before confirming
DECAY_FRAMES       = 5      # frames without observation before removing

# ── Evidence ──────────────────────────────────────────────────────
MARK_INCREMENT      = 8.0
FREE_DECREMENT      = 25.0
CONFIRM_THRESHOLD   = 60.0
HIGH_CONF_THRESHOLD = 150.0
FREE_THRESHOLD      = 10.0
MAX_EVIDENCE        = 300.0

# ── Safety filtering ──────────────────────────────────────────────
SELF_EXCLUSION_RADIUS = 0.8
MIN_POINTS_PER_FRAME  = 20
Z_MIN                 = 0.8
Z_MAX                 = 3.0

# ── Costmap ───────────────────────────────────────────────────────
COSTMAP_RESOLUTION = 0.10
COSTMAP_WIDTH      = 200
COSTMAP_HEIGHT     = 120
COSTMAP_Z_MIN      = 0.5
COSTMAP_Z_MAX      = 2.8

# ── Pose history ──────────────────────────────────────────────────
POSE_HISTORY_SIZE  = 100    # ~2s at 50Hz

# ── Update rate ───────────────────────────────────────────────────
UPDATE_RATE        = 10.0


class VoxelData:
    __slots__ = ['hits', 'misses', 'evidence', 'confirmed', 'high_confidence']
    def __init__(self):
        self.hits           = 0
        self.misses         = 0
        self.evidence       = 0.0
        self.confirmed      = False
        self.high_confidence = False


def _wrap_angle(a):
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a


def _quat_to_rotation_matrix(w, x, y, z):
    """Convert quaternion to 3x3 rotation matrix."""
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)
    return R


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

        # ── Voxel map ─────────────────────────────────────────────
        self.voxels = {}

        # ── Pose history: (ts_ns, x, y, z, qw, qx, qy, qz) ──────
        self.pose_history = deque(maxlen=POSE_HISTORY_SIZE)

        # ── Current drone state ───────────────────────────────────
        self.drone_x   = None
        self.drone_y   = None
        self.drone_z   = 0.0
        self.drone_yaw = 0.0
        self.lidar_z   = 0.0

        # ── VIO ───────────────────────────────────────────────────
        self.vio_x   = None
        self.vio_y   = None
        self.vio_z   = None
        self.slam_pose_available = False

        # ── Trust weight ──────────────────────────────────────────
        self.trust_weight = 1.0

        # ── Safety distances ──────────────────────────────────────
        self.front_dist = float('inf')
        self.left_dist  = float('inf')
        self.right_dist = float('inf')

        # ── Timing ────────────────────────────────────────────────
        self.last_prune_time = 0.0

        # ── Force update ──────────────────────────────────────────
        self.force_update_active  = False
        self.force_update_counter = 0

        # ── Subscribers ───────────────────────────────────────────
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

        # ── Publishers ────────────────────────────────────────────
        self.costmap_pub   = self.create_publisher(OccupancyGrid, '/costmap', 10)
        self.voxel_pub     = self.create_publisher(PointCloud2, '/voxel_map', 10)
        self.distances_pub = self.create_publisher(Float32MultiArray, '/obstacle_distances', 10)

        self.create_timer(1.0 / UPDATE_RATE, self.update_and_publish)

        self.get_logger().info(
            f'OctomapManager started ✓  '
            f'voxel={VOXEL_SIZE}m  consistency={CONSISTENCY_FRAMES}frames  '
            f'window={WINDOW_RADIUS}m')

    # ── Callbacks ─────────────────────────────────────────────────

    def odom_cb(self, msg):
        """Store full pose in history for timestamp sync."""
        # Position
        x = float(msg.position[1]) + SPAWN_X   # PX4 East → World X
        y = float(msg.position[0]) + SPAWN_Y   # PX4 North → World Y
        z = -float(msg.position[2])             # NED down → up

        # Full quaternion — PX4 uses [w, x, y, z]
        qw = float(msg.q[0])
        qx = float(msg.q[1])
        qy = float(msg.q[2])
        qz = float(msg.q[3])

        # Yaw for safety distance calculation
        siny = 2.0*(qw*qz + qx*qy)
        cosy = 1.0 - 2.0*(qy*qy + qz*qz)
        yaw  = _wrap_angle(math.atan2(siny, cosy) - math.pi/2.0)

        # Store in history with PX4 timestamp (microseconds → nanoseconds)
        ts_ns = msg.timestamp * 1000
        self.pose_history.append((ts_ns, x, y, z, qw, qx, qy, qz, yaw))

        # Update current state (used by safety distances)
        if not self.slam_pose_available:
            self.drone_x   = x
            self.drone_y   = y
            self.drone_z   = z
            self.drone_yaw = yaw

    def lidar_cb(self, msg):
        """MTF-01 downward LiDAR — altitude above ground."""
        if msg.ranges and math.isfinite(msg.ranges[0]) and msg.ranges[0] > 0.01:
            self.lidar_z = float(msg.ranges[0])
            # Update Z in latest pose history entry
            if self.pose_history:
                entry = list(self.pose_history[-1])
                entry[3] = self.lidar_z  # replace Z with LiDAR altitude
                self.pose_history[-1] = tuple(entry)

    def slam_pose_cb(self, msg):
        """VIO corrected pose."""
        nx = float(msg.pose.position.x)
        ny = float(msg.pose.position.y)
        nz = float(msg.pose.position.z)
        # Reject large jumps
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

    # ── Pose lookup ───────────────────────────────────────────────

    def _get_pose_at(self, ts_ns):
        """Find pose closest to given timestamp."""
        if not self.pose_history:
            return None
        return min(self.pose_history, key=lambda p: abs(p[0] - ts_ns))

    # ── Voxel helpers ─────────────────────────────────────────────

    def _world_to_voxel(self, wx, wy, wz):
        return (int(math.floor(wx/VOXEL_SIZE)),
                int(math.floor(wy/VOXEL_SIZE)),
                int(math.floor(wz/VOXEL_SIZE)))

    def _voxel_to_world(self, gx, gy, gz):
        return ((gx+0.5)*VOXEL_SIZE,
                (gy+0.5)*VOXEL_SIZE,
                (gz+0.5)*VOXEL_SIZE)

    # ── Point cloud parsing ───────────────────────────────────────

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
            base = i * step
            pts[i,0] = struct.unpack_from('f', data, base+x_off)[0]
            pts[i,1] = struct.unpack_from('f', data, base+y_off)[0]
            pts[i,2] = struct.unpack_from('f', data, base+z_off)[0]
        valid = np.all(np.isfinite(pts), axis=1)
        pts = pts[valid]
        return pts if len(pts) > 0 else None

    # ── Main cloud callback ───────────────────────────────────────

    def cloud_cb(self, msg):
        """
        Project camera-frame points to world frame using full 3D rotation.
        
        Camera frame:  x=right, y=down, z=forward
        Body frame:    x=forward, y=left, z=up
        World frame:   x=East(corridor), y=North, z=up
        
        Transform chain:
          camera → body (fixed rotation + offset)
          body   → world (drone attitude quaternion)
        """
        if self.force_update_active and self.force_update_counter > 0:
            self.force_update_counter -= 1
            if self.force_update_counter == 0:
                self.force_update_active = False

        if self.trust_weight < 0.05:
            return

        # Get synchronized pose from history
        ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        pose  = self._get_pose_at(ts_ns)
        if pose is None:
            return

        _, px, py, pz, qw, qx, qy, qz, yaw = pose

        # Use LiDAR altitude if available
        alt_z = self.lidar_z if self.lidar_z > 0.1 else pz

        # Parse camera-frame points
        cam_pts = self._parse_pointcloud2(msg)
        if cam_pts is None or len(cam_pts) < MIN_POINTS_PER_FRAME:
            return

        # ── Step 1: Camera → Body frame ───────────────────────────
        # Camera: x=right, y=down, z=forward
        # Body:   x=forward, y=left, z=up
        body_x =  cam_pts[:, 2]         # camera z → body x (forward)
        body_y = -cam_pts[:, 0]         # camera -x → body y (left)
        body_z = -cam_pts[:, 1]         # camera -y → body z (up)

        # Add camera mount offset
        body_x += CAM_X
        body_y += CAM_Y
        body_z += CAM_Z

        body_pts = np.stack([body_x, body_y, body_z], axis=1)

        # ── Step 2: Body → World frame using full quaternion ──────
        # PX4 NED body frame: x=North, y=East, z=Down
        # We need to handle the PX4 NED convention properly
        #
        # PX4 quaternion rotates from body to NED world
        # Then we convert NED to our world frame:
        #   world_x = ned_y + SPAWN_X  (East → X)
        #   world_y = ned_x + SPAWN_Y  (North → Y)
        #   world_z = -ned_z           (Down → up)

        R = _quat_to_rotation_matrix(qw, qx, qy, qz)

        # Rotate body points to NED world
        ned_pts = (R @ body_pts.T).T   # shape (N, 3)

        # Convert NED to our world frame + add drone position
        world_x = ned_pts[:, 1] + px   # NED East + drone_x
        world_y = -ned_pts[:, 0] + py   # NED -North + drone_y
        world_z = -ned_pts[:, 2] + alt_z  # NED Down→up + altitude

        # ── Altitude filter ───────────────────────────────────────
        alt_ok  = (world_z >= Z_MIN) & (world_z <= Z_MAX)
        world_x = world_x[alt_ok]
        world_y = world_y[alt_ok]
        world_z = world_z[alt_ok]

        if len(world_x) == 0:
            return

        # ── Self exclusion ────────────────────────────────────────
        dist_2d = np.sqrt((world_x-px)**2 + (world_y-py)**2)
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

        # ── Temporal consistency + evidence ───────────────────────
        seen_voxels = set()
        for i in range(len(world_x)):
            vkey = self._world_to_voxel(world_x[i], world_y[i], world_z[i])
            seen_voxels.add(vkey)

        mark_inc  = MARK_INCREMENT * self.trust_weight
        threshold = CONFIRM_THRESHOLD * 0.4 if self.force_update_active else CONFIRM_THRESHOLD

        for vkey in seen_voxels:
            if vkey not in self.voxels:
                self.voxels[vkey] = VoxelData()
            vd = self.voxels[vkey]
            vd.hits    += 1
            vd.misses   = 0
            vd.evidence = min(MAX_EVIDENCE, vd.evidence + mark_inc)
            if vd.hits >= CONSISTENCY_FRAMES and vd.evidence >= threshold:
                vd.confirmed = True
            if vd.evidence >= HIGH_CONF_THRESHOLD:
                vd.high_confidence = True

        # Decay unseen voxels
        for vkey in list(self.voxels.keys()):
            if vkey not in seen_voxels:
                vd = self.voxels[vkey]
                vd.misses   += 1
                vd.hits      = max(0, vd.hits - 1)
                vd.evidence  = max(0.0, vd.evidence - 3.0)
                if vd.evidence < HIGH_CONF_THRESHOLD:
                    vd.high_confidence = False
                if vd.evidence < FREE_THRESHOLD:
                    vd.confirmed = False

        # ── Safety distances ──────────────────────────────────────
        cos_y = math.cos(-yaw)
        sin_y = math.sin(-yaw)
        front_min = float('inf')
        left_min  = float('inf')
        right_min = float('inf')

        for i in range(len(world_x)):
            bx =  cos_y*(world_x[i]-px) + sin_y*(world_y[i]-py)
            by = -sin_y*(world_x[i]-px) + cos_y*(world_y[i]-py)
            d  = dist_2d[i]
            if bx > 0:
                if by < -0.15:   left_min  = min(left_min,  d)
                elif by > 0.15:  right_min = min(right_min, d)
                else:            front_min = min(front_min, d)

        self.depth_front = front_min
        self.depth_left  = left_min
        self.depth_right = right_min

        self.get_logger().info(
            f'Voxels={len(self.voxels)} '
            f'confirmed={sum(1 for v in self.voxels.values() if v.confirmed)} '
            f'F={front_min:.2f} L={left_min:.2f} R={right_min:.2f}',
            throttle_duration_sec=3.0)

    # ── Pruning ───────────────────────────────────────────────────

    def _prune_voxels(self):
        if self.drone_x is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_prune_time < PRUNE_INTERVAL:
            return
        self.last_prune_time = now

        radius_sq = WINDOW_RADIUS ** 2
        keys_to_remove = []
        for vkey in self.voxels:
            wx, wy, wz = self._voxel_to_world(*vkey)
            dx = wx - self.drone_x
            dy = wy - self.drone_y
            if dx*dx + dy*dy > radius_sq:
                keys_to_remove.append(vkey)
            elif self.voxels[vkey].evidence <= 0.0:
                keys_to_remove.append(vkey)
        for k in keys_to_remove:
            del self.voxels[k]

    # ── Publishers ────────────────────────────────────────────────

    def update_and_publish(self):
        self._prune_voxels()
        self._publish_costmap()
        self._publish_voxel_map()
        self._publish_distances()

    def _publish_costmap(self):
        if self.drone_x is None:
            return
        origin_x = self.drone_x - (COSTMAP_WIDTH  * COSTMAP_RESOLUTION) / 2.0
        origin_y = self.drone_y - (COSTMAP_HEIGHT * COSTMAP_RESOLUTION) / 2.0
        grid = np.zeros((COSTMAP_HEIGHT, COSTMAP_WIDTH), dtype=np.int8)

        for vkey, vd in self.voxels.items():
            if not vd.high_confidence:
                continue
            wx, wy, wz = self._voxel_to_world(*vkey)
            if wz < COSTMAP_Z_MIN or wz > COSTMAP_Z_MAX:
                continue
            cx = int((wx - origin_x) / COSTMAP_RESOLUTION)
            cy = int((wy - origin_y) / COSTMAP_RESOLUTION)
            if 0 <= cx < COSTMAP_WIDTH and 0 <= cy < COSTMAP_HEIGHT:
                grid[cy, cx] = 100

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info.resolution = COSTMAP_RESOLUTION
        msg.info.width      = COSTMAP_WIDTH
        msg.info.height     = COSTMAP_HEIGHT
        msg.info.origin     = Pose()
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self.costmap_pub.publish(msg)

    def _publish_voxel_map(self):
        confirmed = [k for k, v in self.voxels.items() if v.confirmed]
        if not confirmed:
            return
        pts = np.array([self._voxel_to_world(*k) for k in confirmed],
                       dtype=np.float32)
        msg = PointCloud2()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.height = 1; msg.width = len(pts)
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
        front = min(self.front_dist, getattr(self, 'depth_front', float('inf')))
        left  = min(self.left_dist,  getattr(self, 'depth_left',  float('inf')))
        right = min(self.right_dist, getattr(self, 'depth_right', float('inf')))
        msg = Float32MultiArray()
        msg.data = [float(front), float(left), float(right),
                    float(getattr(self, 'depth_front', float('inf'))),
                    float(getattr(self, 'depth_left',  float('inf'))),
                    float(getattr(self, 'depth_right', float('inf')))]
        self.distances_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OctomapManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()