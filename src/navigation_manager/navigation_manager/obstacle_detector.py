import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleOdometry
import numpy as np
import math
import struct

class ObstacleDetector(Node):
    # Corridor dimensions
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

    # SLAM evidence
    DECAY_RATE          = 1.0
    MARK_INCREMENT      = 10.0
    FREE_DECREMENT      = 5.0
    CONFIRM_THRESHOLD   = 100.0
    FREE_THRESHOLD      = 10.0
    MARK_RADIUS         = 1
    MIN_CLUSTER_NEIGHBORS = 4

    # Safety
    SELF_EXCLUSION_RADIUS = 1.0

    def __init__(self):
        super().__init__('obstacle_detector')

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        qos_besteffort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 3D evidence + slam grids
        self.evidence_grid = np.zeros(
            (self.GRID_Z, self.GRID_H, self.GRID_W), dtype=np.float32
        )
        self.slam_map = np.zeros(
            (self.GRID_Z, self.GRID_H, self.GRID_W), dtype=np.float32
        )

        # Drone state
        self.drone_x   = 0.0
        self.drone_y   = 0.0
        self.drone_z   = 0.0
        self.drone_yaw = 0.0

        # Distances
        self.front_dist       = float('inf')
        self.left_dist        = float('inf')
        self.right_dist       = float('inf')
        self.depth_front_dist = float('inf')
        self.depth_left_dist  = float('inf')
        self.depth_right_dist = float('inf')

        # Subscribers
        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.odom_callback, qos_px4
        )
        self.create_subscription(
            PointCloud2, '/pointcloud/filtered',
            self.cloud_callback, qos_reliable
        )
        self.create_subscription(
            LaserScan, '/front_lidar/scan',
            self.front_callback, qos_besteffort
        )
        self.create_subscription(
            LaserScan, '/left_lidar/scan',
            self.left_callback, qos_besteffort
        )
        self.create_subscription(
            LaserScan, '/right_lidar/scan',
            self.right_callback, qos_besteffort
        )

        # Publishers
        self.costmap_pub   = self.create_publisher(OccupancyGrid, '/costmap', 10)
        self.voxel_pub     = self.create_publisher(PointCloud2, '/voxel_map', 10)
        self.distances_pub = self.create_publisher(Float32MultiArray, '/obstacle_distances', 10)

        self.create_timer(0.1, self.update_and_publish)
        self.get_logger().info(
            f'ObstacleDetector 3D started ✓  '
            f'Grid={self.GRID_W}x{self.GRID_H}x{self.GRID_Z}'
        )

    # ==================================================================
    # Helpers
    # ==================================================================
    def _wrap_angle(self, a):
        while a >  math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def _world_to_grid(self, wx, wy, wz):
        gx = int((wx - self.ORIGIN_X) / self.RESOLUTION)
        gy = int((wy - self.ORIGIN_Y) / self.RESOLUTION)
        gz = int((wz - self.ORIGIN_Z) / self.RESOLUTION)
        if (0 <= gx < self.GRID_W and
            0 <= gy < self.GRID_H and
            0 <= gz < self.GRID_Z):
            return gx, gy, gz
        return None, None, None

    def _bresenham3d(self, x0, y0, z0, x1, y1, z1):
        """3D Bresenham ray walking."""
        cells = []
        dx, dy, dz = abs(x1-x0), abs(y1-y0), abs(z1-z0)
        sx = 1 if x0<x1 else -1
        sy = 1 if y0<y1 else -1
        sz = 1 if z0<z1 else -1
        if dx >= dy and dx >= dz:
            p1, p2 = 2*dy-dx, 2*dz-dx
            while x0 != x1:
                if (0<=x0<self.GRID_W and 0<=y0<self.GRID_H and 0<=z0<self.GRID_Z):
                    cells.append((x0,y0,z0))
                else:
                    break
                if p1>=0: y0+=sy; p1-=2*dx
                if p2>=0: z0+=sz; p2-=2*dx
                p1+=2*dy; p2+=2*dz; x0+=sx
        elif dy >= dx and dy >= dz:
            p1, p2 = 2*dx-dy, 2*dz-dy
            while y0 != y1:
                if (0<=x0<self.GRID_W and 0<=y0<self.GRID_H and 0<=z0<self.GRID_Z):
                    cells.append((x0,y0,z0))
                else:
                    break
                if p1>=0: x0+=sx; p1-=2*dy
                if p2>=0: z0+=sz; p2-=2*dy
                p1+=2*dx; p2+=2*dz; y0+=sy
        else:
            p1, p2 = 2*dx-dz, 2*dy-dz
            while z0 != z1:
                if (0<=x0<self.GRID_W and 0<=y0<self.GRID_H and 0<=z0<self.GRID_Z):
                    cells.append((x0,y0,z0))
                else:
                    break
                if p1>=0: x0+=sx; p1-=2*dz
                if p2>=0: y0+=sy; p2-=2*dz
                p1+=2*dx; p2+=2*dy; z0+=sz
        return cells

    # ==================================================================
    # Odometry
    # ==================================================================
    def odom_callback(self, msg):
        self.drone_x = float(msg.position[1]) + 1.0
        self.drone_y = float(msg.position[0]) + 3.0
        self.drone_z = float(msg.position[2])
        q = msg.q
        siny = 2.0*(q[0]*q[3]+q[1]*q[2])
        cosy = 1.0-2.0*(q[2]*q[2]+q[3]*q[3])
        self.drone_yaw = self._wrap_angle(
            math.atan2(siny, cosy) - math.pi/2.0
        )

    # ==================================================================
    # LiDAR
    # ==================================================================
    def front_callback(self, msg):
        r = [x for x in msg.ranges if math.isfinite(x) and x > 0.05]
        self.front_dist = min(r) if r else float('inf')

    def left_callback(self, msg):
        r = [x for x in msg.ranges if math.isfinite(x) and x > 0.05]
        self.left_dist = min(r) if r else float('inf')

    def right_callback(self, msg):
        r = [x for x in msg.ranges if math.isfinite(x) and x > 0.05]
        self.right_dist = min(r) if r else float('inf')

    # ==================================================================
    # Filtered pointcloud — 3D mapping
    # ==================================================================
    def cloud_callback(self, msg):
        points = self._parse_pointcloud2(msg)
        if points is None or len(points) == 0:
            return

        drone_alt = -self.drone_z
        dgx, dgy, dgz = self._world_to_grid(
            self.drone_x, self.drone_y, drone_alt
        )
        if dgx is None:
            return

        left_min  = float('inf')
        front_min = float('inf')
        right_min = float('inf')

        for wx, wy, wz in points:
            dist_2d = math.sqrt(
                (wx-self.drone_x)**2 + (wy-self.drone_y)**2
            )
            if dist_2d < self.SELF_EXCLUSION_RADIUS:
                continue

            hgx, hgy, hgz = self._world_to_grid(wx, wy, wz)
            if hgx is None:
                continue

            # 3D Bresenham free-space clearing
            free_cells = self._bresenham3d(dgx, dgy, dgz, hgx, hgy, hgz)
            for fx, fy, fz in free_cells:
                self.evidence_grid[fz, fy, fx] = max(
                    0.0,
                    self.evidence_grid[fz, fy, fx] - self.FREE_DECREMENT
                )

            # Mark obstacle voxel
            for dx in range(-self.MARK_RADIUS, self.MARK_RADIUS+1):
                for dy in range(-self.MARK_RADIUS, self.MARK_RADIUS+1):
                    for dz in range(-self.MARK_RADIUS, self.MARK_RADIUS+1):
                        nx = hgx+dx
                        ny = hgy+dy
                        nz = hgz+dz
                        if (0<=nx<self.GRID_W and
                            0<=ny<self.GRID_H and
                            0<=nz<self.GRID_Z):
                            self.evidence_grid[nz, ny, nx] = min(
                                100.0,
                                self.evidence_grid[nz, ny, nx] + self.MARK_INCREMENT
                            )

            # Zone distances
            body_x = wx - self.drone_x
            body_y = wy - self.drone_y
            cos_y =  math.cos(-self.drone_yaw)
            sin_y =  math.sin(-self.drone_yaw)
            fwd  =  cos_y*body_x + sin_y*body_y
            side = -sin_y*body_x + cos_y*body_y

            if fwd > 0:
                if side < -0.15:
                    left_min  = min(left_min,  dist_2d)
                elif side > 0.15:
                    right_min = min(right_min, dist_2d)
                else:
                    front_min = min(front_min, dist_2d)

        self.depth_front_dist = front_min
        self.depth_left_dist  = left_min
        self.depth_right_dist = right_min

    def _parse_pointcloud2(self, msg):
        field_offsets = {f.name: f.offset for f in msg.fields}
        if 'x' not in field_offsets:
            return None
        x_off = field_offsets['x']
        y_off = field_offsets['y']
        z_off = field_offsets['z']
        step  = msg.point_step
        data  = msg.data
        n     = msg.width * msg.height
        points = []
        for i in range(n):
            base = i * step
            x = struct.unpack_from('f', data, base+x_off)[0]
            y = struct.unpack_from('f', data, base+y_off)[0]
            z = struct.unpack_from('f', data, base+z_off)[0]
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                points.append((x, y, z))
        return points

    # ==================================================================
    # 3D SLAM map sync
    # ==================================================================
    def _sync_slam_map(self):
        confirmed = self.evidence_grid >= self.CONFIRM_THRESHOLD
        cleared   = self.evidence_grid <  self.FREE_THRESHOLD

        # 3D neighbor count for cluster filter
        confirmed_u8 = confirmed.astype(np.uint8)
        pad = np.pad(confirmed_u8, 1, mode='constant', constant_values=0)
        neighbor_count = (
            pad[0:-2,0:-2,0:-2] + pad[0:-2,0:-2,1:-1] + pad[0:-2,0:-2,2:] +
            pad[0:-2,1:-1,0:-2] + pad[0:-2,1:-1,1:-1] + pad[0:-2,1:-1,2:] +
            pad[0:-2,2:,  0:-2] + pad[0:-2,2:,  1:-1] + pad[0:-2,2:,  2:] +
            pad[1:-1,0:-2,0:-2] + pad[1:-1,0:-2,1:-1] + pad[1:-1,0:-2,2:] +
            pad[1:-1,1:-1,0:-2] +                        pad[1:-1,1:-1,2:] +
            pad[1:-1,2:,  0:-2] + pad[1:-1,2:,  1:-1] + pad[1:-1,2:,  2:] +
            pad[2:,  0:-2,0:-2] + pad[2:,  0:-2,1:-1] + pad[2:,  0:-2,2:] +
            pad[2:,  1:-1,0:-2] + pad[2:,  1:-1,1:-1] + pad[2:,  1:-1,2:] +
            pad[2:,  2:,  0:-2] + pad[2:,  2:,  1:-1] + pad[2:,  2:,  2:]
        )
        clustered = confirmed & (neighbor_count >= self.MIN_CLUSTER_NEIGHBORS)
        self.slam_map[cleared]   = 0.0
        self.slam_map[clustered] = 100.0

    # ==================================================================
    # Publishers
    # ==================================================================
    def update_and_publish(self):
        self._sync_slam_map()
        self._publish_costmap_2d()
        self._publish_voxel_map()
        self._publish_distances()

    def _publish_costmap_2d(self):
        """Project 3D map to 2D for A* planner (max over Z)."""
        flat = np.max(self.slam_map, axis=0)
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info.resolution = self.RESOLUTION
        msg.info.width  = self.GRID_W
        msg.info.height = self.GRID_H
        msg.info.origin = Pose()
        msg.info.origin.position.x = self.ORIGIN_X
        msg.info.origin.position.y = self.ORIGIN_Y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = flat.astype(np.int8).flatten().tolist()
        self.costmap_pub.publish(msg)

    def _publish_voxel_map(self):
        """Publish occupied voxels as PointCloud2 for RViz."""
        occupied = np.argwhere(self.slam_map > 50)
        if len(occupied) == 0:
            return

        pts = np.zeros((len(occupied), 3), dtype=np.float32)
        pts[:, 0] = occupied[:, 2]*self.RESOLUTION + self.ORIGIN_X
        pts[:, 1] = occupied[:, 1]*self.RESOLUTION + self.ORIGIN_Y
        pts[:, 2] = occupied[:, 0]*self.RESOLUTION + self.ORIGIN_Z

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
            f'Distances — F={fused_front:.2f} L={fused_left:.2f} R={fused_right:.2f}',
            throttle_duration_sec=2.0
        )

def main():
    rclpy.init()
    node = ObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()