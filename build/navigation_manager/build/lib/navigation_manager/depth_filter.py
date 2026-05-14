import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from px4_msgs.msg import VehicleOdometry
from cv_bridge import CvBridge
import numpy as np
import math
import struct

class DepthFilter(Node):
    # Camera intrinsics
    CX = 160.0
    CY = 120.0
    FX = 161.4
    FY = 161.4
    IMG_W = 320
    IMG_H = 240

    # Camera mount
    CAM_X = 0.12 + 0.01233
    CAM_Y = 0.0 + (-0.0375)
    CAM_Z = 0.06 + 0.01878

    # Filter parameters
    MIN_DEPTH     = 1.0    # ignore very close points
    MAX_DEPTH     = 3.5    # ignore far/max range
    Z_MIN         = 0.8    # cut floor
    Z_MAX         = 2.7    # cut ceiling
    VOXEL_SIZE    = 0.15   # bigger voxels = fewer points
    RADIUS        = 0.25   # bigger radius for outlier check
    MIN_NEIGHBORS = 5      # stricter — needs 5 neighbors
    SUBSAMPLE     = 5      # skip more pixels

    def __init__(self):
        super().__init__('depth_filter')

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_yaw = 0.0

        # Yaw rate
        self.last_yaw = None
        self.last_yaw_time = None
        self.yaw_rate = 0.0
        self.MAX_YAW_RATE = 0.15

        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_px4
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/oakd/depth/image',
            self.depth_callback,
            qos_reliable
        )
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            '/pointcloud/filtered',
            10
        )

        self.get_logger().info('DepthFilter started ✓')

    def _wrap_angle(self, a):
        while a > math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def odom_callback(self, msg):
        self.drone_x = float(msg.position[1]) + 1.0
        self.drone_y = float(msg.position[0]) + 3.0
        self.drone_z = float(msg.position[2])
        q = msg.q
        siny = 2.0*(q[0]*q[3] + q[1]*q[2])
        cosy = 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3])
        self.drone_yaw = self._wrap_angle(
            math.atan2(siny, cosy) - math.pi/2.0
        )
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_yaw is not None and self.last_yaw_time is not None:
            dt = now - self.last_yaw_time
            if dt > 0.001:
                dyaw = self._wrap_angle(self.drone_yaw - self.last_yaw)
                self.yaw_rate = 0.7*self.yaw_rate + 0.3*(dyaw/dt)
        self.last_yaw = self.drone_yaw
        self.last_yaw_time = now

    def depth_callback(self, msg):
        # Yaw gate
        if abs(self.yaw_rate) > self.MAX_YAW_RATE:
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')
            return

        h, w = depth.shape
        cos_y = math.cos(self.drone_yaw)
        sin_y = math.sin(self.drone_yaw)
        drone_alt = -self.drone_z

        points = []

        for row in range(0, h, self.SUBSAMPLE):
            for col in range(0, w, self.SUBSAMPLE):
                d = float(depth[row, col])

                # PassThrough filter — reject invalid depths
                if not math.isfinite(d) or d < self.MIN_DEPTH or d >= self.MAX_DEPTH:
                    continue

                # Camera -> body frame
                z_cam = d
                x_cam = (col - self.CX) * d / self.FX
                y_cam = (row - self.CY) * d / self.FY
                body_x = z_cam + self.CAM_X
                body_y = -x_cam + self.CAM_Y
                body_z = -y_cam + self.CAM_Z

                # Altitude PassThrough filter
                world_z = drone_alt + body_z
                if world_z < self.Z_MIN or world_z > self.Z_MAX:
                    continue

                # Body -> world
                world_x = self.drone_x + cos_y*body_x - sin_y*body_y
                world_y = self.drone_y + sin_y*body_x + cos_y*body_y

                points.append([world_x, world_y, world_z])

        if not points:
            return

        pts = np.array(points, dtype=np.float32)

        # VoxelGrid downsampling
        pts = self._voxel_grid(pts, self.VOXEL_SIZE)

        # RadiusOutlierRemoval
        pts = self._radius_outlier_removal(pts, self.RADIUS, self.MIN_NEIGHBORS)

        if len(pts) == 0:
            return

        self._publish_cloud(pts)

    def _voxel_grid(self, pts, voxel_size):
        if len(pts) == 0:
            return pts
        voxel_indices = np.floor(pts / voxel_size).astype(np.int32)
        unique_voxels, inverse = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )
        downsampled = np.zeros(
            (len(unique_voxels), 3), dtype=np.float32
        )
        np.add.at(downsampled, inverse, pts)
        counts = np.bincount(inverse)
        downsampled /= counts[:, np.newaxis]
        return downsampled

    def _radius_outlier_removal(self, pts, radius, min_neighbors):
        if len(pts) < min_neighbors + 1:
            return pts
        keep = []
        for i, p in enumerate(pts):
            dists = np.linalg.norm(pts - p, axis=1)
            neighbors = np.sum(dists < radius) - 1  # exclude self
            if neighbors >= min_neighbors:
                keep.append(i)
        return pts[keep] if keep else np.array([], dtype=np.float32)

    def _publish_cloud(self, pts):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.height = 1
        msg.width = len(pts)
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(pts)
        msg.is_dense = True
        msg.data = pts.tobytes()
        self.cloud_pub.publish(msg)

def main():
    rclpy.init()
    node = DepthFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
