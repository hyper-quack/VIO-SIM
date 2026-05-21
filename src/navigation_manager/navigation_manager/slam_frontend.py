#!/usr/bin/env python3
"""
slam_frontend.py — Visual SLAM front-end node

Pipeline:
  /oakd/sync/left/image  ──┐
  /oakd/sync/right/image ──┤→ ORB extract → KLT track → PnP RANSAC → /slam/pose
  /imu/filtered          ──┤                                         → /slam/keyframe
  /fmu/out/vehicle_odometry┤ (PX4 fallback for XY, yaw always)      → /slam/tracking_quality
  /mtf01/lidar           ──┘ (Z always from LiDAR)                  → /slam/debug/feature_count

Coordinate convention (same as all other nodes):
  world_x = px4_position[1] + SPAWN_X   (PX4 East  → World X)
  world_y = px4_position[0] + SPAWN_Y   (PX4 North → World Y)
  world_z = lidar range reading          (MTF-01 altitude)
  yaw     = atan2(siny,cosy) - pi/2     (GPS mode, from PX4 IMU)
"""

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu, LaserScan
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Int32
from px4_msgs.msg import VehicleOdometry
from cv_bridge import CvBridge

# ── ORB ───────────────────────────────────────────────────────────────────────
N_FEATURES           = 500
FAST_THRESHOLD       = 20
SCALE_FACTOR         = 1.2
N_LEVELS             = 8

# ── KLT ───────────────────────────────────────────────────────────────────────
LK_WIN_SIZE          = (21, 21)
LK_MAX_LEVEL         = 3
LK_MAX_ITER          = 30
LK_EPSILON           = 0.01
MIN_TRACKED_POINTS   = 20

# ── Keyframe decision ─────────────────────────────────────────────────────────
MIN_PARALLAX         = 8.0    # pixels — median flow to trigger keyframe
MIN_FEATURE_RATIO    = 0.6    # unused currently, kept for future
MAX_KEYFRAME_DIST    = 0.5    # metres — force keyframe if moved this far

# ── PnP RANSAC ────────────────────────────────────────────────────────────────
PNP_ITERATIONS       = 100
PNP_REPROJECTION_ERR = 4.0    # pixels
PNP_CONFIDENCE       = 0.99
MIN_PNP_INLIERS      = 10

# ── IMU gate ──────────────────────────────────────────────────────────────────
MAX_YAW_RATE         = 0.5    # rad/s — skip frame if rotating faster

# ── Stereo ────────────────────────────────────────────────────────────────────
BASELINE             = 0.075  # metres — OAK-D stereo baseline
MIN_STEREO_DEPTH     = 0.3    # metres
MAX_STEREO_DEPTH     = 8.0    # metres

# ── Camera intrinsics (OAK-D at 320×240) ──────────────────────────────────────
FX = 161.4
FY = 161.4
CX = 160.0
CY = 120.0

# ── Complementary filter weight ───────────────────────────────────────────────
ALPHA = 0.7   # visual weight; (1-ALPHA) = PX4 weight for XY

# ── World frame spawn offsets ─────────────────────────────────────────────────
SPAWN_X = 1.0
SPAWN_Y = 3.0


def _wrap_angle(a: float) -> float:
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


class SlamFrontend(Node):

    def __init__(self):
        super().__init__('slam_frontend')

        # ── QoS profiles ──────────────────────────────────────────────────────
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        qos_image = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        qos_imu = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Image, '/oakd/sync/left/image',
            self._left_cb, qos_image)

        self.create_subscription(
            Image, '/oakd/sync/right/image',
            self._right_cb, qos_image)

        self.create_subscription(
            Imu, '/imu/filtered',
            self._imu_cb, qos_imu)

        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self._px4_odom_cb, qos_px4)

        self.create_subscription(
            LaserScan, '/mtf01/lidar',
            self._mtf01_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.slam_pose_pub    = self.create_publisher(
            PoseStamped, '/slam/pose', qos_reliable)
        self.slam_kf_pub      = self.create_publisher(
            PoseStamped, '/slam/keyframe', qos_reliable)
        self.slam_quality_pub = self.create_publisher(
            Float32, '/slam/tracking_quality', qos_reliable)
        self.slam_feat_pub    = self.create_publisher(
            Int32, '/slam/debug/feature_count', qos_reliable)

        # ── OpenCV ────────────────────────────────────────────────────────────
        self.orb = cv2.ORB_create(
            nfeatures=N_FEATURES,
            scaleFactor=SCALE_FACTOR,
            nlevels=N_LEVELS,
            fastThreshold=FAST_THRESHOLD)

        self.lk_params = dict(
            winSize=LK_WIN_SIZE,
            maxLevel=LK_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      LK_MAX_ITER, LK_EPSILON))

        self.bridge = CvBridge()

        self.K = np.array([
            [FX,  0, CX],
            [ 0, FY, CY],
            [ 0,  0,  1]], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1))

        # ── World-frame pose (what we publish) ────────────────────────────────
        self.pose_x   = SPAWN_X
        self.pose_y   = SPAWN_Y
        self.pose_z   = 0.0
        self.pose_yaw = 0.0

        # ── Keyframe state ────────────────────────────────────────────────────
        self.kf_left_img    = None
        self.kf_points_2d   = None   # (N,2) 2D pixel coords at keyframe
        self.kf_points_3d   = None   # (N,3) 3D world points at keyframe
        self.kf_pose        = None   # (x, y, z, yaw) at last keyframe
        self.kf_descriptors = None   # ORB descriptors (for loop closure later)

        # ── Per-frame tracking state ──────────────────────────────────────────
        self.prev_left_img = None    # grayscale left image, previous frame
        self.prev_pts      = None    # (N,2) tracked 2D points
        self.prev_pts_3d   = None    # (N,3) corresponding 3D world points

        # ── PX4 fallback (always updated from odometry) ───────────────────────
        self.px4_x   = None
        self.px4_y   = None
        self.px4_z   = 0.0
        self.px4_yaw = 0.0

        # ── Sensors ───────────────────────────────────────────────────────────
        self.yaw_rate = 0.0   # from IMU — used for gate
        self.lidar_z  = 0.0   # from MTF-01 — used for Z always
        self.right_img = None  # latest right stereo image

        # ── State flags ───────────────────────────────────────────────────────
        self.tracking_ok    = False
        self.lost_count     = 0
        self.frame_count    = 0
        self.keyframe_count = 0

        self.get_logger().info('SLAM Frontend started ✓')
        self.get_logger().info('Waiting for stereo images and odometry...')

    # ═════════════════════════════════════════════════════════════════════════
    # Subscriber callbacks
    # ═════════════════════════════════════════════════════════════════════════

    def _right_cb(self, msg: Image):
        """Store latest right stereo image for depth computation."""
        self.right_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

    def _imu_cb(self, msg: Imu):
        """Extract yaw rate for IMU gate."""
        self.yaw_rate = msg.angular_velocity.z

    def _mtf01_cb(self, msg: LaserScan):
        """MTF-01 downward LiDAR — single range = altitude above ground."""
        if msg.ranges and math.isfinite(msg.ranges[0]) and msg.ranges[0] > 0.01:
            self.lidar_z = float(msg.ranges[0])

    def _px4_odom_cb(self, msg: VehicleOdometry):
        """
        PX4 NED odometry → world frame.
        msg.position = [North, East, Down]
        msg.q        = [w, x, y, z]
        """
        self.px4_x = float(msg.position[1]) + SPAWN_X   # East  → World X
        self.px4_y = float(msg.position[0]) + SPAWN_Y   # North → World Y
        self.px4_z = -float(msg.position[2])             # Down  → World Z (unused, lidar used instead)

        q = msg.q  # [w, x, y, z]
        siny = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy = 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
        self.px4_yaw = _wrap_angle(math.atan2(siny, cosy) - math.pi / 2.0)

        # Before any visual tracking, seed pose from PX4
        if not self.tracking_ok and self.prev_left_img is None:
            self.pose_x   = self.px4_x
            self.pose_y   = self.px4_y
            self.pose_z   = self.lidar_z
            self.pose_yaw = self.px4_yaw

    # ═════════════════════════════════════════════════════════════════════════
    # Main tracking callback
    # ═════════════════════════════════════════════════════════════════════════

    def _left_cb(self, msg: Image):
        """Process each left stereo frame."""
        gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self.frame_count += 1

        # ── IMU gate: skip frame during fast rotation ─────────────────────────
        if abs(self.yaw_rate) > MAX_YAW_RATE:
            self.get_logger().info(
                f'IMU gate: yaw_rate={self.yaw_rate:.2f} rad/s — skipping',
                throttle_duration_sec=2.0)
            # Still publish last known pose — do not go silent
            self.pose_z   = self.lidar_z
            self.pose_yaw = self.px4_yaw
            self._publish_pose()
            self._publish_quality(0.0, 0)
            return

        # ── First frame: initialize ───────────────────────────────────────────
        if self.prev_left_img is None:
            self._initialize(gray)
            return

        # ── Track features with KLT + PnP ────────────────────────────────────
        n_tracked, curr_pts, curr_pts_3d, n_inliers = self._track(gray)

        if n_tracked < MIN_TRACKED_POINTS:
            self.get_logger().info(
                f'Too few points ({n_tracked}) — reinitializing',
                throttle_duration_sec=2.0)
            self._use_px4_fallback()
            self._initialize(gray)
        else:
            self.tracking_ok   = True
            self.prev_left_img = gray
            self.prev_pts      = curr_pts
            self.prev_pts_3d   = curr_pts_3d

        # ── Keyframe decision ─────────────────────────────────────────────────
        curr_pose = (self.pose_x, self.pose_y, self.pose_z, self.pose_yaw)
        kf_pts    = self.kf_points_2d
        check_pts = curr_pts if n_tracked >= MIN_TRACKED_POINTS else np.empty((0, 2))
        if self._is_keyframe(check_pts, kf_pts, curr_pose):
            self._store_keyframe(gray)
            self._publish_keyframe()

        # ── Publish ───────────────────────────────────────────────────────────
        self._publish_pose()
        quality = self._tracking_quality(n_inliers, n_tracked)
        self._publish_quality(quality, n_tracked)

    # ═════════════════════════════════════════════════════════════════════════
    # Initialization
    # ═════════════════════════════════════════════════════════════════════════

    def _initialize(self, gray: np.ndarray):
        """
        Extract ORB features on first frame (or after reinitialization).
        Compute stereo depth for each keypoint.
        Store as previous frame state.
        """
        kps, descs = self.orb.detectAndCompute(gray, None)
        if not kps:
            self.prev_left_img = gray
            return

        pts_2d = np.array([kp.pt for kp in kps], dtype=np.float32)
        depths = self._stereo_depth(gray, self.right_img, pts_2d)

        pts_3d, valid = self._unproject_to_world(
            pts_2d, depths,
            self.pose_x, self.pose_y, self.pose_z, self.pose_yaw)

        # Keep only points with valid stereo depth
        if valid.sum() > 0:
            pts_2d = pts_2d[valid]
            pts_3d = pts_3d[valid]

        self.prev_left_img = gray
        self.prev_pts      = pts_2d
        self.prev_pts_3d   = pts_3d
        self.kf_descriptors = descs

        feat_msg      = Int32()
        feat_msg.data = len(kps)
        self.slam_feat_pub.publish(feat_msg)

        self.get_logger().info(
            f'Initialized: {len(pts_2d)} points with valid depth',
            throttle_duration_sec=2.0)

    # ═════════════════════════════════════════════════════════════════════════
    # KLT tracking + PnP
    # ═════════════════════════════════════════════════════════════════════════

    def _track(self, gray: np.ndarray):
        """
        1. KLT optical flow from prev to curr frame
        2. solvePnPRansac to estimate camera pose change
        3. Augment with new ORB points to keep feature count healthy
        Returns (n_tracked, curr_pts_2d, curr_pts_3d, n_inliers)
        """
        if self.prev_pts is None or len(self.prev_pts) == 0:
            return 0, np.empty((0, 2)), np.empty((0, 3)), 0

        # ── KLT ──────────────────────────────────────────────────────────────
        prev_klt = self.prev_pts.reshape(-1, 1, 2).astype(np.float32)
        curr_klt, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_left_img, gray, prev_klt, None, **self.lk_params)

        if status is None:
            return 0, np.empty((0, 2)), np.empty((0, 3)), 0

        status   = status.reshape(-1)
        good     = status == 1
        curr_pts = curr_klt.reshape(-1, 2)[good]
        prev_3d  = (self.prev_pts_3d[good]
                    if self.prev_pts_3d is not None and len(self.prev_pts_3d) > 0
                    else np.empty((0, 3)))

        n_tracked = len(curr_pts)

        if n_tracked < MIN_TRACKED_POINTS or len(prev_3d) < MIN_PNP_INLIERS:
            return n_tracked, curr_pts, prev_3d, 0

        # ── PnP RANSAC ────────────────────────────────────────────────────────
        n_inliers = self._estimate_pose(curr_pts, prev_3d)

        # ── Refresh 3D coords using updated pose ──────────────────────────────
        depths = self._stereo_depth(gray, self.right_img, curr_pts)
        curr_pts_3d, _ = self._unproject_to_world(
            curr_pts, depths,
            self.pose_x, self.pose_y, self.pose_z, self.pose_yaw)

        # ── Augment with new ORB detections ───────────────────────────────────
        curr_pts, curr_pts_3d = self._augment_points(gray, curr_pts, curr_pts_3d)

        feat_msg      = Int32()
        feat_msg.data = len(curr_pts)
        self.slam_feat_pub.publish(feat_msg)

        return len(curr_pts), curr_pts, curr_pts_3d, n_inliers

    def _estimate_pose(self, pts_2d: np.ndarray, pts_3d: np.ndarray) -> int:
        """
        solvePnPRansac with world-frame 3D points.
        Updates self.pose_x / pose_y using complementary filter.
        Z always from LiDAR. Yaw always from PX4 IMU.
        Returns number of inliers (0 on failure).
        """
        if len(pts_2d) < MIN_PNP_INLIERS:
            self._use_px4_fallback()
            return 0

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d.astype(np.float64),
                pts_2d.astype(np.float64),
                self.K,
                self.dist_coeffs,
                iterationsCount=PNP_ITERATIONS,
                reprojectionError=PNP_REPROJECTION_ERR,
                confidence=PNP_CONFIDENCE,
                flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            self._use_px4_fallback()
            return 0

        n_inliers = len(inliers) if inliers is not None else 0

        if not success or n_inliers < MIN_PNP_INLIERS:
            self._use_px4_fallback()
            return n_inliers

        # Invert world→camera transform to get camera position in world frame
        R, _   = cv2.Rodrigues(rvec)
        cam_pos = -R.T @ tvec   # shape (3,1)
        new_x   = float(cam_pos[0])
        new_y   = float(cam_pos[1])

        # Sanity check: reject jumps > 1m per frame (bad PnP solution)
        if abs(new_x - self.pose_x) > 1.0 or abs(new_y - self.pose_y) > 1.0:
            self.get_logger().info(
                f'PnP jump rejected ({new_x:.2f},{new_y:.2f}) → PX4 fallback',
                throttle_duration_sec=2.0)
            self._use_px4_fallback()
            return n_inliers

        # Complementary filter on XY: blend visual + PX4
        # Z: always LiDAR
        # Yaw: always PX4 IMU
        if self.px4_x is not None:
            self.pose_x = ALPHA * new_x + (1.0 - ALPHA) * self.px4_x
            self.pose_y = ALPHA * new_y + (1.0 - ALPHA) * self.px4_y
        else:
            self.pose_x = new_x
            self.pose_y = new_y

        self.pose_z   = self.lidar_z
        self.pose_yaw = self.px4_yaw

        self.tracking_ok = True
        self.lost_count  = 0

        return n_inliers

    # ═════════════════════════════════════════════════════════════════════════
    # Feature augmentation
    # ═════════════════════════════════════════════════════════════════════════

    def _augment_points(self, gray, curr_pts, curr_pts_3d):
        """
        Detect new ORB points not close to existing tracked points.
        Merge into tracked set up to N_FEATURES total.
        """
        kps, _ = self.orb.detectAndCompute(gray, None)
        if not kps:
            return curr_pts, curr_pts_3d

        new_pts_2d = np.array([kp.pt for kp in kps], dtype=np.float32)

        # Remove new points too close to already-tracked ones
        if len(curr_pts) > 0:
            keep = []
            for pt in new_pts_2d:
                dists = np.linalg.norm(curr_pts - pt, axis=1)
                if dists.min() > 10.0:
                    keep.append(pt)
            if not keep:
                return curr_pts, curr_pts_3d
            new_pts_2d = np.array(keep, dtype=np.float32)

        budget = N_FEATURES - len(curr_pts)
        if budget <= 0:
            return curr_pts, curr_pts_3d

        new_pts_2d = new_pts_2d[:budget]
        depths     = self._stereo_depth(gray, self.right_img, new_pts_2d)
        new_pts_3d, _ = self._unproject_to_world(
            new_pts_2d, depths,
            self.pose_x, self.pose_y, self.pose_z, self.pose_yaw)

        all_pts_2d = np.vstack([curr_pts,    new_pts_2d])
        all_pts_3d = np.vstack([curr_pts_3d, new_pts_3d])
        return all_pts_2d, all_pts_3d

    # ═════════════════════════════════════════════════════════════════════════
    # Stereo depth
    # ═════════════════════════════════════════════════════════════════════════

    def _stereo_depth(self, left_img, right_img, pts_2d):
        """
        For each 2D point, search along epipolar line in right image using SAD.
        Returns array of depths (nan where stereo fails).
        depth = FX * BASELINE / disparity
        """
        depths = np.full(len(pts_2d), np.nan)
        if right_img is None:
            return depths

        for i, (u, v) in enumerate(pts_2d):
            u, v = int(u), int(v)
            if not (5 <= u < left_img.shape[1] - 5 and
                    5 <= v < left_img.shape[0] - 5):
                continue

            patch_l   = left_img[v-5:v+6, u-5:u+6].astype(np.float32)
            best_sad  = float('inf')
            best_disp = -1

            for d in range(4, min(u, 128)):
                ur = u - d
                if ur < 5:
                    break
                patch_r = right_img[v-5:v+6, ur-5:ur+6].astype(np.float32)
                sad     = np.sum(np.abs(patch_l - patch_r))
                if sad < best_sad:
                    best_sad  = sad
                    best_disp = d

            if best_disp > 0 and best_sad < 5000:
                depth = FX * BASELINE / best_disp
                if MIN_STEREO_DEPTH <= depth <= MAX_STEREO_DEPTH:
                    depths[i] = depth

        return depths

    # ═════════════════════════════════════════════════════════════════════════
    # 3D unprojection
    # ═════════════════════════════════════════════════════════════════════════

    def _unproject_to_world(self, pts_2d, depths, pose_x, pose_y, pose_z, pose_yaw):
        """
        Convert 2D pixel + stereo depth → 3D world point.

        Step 1 — pixel to camera frame:
            x_cam = (u - CX) * d / FX
            y_cam = (v - CY) * d / FY
            z_cam = d

        Step 2 — camera to body frame (OAK-D faces forward):
            body_x =  z_cam   (forward)
            body_y = -x_cam   (left)
            body_z = -y_cam   (up)

        Step 3 — body to world frame (rotate by drone yaw, translate by pose):
            world_x = pose_x + cos(yaw)*body_x - sin(yaw)*body_y
            world_y = pose_y + sin(yaw)*body_x + cos(yaw)*body_y
            world_z = pose_z + body_z
        """
        pts_3d = []
        valid  = []
        cos_y  = math.cos(pose_yaw)
        sin_y  = math.sin(pose_yaw)

        for i, (u, v) in enumerate(pts_2d):
            d = depths[i]
            if not np.isfinite(d) or d <= 0:
                valid.append(False)
                pts_3d.append([0.0, 0.0, 0.0])
                continue

            x_cam =  (u - CX) * d / FX
            y_cam =  (v - CY) * d / FY
            z_cam =  d

            body_x =  z_cam
            body_y = -x_cam
            body_z = -y_cam

            wx = pose_x + cos_y * body_x - sin_y * body_y
            wy = pose_y + sin_y * body_x + cos_y * body_y
            wz = pose_z + body_z

            pts_3d.append([wx, wy, wz])
            valid.append(True)

        return (np.array(pts_3d, dtype=np.float64),
                np.array(valid,  dtype=bool))

    # ═════════════════════════════════════════════════════════════════════════
    # Keyframe logic
    # ═════════════════════════════════════════════════════════════════════════

    def _is_keyframe(self, curr_pts, kf_pts_2d, curr_pose):
        """
        Decide if current frame should become a keyframe.
        Three conditions (any one triggers):
          1. No keyframe exists yet
          2. Median pixel parallax vs last keyframe > MIN_PARALLAX
          3. Euclidean distance from last keyframe pose > MAX_KEYFRAME_DIST
        """
        # Condition 1 — first keyframe
        if self.kf_pose is None:
            return True

        # Condition 2 — parallax
        if (kf_pts_2d is not None and
                len(curr_pts) > 0 and
                len(kf_pts_2d) > 0):
            n    = min(len(curr_pts), len(kf_pts_2d))
            flow = np.linalg.norm(curr_pts[:n] - kf_pts_2d[:n], axis=1)
            if np.median(flow) > MIN_PARALLAX:
                return True

        # Condition 3 — distance
        dist = math.sqrt(
            (curr_pose[0] - self.kf_pose[0]) ** 2 +
            (curr_pose[1] - self.kf_pose[1]) ** 2)
        if dist > MAX_KEYFRAME_DIST:
            return True

        return False

    def _store_keyframe(self, gray: np.ndarray):
        """Save current frame as keyframe reference."""
        self.kf_left_img  = gray.copy()
        self.kf_points_2d = (self.prev_pts.copy()
                             if self.prev_pts is not None else None)
        self.kf_points_3d = (self.prev_pts_3d.copy()
                             if self.prev_pts_3d is not None else None)
        self.kf_pose      = (self.pose_x, self.pose_y,
                             self.pose_z, self.pose_yaw)

    # ═════════════════════════════════════════════════════════════════════════
    # Fallback and quality
    # ═════════════════════════════════════════════════════════════════════════

    def _use_px4_fallback(self):
        """When visual tracking fails, fall back to PX4 odometry for XY."""
        if self.px4_x is not None:
            self.pose_x   = self.px4_x
            self.pose_y   = self.px4_y
        self.pose_z       = self.lidar_z    # always LiDAR
        self.pose_yaw     = self.px4_yaw    # always IMU
        self.tracking_ok  = False
        self.lost_count  += 1

    def _tracking_quality(self, n_inliers: int, n_tracked: int) -> float:
        """Returns 0.0 (lost) to 1.0 (perfect)."""
        if not self.tracking_ok:
            return 0.0
        return float(
            min(1.0, n_inliers / MIN_PNP_INLIERS) *
            min(1.0, n_tracked / MIN_TRACKED_POINTS))

    # ═════════════════════════════════════════════════════════════════════════
    # Publishers
    # ═════════════════════════════════════════════════════════════════════════

    def _publish_pose(self):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.position.x = float(self.pose_x)
        msg.pose.position.y = float(self.pose_y)
        msg.pose.position.z = float(self.pose_z)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(self.pose_yaw / 2.0)
        msg.pose.orientation.w = math.cos(self.pose_yaw / 2.0)
        self.slam_pose_pub.publish(msg)

    def _publish_keyframe(self):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.position.x = float(self.pose_x)
        msg.pose.position.y = float(self.pose_y)
        msg.pose.position.z = float(self.pose_z)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(self.pose_yaw / 2.0)
        msg.pose.orientation.w = math.cos(self.pose_yaw / 2.0)
        self.slam_kf_pub.publish(msg)
        self.keyframe_count += 1
        self.get_logger().info(
            f'Keyframe #{self.keyframe_count} at '
            f'({self.pose_x:.2f}, {self.pose_y:.2f}, {self.pose_z:.2f})')

    def _publish_quality(self, quality: float, n_tracked: int):
        q_msg      = Float32()
        q_msg.data = quality
        self.slam_quality_pub.publish(q_msg)
        self.get_logger().info(
            f'quality={quality:.2f} tracked={n_tracked} '
            f'lost={self.lost_count} lidar_z={self.lidar_z:.2f}',
            throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = SlamFrontend()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()