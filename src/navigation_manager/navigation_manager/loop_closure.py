import os
import math
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR        = os.path.expanduser('~/csky_ws/models')
SUPERPOINT_MODEL = os.path.join(MODEL_DIR, 'superpoint.onnx')
SUPERGLUE_MODEL  = os.path.join(MODEL_DIR, 'superglue_indoor.onnx')

# ── ORB ───────────────────────────────────────────────────────────────────────
N_FEATURES          = 500
ORB_SCORE_THRESHOLD = 0.75

# ── Loop closure gate ─────────────────────────────────────────────────────────
QUALITY_THRESHOLD = 0.8
MIN_KEYFRAME_GAP  = 10
MIN_MATCHES_ORB   = 20
MIN_MATCHES_VERIFY = 15
MAX_CANDIDATES    = 3

# ── Geometric verification ────────────────────────────────────────────────────
RANSAC_THRESHOLD = 3.0
MIN_INLIERS      = 12

# ── Database ──────────────────────────────────────────────────────────────────
MAX_KEYFRAMES = 500

# ── Camera intrinsics (OAK-D 320×240) ────────────────────────────────────────
FX = 161.4
FY = 161.4
CX = 160.0
CY = 120.0

# ── Spawn offsets ─────────────────────────────────────────────────────────────
SPAWN_X = 1.0
SPAWN_Y = 3.0


class KeyframeEntry:
    """One entry in the keyframe database."""
    __slots__ = ['kf_id', 'pose_x', 'pose_y', 'pose_z', 'pose_yaw',
                 'image', 'keypoints', 'descriptors', 'timestamp']

    def __init__(self, kf_id, pose_x, pose_y, pose_z, pose_yaw,
                 image, keypoints, descriptors):
        self.kf_id       = kf_id
        self.pose_x      = pose_x
        self.pose_y      = pose_y
        self.pose_z      = pose_z
        self.pose_yaw    = pose_yaw
        self.image       = image
        self.keypoints   = keypoints
        self.descriptors = descriptors
        self.timestamp   = time.time()


class LoopClosure(Node):

    def __init__(self):
        super().__init__('loop_closure')

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        qos_image = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self.create_subscription(PoseStamped, '/slam/keyframe',
                                 self._keyframe_cb, qos_reliable)
        self.create_subscription(Float32, '/slam/tracking_quality',
                                 self._quality_cb, qos_reliable)
        self.create_subscription(Image, '/oakd/sync/left/image',
                                 self._image_cb, qos_image)

        self.loop_edge_pub  = self.create_publisher(
            PoseStamped, '/slam/loop_edge', qos_reliable)
        self.loop_debug_pub = self.create_publisher(
            String, '/slam/loop_debug', qos_reliable)

        self.orb    = cv2.ORB_create(nfeatures=N_FEATURES)
        self.bf     = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.bridge = CvBridge()

        self.keyframe_db     = []
        self.kf_counter      = 0
        self.current_quality = 1.0
        self.latest_image    = None

        self.sp_session = None
        self.sg_session = None
        self._load_models()

        self.get_logger().info('Loop Closure node started ✓')
        if self.sg_session is not None:
            self.get_logger().info('SuperGlue loaded — geometric verification active')
        else:
            self.get_logger().info('SuperGlue not available — using ORB+RANSAC only')

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_models(self):
        """
        Load SuperPoint and SuperGlue ONNX models if available.
        Never crashes — models are optional; ORB+RANSAC is the fallback.
        """
        if not ONNX_AVAILABLE:
            self.get_logger().warn('onnxruntime not installed — using ORB fallback')
            return

        if os.path.exists(SUPERPOINT_MODEL) and os.path.getsize(SUPERPOINT_MODEL) > 100_000:
            try:
                self.sp_session = ort.InferenceSession(
                    SUPERPOINT_MODEL, providers=['CPUExecutionProvider'])
                self.get_logger().info(f'SuperPoint loaded: {SUPERPOINT_MODEL}')
            except Exception as e:
                self.get_logger().warn(f'SuperPoint load failed: {e}')
                self.sp_session = None
        else:
            self.get_logger().warn(
                f'SuperPoint model not found or too small: {SUPERPOINT_MODEL}')

        if os.path.exists(SUPERGLUE_MODEL) and os.path.getsize(SUPERGLUE_MODEL) > 100_000:
            try:
                self.sg_session = ort.InferenceSession(
                    SUPERGLUE_MODEL, providers=['CPUExecutionProvider'])
                self.get_logger().info(f'SuperGlue loaded: {SUPERGLUE_MODEL}')
            except Exception as e:
                self.get_logger().warn(f'SuperGlue load failed: {e}')
                self.sg_session = None
        else:
            self.get_logger().warn(
                f'SuperGlue model not found or too small: {SUPERGLUE_MODEL}')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _quality_cb(self, msg: Float32):
        self.current_quality = float(msg.data)

    def _image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f'image_cb error: {e}')

    def _keyframe_cb(self, msg: PoseStamped):
        """
        Process incoming keyframe:
        1. Extract pose
        2. Extract ORB features from latest image
        3. Store in database
        4. If quality < threshold: attempt loop closure
        """
        pose_x = float(msg.pose.position.x)
        pose_y = float(msg.pose.position.y)
        pose_z = float(msg.pose.position.z)

        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        pose_yaw = 2.0 * math.atan2(qz, qw)

        if self.latest_image is None:
            self.get_logger().warn('No image yet — skipping keyframe')
            return

        image = self.latest_image.copy()

        kps, descs = self.orb.detectAndCompute(image, None)
        if descs is None or len(kps) < 10:
            self.get_logger().info(
                f'Too few features ({len(kps) if kps else 0}) — skipping keyframe',
                throttle_duration_sec=2.0)
            return

        entry = KeyframeEntry(
            kf_id=self.kf_counter,
            pose_x=pose_x, pose_y=pose_y,
            pose_z=pose_z, pose_yaw=pose_yaw,
            image=image,
            keypoints=kps,
            descriptors=descs)

        self.keyframe_db.append(entry)
        self.kf_counter += 1

        if len(self.keyframe_db) > MAX_KEYFRAMES:
            self.keyframe_db.pop(0)

        self.get_logger().info(
            f'Keyframe #{self.kf_counter} stored '
            f'({pose_x:.2f},{pose_y:.2f}) '
            f'quality={self.current_quality:.2f} '
            f'db_size={len(self.keyframe_db)}',
            throttle_duration_sec=2.0)

        if self.current_quality >= QUALITY_THRESHOLD:
            return

        self.get_logger().info(
            f'Quality={self.current_quality:.2f} < {QUALITY_THRESHOLD} '
            f'— attempting loop closure')

        self._attempt_loop_closure(entry)

    # ── ORB candidate search ───────────────────────────────────────────────────

    def _find_orb_candidates(self, query_entry):
        """
        Search database for visually similar keyframes using ORB + Lowe ratio test.
        Skips keyframes within MIN_KEYFRAME_GAP to avoid matching consecutive frames.
        Returns top-k (score, KeyframeEntry) sorted by match count descending.
        """
        candidates = []

        for entry in self.keyframe_db:
            if abs(entry.kf_id - query_entry.kf_id) < MIN_KEYFRAME_GAP:
                continue
            if entry.descriptors is None or len(entry.descriptors) < 10:
                continue

            try:
                matches = self.bf.knnMatch(
                    query_entry.descriptors, entry.descriptors, k=2)
            except Exception:
                continue

            good = []
            for m_list in matches:
                if len(m_list) == 2:
                    m, n = m_list
                    if m.distance < ORB_SCORE_THRESHOLD * n.distance:
                        good.append(m)

            if len(good) >= MIN_MATCHES_ORB:
                candidates.append((len(good), entry))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:MAX_CANDIDATES]

    # ── Geometric verification: ORB + RANSAC ──────────────────────────────────

    def _verify_with_ransac(self, query_entry, candidate_entry):
        """
        Verify loop candidate with homography RANSAC on ORB matches.
        Returns (confirmed: bool, n_inliers: int).
        """
        try:
            matches = self.bf.knnMatch(
                query_entry.descriptors, candidate_entry.descriptors, k=2)
        except Exception:
            return False, 0

        good = []
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < ORB_SCORE_THRESHOLD * n.distance:
                    good.append(m)

        if len(good) < MIN_INLIERS:
            return False, len(good)

        src_pts = np.float32(
            [query_entry.keypoints[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [candidate_entry.keypoints[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)

        if mask is None:
            return False, 0

        n_inliers = int(mask.sum())
        confirmed = (H is not None and n_inliers >= MIN_INLIERS)
        return confirmed, n_inliers

    # ── Geometric verification: SuperGlue ─────────────────────────────────────

    def _verify_with_superglue(self, query_entry, candidate_entry):
        """
        Verify loop candidate using SuperPoint + SuperGlue learned matching.
        Falls back to RANSAC on any inference error.
        Returns (confirmed: bool, n_matches: int).

        NOTE: SuperGlue input names are read dynamically from the ONNX session
        to handle different export variants — never hardcoded.
        """
        if self.sg_session is None or self.sp_session is None:
            return self._verify_with_ransac(query_entry, candidate_entry)

        try:
            img0 = query_entry.image.astype(np.float32) / 255.0
            img1 = candidate_entry.image.astype(np.float32) / 255.0

            img0 = img0[np.newaxis, np.newaxis]
            img1 = img1[np.newaxis, np.newaxis]

            sp_input_name = self.sp_session.get_inputs()[0].name

            sp_out0 = self.sp_session.run(None, {sp_input_name: img0})
            sp_out1 = self.sp_session.run(None, {sp_input_name: img1})

            kpts0, scores0, desc0 = sp_out0[0], sp_out0[1], sp_out0[2]
            kpts1, scores1, desc1 = sp_out1[0], sp_out1[1], sp_out1[2]

            if kpts0.shape[1] < 4 or kpts1.shape[1] < 4:
                self.get_logger().warn(
                    'SuperPoint: too few keypoints — RANSAC fallback')
                return self._verify_with_ransac(query_entry, candidate_entry)

            sg_inputs = {
                'keypoints0':   kpts0.astype(np.float32),
                'keypoints1':   kpts1.astype(np.float32),
                'scores0':      scores0.astype(np.float32),
                'scores1':      scores1.astype(np.float32),
                'descriptors0': desc0.astype(np.float32),
                'descriptors1': desc1.astype(np.float32),
                'image0':       img0,
                'image1':       img1,
            }

            # Match only the input names that this ONNX export actually expects
            sg_input_names = [i.name for i in self.sg_session.get_inputs()]
            sg_feed = {k: v for k, v in sg_inputs.items() if k in sg_input_names}

            sg_out = self.sg_session.run(None, sg_feed)

            # matches0: (1, N) — index of match in image1, -1 = unmatched
            matches0  = sg_out[0].flatten()
            n_matches = int((matches0 > -1).sum())

            confirmed = n_matches >= MIN_MATCHES_VERIFY
            return confirmed, n_matches

        except Exception as e:
            self.get_logger().warn(
                f'SuperGlue inference failed: {e} — RANSAC fallback')
            return self._verify_with_ransac(query_entry, candidate_entry)

    # ── Loop closure pipeline ─────────────────────────────────────────────────

    def _attempt_loop_closure(self, query_entry):
        """
        Full loop closure pipeline:
        1. ORB candidate retrieval
        2. Geometric verification (SuperGlue or RANSAC)
        3. Publish confirmed loop edge
        """
        if len(self.keyframe_db) < MIN_KEYFRAME_GAP + 1:
            return

        candidates = self._find_orb_candidates(query_entry)

        if not candidates:
            self.get_logger().info(
                'Loop closure: no ORB candidates found',
                throttle_duration_sec=3.0)
            self._publish_debug('NO_CANDIDATES')
            return

        self.get_logger().info(
            f'Loop closure: {len(candidates)} ORB candidates found')

        best_confirmed = False
        best_entry     = None
        best_inliers   = 0

        for score, candidate in candidates:
            if self.sg_session is not None:
                confirmed, n_inliers = self._verify_with_superglue(
                    query_entry, candidate)
            else:
                confirmed, n_inliers = self._verify_with_ransac(
                    query_entry, candidate)

            self.get_logger().info(
                f'  Candidate kf#{candidate.kf_id}: '
                f'confirmed={confirmed} inliers={n_inliers}')

            if confirmed and n_inliers > best_inliers:
                best_confirmed = True
                best_entry     = candidate
                best_inliers   = n_inliers

        if not best_confirmed or best_entry is None:
            self.get_logger().info(
                'Loop closure: verification failed for all candidates')
            self._publish_debug('VERIFY_FAILED')
            return

        dist = math.sqrt(
            (query_entry.pose_x - best_entry.pose_x) ** 2 +
            (query_entry.pose_y - best_entry.pose_y) ** 2)

        self.get_logger().info(
            f'LOOP CLOSURE CONFIRMED! '
            f'kf#{query_entry.kf_id} → kf#{best_entry.kf_id} '
            f'inliers={best_inliers} '
            f'dist={dist:.2f}m')

        self._publish_loop_edge(query_entry, best_entry)
        self._publish_debug(
            f'CONFIRMED kf#{query_entry.kf_id}→kf#{best_entry.kf_id} '
            f'inliers={best_inliers}')

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_loop_edge(self, query_entry, match_entry):
        """
        Publish confirmed loop edge as PoseStamped.

        position   = query keyframe pose (current frame)
        orientation = match keyframe pose encoded compactly:
            orientation.x = match_entry.pose_x
            orientation.y = match_entry.pose_y
            orientation.z = match_entry.pose_z
            orientation.w = match_entry.pose_yaw
        This is NOT a unit quaternion — it is a 4-field compact encoding
        for the pose_graph node to unpack. Do not normalize it.
        """
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.pose.position.x = float(query_entry.pose_x)
        msg.pose.position.y = float(query_entry.pose_y)
        msg.pose.position.z = float(query_entry.pose_z)

        msg.pose.orientation.x = float(match_entry.pose_x)
        msg.pose.orientation.y = float(match_entry.pose_y)
        msg.pose.orientation.z = float(match_entry.pose_z)
        msg.pose.orientation.w = float(match_entry.pose_yaw)

        self.loop_edge_pub.publish(msg)

    def _publish_debug(self, status: str):
        msg = String()
        msg.data = f'[loop_closure] {status}'
        self.loop_debug_pub.publish(msg)
        self.get_logger().info(msg.data, throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = LoopClosure()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
