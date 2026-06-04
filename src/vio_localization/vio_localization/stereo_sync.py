import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber


class StereoSync(Node):
    def __init__(self):
        super().__init__('stereo_sync')

        # Camera_info is static — cache the first message received, then
        # publish it independently.  This avoids including camera_info in the
        # 4-topic synchronizer, which requires all four topics to arrive within
        # slop of each other and was rejecting ~80% of image pairs.
        self._left_info  = None
        self._right_info = None
        self._frame_count   = 0
        # publish every pair — raw rate ~14-17Hz (was 3 = decimation for 60Hz)
        self._publish_every = 1

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Camera_info subscriptions — regular (not message_filters), cache only
        self.create_subscription(
            CameraInfo, '/oakd/left/camera_info',  self._left_info_cb,  10)
        self.create_subscription(
            CameraInfo, '/oakd/right/camera_info', self._right_info_cb, 10)

        # Sync only the two image topics — much easier to match at full rate
        self.left_img_sub  = Subscriber(
            self, Image, '/oakd/left/image',  qos_profile=qos_be)
        self.right_img_sub = Subscriber(
            self, Image, '/oakd/right/image', qos_profile=qos_be)

        # slop=0.10 s is achievable with only 2 topics: gz-bridge serialises
        # left then right within a few ms at 60 Hz (inter-frame ~17 ms).
        # _publish_every=1: publish every pair at full raw rate (~58 Hz).
        self.sync = ApproximateTimeSynchronizer(
            [self.left_img_sub, self.right_img_sub],
            queue_size=10,
            slop=0.10,
        )
        self.sync.registerCallback(self.sync_callback)

        # Publishers
        self.left_img_pub   = self.create_publisher(
            Image,      '/oakd/sync/left/image',        qos_be)
        self.right_img_pub  = self.create_publisher(
            Image,      '/oakd/sync/right/image',       qos_be)
        self.left_info_pub  = self.create_publisher(
            CameraInfo, '/oakd/sync/left/camera_info',  qos_be)
        self.right_info_pub = self.create_publisher(
            CameraInfo, '/oakd/sync/right/camera_info', qos_be)

        self.get_logger().info('StereoSync started ✓ — 2-topic image sync')

    # ── camera_info cache ────────────────────────────────────────────────────

    def _left_info_cb(self, msg: CameraInfo):
        self._left_info = msg

    def _right_info_cb(self, msg: CameraInfo):
        if self._right_info is None:
            # Inject baseline correction once (Tx = -fx * 0.075)
            msg.p[3] = -msg.p[0] * 0.075
        self._right_info = msg

    # ── image sync callback ──────────────────────────────────────────────────

    def sync_callback(self, left_img: Image, right_img: Image):
        self._frame_count += 1
        if self._frame_count % self._publish_every != 0:
            return

        t = left_img.header.stamp

        left_img.header.stamp  = t
        right_img.header.stamp = t

        self.left_img_pub.publish(left_img)
        self.right_img_pub.publish(right_img)

        # Co-publish camera_info with the same timestamp so rtabmap_odom
        # receives image + info with identical stamps in the same callback tick.
        if self._left_info is not None:
            self._left_info.header.stamp = t
            self.left_info_pub.publish(self._left_info)
        if self._right_info is not None:
            self._right_info.header.stamp = t
            self.right_info_pub.publish(self._right_info)

        self.get_logger().info(
            'Stereo sync OK', throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = StereoSync()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
