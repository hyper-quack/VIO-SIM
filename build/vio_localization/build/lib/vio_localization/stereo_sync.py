import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber


class StereoSync(Node):
    def __init__(self):
        super().__init__('stereo_sync')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.left_img_sub   = Subscriber(self, Image,      '/oakd/left/image',        qos_profile=qos)
        self.right_img_sub  = Subscriber(self, Image,      '/oakd/right/image',       qos_profile=qos)
        self.left_info_sub  = Subscriber(self, CameraInfo, '/oakd/left/camera_info',  qos_profile=qos)
        self.right_info_sub = Subscriber(self, CameraInfo, '/oakd/right/camera_info', qos_profile=qos)

        self.sync = ApproximateTimeSynchronizer(
            [self.left_img_sub, self.right_img_sub,
             self.left_info_sub, self.right_info_sub],
            queue_size=50,
            slop=10.0
        )
        self.sync.registerCallback(self.sync_callback)

        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.left_img_pub   = self.create_publisher(Image,      '/oakd/sync/left/image',        qos_pub)
        self.right_img_pub  = self.create_publisher(Image,      '/oakd/sync/right/image',       qos_pub)
        self.left_info_pub  = self.create_publisher(CameraInfo, '/oakd/sync/left/camera_info',  qos_pub)
        self.right_info_pub = self.create_publisher(CameraInfo, '/oakd/sync/right/camera_info', qos_pub)

        self.get_logger().info('Stereo Sync node démarré ✓')

    def sync_callback(self, left_img, right_img, left_info, right_info):
        t = self.get_clock().now().to_msg()

        left_img.header.stamp   = t
        right_img.header.stamp  = t
        left_info.header.stamp  = t
        right_info.header.stamp = t

        # Baseline OAK-D Lite = 7.5cm
        right_info.p[3] = -right_info.p[0] * 0.075

        self.left_img_pub.publish(left_img)
        self.right_img_pub.publish(right_img)
        self.left_info_pub.publish(left_info)
        self.right_info_pub.publish(right_info)
        self.get_logger().info('Stereo sync OK', throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = StereoSync()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()