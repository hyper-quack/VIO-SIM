import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge

class RGBResizer(Node):
    def __init__(self):
        super().__init__('rgb_resizer')
        
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(
            Image, '/oakd_lite/rgb/image', self.image_callback, qos_sub)
        
        self.info_sub = self.create_subscription(
            CameraInfo, '/oakd_lite/rgb/camera_info', self.info_callback, qos_sub)

        self.image_pub = self.create_publisher(
            Image, '/oakd_lite/rgb/image_resized', qos_pub)
        
        self.info_pub = self.create_publisher(
            CameraInfo, '/oakd_lite/rgb/camera_info_resized', qos_pub)

        self.get_logger().info('RGB Resizer démarré 1920x1080 → 640x480')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        resized = cv2.resize(cv_image, (640, 480))
        out = self.bridge.cv2_to_imgmsg(resized, 'bgr8')
        out.header = msg.header
        self.image_pub.publish(out)
        self.get_logger().info('Image resizée publiée', throttle_duration_sec=2.0)

    def info_callback(self, msg):
        scale_x = 640.0 / 1920.0
        scale_y = 480.0 / 1080.0
        msg.width = 640
        msg.height = 480
        msg.k[0] *= scale_x
        msg.k[2] *= scale_x
        msg.k[4] *= scale_y
        msg.k[5] *= scale_y
        msg.p[0] *= scale_x
        msg.p[2] *= scale_x
        msg.p[5] *= scale_y
        msg.p[6] *= scale_y
        self.info_pub.publish(msg)

def main():
    rclpy.init()
    node = RGBResizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()