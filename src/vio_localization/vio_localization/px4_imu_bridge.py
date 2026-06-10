#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import SensorCombined, VehicleAttitude
from sensor_msgs.msg import Imu

class PX4ImuBridge(Node):
    def __init__(self):
        super().__init__('px4_imu_bridge')
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
        self._attitude = None
        self.create_subscription(SensorCombined,
            '/fmu/out/sensor_combined', self._sensor_cb, qos_px4)
        self.create_subscription(VehicleAttitude,
            '/fmu/out/vehicle_attitude', self._att_cb, qos_px4)
        self._pub = self.create_publisher(Imu, '/px4/imu', 10)
        self.get_logger().info('px4_imu_bridge started')

    def _att_cb(self, msg):
        self._attitude = msg

    def _sensor_cb(self, msg):
        imu = Imu()
        imu.header.stamp = self.get_clock().now().to_msg()
        imu.header.frame_id = 'base_link'
        # FRD → FLU
        imu.angular_velocity.x =  float(msg.gyro_rad[0])
        imu.angular_velocity.y = -float(msg.gyro_rad[1])
        imu.angular_velocity.z = -float(msg.gyro_rad[2])
        imu.linear_acceleration.x =  float(msg.accelerometer_m_s2[0])
        imu.linear_acceleration.y = -float(msg.accelerometer_m_s2[1])
        imu.linear_acceleration.z = -float(msg.accelerometer_m_s2[2])
        if self._attitude is not None:
            imu.orientation.w =  float(self._attitude.q[0])
            imu.orientation.x =  float(self._attitude.q[1])
            imu.orientation.y = -float(self._attitude.q[2])
            imu.orientation.z = -float(self._attitude.q[3])
        else:
            imu.orientation.w = 1.0
        imu.angular_velocity_covariance[0] = 1e-6
        imu.angular_velocity_covariance[4] = 1e-6
        imu.angular_velocity_covariance[8] = 1e-6
        imu.linear_acceleration_covariance[0] = 1e-4
        imu.linear_acceleration_covariance[4] = 1e-4
        imu.linear_acceleration_covariance[8] = 1e-4
        imu.orientation_covariance[0] = 1e-6
        imu.orientation_covariance[4] = 1e-6
        imu.orientation_covariance[8] = 1e-6
        self._pub.publish(imu)

def main():
    rclpy.init()
    rclpy.spin(PX4ImuBridge())

if __name__ == '__main__':
    main()
