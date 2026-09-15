#!/usr/bin/env python3
"""
Stays in place, rotates in a full circle scanning for the target-coloured
book, computes its 3D position via depth once found, and publishes it on
/erc/target_book_point - the exact topic manipulation_node.py is already
subscribed to, so it picks up the grasp automatically.

Does NOT drive toward the shelf at all - purely: twist, spot the book,
hand off to the existing arm/grasp pipeline.
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge

RGB_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
DEPTH_TOPIC = '/head_front_camera/head_front_camera/depth/image_rect_raw'
CAMERA_INFO_TOPIC = '/head_front_camera/head_front_camera/color/camera_info'
GRASP_POINT_TOPIC = '/erc/target_book_point'
CAMERA_OPTICAL_FRAME = 'head_front_camera_color_optical_frame'

TILT = 0.0
ROTATE_SPEED = 0.3
MIN_CONTOUR_AREA = 500

COLOR_RANGES = {
    "red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    "blue": [(np.array([100, 120, 70]), np.array([130, 255, 255]))],
    "green": [(np.array([40, 70, 70]), np.array([80, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
}


class TwistBookFinder(Node):
    def __init__(self):
        super().__init__('twist_book_finder')
        self.declare_parameter('book_colour', 'red')
        self.target_colour = self.get_parameter('book_colour').value

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.fx = self.fy = self.cx = self.cy = None
        self.found = False

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(Image, RGB_TOPIC, self.rgb_cb, qos_profile_sensor_data)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.head_pub = self.create_publisher(JointTrajectory, '/head_controller/joint_trajectory', 10)
        self.point_pub = self.create_publisher(PointStamped, GRASP_POINT_TOPIC, 10)

        self._head_sent = False
        self.get_logger().info(f'Twisting in place, scanning for {self.target_colour} book...')
        self.create_timer(0.1, self.control_loop)

    def camera_info_cb(self, msg):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def rgb_cb(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def send_head(self, pan, tilt):
        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start = Duration(sec=1, nanosec=0)
        traj.points = [point]
        self.head_pub.publish(traj)

    def detect_colour(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in COLOR_RANGES[self.target_colour]:
            mask |= cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
            return None
        x, y, w, h = cv2.boundingRect(largest)
        return (x + w // 2, y + h // 2)

    def pixel_to_point(self, px, py, depth):
        x = (px - self.cx) * depth / self.fx
        y = (py - self.cy) * depth / self.fy
        z = depth
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = CAMERA_OPTICAL_FRAME
        point_msg.point.x = x
        point_msg.point.y = y
        point_msg.point.z = z
        return point_msg

    def control_loop(self):
        if self.found:
            return

        if not self._head_sent:
            self.send_head(0.0, TILT)
            self._head_sent = True
            return

        if self.latest_rgb is not None and self.latest_depth is not None and self.fx is not None:
            centre = self.detect_colour(self.latest_rgb)
            if centre is not None:
                cx_px, cy_px = centre
                depth_value = float(self.latest_depth[cy_px, cx_px])
                if depth_value > 0.0 and not np.isnan(depth_value):
                    self.cmd_pub.publish(Twist())  # stop rotating
                    point_msg = self.pixel_to_point(cx_px, cy_px, depth_value)
                    self.point_pub.publish(point_msg)
                    self.found = True
                    self.get_logger().info(
                        f'Spotted {self.target_colour} book, depth={depth_value:.2f}m, '
                        f'publishing point for manipulation_node to grasp.'
                    )
                    return

        twist = Twist()
        twist.angular.z = ROTATE_SPEED
        self.cmd_pub.publish(twist)


def main():
    rclpy.init()
    node = TwistBookFinder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
