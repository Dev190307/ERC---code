"""
perception_book_node.py — Detects the target book (colour) and its 3D position.

Subscribes to RGB + depth camera. Publishes row number to
/erc/shelf_row_identification (scored topic), saves annotated images (scored),
and publishes the book's 3D position (in the camera frame) on a custom topic
your manipulation_node can consume for grasp planning.

NOTE: HSV ranges are starting points from standalone testing -- re-tune
against real Gazebo footage once you have sim access.
"""

import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

RGB_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
DEPTH_TOPIC = '/head_front_camera/head_front_camera/depth/image_rect_raw'
CAMERA_INFO_TOPIC = '/head_front_camera/head_front_camera/color/camera_info'
RESULT_TOPIC = '/erc/shelf_row_identification'
GRASP_POINT_TOPIC = '/erc/target_book_point'  # custom -- consumed by manipulation_node
SAVE_DIR = os.path.expanduser('~/ros2_ws/src/team_falcon_pkg/erc_images')

COLOR_RANGES = {
    "red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    "blue": [(np.array([100, 120, 70]), np.array([130, 255, 255]))],
    "green": [(np.array([40, 70, 70]), np.array([80, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
}
MIN_CONTOUR_AREA = 500

# Shelf rows are the middle 4 rows (per Overview doc); row assignment here
# is a simple vertical-position bucket -- refine once you know real pixel
# boundaries for each row from actual footage.
NUM_ROWS = 4


def detect_book_colour(frame, target_colour):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_RANGES[target_colour]:
        mask |= cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_CONTOUR_AREA:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return {"colour": target_colour, "bbox": (x, y, w, h), "centre": (x + w // 2, y + h // 2)}


def estimate_row(centre_y, frame_height):
    row = int((centre_y / frame_height) * NUM_ROWS) + 1
    return max(1, min(NUM_ROWS, row))


class PerceptionBookNode(Node):
    def __init__(self):
        super().__init__('perception_book_node')

        self.declare_parameter('book_colour', 'red')
        self.target_colour = self.get_parameter('book_colour').value

        self.bridge = CvBridge()
        self.latest_depth = None
        self.fx = self.fy = self.cx = self.cy = None
        self.found_target = False

        self.result_pub = self.create_publisher(Int32, RESULT_TOPIC, 10)
        self.point_pub = self.create_publisher(PointStamped, GRASP_POINT_TOPIC, 10)

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)
        self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 10)

        os.makedirs(SAVE_DIR, exist_ok=True)
        self.get_logger().info(f'Looking for {self.target_colour} book')

    def camera_info_callback(self, msg):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def rgb_callback(self, msg):
        if self.found_target or self.latest_depth is None or self.fx is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        detection = detect_book_colour(frame, self.target_colour)
        if detection is None:
            return

        cx_px, cy_px = detection["centre"]
        row = estimate_row(cy_px, frame.shape[0])

        result_msg = Int32()
        result_msg.data = row
        self.result_pub.publish(result_msg)

        depth_value = float(self.latest_depth[cy_px, cx_px])
        if depth_value <= 0.0 or np.isnan(depth_value):
            return

        point_msg = self._pixel_to_point(cx_px, cy_px, depth_value)
        self.point_pub.publish(point_msg)

        self._save_annotated_image(frame, detection, row)
        self.found_target = True
        self.get_logger().info(
            f'Confirmed {self.target_colour} book at row {row}, '
            f'3D point ({point_msg.point.x:.2f}, {point_msg.point.y:.2f}, {point_msg.point.z:.2f})'
        )

    def _pixel_to_point(self, px, py, depth):
        # Pinhole camera projection: pixel + depth -> 3D point in the
        # camera's own optical frame. manipulation_node must transform
        # this into 'base_link' (or the MoveIt2 planning frame) via tf2
        # before using it as a grasp goal.
        x = (px - self.cx) * depth / self.fx
        y = (py - self.cy) * depth / self.fy
        z = depth

        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'head_front_camera_color_optical_frame'
        point_msg.point.x = x
        point_msg.point.y = y
        point_msg.point.z = z
        return point_msg

    def _save_annotated_image(self, frame, detection, row):
        x, y, w, h = detection["bbox"]
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f'{detection["colour"]} row {row}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(SAVE_DIR, f'book_{timestamp}.png')
        cv2.imwrite(filepath, annotated)
        self.get_logger().info(f'Saved annotated image: {filepath}')


def main():
    rclpy.init()
    node = PerceptionBookNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

