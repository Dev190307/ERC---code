#!/usr/bin/env python3
"""
ERC 2026 Phase 1b - find the target-coloured book on the current shelf
column by tilting the head through each row, then publish its 3D position
(in the camera's own optical frame) on /erc/target_book_point.

manipulation_node.py is already subscribed to that topic and does its own
tf2 transform into the planning frame -- this node's only job is
"find the book, work out where it is in 3D, publish it."

Assumes the robot is already stopped and facing the correct shelf column
(i.e. run this after shelf_locator_node reaches DONE).
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge

RGB_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
DEPTH_TOPIC = '/head_front_camera/head_front_camera/depth/image_rect_raw'
CAMERA_INFO_TOPIC = '/head_front_camera/head_front_camera/color/camera_info'
RESULT_TOPIC = '/erc/shelf_row_identification'
GRASP_POINT_TOPIC = '/erc/target_book_point'
CAMERA_OPTICAL_FRAME = 'head_front_camera_color_optical_frame'

# Head tilt positions to sweep through, top row to bottom row.
# Starting guesses -- retune against your actual shelf/row spacing.
TILT_POSITIONS = [-0.2, 0.0, 0.2, 0.4, 0.6]
DWELL_SEC = 2.0  # time to hold each tilt before giving up and moving to the next

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
    return {"bbox": (x, y, w, h), "centre": (x + w // 2, y + h // 2)}


class BookLocatorNode(Node):
    def __init__(self):
        super().__init__('book_locator_node')
        self.declare_parameter('book_colour', 'red')
        self.target_colour = self.get_parameter('book_colour').value

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.fx = self.fy = self.cx = self.cy = None

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(Image, RGB_TOPIC, self.rgb_cb, qos_profile_sensor_data)

        self.head_pub = self.create_publisher(JointTrajectory, '/head_controller/joint_trajectory', 10)
        self.row_pub = self.create_publisher(Int32, RESULT_TOPIC, 10)
        self.point_pub = self.create_publisher(PointStamped, GRASP_POINT_TOPIC, 10)

        self.tilt_index = 0
        self.dwell_elapsed = 0.0
        self.found = False
        self.shelf_reached = False
        self.state = 'WAIT_FOR_SHELF'

        self.create_subscription(Bool, '/erc/shelf_reached', self.shelf_reached_cb, 10)

        self.get_logger().info('Waiting for robot to reach the shelf before searching for the book...')
        self.create_timer(0.1, self.control_loop)

    def shelf_reached_cb(self, msg):
        if msg.data and not self.shelf_reached:
            self.shelf_reached = True
            self.state = 'MOVE_HEAD'
            self.get_logger().info(f'Shelf reached. Looking for {self.target_colour} book, sweeping {len(TILT_POSITIONS)} rows...')

    def camera_info_cb(self, msg):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def rgb_cb(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def send_head_tilt(self, tilt):
        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.0, tilt]
        point.time_from_start = Duration(sec=1, nanosec=0)
        traj.points = [point]
        self.head_pub.publish(traj)

    def control_loop(self):
        if self.state == 'WAIT_FOR_SHELF':
            return

        if self.found:
            return

        if self.tilt_index >= len(TILT_POSITIONS):
            self.get_logger().error(f'Swept all rows, never found {self.target_colour} book.')
            self.state = 'FAILED'
            return

        if self.state == 'MOVE_HEAD':
            self.send_head_tilt(TILT_POSITIONS[self.tilt_index])
            self.dwell_elapsed = 0.0
            self.state = 'LOOK'
            self.get_logger().info(f'Row {self.tilt_index + 1}/{len(TILT_POSITIONS)}: tilt={TILT_POSITIONS[self.tilt_index]}')
            return

        if self.state == 'LOOK':
            self.dwell_elapsed += 0.1

            # Head trajectory takes ~1s to physically settle -- don't trust
            # detections from before that, or we sample mid-motion frames
            # and mislabel which row we're actually looking at.
            if self.dwell_elapsed < 1.2:
                return

            if self.latest_rgb is not None and self.latest_depth is not None and self.fx is not None:
                detection = detect_book_colour(self.latest_rgb, self.target_colour)
                if detection is not None:
                    cx_px, cy_px = detection["centre"]
                    depth_value = float(self.latest_depth[cy_px, cx_px])
                    self.get_logger().info(
                        f'DEBUG: pixel=({cx_px},{cy_px}) raw_depth={depth_value:.4f} '
                        f'fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}')
                    if depth_value > 0.0 and not np.isnan(depth_value):
                        point_msg = self._pixel_to_point(cx_px, cy_px, depth_value)
                        self.point_pub.publish(point_msg)
                        self.row_pub.publish(Int32(data=self.tilt_index + 1))
                        self.found = True
                        self.state = 'DONE'
                        self.get_logger().info(
                            f'Found {self.target_colour} book at row {self.tilt_index + 1}, '
                            f'3D point ({point_msg.point.x:.2f}, {point_msg.point.y:.2f}, {point_msg.point.z:.2f})'
                        )
                        return

            if self.dwell_elapsed > DWELL_SEC:
                self.tilt_index += 1
                self.state = 'MOVE_HEAD'
            return

    def _pixel_to_point(self, px, py, depth):
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


def main():
    rclpy.init()
    node = BookLocatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
