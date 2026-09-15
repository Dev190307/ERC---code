#!/usr/bin/env python3
"""
ERC 2026 - consolidated final approach node.

Combines every piece confirmed working this session:
  1. Rotate (confirmed CW direction) + center on the shelf number (OCR)
  2. Sweep head tilts row by row, looking for the target book colour,
     to get its approximate 3D position
  3. STRAFE sideways only (zero rotation - orientation stays frozen at the
     heading confirmed correct during centering) to align precisely with
     the BOOK's actual lateral position - not just the shelf number's,
     since books are randomised within their row and can sit off to either
     side of where the number itself was centered
  4. Drive STRAIGHT forward only (zero rotation) until within arm's reach
  5. One final close-range colour scan for an accurate position, then
     publish it on /erc/target_book_point for manipulation_node to grasp

Never rotates again after the initial centering step - avoids the "ends up
tilted" problem entirely by construction, since translation-only movement
cannot change the robot's heading.
"""
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge

try:
    import pytesseract
    HAVE_TESSERACT = True
except ImportError:
    HAVE_TESSERACT = False

# ============================== CONSTANTS ==============================
RGB_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
DEPTH_TOPIC = '/head_front_camera/head_front_camera/depth/image_rect_raw'
CAMERA_INFO_TOPIC = '/head_front_camera/head_front_camera/color/camera_info'
CAMERA_OPTICAL_FRAME = 'head_front_camera_color_optical_frame'
GRASP_POINT_TOPIC = '/erc/target_book_point'
COL_TOPIC = '/erc/shelf_column_identification'
ROW_TOPIC = '/erc/shelf_row_identification'

TILT_UP = 0.3
ROTATE_SPEED = -0.4        # confirmed clockwise (REP103: negative = CW)
CENTER_TOLERANCE_PX = 15
CENTER_GAIN = 0.0015
SEARCH_TIMEOUT_S = 40.0

BOOK_TILT_POSITIONS = [-0.2, 0.0, 0.2, 0.4, 0.6]  # top row to bottom row
BOOK_DWELL_SEC = 2.0
MIN_CONTOUR_AREA = 100

# Known robot/shelf dimensions (per the competition spec):
#   - Shelf depth: only 30cm - books sit within a shallow band
#   - TIAGo Pro per-arm horizontal reach: ~118cm (236cm combined / 2 arms)
#   - We stop with the shelf FACE this far away (LIDAR-measured) - leaves
#     enough margin that the book (up to 30cm further back) is still
#     comfortably within the arm's reach, with some clearance for the
#     resting arm pose not to touch the shelf while driving
STOP_DISTANCE_M = 0.75
WAYPOINT_POS_TOLERANCE_M = 0.05
WAYPOINT_MAX_LINEAR = 0.3

COLOR_RANGES = {
    "red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    "blue": [(np.array([100, 120, 70]), np.array([130, 255, 255]))],
    "green": [(np.array([40, 70, 70]), np.array([80, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
}
# =========================================================================


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class FinalApproachNode(Node):
    def __init__(self):
        super().__init__('final_approach_node')
        self.declare_parameter('target_column', 1)
        self.declare_parameter('book_colour', 'red')
        self.target_column = self.get_parameter('target_column').value
        self.target_colour = self.get_parameter('book_colour').value

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.fx = self.fy = self.cx = self.cy = None
        self.front_range = None
        self.current_x = self.current_y = self.current_yaw = None
        self.digit_templates = self._build_digit_templates()

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(Image, RGB_TOPIC, self.rgb_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan_front_raw', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.head_pub = self.create_publisher(JointTrajectory, '/head_controller/joint_trajectory', 10)
        self.col_pub = self.create_publisher(Int32, COL_TOPIC, 10)
        self.row_pub = self.create_publisher(Int32, ROW_TOPIC, 10)
        self.point_pub = self.create_publisher(PointStamped, GRASP_POINT_TOPIC, 10)

        self.state = 'HEAD_INIT'
        self._wait_ticks = int(1.2 / 0.1)
        self._search_elapsed = 0.0
        self._lost_ticks = 0
        self._book_tilt_idx = 0
        self._book_dwell = 0.0
        self._book_tilt_sent = False
        self._frozen_yaw = None
        self._lateral_offset_m = None   # +ve = book is to the robot's right
        self._book_row = None

        self.get_logger().info(f'Searching for shelf {self.target_column}, then {self.target_colour} book...')
        self.create_timer(0.1, self.control_loop)

    # ---------------------------------------------------------- callbacks
    def camera_info_cb(self, msg):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def rgb_cb(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def scan_cb(self, msg):
        n = len(msg.ranges)
        if n == 0:
            return
        mid = n // 2
        span = max(1, int(n * (15.0 / 180.0)))
        window = [r for r in msg.ranges[mid - span:mid + span]
                  if r > 0.05 and not np.isinf(r) and not np.isnan(r)]
        self.front_range = min(window) if window else None

    def odom_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quat(msg.pose.pose.orientation)

    # ---------------------------------------------------------- digit OCR
    def _build_digit_templates(self):
        templates = {}
        for d in range(1, 6):
            canvas = np.zeros((64, 64), dtype=np.uint8)
            cv2.putText(canvas, str(d), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.8, 255, 4, cv2.LINE_AA)
            templates[d] = canvas
        return templates

    def _ocr_digit(self, crop_gray):
        scale = max(1, 100 // max(crop_gray.shape[0], 1))
        big = cv2.resize(crop_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, big_bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if HAVE_TESSERACT:
            for cfg in ['--psm 7 -c tessedit_char_whitelist=12345',
                        '--psm 8 -c tessedit_char_whitelist=12345']:
                txt = pytesseract.image_to_string(big_bw, config=cfg).strip()
                if txt.isdigit() and 1 <= int(txt) <= 5:
                    return int(txt)
        best_score, best_digit = -1, None
        resized = cv2.resize(big_bw, (64, 64))
        for d, tmpl in self.digit_templates.items():
            res = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
            score = res.max()
            if score > best_score:
                best_score, best_digit = score, d
        return best_digit if best_score > 0.35 else None

    def find_target_digit_x(self):
        if self.latest_rgb is None:
            return None
        gray = cv2.cvtColor(self.latest_rgb, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi = gray[0:int(h * 0.6), :]
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 8 or ch < 12 or cw > 200 or ch > 200:
                continue
            crop = roi[y:y + ch, x:x + cw]
            digit = self._ocr_digit(crop)
            if digit == self.target_column:
                return x + cw // 2
        return None

    # ---------------------------------------------------------- colour + depth
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

    def pixel_to_camera_point(self, px, py, depth):
        """Returns (x, y, z) in the camera's OWN optical frame: x = right,
        y = down, z = forward (standard optical frame convention)."""
        x = (px - self.cx) * depth / self.fx
        y = (py - self.cy) * depth / self.fy
        return x, y, depth

    # ---------------------------------------------------------- head/motion helpers
    def send_head(self, pan, tilt):
        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start = Duration(sec=1, nanosec=0)
        traj.points = [point]
        self.head_pub.publish(traj)

    def publish_point(self, cam_x, cam_y, cam_z):
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = CAMERA_OPTICAL_FRAME
        point_msg.point.x = cam_x
        point_msg.point.y = cam_y
        point_msg.point.z = cam_z
        self.point_pub.publish(point_msg)

    # ---------------------------------------------------------- main state machine
    def control_loop(self):
        if self.state == 'HEAD_INIT':
            self.send_head(0.0, TILT_UP)
            self._wait_ticks -= 1
            if self._wait_ticks <= 0:
                self.state = 'SEARCH'
                self.get_logger().info('Head set. Rotating (CW) to search for shelf number...')
            return

        if self.state == 'SEARCH':
            self._search_elapsed += 0.1
            if self._search_elapsed > SEARCH_TIMEOUT_S:
                self.cmd_pub.publish(Twist())
                self.get_logger().error(f'Search timed out, never found shelf {self.target_column}.')
                self.state = 'FAILED'
                return
            cx = self.find_target_digit_x()
            if cx is not None:
                self.cmd_pub.publish(Twist())
                self.col_pub.publish(Int32(data=self.target_column))
                self.get_logger().info(f'Found shelf {self.target_column} at pixel x={cx}. Centering...')
                self.state = 'CENTER'
                return
            twist = Twist()
            twist.angular.z = ROTATE_SPEED
            self.cmd_pub.publish(twist)
            return

        if self.state == 'CENTER':
            cx = self.find_target_digit_x()
            if self.latest_rgb is None:
                return
            img_center = self.latest_rgb.shape[1] / 2.0
            if cx is None:
                self._lost_ticks += 1
                twist = Twist()
                twist.angular.z = ROTATE_SPEED * 0.6
                self.cmd_pub.publish(twist)
                if self._lost_ticks > 80:
                    self.get_logger().warn('Lost target too long, resuming SEARCH')
                    self.state = 'SEARCH'
                    self._lost_ticks = 0
                return
            self._lost_ticks = 0
            error_px = cx - img_center
            if abs(error_px) < CENTER_TOLERANCE_PX:
                self.cmd_pub.publish(Twist())
                if self.current_yaw is None:
                    self.get_logger().warn('No odom yet, waiting...')
                    return
                # FREEZE this heading - nothing after this point ever rotates again
                self._frozen_yaw = self.current_yaw
                self.get_logger().info(
                    f'Centered on shelf {self.target_column} (yaw frozen at {self._frozen_yaw:.2f}). '
                    f'Sweeping rows for {self.target_colour} book...')
                self.state = 'BOOK_SWEEP'
                return
            twist = Twist()
            twist.angular.z = max(min(-CENTER_GAIN * error_px, 0.5), -0.5)
            self.cmd_pub.publish(twist)
            return

        if self.state == 'BOOK_SWEEP':
            if self._book_tilt_idx >= len(BOOK_TILT_POSITIONS):
                self.get_logger().error(f'Swept all rows, never found {self.target_colour} book.')
                self.state = 'FAILED'
                return
            if not self._book_tilt_sent:
                tilt = BOOK_TILT_POSITIONS[self._book_tilt_idx]
                self.send_head(0.0, tilt)
                self._book_dwell = 0.0
                self._book_tilt_sent = True
                self.get_logger().info(f'Row {self._book_tilt_idx + 1}/{len(BOOK_TILT_POSITIONS)}: tilt={tilt}')
                return
            self._book_dwell += 0.1
            if self._book_dwell < 1.2:
                return
            if self.latest_rgb is not None and self.latest_depth is not None and self.fx is not None:
                centre = self.detect_colour(self.latest_rgb)
                if centre is not None:
                    px, py = centre
                    depth = float(self.latest_depth[py, px])
                    if depth > 0.0 and not np.isnan(depth):
                        cam_x, cam_y, cam_z = self.pixel_to_camera_point(px, py, depth)
                        # camera optical frame: +x = right. This IS the
                        # lateral offset we need to strafe to align with
                        # the book (camera looks straight ahead, aligned
                        # with the robot's forward axis at this point).
                        self._lateral_offset_m = cam_x
                        self._book_row = self._book_tilt_idx + 1
                        self.row_pub.publish(Int32(data=self._book_row))
                        self.get_logger().info(
                            f'Found {self.target_colour} book in row {self._book_row}, '
                            f'depth={depth:.2f}m, lateral_offset={cam_x:.2f}m. Strafing to align...')
                        self.state = 'STRAFE_TO_BOOK'
                        return
            if self._book_dwell > BOOK_DWELL_SEC:
                self._book_tilt_idx += 1
                self._book_tilt_sent = False
            return

        if self.state == 'STRAFE_TO_BOOK':
            # Pure sideways motion, ZERO rotation - orientation stays at
            # _frozen_yaw throughout. This is what avoids "ends up tilted":
            # there is no rotation left to introduce drift.
            if self.current_x is None:
                return
            remaining = self._lateral_offset_m
            if abs(remaining) < WAYPOINT_POS_TOLERANCE_M:
                self.cmd_pub.publish(Twist())
                self.get_logger().info('Aligned laterally with book. Driving straight in...')
                self.state = 'APPROACH_STRAIGHT'
                return
            twist = Twist()
            # camera +x (right) maps to robot -y (mecanum convention: +y is
            # left) - strafe toward the book
            speed = max(min(-remaining, WAYPOINT_MAX_LINEAR), -WAYPOINT_MAX_LINEAR)
            twist.linear.y = speed
            self.cmd_pub.publish(twist)
            # decay the remaining offset estimate proportionally to commanded
            # motion per tick (simple open-loop decrement, since we're not
            # re-measuring the offset continuously here)
            self._lateral_offset_m -= speed * 0.1
            return

        if self.state == 'APPROACH_STRAIGHT':
            if self.front_range is None:
                self.get_logger().warn('No LIDAR range yet, waiting...')
                return
            if self.front_range < STOP_DISTANCE_M:
                self.cmd_pub.publish(Twist())
                self.get_logger().info(
                    f'Reached grasp range (front_range={self.front_range:.2f}m). '
                    f'Doing final close-range scan...')
                self.state = 'FINAL_SCAN'
                return
            twist = Twist()
            twist.linear.x = WAYPOINT_MAX_LINEAR
            self.cmd_pub.publish(twist)
            return

        if self.state == 'FINAL_SCAN':
            # one more accurate depth read now that we're close, before
            # handing off to manipulation_node
            if self.latest_rgb is not None and self.latest_depth is not None and self.fx is not None:
                centre = self.detect_colour(self.latest_rgb)
                if centre is not None:
                    px, py = centre
                    depth = float(self.latest_depth[py, px])
                    if depth > 0.0 and not np.isnan(depth):
                        cam_x, cam_y, cam_z = self.pixel_to_camera_point(px, py, depth)
                        self.publish_point(cam_x, cam_y, cam_z)
                        self.get_logger().info(
                            f'=== Final book position published: ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) '
                            f'in camera frame. manipulation_node should grasp now. ===')
                        self.state = 'DONE'
                        return
            self.get_logger().warn('Could not re-spot book at close range - trying again next tick.')
            return

        if self.state in ('DONE', 'FAILED'):
            self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = FinalApproachNode()
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
