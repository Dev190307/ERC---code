#!/usr/bin/env python3
"""
ERC 2026 Phase 1 - find target shelf column, face it, approach, stop.
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d
from std_msgs.msg import Int32, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge

try:
    import pytesseract
    HAVE_TESSERACT = True
except ImportError:
    HAVE_TESSERACT = False

TILT_UP = 0.3
ROTATE_SPEED = -0.4
CENTER_TOLERANCE_PX = 15
CENTER_GAIN = 0.0015
APPROACH_STOP_DISTANCE_M = 0.4
SEARCH_TIMEOUT_S = 40.0
WAYPOINT_POS_TOLERANCE_M = 0.05
WAYPOINT_YAW_TOLERANCE_RAD = 0.03
WAYPOINT_MAX_LINEAR = 0.4
WAYPOINT_MAX_ANGULAR = 0.3
APPROACH_TIMEOUT_S = 25.0


class ShelfLocatorNode(Node):
    def __init__(self):
        super().__init__('shelf_locator_node')
        self.declare_parameter('target_column', 1)
        self.target_column = self.get_parameter('target_column').value

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.front_range = None

        self.create_subscription(
            Image, '/head_front_camera/head_front_camera/color/image_raw',
            self.rgb_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan_front_raw', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self._waypoint = None
        self._approach_elapsed = 0.0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.head_pub = self.create_publisher(JointTrajectory, '/head_controller/joint_trajectory', 10)
        self.col_pub = self.create_publisher(Int32, '/erc/shelf_column_identification', 10)
        self.reached_pub = self.create_publisher(Bool, '/erc/shelf_reached', 10)

        self.digit_templates = self._build_digit_templates()
        self.state = 'HEAD_INIT'
        self._wait_ticks = int(1.2 / 0.1)
        self._search_elapsed = 0.0
        self._lost_ticks = 0

        self.get_logger().info(f'Searching for shelf column {self.target_column}...')
        self.create_timer(0.1, self.control_loop)

    def rgb_cb(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def odom_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quat(msg.pose.pose.orientation)

    def scan_cb(self, msg):
        n = len(msg.ranges)
        if n == 0:
            return
        mid = n // 2
        span = max(1, int(n * (15.0 / 180.0)))
        window = [r for r in msg.ranges[mid - span:mid + span]
                  if r > 0.05 and not np.isinf(r) and not np.isnan(r)]
        self.front_range = min(window) if window else None

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

    def send_head(self, pan, tilt):
        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start = Duration(sec=1, nanosec=0)
        traj.points = [point]
        self.head_pub.publish(traj)

    def control_loop(self):
        if self.state == 'HEAD_INIT':
            self.send_head(0.0, TILT_UP)
            self._wait_ticks -= 1
            if self._wait_ticks <= 0:
                self.state = 'SEARCH'
                self.get_logger().info('Head set. Rotating to search...')
            return

        if self.state == 'SEARCH':
            self._search_elapsed += 0.1
            if self._search_elapsed > SEARCH_TIMEOUT_S:
                self.cmd_pub.publish(Twist())
                self.get_logger().error(f'Search timed out, never found column {self.target_column}.')
                self.state = 'FAILED'
                return
            cx = self.find_target_digit_x()
            if cx is not None:
                self.cmd_pub.publish(Twist())
                self.col_pub.publish(Int32(data=self.target_column))
                self.get_logger().info(f'Found column {self.target_column} at pixel x={cx}. Centering...')
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
                self.get_logger().info(f'CENTER: lost target, nudging (lost_ticks={self._lost_ticks})')
                if self._lost_ticks > 80:
                    self.get_logger().warn('CENTER: lost target too long, resuming full SEARCH')
                    self.state = 'SEARCH'
                    self._lost_ticks = 0
                return
            self._lost_ticks = 0
            error_px = cx - img_center
            self.get_logger().info(f'CENTER: cx={cx} error_px={error_px:.0f}')
            if abs(error_px) < CENTER_TOLERANCE_PX:
                self.cmd_pub.publish(Twist())
                if self.current_x is None:
                    self.get_logger().warn('No odom yet, waiting to compute waypoint...')
                    return
                dist_to_travel = self.front_range - APPROACH_STOP_DISTANCE_M if self.front_range else 2.0
                dist_to_travel = max(dist_to_travel, 0.3)
                wp_x = self.current_x + dist_to_travel * math.cos(self.current_yaw)
                wp_y = self.current_y + dist_to_travel * math.sin(self.current_yaw)
                self._waypoint = (wp_x, wp_y, self.current_yaw)
                self.get_logger().info(
                    f'Centered. Recorded waypoint ({wp_x:.2f}, {wp_y:.2f}), '
                    f'dist={dist_to_travel:.2f}m. Driving to it...')
                self._approach_elapsed = 0.0
                self.state = 'WAYPOINT_APPROACH'
                return
            twist = Twist()
            twist.angular.z = max(min(-CENTER_GAIN * error_px, 0.5), -0.5)
            self.cmd_pub.publish(twist)
            return

        if self.state == 'WAYPOINT_APPROACH':
            # ORIENTATION IS FROZEN at wp_yaw (the heading confirmed correct
            # during CENTER) for this entire phase - angular.z is NEVER
            # touched here. We reach the target using pure translation on
            # TIAGo's mecanum base: forward/backward (linear.x) first, then
            # strictly sideways strafe (linear.y) second. Since rotation is
            # never used, the robot is mathematically guaranteed to still be
            # squared on the shelf when it stops - there is no rotation step
            # left that could introduce drift.
            self._approach_elapsed += 0.1
            if self._approach_elapsed > APPROACH_TIMEOUT_S:
                self.cmd_pub.publish(Twist())
                self.get_logger().error('WAYPOINT_APPROACH safety timeout - stopping.')
                self.state = 'FAILED'
                return

            if self.current_x is None or self._waypoint is None:
                return
            wp_x, wp_y, wp_yaw = self._waypoint
            dx = wp_x - self.current_x
            dy = wp_y - self.current_y

            # convert world-frame remaining displacement into the ROBOT'S
            # LOCAL frame, fixed at wp_yaw (forward = local x, left = local y)
            local_forward = dx * math.cos(wp_yaw) + dy * math.sin(wp_yaw)
            local_left = -dx * math.sin(wp_yaw) + dy * math.cos(wp_yaw)

            if abs(local_forward) > WAYPOINT_POS_TOLERANCE_M:
                # PHASE 1: forward/backward only
                twist = Twist()
                speed = max(min(local_forward, WAYPOINT_MAX_LINEAR), -WAYPOINT_MAX_LINEAR)
                twist.linear.x = speed
                self.cmd_pub.publish(twist)
                return

            if abs(local_left) > WAYPOINT_POS_TOLERANCE_M:
                # PHASE 2: sideways strafe only (forward axis already done)
                twist = Twist()
                speed = max(min(local_left, WAYPOINT_MAX_LINEAR), -WAYPOINT_MAX_LINEAR)
                twist.linear.y = speed
                self.cmd_pub.publish(twist)
                return

            # both axes done - final heading check, should already match
            # wp_yaw exactly since we never rotated, but confirm before DONE
            yaw_err = angle_diff(wp_yaw, self.current_yaw)
            if abs(yaw_err) < WAYPOINT_YAW_TOLERANCE_RAD:
                self.cmd_pub.publish(Twist())
                self.get_logger().info(
                    f'Reached shelf column {self.target_column}, squared on the column (no rotation used '
                    f'in final approach). Stopped.')
                self.state = 'DONE'
                self.reached_pub.publish(Bool(data=True))
                return
            twist = Twist()
            twist.angular.z = max(min(WAYPOINT_MAX_ANGULAR * yaw_err, WAYPOINT_MAX_ANGULAR), -WAYPOINT_MAX_ANGULAR)
            self.cmd_pub.publish(twist)
            return

        if self.state in ('DONE', 'FAILED'):
            self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = ShelfLocatorNode()
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
