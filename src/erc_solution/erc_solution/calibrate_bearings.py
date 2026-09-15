#!/usr/bin/env python3
"""One-time calibration: search+center on EACH digit 1-5 in turn, printing
the real resulting yaw for each - directly measured, not geometrically
estimated. Run once, note down the 5 yaw values alongside which physical
slot (left to right) each corresponds to."""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import math

try:
    import pytesseract
    HAVE_TESSERACT = True
except ImportError:
    HAVE_TESSERACT = False

TILT_UP = 0.3
ROTATE_SPEED = -0.4
CENTER_TOLERANCE_PX = 15
CENTER_GAIN = 0.0015


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y * 2 - 1.0)
    return math.atan2(siny_cosp, 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class CalibrateBearings(Node):
    def __init__(self):
        super().__init__('calibrate_bearings')
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.current_yaw = None
        self.digit_templates = self._build_digit_templates()

        self.create_subscription(Image, '/head_front_camera/head_front_camera/color/image_raw',
                                  self.rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.head_pub = self.create_publisher(JointTrajectory, '/head_controller/joint_trajectory', 10)

    def rgb_cb(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def send_head(self, pan, tilt):
        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start = Duration(sec=1, nanosec=0)
        traj.points = [point]
        self.head_pub.publish(traj)

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

    def find_digit_x(self, target):
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
            if digit == target:
                return x + cw // 2
        return None

    def search_and_center(self, target):
        self.get_logger().info(f'--- Searching for digit {target} ---')
        # search
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            cx = self.find_digit_x(target)
            if cx is not None:
                self.cmd_pub.publish(Twist())
                break
            twist = Twist()
            twist.angular.z = ROTATE_SPEED
            self.cmd_pub.publish(twist)
        # center
        lost = 0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            cx = self.find_digit_x(target)
            if self.latest_rgb is None:
                continue
            img_center = self.latest_rgb.shape[1] / 2.0
            if cx is None:
                lost += 1
                twist = Twist()
                twist.angular.z = ROTATE_SPEED * 0.6
                self.cmd_pub.publish(twist)
                if lost > 80:
                    self.get_logger().warn('lost too long, breaking')
                    break
                continue
            lost = 0
            error_px = cx - img_center
            if abs(error_px) < CENTER_TOLERANCE_PX:
                self.cmd_pub.publish(Twist())
                break
            twist = Twist()
            twist.angular.z = max(min(-CENTER_GAIN * error_px, 0.5), -0.5)
            self.cmd_pub.publish(twist)
        self.get_logger().info(f'>>> Digit {target} centered at yaw={self.current_yaw:.4f} <<<')
        return self.current_yaw


def main():
    rclpy.init()
    node = CalibrateBearings()
    node.send_head(0.0, TILT_UP)
    for _ in range(15):
        rclpy.spin_once(node, timeout_sec=0.1)

    results = {}
    for digit in [1, 2, 3, 4, 5]:
        yaw = node.search_and_center(digit)
        results[digit] = yaw
        input(f'Digit {digit} done (yaw={yaw:.4f}). Note which PHYSICAL slot this is (left-to-right), then press Enter to continue to next digit...')

    node.get_logger().info(f'=== ALL RESULTS: {results} ===')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
