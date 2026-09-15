"""
perception_shelf_node.py — DEBUG VERSION for diagnosing why detection isn't firing.

Adds verbose logging at every stage of detect_shelf_number() and periodically
saves raw + thresholded frames to disk for visual inspection. Once the real
issue is identified, strip this logging back out.
"""

import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge

IMAGE_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
RESULT_TOPIC = '/erc/shelf_column_identification'
SAVE_DIR = os.path.expanduser('~/ros2_ws/src/team_falcon_pkg/erc_images')
DEBUG_DIR = os.path.expanduser('~/erc_debug_frames')  # new — debug output goes here

DIGITS = ["1", "2", "3", "4", "5"]
TEMPLATE_SIZE = (50, 50)
MIN_CONTOUR_AREA = 200


def _generate_templates():
    templates = {}
    for digit in DIGITS:
        canvas = np.zeros(TEMPLATE_SIZE, dtype=np.uint8)
        cv2.putText(canvas, digit, (5, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, 255, 3, cv2.LINE_AA)
        templates[digit] = canvas
    return templates


TEMPLATES = _generate_templates()


def detect_shelf_number(frame, logger, save_debug=False, frame_count=0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger.info(f'[DEBUG] Frame {frame_count}: found {len(contours)} total contours')

    if save_debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, f'raw_{frame_count}.png'), frame)
        cv2.imwrite(os.path.join(DEBUG_DIR, f'thresh_{frame_count}.png'), thresh)
        logger.info(f'[DEBUG] Saved raw_{frame_count}.png and thresh_{frame_count}.png to {DEBUG_DIR}')

    best_match = None
    best_score = -1
    rejected_area = 0
    rejected_aspect = 0
    considered = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            rejected_area += 1
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if aspect_ratio > 1.2 or aspect_ratio < 0.15:
            rejected_aspect += 1
            continue

        considered += 1
        candidate = cv2.resize(thresh[y:y + h, x:x + w], TEMPLATE_SIZE)
        for digit, template in TEMPLATES.items():
            score = cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED).max()
            if score > best_score:
                best_score = score
                best_match = {"number": int(digit), "bbox": (x, y, w, h), "confidence": float(score)}

    logger.info(
        f'[DEBUG] Frame {frame_count}: {rejected_area} rejected (too small, area<{MIN_CONTOUR_AREA}), '
        f'{rejected_aspect} rejected (aspect ratio out of range), '
        f'{considered} passed both filters and were template-matched'
    )
    if best_match:
        logger.info(
            f'[DEBUG] Frame {frame_count}: best match = digit {best_match["number"]}, '
            f'confidence {best_match["confidence"]:.3f} (publish threshold is not applied here, '
            f'save/lock-in threshold is 0.4)'
        )
    else:
        logger.info(f'[DEBUG] Frame {frame_count}: no contour passed both filters, nothing to match')

    if best_match and best_match["confidence"] > 0.25:
        return best_match
    return None


class PerceptionShelfNode(Node):
    def __init__(self):
        super().__init__('perception_shelf_node')

        self.declare_parameter('shelf_column_number', 0)
        self.target_column = self.get_parameter('shelf_column_number').value

        self.bridge = CvBridge()
        self.found_target = False
        self.frame_count = 0

        self.result_pub = self.create_publisher(Int32, RESULT_TOPIC, 10)
        self.image_sub = self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)

        os.makedirs(SAVE_DIR, exist_ok=True)
        self.get_logger().info(f'Looking for shelf column {self.target_column}')
        self.get_logger().info(f'[DEBUG] Debug frames will be saved to {DEBUG_DIR}')

    def image_callback(self, msg):
        if self.found_target:
            return

        self.frame_count += 1
        # Only save debug images every 30 frames so we don't flood disk/logs
        save_debug = (self.frame_count % 30 == 1)

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        detection = detect_shelf_number(frame, self.get_logger(), save_debug, self.frame_count)

        if detection is None:
            return

        result_msg = Int32()
        result_msg.data = detection["number"]
        self.result_pub.publish(result_msg)

        if detection["number"] == self.target_column and detection["confidence"] > 0.4:
            self._save_annotated_image(frame, detection)
            self.found_target = True
            self.get_logger().info(f'Confirmed target column {self.target_column}')

    def _save_annotated_image(self, frame, detection):
        x, y, w, h = detection["bbox"]
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f'col {detection["number"]}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(SAVE_DIR, f'shelf_column_{timestamp}.png')
        cv2.imwrite(filepath, annotated)
        self.get_logger().info(f'Saved annotated image: {filepath}')


def main():
    rclpy.init()
    node = PerceptionShelfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
