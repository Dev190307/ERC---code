#!/usr/bin/env python3
"""
Drives the robot to a fixed, hardcoded (x, y, yaw) waypoint using pure
closed-loop odometry feedback. Classic 3-phase approach:
  1. Rotate to face the waypoint direction
  2. Drive straight forward until position is reached
  3. Rotate to match the target's final orientation exactly

Edit TARGET_X / TARGET_Y / TARGET_QZ / TARGET_QW below for a different
waypoint (values straight from `ros2 topic echo /odom --once`).
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# ===== EDIT THESE for a different waypoint =====
TARGET_X = 2.0216082098509567
TARGET_Y = -1.3497135969895155
TARGET_QZ = -0.7143501371789682
TARGET_QW = 0.6997884548293072
# =================================================

POS_TOLERANCE_M = 0.05
YAW_TOLERANCE_RAD = 0.03
MAX_LINEAR = 0.4
MAX_ANGULAR = 0.3


def yaw_from_quat_zw(qz, qw):
    return 2.0 * math.atan2(qz, qw)


def angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


class GotoWaypoint(Node):
    def __init__(self):
        super().__init__('goto_waypoint')
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.target_yaw = yaw_from_quat_zw(TARGET_QZ, TARGET_QW)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.state = 'ROTATE_TO_FACE'
        self.get_logger().info(
            f'Target: x={TARGET_X:.2f}, y={TARGET_Y:.2f}, yaw={self.target_yaw:.2f} rad')
        self.create_timer(0.1, self.control_loop)

    def odom_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        if self.current_x is None:
            return

        dx = TARGET_X - self.current_x
        dy = TARGET_Y - self.current_y
        dist = math.hypot(dx, dy)

        if self.state == 'ROTATE_TO_FACE':
            if dist < POS_TOLERANCE_M:
                self.get_logger().info('Already at target position, skipping to final rotation.')
                self.state = 'ROTATE_FINAL'
                return
            heading_to_target = math.atan2(dy, dx)
            err = angle_diff(heading_to_target, self.current_yaw)
            if abs(err) < YAW_TOLERANCE_RAD:
                self.cmd_pub.publish(Twist())
                self.get_logger().info('Facing target. Driving straight...')
                self.state = 'DRIVE_STRAIGHT'
                return
            twist = Twist()
            twist.angular.z = max(min(0.8 * err, MAX_ANGULAR), -MAX_ANGULAR)
            self.cmd_pub.publish(twist)
            return

        if self.state == 'DRIVE_STRAIGHT':
            # Use SIGNED forward distance (relative to current heading), not
            # raw distance - this lets us detect overshoot (target now
            # behind us) and correct with reverse motion instead of driving
            # forward forever. Also apply a small heading correction to
            # prevent drift off the straight line.
            local_forward = dx * math.cos(self.current_yaw) + dy * math.sin(self.current_yaw)

            if abs(local_forward) < POS_TOLERANCE_M and dist < POS_TOLERANCE_M * 2:
                self.cmd_pub.publish(Twist())
                self.get_logger().info('Reached target position. Final rotation...')
                self.state = 'ROTATE_FINAL'
                return

            heading_to_target = math.atan2(dy, dx)
            heading_err = angle_diff(heading_to_target, self.current_yaw)
            # if we've overshot (local_forward negative), heading_to_target
            # points backward - don't fight it with a big turn, just reverse
            if local_forward < 0:
                heading_err = 0.0

            twist = Twist()
            twist.linear.x = max(min(local_forward, MAX_LINEAR), -MAX_LINEAR)
            if abs(twist.linear.x) < 0.05:
                twist.linear.x = 0.05 if local_forward > 0 else -0.05
            twist.angular.z = max(min(0.5 * heading_err, 0.15), -0.15)
            self.cmd_pub.publish(twist)
            return

        if self.state == 'ROTATE_FINAL':
            err = angle_diff(self.target_yaw, self.current_yaw)
            if abs(err) < YAW_TOLERANCE_RAD:
                self.cmd_pub.publish(Twist())
                self.get_logger().info('=== WAYPOINT REACHED (position + orientation). ===')
                self.state = 'DONE'
                return
            twist = Twist()
            twist.angular.z = max(min(0.8 * err, MAX_ANGULAR), -MAX_ANGULAR)
            self.cmd_pub.publish(twist)
            return

        if self.state == 'DONE':
            self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = GotoWaypoint()
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
