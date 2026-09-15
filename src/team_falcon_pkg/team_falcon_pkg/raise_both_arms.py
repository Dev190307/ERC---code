#!/usr/bin/env python3
"""Slightly raises both shoulder joints via direct joint_trajectory publish.
The controller requires ALL joints in the trajectory message (rejects
partial/single-joint commands), so this reads current joint states first
and sends a full 7-joint command, only changing the shoulder value."""
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

SHOULDER_RAISE = 1.5

LEFT_JOINTS = ['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
               'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint']
RIGHT_JOINTS = ['arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint', 'arm_right_4_joint',
                'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint']


class RaiseBothArms(Node):
    def __init__(self):
        super().__init__('raise_both_arms')
        self.left_pub = self.create_publisher(JointTrajectory, '/arm_left_controller/joint_trajectory', 10)
        self.right_pub = self.create_publisher(JointTrajectory, '/arm_right_controller/joint_trajectory', 10)
        self.joint_positions = {}
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def send_full_trajectory(self, pub, joint_names, shoulder_joint_name, shoulder_value):
        positions = []
        for name in joint_names:
            if name == shoulder_joint_name:
                positions.append(shoulder_value)
            else:
                positions.append(self.joint_positions.get(name, 0.0))

        traj = JointTrajectory()
        traj.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=2, nanosec=0)
        traj.points = [point]
        pub.publish(traj)
        self.get_logger().info(f'Sent to {joint_names[0].split("_")[1]} arm: {positions}')


def main():
    rclpy.init()
    node = RaiseBothArms()

    # wait for at least one /joint_states message so we have real current values
    node.get_logger().info('Waiting for joint states...')
    while rclpy.ok() and not node.joint_positions:
        rclpy.spin_once(node, timeout_sec=0.5)

    node.get_logger().info(f'Raising both shoulders by {SHOULDER_RAISE} rad...')
    node.send_full_trajectory(node.left_pub, LEFT_JOINTS, 'arm_left_2_joint', SHOULDER_RAISE)
    node.send_full_trajectory(node.right_pub, RIGHT_JOINTS, 'arm_right_2_joint', SHOULDER_RAISE)

    time.sleep(3)
    node.get_logger().info('Done.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
