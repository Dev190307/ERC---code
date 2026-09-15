#!/usr/bin/env python3
"""
Simplest possible MoveIt2 planning test: pure joint-space goal, no TF, no
frames, no Cartesian poses at all. Targets the SRDF's own predefined
'arm_left_home' state (all 7 arm_left joints = 0.0). If this fails too,
the problem is deeper than the Cartesian/frame trick - likely the SRDF
itself, joint limits, or a collision/config issue.
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint, PlanningOptions

PLANNING_GROUP = 'arm_left'
ARM_JOINTS = [
    'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
    'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint',
]
JOINT_TOLERANCE = 0.05


def joint_goal_constraints(target_values):
    constraints = Constraints()
    for name, value in zip(ARM_JOINTS, target_values):
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = value
        jc.tolerance_above = JOINT_TOLERANCE
        jc.tolerance_below = JOINT_TOLERANCE
        jc.weight = 1.0
        constraints.joint_constraints.append(jc)
    return constraints


class ArmJointTest(Node):
    def __init__(self):
        super().__init__('arm_joint_test')
        self._client = ActionClient(self, MoveGroup, 'move_action')

    def move_to_joints(self, target_values) -> bool:
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('move_group action server not available')
            return False

        goal = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = PLANNING_GROUP
        request.num_planning_attempts = 5
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = 0.3
        request.max_acceleration_scaling_factor = 0.3
        request.goal_constraints = [joint_goal_constraints(target_values)]
        goal.request = request

        planning_options = PlanningOptions()
        planning_options.plan_only = False
        goal.planning_options = planning_options

        self.get_logger().info(f'Requesting joint-space move to: {target_values}')
        send_goal_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None or result.result.error_code.val != 1:
            code = result.result.error_code.val if result else 'unknown'
            self.get_logger().error(f'Move failed, error code: {code}')
            return False

        self.get_logger().info('Joint-space move SUCCEEDED!')
        return True


def main():
    rclpy.init()
    node = ArmJointTest()
    # Target: the SRDF's own predefined 'arm_left_home' state - all zeros
    node.move_to_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
