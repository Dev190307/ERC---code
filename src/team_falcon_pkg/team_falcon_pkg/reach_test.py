#!/usr/bin/env python3
"""Tests reach at increasing distances, using base_link frame with an
identity/neutral orientation (loose tolerance) - the SAME reliable method
manipulation_node.py uses for real grasps, not the gripper-relative frame
trick (which we have separate evidence may be unreliable on its own)."""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, PlanningOptions
)
from shape_msgs.msg import SolidPrimitive

PLANNING_GROUP = 'arm_left'
END_EFFECTOR_LINK = 'gripper_left_grasping_link'
PLANNING_FRAME = 'base_link'
POSITION_TOLERANCE = 0.03
ORIENTATION_TOLERANCE = 3.14  # loose - only testing position reachability


def make_constraints(x, y, z):
    pose = PoseStamped()
    pose.header.frame_id = PLANNING_FRAME
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation.w = 1.0

    constraints = Constraints()
    pos_constraint = PositionConstraint()
    pos_constraint.header = pose.header
    pos_constraint.link_name = END_EFFECTOR_LINK
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [POSITION_TOLERANCE]
    bounding_volume = BoundingVolume()
    bounding_volume.primitives = [sphere]
    bounding_volume.primitive_poses = [pose.pose]
    pos_constraint.constraint_region = bounding_volume
    pos_constraint.weight = 1.0

    orient_constraint = OrientationConstraint()
    orient_constraint.header = pose.header
    orient_constraint.link_name = END_EFFECTOR_LINK
    orient_constraint.orientation = pose.pose.orientation
    orient_constraint.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE
    orient_constraint.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE
    orient_constraint.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE
    orient_constraint.weight = 1.0

    constraints.position_constraints = [pos_constraint]
    constraints.orientation_constraints = [orient_constraint]
    return constraints


class ReachTest(Node):
    def __init__(self):
        super().__init__('reach_test')
        self._client = ActionClient(self, MoveGroup, 'move_action')

    def try_reach(self, x, y, z) -> bool:
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('move_group not available')
            return False
        goal = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = PLANNING_GROUP
        request.num_planning_attempts = 5
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = 0.3
        request.max_acceleration_scaling_factor = 0.3
        request.goal_constraints = [make_constraints(x, y, z)]
        goal.request = request
        planning_options = PlanningOptions()
        planning_options.plan_only = False
        goal.planning_options = planning_options

        self.get_logger().info(f'--- Trying ({x}, {y}, {z}) in base_link frame ---')
        send_goal_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        success = result is not None and result.result.error_code.val == 1
        self.get_logger().info(f'    -> {"SUCCESS" if success else "FAILED"}')
        return success


def main():
    rclpy.init()
    node = ReachTest()
    # sweep forward distance, modest height, centered laterally - typical
    # TIAGo arm mount is roughly torso-height, so z~1.0-1.2 is reasonable
    test_points = [
        (0.4, 0.0, 1.0),
        (0.5, 0.0, 1.0),
        (0.6, 0.0, 1.0),
        (0.7, 0.0, 1.0),
        (0.8, 0.0, 1.0),
    ]
    for x, y, z in test_points:
        node.try_reach(x, y, z)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
