#!/usr/bin/env python3
"""
Standalone test: move the left gripper a small amount RELATIVE TO ITSELF
(not a global/base_link position), then close the fingers.

Key idea: setting the goal PoseStamped's header.frame_id to the end-effector
link name itself (instead of base_link) makes MoveIt interpret x/y/z as an
OFFSET from wherever the gripper currently is - "the arm is the reference
frame" - rather than an absolute world coordinate.

Run this with the robot in its normal resting pose, nowhere near the
shelf, just to sanity check relative arm motion works at all.
"""
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
from std_srvs.srv import Empty

PLANNING_GROUP = 'arm_left'
END_EFFECTOR_LINK = 'gripper_left_grasping_link'
POSITION_TOLERANCE = 0.03
ORIENTATION_TOLERANCE = 3.14  # loose - position-only concern for this test


def relative_pose_constraints(dx, dy, dz):
    """A goal pose expressed RELATIVE to the end effector's own current
    pose - since orientation is 0,0,0,1 (identity) in this frame, dx/dy/dz
    are a pure offset along the gripper's own current local axes."""
    pose = PoseStamped()
    pose.header.frame_id = END_EFFECTOR_LINK  # <-- the key part: arm's own frame
    pose.pose.position.x = dx
    pose.pose.position.y = dy
    pose.pose.position.z = dz
    pose.pose.orientation.w = 1.0  # identity = "don't change current orientation"

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


class ArmRelativeTest(Node):
    def __init__(self):
        super().__init__('arm_relative_test')
        self._client = ActionClient(self, MoveGroup, 'move_action')
        self._grasp_client = self.create_client(Empty, '/gripper_left_grasper_srv/grasp')
        self._release_client = self.create_client(Empty, '/gripper_left_grasper_srv/release')

    def move_relative(self, dx, dy, dz) -> bool:
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
        request.goal_constraints = [relative_pose_constraints(dx, dy, dz)]
        goal.request = request

        planning_options = PlanningOptions()
        planning_options.plan_only = False
        goal.planning_options = planning_options

        self.get_logger().info(f'Requesting relative move: dx={dx}, dy={dy}, dz={dz} (in gripper frame)')
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
            self.get_logger().error(f'Move failed, error code: {result.result.error_code.val if result else "unknown"}')
            return False

        self.get_logger().info('Relative move succeeded!')
        return True

    def call_grasp(self):
        if not self._grasp_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('grasp service not available')
            return
        self._grasp_client.call_async(Empty.Request())
        self.get_logger().info('Grasp (close) requested')


def main():
    rclpy.init()
    node = ArmRelativeTest()

    # Step 1: move the gripper a small amount UP (positive Z in the
    # gripper's OWN frame) and slightly BACK (negative X, away from
    # whatever's in front of it) - a conservative first move to see it
    # working without risking a shelf collision.
    node.move_relative(dx=0.20, dy=0.0, dz=0.0)  # 20cm forward reach test

    import time
    time.sleep(1.0)

    # Step 2: close the fingers
    node.call_grasp()
    rclpy.spin_once(node, timeout_sec=2.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
