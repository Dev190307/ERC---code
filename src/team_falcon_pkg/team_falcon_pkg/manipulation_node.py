"""
manipulation_node.py — Plans and executes the grasp and the placement.

Uses MoveIt2's `move_group` action interface directly (moveit_msgs/action/MoveGroup),
since moveit_py (the Python convenience wrapper) is not available as a binary
package for ROS2 Humble and building it from source was out of scope for this
timeline. This is more verbose than moveit_py but uses the same underlying
MoveIt2 planning/execution pipeline — collision-aware IK and trajectory
execution are unchanged.

IMPORTANT -- still a starting scaffold, not tested against real hardware:
- The tf2 transform step (camera frame -> planning frame) is unchanged from
  the original and still needs isolated testing -- this remains the step
  flagged as where teams lose the most time.
- _move_to_pose blocks with spin_until_future_complete inside a callback.
  This is fine for a single sequential grasp, but if you ever need this node
  to do anything concurrently (e.g. respond to a cancel), it needs a
  multi-threaded executor or a proper async rewrite instead.
- Position/orientation tolerances in pose_to_constraints are starting
  guesses -- tune once you can see real grasp attempts in sim.
- Gripper open/close still uses the joint_trajectory topic shortcut, same
  as the original -- unaffected by this change.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
import math
import time
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Bool
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 -- registers PointStamped transform support

from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionIK
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, PlanningOptions
)
from shape_msgs.msg import SolidPrimitive

PLANNING_GROUP = 'arm_left'
PLANNING_FRAME = 'base_link'
END_EFFECTOR_LINK = 'gripper_left_grasping_link'
GRIPPER_TOPIC = '/gripper_left_controller_raw/joint_trajectory'
GRIPPER_JOINT = 'gripper_left_finger_joint'

ARM_LEFT_TRAJ_TOPIC = '/arm_left_controller/joint_trajectory'

ARM_LEFT_JOINT_NAMES = [
    'arm_left_1_joint',
    'arm_left_2_joint',
    'arm_left_3_joint',
    'arm_left_4_joint',
    'arm_left_5_joint',
    'arm_left_6_joint',
    'arm_left_7_joint',
]

WRIST_ROLL_JOINT = 'arm_left_7_joint'
WRIST_ROLL_ANGLE_RAD = -math.pi / 2.0

# Center wrist before creep so a full clockwise 90-degree
# movement is available after the book is grasped.
WRIST_PREGRASP_POSITION = 0.0
   # 90 deg clockwise


# Torso control
TORSO_ACTION = '/torso_controller/follow_joint_trajectory'
TORSO_JOINT = 'torso_lift_joint'

# Row boundaries (base_link Z), derived from simulation.launch.py's own
# book-spawn formula: row1 > 1.35, row2 in (1.02, 1.35], row3 in (0.69, 1.02],
# row4 <= 0.69. Setting the threshold to 1.02 makes the torso raise for
# BOTH row 1 and row 2 (anything above the row2/row3 boundary), while
# rows 3 and 4 stay arm-only.
TOP_SHELF_Z_THRESHOLD = 1.02

# IMPORTANT: confirm this is within the torso joint limits in the URDF.
TOP_SHELF_TORSO_POSITION = 0.35 * 0.85  # 85% of max height, reduces travel distance/time; our max-reach search will still find a reachable X even with the slightly lower shoulder height
GRASP_POINT_TOPIC = '/erc/target_book_point'
GRASP_RESULT_TOPIC = '/erc/grasp_result'
READY_TO_CREEP_TOPIC = '/erc/ready_to_creep'
CREEP_CONTACT_TOPIC = '/erc/creep_contact_made'
BACKOUT_COMPLETE_TOPIC = '/erc/backout_complete'
LIFT_RESULT_TOPIC = '/erc/lift_result'
BIN_POINT_TOPIC = '/erc/bin_point'
PLACE_RESULT_TOPIC = '/erc/place_result'

STANDOFF_X_METERS = 0.5
LIFT_HEIGHT_METERS = 0.15

# Full clockwise wrist roll after grabbing the book.
BOOK_SUPPORT_ROLL_DEG = 90.0

GRIPPER_RIGHT_OFFSET_METERS = 0.013

# Lowest shelf only:
# shift grasp target DOWN by 5 cm so the gripper aims closer
# to the middle of the book instead of its upper corner.
LOW_SHELF_Z_OFFSET_METERS = 0.05

# Safety floor for the lowest shelf.
# Even with the requested 5 cm downward offset, never command
# the gripper lower than this in base_link coordinates.
LOW_SHELF_MIN_TARGET_Z = 0.56

  # shift grasp target 1.2 cm to robot's right (-Y in base_link)
GRIPPER_CLOSE_SETTLE_SEC = 1.2       # allow fingers to fully close before lifting/moving away
BIN_APPROACH_Z_OFFSET = 0.20         # approach safely above the detected red bin
BIN_RELEASE_Z_OFFSET = 0.10          # release above the bin so the book drops inside

POSITION_TOLERANCE = 0.02   # metres
ORIENTATION_TOLERANCE = 0.05  # radians (~2.9 degrees); keeps gripper straight


def horizontal_approach_orientation(yaw_rad: float = 0.0) -> Quaternion:
    """Gripper orientation for the approach when the target is at/above
    shoulder height. CONFIRMED CORRECT via live Gazebo test: identity (no
    rotation) is correct for mid/high targets."""
    return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)


def approach_orientation_for_height(z_base_link: float) -> Quaternion:
    """Gripper orientation adapted to target height. CONFIRMED via live IK
    testing: identity (level) works for mid/high targets, but low targets
    (row 4, near/below shoulder height) have NO valid IK solution held
    perfectly level -- the solver succeeds immediately once the gripper is
    allowed to angle downward instead. Blends smoothly between level (at/
    above shoulder height, ~0.68m) and ~60 degrees down (well below
    shoulder height, ~0.4m and lower)."""
    SHOULDER_Z = 0.677
    LOW_Z = 0.4
    pitch_max = math.radians(60)
    if z_base_link >= SHOULDER_Z:
        pitch = 0.0
    elif z_base_link <= LOW_Z:
        pitch = pitch_max
    else:
        frac = (SHOULDER_Z - z_base_link) / (SHOULDER_Z - LOW_Z)
        pitch = frac * pitch_max
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    return Quaternion(x=0.0, y=sp, z=0.0, w=cp)


def pose_to_constraints(pose: PoseStamped, link_name: str) -> Constraints:
    """Build a Constraints message MoveGroup can plan against, from a target pose."""
    constraints = Constraints()

    pos_constraint = PositionConstraint()
    pos_constraint.header = pose.header
    pos_constraint.link_name = link_name
    pos_constraint.target_point_offset.x = 0.0
    pos_constraint.target_point_offset.y = 0.0
    pos_constraint.target_point_offset.z = 0.0

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
    orient_constraint.link_name = link_name
    orient_constraint.orientation = pose.pose.orientation
    orient_constraint.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE
    orient_constraint.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE
    orient_constraint.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE
    orient_constraint.weight = 1.0

    constraints.position_constraints = [pos_constraint]
    constraints.orientation_constraints = [orient_constraint]
    return constraints


class ManipulationNode(Node):

    def _wait_for_future(self, future, timeout_sec=None):
        start = time.monotonic()

        while rclpy.ok() and not future.done():
            if timeout_sec is not None and time.monotonic() - start >= timeout_sec:
                return False
            time.sleep(0.01)

        return future.done()

    def __init__(self):
        super().__init__('manipulation_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._action_callback_group = ReentrantCallbackGroup()
        self._move_group_client = ActionClient(
            self, MoveGroup, 'move_action', callback_group=self._action_callback_group)

        self._ik_client = self.create_client(
            GetPositionIK, 'compute_ik', callback_group=self._action_callback_group)

        self._torso_client = ActionClient(
            self,
            FollowJointTrajectory,
            TORSO_ACTION,
            callback_group=self._action_callback_group
        )

        self.gripper_pub = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)

        # Direct left-arm trajectory publisher.
        # Used specifically for the 90-degree wrist roll.
        self.arm_left_traj_pub = self.create_publisher(
            JointTrajectory,
            ARM_LEFT_TRAJ_TOPIC,
            10
        )
        self.result_pub = self.create_publisher(Bool, GRASP_RESULT_TOPIC, 10)
        self.ready_pub = self.create_publisher(Bool, READY_TO_CREEP_TOPIC, 10)
        self.create_subscription(PointStamped, GRASP_POINT_TOPIC, self.grasp_point_callback, 10)
        self.create_subscription(Bool, CREEP_CONTACT_TOPIC, self.creep_contact_callback, 10)
        self.create_subscription(
            Bool,
            BACKOUT_COMPLETE_TOPIC,
            self.backout_complete_callback,
            10
        )
        self.lift_result_pub = self.create_publisher(
            Bool,
            LIFT_RESULT_TOPIC,
            10
        )
        self.create_subscription(PointStamped, BIN_POINT_TOPIC, self.bin_point_callback, 10)
        self.place_result_pub = self.create_publisher(Bool, PLACE_RESULT_TOPIC, 10)
        self._standoff_pose = None
        self._waiting_for_backout = False
        self._torso_position = None
        self._arm_joint_positions = {}
        from sensor_msgs.msg import JointState
        self.create_subscription(JointState, '/joint_states', self._joint_states_cb, 10, callback_group=self._action_callback_group)

        self.get_logger().info('manipulation_node ready, waiting for a grasp target')

    def _joint_states_cb(self, msg):
        # Keep latest positions for all left-arm joints.
        for name, position in zip(msg.name, msg.position):
            if name in ARM_LEFT_JOINT_NAMES:
                self._arm_joint_positions[name] = position

        try:
            idx = msg.name.index(TORSO_JOINT)
            self._torso_position = msg.position[idx]
        except ValueError:
            pass


    def grasp_point_callback(self, point_msg: PointStamped):
        point_msg.header.stamp = rclpy.time.Time().to_msg()
        try:
            point_in_base = self.tf_buffer.transform(
                point_msg, PLANNING_FRAME, timeout=Duration(seconds=2.0)
            )
        except Exception as e:
            self.get_logger().error(f'tf2 transform failed: {e}')
            self._publish_result(False)
            return

        self.get_logger().info(
            f'Book point in {PLANNING_FRAME} frame: '
            f'x={point_in_base.point.x:.3f}, y={point_in_base.point.y:.3f}, z={point_in_base.point.z:.3f}'
        )

        if point_in_base.point.z >= TOP_SHELF_Z_THRESHOLD:
            self.get_logger().info(
                f'Top-shelf target detected (z={point_in_base.point.z:.3f} m). '
                f'Raising torso to {TOP_SHELF_TORSO_POSITION:.3f} m.'
            )
            self.get_logger().info('Torso moving up, please wait...')
            if not self._move_torso(TOP_SHELF_TORSO_POSITION):
                self.get_logger().error(
                    'Failed to raise torso for top-shelf grasp -- aborting grasp.'
                )
                self._publish_result(False)
                return

        self._start_standoff_approach(point_in_base)

    def _move_torso(self, position: float) -> bool:
        if not self._torso_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f'Torso trajectory action server not available at {TORSO_ACTION}'
            )
            return False

        # Was 0.07 m/s per the joint limits spec sheet, but LIVE MEASUREMENT during
# testing showed the real achieved rate is only ~0.013 m/s (over 5x slower).
# Using the measured value here so both the commanded trajectory duration
# and our settle-verification timeout are realistic. This mismatch is worth
# flagging separately -- likely a torso_controller gain/velocity-limit
# config issue, not something fixable from this node.
        TORSO_MAX_VELOCITY = 0.013  # m/s, measured live (see chat)
        current = self._torso_position if self._torso_position is not None else 0.0
        distance = abs(position - current)
        # Physically-realistic duration with margin for accel/decel settling
        duration_s = max(distance / TORSO_MAX_VELOCITY * 1.3, 1.0)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [TORSO_JOINT]

        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = [float(position)]
        trajectory_point.time_from_start = Duration(seconds=duration_s).to_msg()
        goal.trajectory.points = [trajectory_point]

        self.get_logger().info(
            f'Sending torso goal: {TORSO_JOINT} -> {position:.3f} '
            f'(duration={duration_s:.1f}s, physically realistic for {TORSO_MAX_VELOCITY} m/s max)'
        )

        send_goal_future = self._torso_client.send_goal_async(goal)
        self._wait_for_future(send_goal_future)
        goal_handle = send_goal_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Torso controller rejected the goal')
            return False

        result_future = goal_handle.get_result_async()
        self._wait_for_future(result_future)
        wrapped_result = result_future.result()

        if wrapped_result is None:
            self.get_logger().error('No result returned by torso controller')
            return False

        if wrapped_result.result.error_code != 0:
            self.get_logger().error(
                f'Torso trajectory failed with error code '
                f'{wrapped_result.result.error_code}: '
                f'{wrapped_result.result.error_string}'
            )
            return False

        # HARD VERIFICATION: don't trust the action's completion signal
        # alone -- poll the torso's real live position from /joint_states
        # until it's actually settled at the target, before letting the
        # arm move. This is what actually fixes "torso still moving while
        # arm adjusts" regardless of any controller timing quirks.
        POSITION_TOLERANCE_M = 0.01
        # Must be at least as long as the commanded trajectory duration
        # itself, plus real margin -- was a fixed 5.0s, shorter than a
        # 6.5s+ torso move, so it was giving up before the torso could
        # possibly have finished.
        settle_timeout_s = duration_s + 5.0
        elapsed = 0.0
        last_logged_second = -1
        while elapsed < settle_timeout_s:
            if self._torso_position is not None and abs(self._torso_position - position) < POSITION_TOLERANCE_M:
                self.get_logger().info(
                    f'Torso verified settled at {self._torso_position:.3f} m '
                    f'(target {position:.3f} m). Continuing with arm motion.'
                )
                return True
            if int(elapsed) != last_logged_second:
                last_logged_second = int(elapsed)
                self.get_logger().info(
                    f'Waiting for torso to settle... current position: {self._torso_position}'
                )
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed += 0.1

        self.get_logger().error(
            f'Torso did not settle at target within {settle_timeout_s:.1f}s '
            f'(last known position: {self._torso_position}). Aborting to avoid '
            f'moving the arm while the torso is still in motion.'
        )
        return False

    def _find_max_reach_x(self, y: float, z: float, orientation: Quaternion) -> float:
        """Search for the farthest reachable standoff X (arm extended as far
        as possible toward the book, staying aligned on the same y/z line),
        by checking IK feasibility at decreasing distances via /compute_ik,
        without actually moving the arm yet."""
        candidate_x_values = [0.85, 0.75, 0.65, 0.55, STANDOFF_X_METERS]
        for x in candidate_x_values:
            pose = PoseStamped()
            pose.header.frame_id = PLANNING_FRAME
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation = orientation

            request = GetPositionIK.Request()
            request.ik_request.group_name = PLANNING_GROUP
            request.ik_request.pose_stamped = pose
            request.ik_request.avoid_collisions = True
            request.ik_request.timeout = Duration(seconds=1.0).to_msg()

            future = self._ik_client.call_async(request)
            self._wait_for_future(future, timeout_sec=1.5)
            result = future.result()

            if result is not None and result.error_code.val == 1:
                self.get_logger().info(f'Max reach check: x={x:.2f} is reachable -- using this.')
                return x
            else:
                self.get_logger().info(f'Max reach check: x={x:.2f} NOT reachable, trying closer...')

        self.get_logger().warn(
            f'No candidate X was confirmed reachable, falling back to {STANDOFF_X_METERS}m'
        )
        return STANDOFF_X_METERS

    def _start_standoff_approach(self, point: PointStamped):
        # Rows 1-3: keep the gripper straight.
        # Row 4: slight downward pitch so arm_left_4_link clears the shelf.
        if point.point.z <= 0.69:
            pitch = math.radians(10.0)
            orientation = Quaternion(
                x=0.0,
                y=math.sin(pitch / 2.0),
                z=0.0,
                w=math.cos(pitch / 2.0)
            )
            self.get_logger().info(
                'Low shelf detected: using 10 degree downward gripper pitch.'
            )
        else:
            orientation = horizontal_approach_orientation()

        # In base_link coordinates: +Y is robot-left, so subtracting from Y
        # shifts the whole gripper target toward the robot's right.
        target_y = point.point.y - GRIPPER_RIGHT_OFFSET_METERS

        # Lowest shelf only:
        # aim 5 cm lower so the gripper grabs closer to the
        # centre of the book instead of its upper corner.
        if point.point.z <= 0.69:
            requested_z = (
                point.point.z - LOW_SHELF_Z_OFFSET_METERS
            )

            # Apply requested downward offset, but never allow the
            # arm/gripper target to drop below our safe clearance.
            target_z = max(
                requested_z,
                LOW_SHELF_MIN_TARGET_Z
            )

            self.get_logger().info(
                f'Low shelf: detected_z={point.point.z:.3f}, '
                f'requested_z={requested_z:.3f}, '
                f'safe_target_z={target_z:.3f}.'
            )

            if requested_z < LOW_SHELF_MIN_TARGET_Z:
                self.get_logger().warn(
                    f'Low-shelf 5 cm offset would be too low. '
                    f'Clamping target Z to '
                    f'{LOW_SHELF_MIN_TARGET_Z:.3f} m for shelf clearance.'
                )
        else:
            target_z = point.point.z

        best_x = self._find_max_reach_x(target_y, target_z, orientation)

        standoff_pose = PoseStamped()
        standoff_pose.header.frame_id = PLANNING_FRAME
        standoff_pose.pose.position.x = best_x
        standoff_pose.pose.position.y = target_y
        standoff_pose.pose.position.z = target_z
        standoff_pose.pose.orientation = orientation

        if not self._move_to_pose(standoff_pose):
            self.get_logger().error('Failed to plan standoff pose -- aborting grasp')
            self._publish_result(False)
            return

        self._set_gripper(open_gripper=True)
        self._standoff_pose = standoff_pose
        self.get_logger().info(
            f'Arm fully extended to x={best_x:.2f} '
            f'(detected_y={point.point.y:.3f}, shifted_y={target_y:.3f}, '
            f'right_offset={GRIPPER_RIGHT_OFFSET_METERS:.3f}m, '
            f'z={target_z:.3f}), gripper open. '
            'NOW signaling base to creep forward.')
        # Center wrist BEFORE allowing the base to creep.

        self.ready_pub.publish(Bool(data=True))

    def _set_wrist_pregrasp(self) -> bool:
        """
        Move only arm_left_7_joint to 0 rad before creep.

        Joints 1-6 stay at their current values.
        """

        missing = [
            j for j in ARM_LEFT_JOINT_NAMES
            if j not in self._arm_joint_positions
        ]

        if missing:
            self.get_logger().error(
                f'Cannot center wrist. Missing joints: {missing}'
            )
            return False

        current = [
            self._arm_joint_positions[j]
            for j in ARM_LEFT_JOINT_NAMES
        ]

        target = list(current)

        wrist_index = ARM_LEFT_JOINT_NAMES.index(
            WRIST_ROLL_JOINT
        )

        old_wrist = current[wrist_index]
        target[wrist_index] = WRIST_PREGRASP_POSITION

        self.get_logger().info(
            f'PRE-GRASP WRIST: '
            f'{old_wrist:.3f} -> 0.000 rad.'
        )

        traj = JointTrajectory()
        traj.joint_names = ARM_LEFT_JOINT_NAMES

        pt = JointTrajectoryPoint()
        pt.positions = target
        pt.time_from_start = Duration(seconds=2.0).to_msg()

        traj.points = [pt]

        self.arm_left_traj_pub.publish(traj)

        time.sleep(2.5)

        actual = self._arm_joint_positions.get(
            WRIST_ROLL_JOINT,
            old_wrist
        )

        self.get_logger().info(
            f'PRE-GRASP WRIST actual = {actual:.3f} rad.'
        )

        if abs(actual) > 0.15:
            self.get_logger().error(
                'Wrist did not reach neutral position.'
            )
            return False

        return True


    def _roll_wrist_clockwise_90(self) -> bool:
        """
        Rotate ONLY arm_left_7_joint by 90 degrees clockwise.

        Joints 1-6 are commanded to remain exactly at their current
        positions so the arm does not lift, translate or retract.
        """

        missing = [
            j for j in ARM_LEFT_JOINT_NAMES
            if j not in self._arm_joint_positions
        ]

        if missing:
            self.get_logger().error(
                f'Cannot roll wrist: missing joint states: {missing}'
            )
            return False

        current = [
            self._arm_joint_positions[j]
            for j in ARM_LEFT_JOINT_NAMES
        ]

        target = list(current)

        wrist_index = ARM_LEFT_JOINT_NAMES.index(
            WRIST_ROLL_JOINT
        )

        old_wrist = current[wrist_index]

        # Direct relative 90-degree clockwise roll.
        new_wrist = -math.pi / 2.0

        target[wrist_index] = new_wrist

        self.get_logger().info(
            f'DIRECT WRIST ROLL: {WRIST_ROLL_JOINT} '
            f'{old_wrist:.3f} -> {new_wrist:.3f} rad '
            '(90 degrees clockwise).'
        )

        traj = JointTrajectory()
        traj.joint_names = ARM_LEFT_JOINT_NAMES

        pt = JointTrajectoryPoint()
        pt.positions = target

        # Slow controlled roll so the book settles onto fingertips.
        pt.time_from_start = Duration(seconds=3.0).to_msg()

        traj.points = [pt]

        self.arm_left_traj_pub.publish(traj)

        # Wait until the commanded wrist motion has physically had
        # enough time to finish before allowing the base to reverse.
        time.sleep(3.5)

        actual = self._arm_joint_positions.get(
            WRIST_ROLL_JOINT,
            old_wrist
        )

        moved = abs(actual - old_wrist)

        self.get_logger().info(
            f'Wrist roll finished. Actual movement = '
            f'{math.degrees(moved):.1f} degrees.'
        )

        # Require substantial physical movement before backout.
        if moved < math.radians(70.0):
            self.get_logger().error(
                'Wrist did NOT complete the required roll. '
                'Refusing to back out.'
            )
            return False

        return True


    def creep_contact_callback(self, msg: Bool):
        if not msg.data:
            return

        if self._standoff_pose is None:
            self.get_logger().error(
                'Contact received but no stored grasp pose exists.'
            )
            self._publish_result(False)
            return

        self.get_logger().info(
            'Contact confirmed -- closing gripper firmly. '
            'NO wrist rotation. NO arm movement.'
        )

        # ======================================================
        # HARD / RELIABLE CLAMP
        #
        # This gripper is position controlled.
        # 0.0 is already its fully closed position.
        #
        # Therefore we do NOT command a negative joint value.
        # Instead we repeatedly command full-close so the
        # simulated fingers remain firmly clamped around the book.
        # ======================================================

        self._set_gripper(open_gripper=False)
        time.sleep(0.40)

        self._set_gripper(open_gripper=False)
        time.sleep(0.40)

        self._set_gripper(open_gripper=False)
        time.sleep(0.40)

        self._set_gripper(open_gripper=False)
        time.sleep(0.40)

        # Final settling time before the base moves.
        self._set_gripper(open_gripper=False)
        time.sleep(1.00)

        self.get_logger().info(
            'GRIPPER FULLY CLAMPED. '
            'Holding arm fixed and pulling book straight out.'
        )

        # Important:
        # DO NOT lift here.
        # DO NOT rotate here.
        # DO NOT move the arm here.
        #
        # Signal waypoint node that it may begin reversing.
        self._waiting_for_backout = True
        self._publish_result(True)



    def backout_complete_callback(self, msg: Bool):
        if not msg.data:
            return

        if not self._waiting_for_backout or self._standoff_pose is None:
            self.get_logger().warn(
                'Received backout_complete but no grasp is waiting for lift.'
            )
            return

        self.get_logger().info(
            'Base backout complete. Robot is clear of shelf. '
            'NOW lifting the book.'
        )

        lift_pose = PoseStamped()
        lift_pose.header.frame_id = PLANNING_FRAME
        lift_pose.pose.position.x = self._standoff_pose.pose.position.x
        lift_pose.pose.position.y = self._standoff_pose.pose.position.y
        lift_pose.pose.position.z = (
            self._standoff_pose.pose.position.z + LIFT_HEIGHT_METERS
        )
        lift_pose.pose.orientation = self._standoff_pose.pose.orientation

        success = self._move_to_pose(lift_pose)

        if not success:
            self.get_logger().error(
                'Failed to lift book after backout.'
            )
        else:
            self.get_logger().info(
                'Post-backout lift completed successfully.'
            )

        self._waiting_for_backout = False
        self._standoff_pose = None

        self.lift_result_pub.publish(Bool(data=success))


    def bin_point_callback(self, point_msg: PointStamped):
        point_msg.header.stamp = rclpy.time.Time().to_msg()
        try:
            point_in_base = self.tf_buffer.transform(
                point_msg, PLANNING_FRAME, timeout=Duration(seconds=2.0)
            )
        except Exception as e:
            self.get_logger().error(f'tf2 transform failed for bin point: {e}')
            self.place_result_pub.publish(Bool(data=False))
            return

        self.get_logger().info(
            f'Bin point in {PLANNING_FRAME} frame: '
            f'x={point_in_base.point.x:.3f}, y={point_in_base.point.y:.3f}, z={point_in_base.point.z:.3f}'
        )
        success = self.place_in_bin(point_in_base)
        self.place_result_pub.publish(Bool(data=success))

    def _grasp_at(self, point: PointStamped) -> bool:
        approach_pose = self._pose_from_point(point, z_offset=0.15)
        grasp_pose = self._pose_from_point(point, z_offset=0.0)

        if not self._move_to_pose(approach_pose):
            self.get_logger().error('Failed to plan approach pose')
            return False

        self._set_gripper(open_gripper=True)

        if not self._move_to_pose(grasp_pose):
            self.get_logger().error('Failed to plan grasp pose')
            return False

        self._set_gripper(open_gripper=False)

        lift_pose = self._pose_from_point(point, z_offset=0.20)
        if not self._move_to_pose(lift_pose):
            self.get_logger().error('Failed to lift after grasp')
            return False

        self.get_logger().info('Grasp sequence completed')
        return True

    def place_in_bin(self, bin_point: PointStamped) -> bool:
        # Move above the detected centre of the red collection bin first,
        # then lower to a safe release height and open the gripper.
        approach_pose = self._pose_from_point(
            bin_point, z_offset=BIN_APPROACH_Z_OFFSET
        )
        release_pose = self._pose_from_point(
            bin_point, z_offset=BIN_RELEASE_Z_OFFSET
        )

        self.get_logger().info(
            f'Moving above red bin at x={bin_point.point.x:.3f}, '
            f'y={bin_point.point.y:.3f}, z={bin_point.point.z:.3f}.'
        )

        if not self._move_to_pose(approach_pose):
            self.get_logger().error('Failed to reach above-bin approach pose')
            return False

        if not self._move_to_pose(release_pose):
            self.get_logger().error('Failed to lower book into release position over bin')
            return False

        self.get_logger().info('Book over red bin -- opening gripper to drop it inside.')
        self._set_gripper(open_gripper=True)
        time.sleep(1.2)
        self.get_logger().info('Book released into bin.')
        return True

    def _pose_from_point(self, point: PointStamped, z_offset: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = PLANNING_FRAME
        pose.pose.position.x = point.point.x
        pose.pose.position.y = point.point.y
        pose.pose.position.z = point.point.z + z_offset
        pose.pose.orientation = horizontal_approach_orientation()
        return pose

    def _move_to_pose(self, pose: PoseStamped) -> bool:
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('move_group action server not available')
            return False

        goal = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = PLANNING_GROUP
        request.num_planning_attempts = 5
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = 0.5
        request.max_acceleration_scaling_factor = 0.5
        request.goal_constraints = [pose_to_constraints(pose, END_EFFECTOR_LINK)]
        goal.request = request

        planning_options = PlanningOptions()
        planning_options.plan_only = False
        goal.planning_options = planning_options

        send_goal_future = self._move_group_client.send_goal_async(goal)
        self._wait_for_future(send_goal_future)
        goal_handle = send_goal_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('move_group rejected the goal')
            return False

        result_future = goal_handle.get_result_async()
        self._wait_for_future(result_future)
        result = result_future.result()

        if result is None or result.result.error_code.val != 1:
            error_val = result.result.error_code.val if result else 'unknown'
            self.get_logger().error(f'move_group failed, error code: {error_val}')
            return False

        return True

    def _set_gripper(self, open_gripper: bool):
        msg = JointTrajectory()
        msg.joint_names = [GRIPPER_JOINT]
        point = JointTrajectoryPoint()
        point.positions = [0.068] if open_gripper else [0.0]  # true max is 0.07m per joint limits; small margin kept below the hard limit
        point.time_from_start = Duration(seconds=1).to_msg()
        msg.points = [point]
        self.gripper_pub.publish(msg)

    def _publish_result(self, success: bool):
        msg = Bool()
        msg.data = success
        self.result_pub.publish(msg)


def main():
    rclpy.init()
    node = ManipulationNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
