"""
orchestrator.py — State machine sequencing the full Library Assistant task.

Sequence: LOCATE_SHELF -> NAVIGATE_TO_SHELF -> LOCATE_BOOK -> GRASP_BOOK ->
          RETURN_TO_START -> LOCATE_BIN -> PLACE_BOOK -> DONE

This listens to the result topics published by the other nodes (shelf
column, book point, nav result, grasp result) and drives the sequence
forward, with a timeout per state so a stuck state doesn't hang forever.

IMPORTANT -- this is a starting scaffold:
- The pose-lookup step for "where is shelf column N" and "where is the
  bin" are stubbed as placeholders below. You need real coordinates (or a
  computed offset) once you can see the actual sim environment -- this
  can't be finalized without hands-on testing.
- Timeout values (TIMEOUT_SEC) are guesses; tune them against real trial
  timing once you're gathering data per the build plan's Step 8.
"""

import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped, PointStamped

TIMEOUT_SEC = 30.0


class State(Enum):
    LOCATE_SHELF = auto()
    NAVIGATE_TO_SHELF = auto()
    LOCATE_BOOK = auto()
    GRASP_BOOK = auto()
    RETURN_TO_START = auto()
    LOCATE_BIN = auto()
    PLACE_BOOK = auto()
    DONE = auto()
    FAILED = auto()


class Orchestrator(Node):
    def __init__(self):
        super().__init__('orchestrator')

        self.declare_parameter('shelf_column_number', 0)
        self.declare_parameter('book_colour', 'red')
        self.target_column = self.get_parameter('shelf_column_number').value
        self.book_colour = self.get_parameter('book_colour').value

        self.state = State.LOCATE_SHELF
        self.state_entered_at = time.time()

        self.nav_target_pub = self.create_publisher(PoseStamped, '/erc/nav_target', 10)

        self.create_subscription(Int32, '/erc/shelf_column_identification', self.on_shelf_column, 10)
        self.create_subscription(Int32, '/erc/shelf_row_identification', self.on_shelf_row, 10)
        self.create_subscription(Bool, '/erc/nav_result', self.on_nav_result, 10)
        self.create_subscription(Bool, '/erc/grasp_result', self.on_grasp_result, 10)
        self.create_subscription(PointStamped, '/erc/target_book_point', self.on_book_point, 10)

        self.timer = self.create_timer(1.0, self.tick)
        self.get_logger().info(f'Orchestrator started. Target: column {self.target_column}, '
                                f'colour {self.book_colour}')

    def transition(self, new_state: State):
        self.get_logger().info(f'{self.state.name} -> {new_state.name}')
        self.state = new_state
        self.state_entered_at = time.time()

    def tick(self):
        # Timeout watchdog -- explicitly the "error handling" the rubric grades.
        elapsed = time.time() - self.state_entered_at
        if elapsed > TIMEOUT_SEC and self.state not in (State.DONE, State.FAILED):
            self.get_logger().error(f'Timeout in state {self.state.name} after {elapsed:.1f}s')
            self.transition(State.FAILED)
            return

        if self.state == State.NAVIGATE_TO_SHELF and elapsed < 1.0:
            # send the nav goal exactly once on state entry
            self._send_shelf_nav_goal()

    def on_shelf_column(self, msg: Int32):
        if self.state == State.LOCATE_SHELF and msg.data == self.target_column:
            self.transition(State.NAVIGATE_TO_SHELF)

    def on_nav_result(self, msg: Bool):
        if self.state == State.NAVIGATE_TO_SHELF:
            if msg.data:
                self.transition(State.LOCATE_BOOK)
            else:
                self.get_logger().error('Navigation to shelf failed')
                self.transition(State.FAILED)
        elif self.state == State.RETURN_TO_START:
            if msg.data:
                self.transition(State.LOCATE_BIN)
            else:
                self.get_logger().error('Return navigation failed')
                self.transition(State.FAILED)

    def on_shelf_row(self, msg: Int32):
        # Row info published by perception_book_node -- used for logging/report
        # data, not directly for state transitions (the book point drives that).
        self.get_logger().info(f'Book detected at row {msg.data}')

    def on_book_point(self, msg: PointStamped):
        if self.state == State.LOCATE_BOOK:
            self.transition(State.GRASP_BOOK)
            # manipulation_node is already subscribed to the same topic and
            # will act on this independently -- orchestrator just tracks state.

    def on_grasp_result(self, msg: Bool):
        if self.state == State.GRASP_BOOK:
            if msg.data:
                self.transition(State.RETURN_TO_START)
                self._send_return_nav_goal()
            else:
                self.get_logger().error('Grasp failed')
                # A production version should attempt a retry here (e.g. up
                # to 2 more attempts) before giving up -- graded explicitly.
                self.transition(State.FAILED)
        elif self.state == State.PLACE_BOOK:
            if msg.data:
                self.transition(State.DONE)
                self.get_logger().info('Task complete')
            else:
                self.transition(State.FAILED)

    def _send_shelf_nav_goal(self):
        # TODO: replace with a real pose once you know the shelf column
        # spacing/geometry from the actual sim -- this is a placeholder.
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(self.target_column)  # placeholder mapping
        pose.pose.orientation.w = 1.0
        self.nav_target_pub.publish(pose)

    def _send_return_nav_goal(self):
        # TODO: replace with the actual Start/End Zone pose from the sim.
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.orientation.w = 1.0
        self.nav_target_pub.publish(pose)


def main():
    rclpy.init()
    node = Orchestrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
