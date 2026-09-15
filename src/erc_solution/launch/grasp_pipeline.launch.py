"""
grasp_pipeline.launch.py — brings up everything the grasp needs in ONE
terminal: move_group (MoveIt2 planner) + manipulation_node (subscribes
to /erc/target_book_point and drives the arm).

Deliberately does NOT include waypoint_nav_final — keep that running
standalone in its own terminal via:
    ros2 run erc_solution waypoint_nav_final --ros-args -p target_column:=1 -p book_colour:=red
so it stays untouched and easy to verify independently.

Usage (after simulation.launch.py is already running in another terminal):
    ros2 launch erc_solution grasp_pipeline.launch.py
"""
import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    erc_solution_share = get_package_share_directory('erc_solution')
    urdf_path = '/opt/erc_ws/src/erc_description/urdf/tiago_pro.urdf'
    srdf_path = os.path.join(erc_solution_share, 'config', 'tiago_pro.srdf')
    kinematics_path = os.path.join(erc_solution_share, 'config', 'kinematics.yaml')
    controllers_path = os.path.join(erc_solution_share, 'config', 'moveit_controllers.yaml')
    joint_limits_path = os.path.join(erc_solution_share, 'config', 'joint_limits.yaml')
    with open(joint_limits_path, 'r') as f:
        _raw_joint_limits = yaml.safe_load(f)
    # MoveIt2 (OMPL) reads joint limits from 'robot_description_planning.joint_limits',
    # not a bare 'joint_limits' namespace -- wrap it correctly here instead of
    # passing the file path directly as a parameter file.
    joint_limits_params = {
        'robot_description_planning': {
            'joint_limits': _raw_joint_limits['/**']['ros__parameters']['joint_limits']
        }
    }

    with open(urdf_path, 'r') as f:
        robot_description_content = f.read()
    with open(srdf_path, 'r') as f:
        robot_description_semantic_content = f.read()

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            {'robot_description': robot_description_content},
            {'robot_description_semantic': robot_description_semantic_content},
            kinematics_path,
            controllers_path,
            joint_limits_params,
            {'use_sim_time': True},
            {'publish_robot_description_semantic': True},
            {
                'planning_pipelines': ['ompl'],
                'default_planning_pipeline': 'ompl',
                'ompl.planning_plugin': 'ompl_interface/OMPLPlanner',
                'ompl.request_adapters': (
                    'default_planner_request_adapters/AddTimeOptimalParameterization '
                    'default_planner_request_adapters/ResolveConstraintFrames '
                    'default_planner_request_adapters/FixWorkspaceBounds '
                    'default_planner_request_adapters/FixStartStateBounds '
                    'default_planner_request_adapters/FixStartStateCollision '
                    'default_planner_request_adapters/FixStartStatePathConstraints'
                ),
                'ompl.start_state_max_bounds_error': 0.1,
            },
            {'moveit_manage_controllers': True},
            {
                'planning_scene_monitor_options': {
                    'joint_state_topic': '/joint_states',
                    'attached_collision_object_topic': '/moveit_cpp/planning_scene_monitor',
                    'publish_planning_scene_topic': '/moveit_cpp/publish_planning_scene',
                    'monitored_planning_scene_topic': '/monitored_planning_scene',
                    'wait_for_initial_state_timeout': 10.0,
                },
            },
        ],
    )

    manipulation = Node(
        package='team_falcon_pkg',
        executable='manipulation_node',
        name='manipulation_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        move_group,
        manipulation,
    ])
