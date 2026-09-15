import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    erc_solution_share = get_package_share_directory('erc_solution')
    urdf_path = '/opt/erc_ws/src/erc_description/urdf/tiago_pro.urdf'
    srdf_path = os.path.join(erc_solution_share, 'config', 'tiago_pro.srdf')
    kinematics_path = os.path.join(erc_solution_share, 'config', 'kinematics.yaml')
    controllers_path = os.path.join(erc_solution_share, 'config', 'moveit_controllers.yaml')

    with open(urdf_path, 'r') as f:
        robot_description_content = f.read()
    with open(srdf_path, 'r') as f:
        robot_description_semantic_content = f.read()

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            {'robot_description': robot_description_content},
            {'robot_description_semantic': robot_description_semantic_content},
            kinematics_path,
            controllers_path,
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
            {
                'capabilities': '',
                'disable_capabilities': '',
            },
            {'moveit_manage_controllers': True},
        ],
    )

    return LaunchDescription([move_group_node])
