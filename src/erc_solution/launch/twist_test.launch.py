import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    target_column_arg = DeclareLaunchArgument('target_column', default_value='1')
    book_colour_arg = DeclareLaunchArgument('book_colour', default_value='red')

    twist_node = Node(
        package='erc_solution',
        executable='twist_shelf_and_book',
        name='twist_shelf_and_book',
        output='screen',
        parameters=[{
            'target_column': LaunchConfiguration('target_column'),
            'book_colour': LaunchConfiguration('book_colour'),
        }],
    )

    manipulation = Node(
        package='team_falcon_pkg',
        executable='manipulation_node',
        name='manipulation_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    erc_solution_share = get_package_share_directory('erc_solution')
    urdf_path = '/opt/erc_ws/src/erc_description/urdf/tiago_pro.urdf'
    srdf_path = os.path.join(erc_solution_share, 'config', 'tiago_pro.srdf')
    kinematics_path = os.path.join(erc_solution_share, 'config', 'kinematics.yaml')
    controllers_path = os.path.join(erc_solution_share, 'config', 'moveit_controllers.yaml')
    joint_limits_path = os.path.join(erc_solution_share, 'config', 'joint_limits.yaml')

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
            joint_limits_path,
            {'use_sim_time': True},
            {'publish_robot_description_semantic': True},
            {
                'planning_pipelines': ['ompl'],
                'default_planning_pipeline': 'ompl',
                'ompl.planning_plugin': 'ompl_interface/OMPLPlanner',
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

    return LaunchDescription([
        target_column_arg,
        book_colour_arg,
        move_group,
        manipulation,
        twist_node,
    ])
