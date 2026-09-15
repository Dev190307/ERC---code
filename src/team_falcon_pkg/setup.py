from setuptools import find_packages, setup

package_name = 'team_falcon_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
            'arm_joint_test = team_falcon_pkg.arm_joint_test:main',
            'arm_relative_test = team_falcon_pkg.arm_relative_test:main',
            'raise_both_arms = team_falcon_pkg.raise_both_arms:main',
            'goto_waypoint = team_falcon_pkg.goto_waypoint:main',
            'reach_test = team_falcon_pkg.reach_test:main',
        'perception_shelf_node = team_falcon_pkg.perception_shelf_node:main',
        'perception_book_node = team_falcon_pkg.perception_book_node:main',
        'nav_goal_manager = team_falcon_pkg.nav_goal_manager:main',
        'manipulation_node = team_falcon_pkg.manipulation_node:main',
        'orchestrator = team_falcon_pkg.orchestrator:main',
    ],
},
)
