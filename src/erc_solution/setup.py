from setuptools import find_packages, setup

package_name = 'erc_solution'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/erc_solution.launch.py', 'launch/grasp_pipeline.launch.py', 'launch/move_group.launch.py', 'launch/twist_test.launch.py', 'launch/final_test.launch.py']),
        ('share/' + package_name + '/config', [
            'config/tiago_pro.srdf',
            'config/kinematics.yaml',
            'config/moveit_controllers.yaml',
            'config/joint_limits.yaml',
        ]),
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
            'waypoint_nav_final = erc_solution.waypoint_nav_final:main',
            'calibrate_bearings = erc_solution.calibrate_bearings:main',
            'book_locator_node = erc_solution.book_locator_node:main',
            'shelf_locator_node = erc_solution.shelf_locator_node:main',
            'twist_book_finder = erc_solution.twist_book_finder:main',
            'twist_shelf_and_book = erc_solution.twist_shelf_and_book:main',
            'final_approach_node = erc_solution.final_approach:main',
        ],
    },
)
