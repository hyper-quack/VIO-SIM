from setuptools import setup
import os
from glob import glob

package_name = 'navigation_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'waypoints'),
            glob('waypoints/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wassimsf',
    maintainer_email='wassimsf@todo.todo',
    description='Navigation manager for CSKy drone',
    license='MIT',
    entry_points={
    'console_scripts': [
        'a_star_planner = navigation_manager.a_star_planner:main',
        'obstacle_detector = navigation_manager.obstacle_detector:main',
        'path_follower = navigation_manager.path_follower:main',
        'safety_layer = navigation_manager.safety_layer:main',
        'waypoint_manager = navigation_manager.waypoint_manager:main',
        'depth_filter = navigation_manager.depth_filter:main',
        'octomap_manager = navigation_manager.octomap_manager:main',
        'rrt_local_planner = navigation_manager.rrt_local_planner:main',
    ],
},
)