from setuptools import setup
import os
from glob import glob

package_name = 'vio_localization'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wassimsf',
    maintainer_email='wassimsf@todo.todo',
    description='VIO localization with RTAB-Map for CSKy drone',
    license='MIT',
    entry_points={
    'console_scripts': [
        'rgb_resizer = vio_localization.rgb_resizer:main',
        'stereo_sync = vio_localization.stereo_sync:main',
    ],
},
)