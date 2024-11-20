from setuptools import setup
import os
from glob import glob

package_name = 'realsense_pointcloud_py'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Change this line to correctly include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='RealSense PointCloud Processing Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'pointcloud_visualizer = realsense_pointcloud_py.pointcloud_visualizer:main',
        'aruco_detect = realsense_pointcloud_py.aruco_detect:main',
        ],
    },
)
