from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense2_camera_node',
            output='screen'
        ),
        Node(
            package='realsense_pointcloud_py',
            executable='pointcloud_subscriber',
            name='pointcloud_subscriber',
            output='screen'
        )
    ])
    