# realsense_pointcloud
A ROS2 package that integrates a realsense camera and computer vision
Run this command to launch the realsense node first before running this script
'''
ros2 run realsense2_camera realsense2_camera_node --ros-args -p enable_rgbd:=true -p enable_sync:=true -p align_depth.enable:=true -p enable_color:=true -p enable_depth:=true -p pointcloud.enable:=true
'''
