#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import tf2_ros

class ArucoTagDetector(Node):
    def __init__(self):
        super().__init__('aruco_tag_detector')
        self.bridge = CvBridge()
        self.target_ids = {4, 5, 7, 8}
        self.tf_scale = 1.
        self.filtered_translation = np.zeros(3)  # Initial translation
        self.filtered_rotation = np.array([0.0, 0.0, 0.0, 1.0])  # Initial quaternion (identity rotation)
        self.alpha = 0.2  # Smoothing factor for EMA


        # Subscribe to the compressed color image and depth image
        self.color_sub = message_filters.Subscriber(
            self,
            CompressedImage,
            '/camera/camera/color/image_raw/compressed'
        )
        
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw'
        )
        
        self.camera_info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            '/camera/camera/color/camera_info'
        )
        
        # Publisher for markers (centroids)
        self.marker_pub = self.create_publisher(
            Marker,
            'aruco_tag_positions',
            10
        )
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        cv2.namedWindow('Aruco Tag Detection', cv2.WINDOW_NORMAL)
        
        self.ts = message_filters.TimeSynchronizer(
            [self.color_sub, self.depth_sub, self.camera_info_sub],
            10
        )
        self.ts.registerCallback(self.synchronized_callback)

    def detect_aruco_tags(self, image, aruco_dict, parameters):
        """Detect Aruco markers in the image and return their ids and corners."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)
        return corners, ids

    def synchronized_callback(self, color_msg, depth_msg, camera_info_msg):
        try:
            # Convert compressed color image to OpenCV format
            np_arr = np.frombuffer(color_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Convert depth image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            # Initialize Aruco dictionary and parameters
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
            parameters = cv2.aruco.DetectorParameters_create()
            
            # Detect Aruco markers
            corners, ids = self.detect_aruco_tags(cv_image, aruco_dict, parameters)
            
            display_image = cv_image.copy()
            K = np.array(camera_info_msg.k).reshape(3, 3)  # Camera intrinsic matrix
            
            # Map of detected marker positions
            marker_positions = {}
            
            if ids is not None:
                for marker_id, corner in zip(ids.flatten(), corners):
                    # Calculate the centroid of the marker
                    centroid_x = int(np.mean(corner[0][:, 0]))
                    centroid_y = int(np.mean(corner[0][:, 1]))
                    
                    # Get depth value at centroid
                    depth = depth_image[centroid_y, centroid_x] * 0.001  # Convert depth to meters
                    if depth > 0:  # Ensure valid depth
                        # Convert pixel coordinates to 3D
                        position = self.pixel_to_3d(centroid_x, centroid_y, depth, K)
                        marker_positions[marker_id] = position
                
                # Ensure required markers are detected
                if all(mid in marker_positions for mid in [4, 5, 7]):
                    origin = marker_positions[4]  # Tag 4 as the origin
                    vector_45 = marker_positions[5] - origin
                    vector_47 = marker_positions[7] - origin
                    
                    # Calculate plane normal
                    plane_normal = np.cross(vector_45, vector_47)
                    plane_normal /= np.linalg.norm(plane_normal)  # Normalize
                    
                    # Create rotation matrix (Z: plane normal, X: vector_45, Y: orthogonal)
                    z_axis = plane_normal
                    x_axis = vector_45 / np.linalg.norm(vector_45)
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis /= np.linalg.norm(y_axis)
                    
                    # Combine into rotation matrix
                    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
                    
                    # Convert to quaternion
                    quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
                    
                    # Publish the transform
                    self.publish_tf_centroid(origin, quaternion, color_msg.header)
            
            cv2.imshow('Aruco Tag Detection', display_image)
            cv2.waitKey(1)
                        
        except Exception as e:
            self.get_logger().error(f'Error in processing: {str(e)}')

    def rotation_matrix_to_quaternion(self, matrix):
        """Convert a rotation matrix to a quaternion."""
        q_w = np.sqrt(1.0 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 2.0
        q_x = (matrix[2, 1] - matrix[1, 2]) / (4.0 * q_w)
        q_y = (matrix[0, 2] - matrix[2, 0]) / (4.0 * q_w)
        q_z = (matrix[1, 0] - matrix[0, 1]) / (4.0 * q_w)
        return [q_x, q_y, q_z, q_w]

    def pixel_to_3d(self, u, v, depth, K):
        """Convert a 2D pixel to 3D point using camera intrinsics."""
        # Convert pixel coordinates (u, v) to normalized camera coordinates
        x_cam = (u - K[0, 2]) / K[0, 0]
        y_cam = (v - K[1, 2]) / K[1, 1]
        
        # Scale by depth to get 3D coordinates
        x_3d = depth * x_cam
        y_3d = depth * y_cam
        z_3d = depth
        
        return np.array([x_3d, y_3d, z_3d])

    def publish_base_link_tf(self, header):
        """Publish a static transform from aruco_link to base_link."""
        # 180-degree rotation about the X-axis
        rotation_quaternion = [1.0, 0.0, 0.0, 0.0]  # Quaternion for 180° about X

        # Create and broadcast TransformStamped
        transform = TransformStamped()
        transform.header = header
        transform.child_frame_id = "base_link"
        transform.header.frame_id = "aruco_link"
        
        # Translation remains the same (no offset)
        transform.transform.translation.x = 0.1
        transform.transform.translation.y = 0.1
        transform.transform.translation.z = 0.2
        
        # Apply the rotation
        transform.transform.rotation.x = rotation_quaternion[0]
        transform.transform.rotation.y = rotation_quaternion[1]
        transform.transform.rotation.z = rotation_quaternion[2]
        transform.transform.rotation.w = rotation_quaternion[3]
        
        self.tf_broadcaster.sendTransform(transform)


    def publish_tf_centroid(self, origin, quaternion, header):
        """Publish the transform with a low-pass filter."""
        # Apply low-pass filter to translation
        self.filtered_translation = (
            self.alpha * np.array(origin)
            + (1 - self.alpha) * self.filtered_translation
        )
        
        # Apply low-pass filter to rotation
        self.filtered_rotation = (
            self.alpha * np.array(quaternion)
            + (1 - self.alpha) * self.filtered_rotation
        )
        self.filtered_rotation /= np.linalg.norm(self.filtered_rotation)  # Normalize quaternion
        
        # Create and broadcast TransformStamped
        transform = TransformStamped()
        transform.header = header
        transform.child_frame_id = "aruco_link"
        transform.transform.translation.x = self.filtered_translation[0]
        transform.transform.translation.y = self.filtered_translation[1]
        transform.transform.translation.z = self.filtered_translation[2]
        transform.transform.rotation.x = self.filtered_rotation[0]
        transform.transform.rotation.y = self.filtered_rotation[1]
        transform.transform.rotation.z = self.filtered_rotation[2]
        transform.transform.rotation.w = self.filtered_rotation[3]
        
        self.tf_broadcaster.sendTransform(transform)
        self.publish_base_link_tf(header)


    
    # @staticmethod
    # def rotation_matrix_to_quaternion(matrix):
    #     """Convert a rotation matrix to a quaternion."""
    #     q_w = np.sqrt(1.0 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 2.0
    #     q_x = (matrix[2, 1] - matrix[1, 2]) / (4.0 * q_w)
    #     q_y = (matrix[0, 2] - matrix[2, 0]) / (4.0 * q_w)
    #     q_z = (matrix[1, 0] - matrix[0, 1]) / (4.0 * q_w)
    #     return [q_x, q_y, q_z, q_w]

    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoTagDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
