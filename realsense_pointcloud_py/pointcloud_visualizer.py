#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class BlueObjectDetector(Node):
    def __init__(self):
        super().__init__('blue_object_detector')
        self.bridge = CvBridge()
        
        # Subscribers for color image and depth map
        self.color_sub = message_filters.Subscriber(
            self,
            Image,
            '/camera/camera/color/image_raw'
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
            'blue_object_centroids',
            10
        )
        
        cv2.namedWindow('Blue Object Detection', cv2.WINDOW_NORMAL)
        
        self.ts = message_filters.TimeSynchronizer(
            [self.color_sub, self.depth_sub, self.camera_info_sub],
            10
        )
        self.ts.registerCallback(self.synchronized_callback)

    def detect_blue_objects(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))
        
        return bounding_boxes

    def synchronized_callback(self, color_msg, depth_msg, camera_info_msg):
        try:
            # Convert color image
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            # Convert depth image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            bounding_boxes = self.detect_blue_objects(cv_image)
            
            display_image = cv_image.copy()
            centroids = []
            K = np.array(camera_info_msg.k).reshape(3, 3)
            
            for bbox in bounding_boxes:
                x, y, w, h = bbox
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Find the centroid of the bounding box
                centroid_x = x + w // 2
                centroid_y = y + h // 2
                
                # Scale the bounding box by 0.5 around the centroid
                scaled_w = int(w * 0.5)
                scaled_h = int(h * 0.5)
                scaled_x = centroid_x - scaled_w // 2
                scaled_y = centroid_y - scaled_h // 2
                
                points_list = []
                
                # Collect 3D points from the depth map within the scaled bounding box
                for v in range(scaled_y, scaled_y + scaled_h):
                    for u in range(scaled_x, scaled_x + scaled_w):
                        if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
                            depth = depth_image[v, u] * 0.001  # Convert depth to meters (assuming 16-bit depth image)
                            if depth > 0:  # Valid depth value
                                # Convert 2D pixel to 3D point using camera intrinsics
                                point_3d = self.pixel_to_3d(u, v, depth, K)
                                points_list.append(point_3d)
                
                # Calculate mean position of points within the scaled bounding box
                if points_list:
                    points = np.array(points_list)
                    centroid = np.mean(points, axis=0)
                    centroids.append(centroid)
            
            cv2.imshow('Blue Object Detection', display_image)
            cv2.waitKey(1)
            
            # Create and publish marker message with all centroids
            if centroids:
                marker = Marker()
                marker.header = color_msg.header
                marker.ns = "blue_objects"
                marker.id = 0
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.scale.x = 0.05  # Point size
                marker.scale.y = 0.05
                marker.color.a = 1.0   # Alpha
                marker.color.r = 0.0   # Blue color
                marker.color.g = 0.0
                marker.color.b = 1.0
                
                for centroid in centroids:
                    point = Point()
                    point.x = float(centroid[0])
                    point.y = float(centroid[1])
                    point.z = float(centroid[2])
                    marker.points.append(point)
                
                self.marker_pub.publish(marker)
                    
        except Exception as e:
            self.get_logger().error(f'Error in processing: {str(e)}')

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

    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = BlueObjectDetector()
    
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
