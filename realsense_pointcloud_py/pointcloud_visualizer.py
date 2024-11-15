#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import sensor_msgs_py.point_cloud2 as pc2

class BlueObjectDetector(Node):
    def __init__(self):
        super().__init__('blue_object_detector')
        self.bridge = CvBridge()
        
        self.color_sub = message_filters.Subscriber(
            self,
            Image,
            '/camera/camera/color/image_raw'
        )
        
        self.pointcloud_sub = message_filters.Subscriber(
            self,
            PointCloud2,
            '/camera/camera/depth/color/points'
        )
        
        self.camera_info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            '/camera/camera/color/camera_info'
        )
        
        # Changed to Marker publisher for multiple points
        self.marker_pub = self.create_publisher(
            Marker,
            'blue_object_centroids',
            10
        )
        
        cv2.namedWindow('Blue Object Detection', cv2.WINDOW_NORMAL)
        
        self.ts = message_filters.TimeSynchronizer(
            [self.color_sub, self.pointcloud_sub, self.camera_info_sub],
            10
        )
        self.ts.registerCallback(self.synchronized_callback)

    def detect_blue_objects(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((5,5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))
        
        return bounding_boxes
        
    def synchronized_callback(self, color_msg, pointcloud_msg, camera_info_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            bounding_boxes = self.detect_blue_objects(cv_image)
            
            display_image = cv_image.copy()
            centroids = []
            K = np.array(camera_info_msg.k).reshape(3, 3)
            
            for bbox in bounding_boxes:
                x, y, w, h = bbox
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                points_list = []
                for data in pc2.read_points(pointcloud_msg, field_names=['x', 'y', 'z'], skip_nans=True):
                    point_3d = np.array([data[0], data[1], data[2]])
                    point_2d = K @ point_3d
                    point_2d = point_2d / point_2d[2]
                    x_2d, y_2d = point_2d[0], point_2d[1]
                    
                    if (x <= x_2d <= x + w and y <= y_2d <= y + h):
                        points_list.append([data[0], data[1], data[2]])
                
                if points_list:
                    points = np.array(points_list)
                    centroid = np.mean(points, axis=0)
                    centroids.append(centroid)
            
            cv2.imshow('Blue Object Detection', display_image)
            cv2.waitKey(1)
            
            # Create and publish marker message with all points
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