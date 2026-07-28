import traceback
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
import rclpy.time
from tf2_ros import TransformBroadcaster

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import open3d as o3d
import cgn_pytorch as cgn


class GraspNode(Node):
    def init_tf(self):
        self.tf_broadcaster = TransformBroadcaster(self)
    
    def __init__(self, name="grasp_node"):
        super().__init__(name)
        
        self.init_tf()
        
        self.point_cloud_subscriber = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.point_cloud_callback, 10)
        
        self.point_cloud_publisher = self.create_publisher(PointCloud2, '/filtered_point_cloud', 10)
        
        self.get_logger().info("GraspNode initialized.")
        
    def point_cloud_callback(self, msg):
        cloud_generator = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        structured_cloud = np.array(list(cloud_generator))

        if structured_cloud.size == 0:
            return

        x = structured_cloud['x']
        y = structured_cloud['y']
        z = structured_cloud['z']

        mask = (
            (x >= -0.5) & (x <= 0.5) &
            (y >= -0.5) & (y <= 0.5) &
            (z >= 0.0)  & (z <= 1.5)
        )

        cropped_structured_cloud = structured_cloud[mask]

        if cropped_structured_cloud.size == 0:
            self.get_logger().info("No points found inside the bounding box.")
            return

        points = np.column_stack((
            cropped_structured_cloud['x'],
            cropped_structured_cloud['y'],
            cropped_structured_cloud['z']
        )).astype(np.float32)

        if points.size == 0:
            return

        center_x, center_y = 0.0, 0.0
        radius = 0.1
        distances_sq = (points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2
        mask = distances_sq <= radius**2
        filtered_points = points[mask]
        
        header = msg.header
        filtered_msg = pc2.create_cloud_xyz32(header, filtered_points.tolist())
        self.point_cloud_publisher.publish(filtered_msg)
        
                
def main(args=None):
    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        grasp_node = GraspNode()
        
        rclpy.spin(grasp_node)
        
        print("Grasp node is running. Press Ctrl+C to exit.")

                
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:    
        print("Error in main: ", traceback.format_exc())   
        pass
    finally:
        rclpy.shutdown()