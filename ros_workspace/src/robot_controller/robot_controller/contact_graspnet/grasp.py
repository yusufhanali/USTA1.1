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

import cgn_pytorch_src.cgn_utils as cgn_utils
from robot_control.new_controller import spin_thread


class GraspNode(Node):
    def init_tf(self):
        self.tf_broadcaster = TransformBroadcaster(self)
    
    def init_cgn(self, config_file = "cgn_pytorch_src/cgn_pytorch/checkpoints", load_model = True, save_path = "cgn_pytorch_src/cgn_pytorch/checkpoints/current.pth"):
        self.cgn, _, _ = cgn_utils.initialize_net(config_file, load_model, save_path)
    
    def init_pcd(self):
        self.point_cloud_subscriber = self.create_subscription(PointCloud2, '/yifan/wrist/depth/color/points', self.point_cloud_callback, 10)        
        self.point_cloud_publisher = self.create_publisher(PointCloud2, '/filtered_point_cloud', 10)
        
        self.grasp_event = threading.Event()
        self.grasp_event.clear()
    
    def __init__(self, name="grasp_node"):
        super().__init__(name)
        
        self.init_tf()
        self.init_cgn()
        self.init_pcd()
        
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
            (x >= -0.15) & (x <= 0.15) &
            (y >= -0.15) & (y <= 0.15)
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
        
        header = msg.header
        filtered_msg = pc2.create_cloud_xyz32(header, points.tolist())
        self.point_cloud_publisher.publish(filtered_msg)
        
    def run_cgn_inference(self, point_cloud):
        try:
            pred_grasps, pred_success, downsample = cgn_utils.cgn_infer(self.cgn, point_cloud, obj_mask=None, threshold=0.9)
            self.get_logger().info(f"CGN inference completed. Number of grasps found: {pred_grasps.shape[0]}")
            return pred_grasps, pred_success, downsample
        except Exception as e:
            self.get_logger().error(f"Error during CGN inference: {traceback.format_exc()}")
            return None, None, None
                
def main(args=None):
    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        grasp_node = GraspNode()
        
        grasp_spin_thread = threading.Thread(target=spin_thread, args=(grasp_node,))
        grasp_spin_thread.start()
        
        while rclpy.ok():
            grasp_node.grasp_event.wait()
                
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:    
        print("Error in main: ", traceback.format_exc())   
        pass
    
    rclpy.try_shutdown()
    grasp_spin_thread.join()
    
    print("take care of yourself")