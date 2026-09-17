import traceback
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import rclpy.time
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import TransformStamped
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import contact_graspnet.cgn_pytorch_src.cgn_utils as cgn_utils
from robot_control.new_controller import spin_thread
import utilities.linear_algebra as la


class GraspNode(Node):
    def init_tf(self):
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        can_transform = False
        
        while not can_transform:
            try:
                curr_ts = rclpy.time.Time()
                can_transform = self.tfBuffer.can_transform("base", "wrist_3_link", curr_ts) and self.tfBuffer.can_transform("world", "base", curr_ts)
            except Exception as e:
                time.sleep(0.01) #TODO maybe sleep a bit 
                self.get_logger().info(e)
                pass
            finally:
                if can_transform:
                    self.get_logger().info('TF tree received.')
                    break
                else:
                    rclpy.spin_once(self)
    
    def init_cgn(self, config_file = "/home/kovan/USTA1.1/ros_workspace/src/robot_controller/robot_controller/contact_graspnet/cgn_pytorch_src/cgn_pytorch/checkpoints", load_model = True, save_path = "/home/kovan/USTA1.1/ros_workspace/src/robot_controller/robot_controller/contact_graspnet/cgn_pytorch_src/cgn_pytorch/checkpoints/current.pth"):
        self.cgn, _, _ = cgn_utils.initialize_net(config_file, load_model, save_path)
    
    def init_pcd(self):
        self.point_cloud_subscriber = self.create_subscription(PointCloud2, '/yifan/wrist/depth/color/points', self.point_cloud_callback, 10, callback_group=MutuallyExclusiveCallbackGroup())        
        self.point_cloud_publisher = self.create_publisher(PointCloud2, '/filtered_point_cloud', 10)
        self.get_pcd = False
        self.grasp_pcd = None
        self.grasp_pcd_offset = None
        
        self.grasp_event = threading.Event()
        self.grasp_event.clear()
        
        self.grasp_signal_subscriber = self.create_subscription(Bool, "/grasp_signal", self.grasp_signal_callback, 10)
        self.grasp_pose_publisher = self.create_publisher(Float32MultiArray, "/grasp_pose", 10)
    
    def __init__(self, name="grasp_node"):
        super().__init__(name)
        
        self.init_tf()
        self.init_cgn()
        self.init_pcd()
        
        self.base = "base"
                
        self.get_logger().info("GraspNode initialized.")
       
    def grasp_signal_callback(self, msg):
        if msg.data:
            self.get_logger().info("Grasp signal received. Setting grasp_event.")
            self.get_pcd = True
        
    def point_cloud_callback(self, msg):
        if self.get_pcd:
            self.get_pcd = False
            cloud_generator = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            structured_cloud = np.array(list(cloud_generator))

            if structured_cloud.size == 0:
                return

            x = structured_cloud['x']
            y = structured_cloud['y']
            z = structured_cloud['z']

            mask = (
                (x >= -0.12) & (x <= 0.12) &
                (y >= -0.07) & (y <= 0.17) &
                (z >= 0.0) & (z <= 1.0)
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
                        
            camera_to_base_tf = self.tfBuffer.lookup_transform(self.base, msg.header.frame_id, rclpy.time.Time())
            camera_to_base_matrix = la.tf_transform_to_homogeneous_matrix(camera_to_base_tf.transform)
            
            points_in_base = np.array([(camera_to_base_matrix @ np.array([p[0], p[1], p[2], 1.0]).T)[:3] for p in points])
            
            header = msg.header
            header.frame_id = self.base
            filtered_msg = pc2.create_cloud_xyz32(header, points_in_base.tolist())
            self.point_cloud_publisher.publish(filtered_msg)
            
            # Zero Center the point cloud
            centroid = np.mean(points_in_base, axis=0)
            points_in_base -= centroid
            
            self.point_cloud_data = (points_in_base, centroid)
            
            self.grasp_event.set()

    def publish_grasp_pose(self, grasp_pose, grasp_pcd_offset):                
        grasp_orientation = grasp_pose[:3, :3]
        grasp_orientation_quat = R.from_matrix(grasp_orientation).as_quat()
        
        grasp_position = grasp_pose[:3, 3] + grasp_pcd_offset
        #grasp_position -= grasp_orientation @ np.array([0, 0, 0.055]) # No need, changed the eefo position to the tip of the gripper, so the offset is already accounted for in the grasp_pose.
        grasp_position += grasp_orientation @ np.array([0, 0, 0.14]) # Add the offset to the grasp position to account for the gripper's length and models learned position offset
                        
        grasp_pose = np.concatenate((grasp_position, grasp_orientation_quat))
                        
        transform_msg = TransformStamped()
        transform_msg.header.stamp = self.get_clock().now().to_msg()
        transform_msg.header.frame_id = "base"
        transform_msg.child_frame_id = "predicted_grasp"
        transform_msg.transform.translation.x = grasp_pose[0]
        transform_msg.transform.translation.y = grasp_pose[1]
        transform_msg.transform.translation.z = grasp_pose[2]
        transform_msg.transform.rotation.x = grasp_pose[3]
        transform_msg.transform.rotation.y = grasp_pose[4]
        transform_msg.transform.rotation.z = grasp_pose[5]
        transform_msg.transform.rotation.w = grasp_pose[6]
        
        self.tf_broadcaster.sendTransform(transform_msg)
        
        grasp_pose_msg = Float32MultiArray()
        grasp_pose_msg.data = grasp_pose.tolist()
        self.grasp_pose_publisher.publish(grasp_pose_msg)
        self.get_logger().info(f"Published grasp pose: {grasp_pose}")

    def publish_dud_grasp_pose(self):
        dud_pose_msg = Float32MultiArray()
        dud_pose_msg.data = [-1.0] * 7
        self.grasp_pose_publisher.publish(dud_pose_msg)
        self.get_logger().info("Published dud grasp pose.")
    
        
    def run_cgn_inference(self, point_cloud):
        try:
            pred_grasps, pred_success, downsample = cgn_utils.cgn_infer(self.cgn, point_cloud, obj_mask=None, threshold=0.5)
            self.get_logger().info(f"CGN inference completed. Number of grasps found: {pred_grasps.shape[0]}")
            return pred_grasps, pred_success, downsample
        except Exception as e:
            self.get_logger().error(f"Error during CGN inference: {traceback.format_exc()}")
            return None, None, None

    def get_and_publish_grasp_pose(self, max_tries=10):
        tries = 0
        point_cloud, centroid = self.point_cloud_data
        while rclpy.ok() and tries < max_tries:
            pred_grasps, _, _ = self.run_cgn_inference(point_cloud)
            tries += 1
            
            if pred_grasps is not None and pred_grasps.shape[0] > 0:                
                grasp_pose = pred_grasps[0]
                self.publish_grasp_pose(grasp_pose, centroid)
                return grasp_pose
            else:
                self.get_logger().info(f"No valid grasps found at attempt {tries}. Retrying...")
                
        self.get_logger().info("Max tries reached. No valid grasp found.")
        self.publish_dud_grasp_pose()
        
        return None  # Return None if no valid grasp was found after max_tries

    def cleanup(self):
        self.get_logger().info("Cleaning up GraspNode resources.")
        self.destroy_node()
        self.get_logger().info("GraspNode destroyed.")

                
def main(args=None):
    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        grasp_node = GraspNode()
        
        grasp_spin_thread = threading.Thread(target=spin_thread, args=(grasp_node,))
        grasp_spin_thread.start()
        
        while rclpy.ok():
            grasp_node.grasp_event.wait()
            print("Grasp event set. Running CGN inference.")
            
            grasp_node.get_and_publish_grasp_pose()
            
            grasp_node.grasp_event.clear()
                
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:    
        print("Error in main: ", traceback.format_exc())   
        pass
    
    grasp_node.cleanup()
    
    rclpy.try_shutdown()
    grasp_spin_thread.join()
    
    print("take care of yourself")