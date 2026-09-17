import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as R    
import numpy as np

import sys
import tty
import termios
import threading

import utilities.linear_algebra as linalg_utils

class TransformController(Node):
    def __init__(self):
        super().__init__('transform_controller')
        
        self.broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.camera_frame = 'wrist_yifan_camera_link'
        self.wrist_frame = 'wrist_3_link'
        self.world_frame = 'world'
        
        self.wrist_to_world_transform = None
        while self.wrist_to_world_transform is None:
            try:
                self.wrist_to_world_transform = self.tf_buffer.lookup_transform(self.world_frame, self.wrist_frame, rclpy.time.Time())
            except Exception as e:
                self.get_logger().info(f"Waiting for transform from {self.wrist_frame} to {self.world_frame}. Error: {e}")
                rclpy.spin_once(self, timeout_sec=1.0)
        
        self.wrist_to_world_matrix = linalg_utils.transform_to_matrix(self.wrist_to_world_transform)
        
        
        self.head_position = [0.0205, 0.0776, 0.0257]  # x, y, z
        #self.relative_head_orientation = [-1.6007963, -1.5457963, 0.0868] # aerial xyz
        self.relative_head_orientation_quat = [0.49716405, 0.50742581, 0.50039075, -0.49493034]  # xyzw format
        self.change_wrt_world = [0.0, 0.0, 0.0]
        
        self.stdin_fd = sys.stdin.fileno()
        self.translation_step = 0.0001
        self.rotation_step = 0.0001
            
        self.publish_timer = self.create_timer(0.1, self.publish_tf)

    def publish_tf(self):
        current_orientation = R.from_quat(self.relative_head_orientation_quat).as_matrix()
        current_orientation = self.wrist_to_world_matrix[:3, :3] @ current_orientation
        current_orientation = R.from_euler('XYZ', self.change_wrt_world).as_matrix() @ current_orientation
        current_orientation = self.wrist_to_world_matrix[:3, :3].T @ current_orientation
        
        orientation_quat = R.from_matrix(current_orientation).as_quat()  # Convert to xyzw format
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.wrist_frame
        t.child_frame_id = self.camera_frame
        t.transform.translation.x = self.head_position[0]
        t.transform.translation.y = self.head_position[1]
        t.transform.translation.z = self.head_position[2]
        t.transform.rotation.x = orientation_quat[0]
        t.transform.rotation.y = orientation_quat[1]
        t.transform.rotation.z = orientation_quat[2]
        t.transform.rotation.w = orientation_quat[3]
        self.broadcaster.sendTransform(t)
        
        print(f'position={self.head_position}, orientation={self.relative_head_orientation_quat}, orientation_quat={orientation_quat}', end='\r')  # Print on the same line
        
def read_input(self):
    while rclpy.ok():
        if sys.stdin.isatty():
            
            old_settings = termios.tcgetattr(self.stdin_fd)
            try:
                tty.setraw(self.stdin_fd)
                ch = sys.stdin.read(1)
                termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, old_settings)                
                
                if ch == 'w':
                    self.head_position[0] += self.translation_step
                elif ch == 's':
                    self.head_position[0] -= self.translation_step
                elif ch == 'a':
                    self.head_position[1] += self.translation_step
                elif ch == 'd':
                    self.head_position[1] -= self.translation_step
                elif ch == 'q':
                    self.head_position[2] -= self.translation_step
                elif ch == 'e':
                    self.head_position[2] += self.translation_step
                elif ch == 'r':
                    self.change_wrt_world[0] += self.rotation_step
                elif ch == 'f':
                    self.change_wrt_world[0] -= self.rotation_step
                elif ch == 't':
                    self.change_wrt_world[1] += self.rotation_step
                elif ch == 'g':
                    self.change_wrt_world[1] -= self.rotation_step
                elif ch == 'y':
                    self.change_wrt_world[2] += self.rotation_step
                elif ch == 'h':
                    self.change_wrt_world[2] -= self.rotation_step
                elif ch == 'u':
                    self.translation_step += 0.0001
                elif ch == 'j':
                    self.translation_step = max(0.0001, self.translation_step - 0.0001)
                elif ch == 'ı':
                    self.rotation_step += 0.0001
                elif ch == 'k':
                    self.rotation_step = max(0.0001, self.rotation_step - 0.0001)        
                    
                elif ch == 'x':
                    rclpy.shutdown()        
            except:
                termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, old_settings)    
    
    
def main(args=None):
    rclpy.init(args=args)
    node = TransformController()
    
    input_thread = threading.Thread(target=read_input, args=(node,))
    input_thread.daemon = True
    input_thread.start()
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()