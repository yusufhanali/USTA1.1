import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as R    
import numpy as np

import sys
import tty
import termios
import threading

class TransformController(Node):
    def __init__(self):
        super().__init__('transform_controller')
        
        self.broadcaster = TransformBroadcaster(self)
        
        self.head_position = [0.0162, 0.0788, 0.0268]  # x, y, z
        self.head_orientation = [-np.pi/2, -np.pi/2, 0.0] # aerial xyz
        
        self.stdin_fd = sys.stdin.fileno()
        self.translation_step = 0.0005
        self.rotation_step = 0.0002
            
        self.publish_timer = self.create_timer(0.1, self.publish_tf)

    def publish_tf(self):
        orientation_quat = R.from_euler('ZYZ', self.head_orientation).as_quat()  # Convert to xyzw format
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'wrist_3_link'
        t.child_frame_id = 'wrist_yifan_camera_link'
        t.transform.translation.x = self.head_position[0]
        t.transform.translation.y = self.head_position[1]
        t.transform.translation.z = self.head_position[2]
        t.transform.rotation.x = orientation_quat[0]
        t.transform.rotation.y = orientation_quat[1]
        t.transform.rotation.z = orientation_quat[2]
        t.transform.rotation.w = orientation_quat[3]
        self.broadcaster.sendTransform(t)
        
        print(f'position={self.head_position}, orientation={self.head_orientation}, orientation_quat={orientation_quat}', end='\r')  # Print on the same line
        
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
                    self.head_orientation[0] += self.rotation_step
                elif ch == 'f':
                    self.head_orientation[0] -= self.rotation_step
                elif ch == 't':
                    self.head_orientation[1] += self.rotation_step
                elif ch == 'g':
                    self.head_orientation[1] -= self.rotation_step
                elif ch == 'y':
                    self.head_orientation[2] += self.rotation_step
                elif ch == 'h':
                    self.head_orientation[2] -= self.rotation_step
                elif ch == 'u':
                    self.translation_step += 0.1
                elif ch == 'j':
                    self.translation_step = max(0.1, self.translation_step - 0.1)
                elif ch == 'ı':
                    self.rotation_step += 0.1
                elif ch == 'k':
                    self.rotation_step = max(0.1, self.rotation_step - 0.1)        
                    
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