import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as R    

import sys
import tty
import termios
import threading

class FakeFacePublisher(Node):
    def __init__(self):
        super().__init__('fake_face_publisher')
        
        self.broadcaster = TransformBroadcaster(self)
        
        self.head_position = [0.0245, 0.08, 0.029]  # x, y, z
        self.head_orientation = [-1.2091996, -1.2091996, -1.2091996] # axis-angle representation (roll, pitch, yaw), rotvec basically
        
        self.stdin_fd = sys.stdin.fileno()
        self.translation_step = 0.0005
        self.rotation_step = 0.01
            
        self.publish_timer = self.create_timer(0.1, self.publish_fake_face)

    def publish_fake_face(self):        
        orientation_quat = R.from_rotvec(self.head_orientation).as_quat()
        
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
        
        self.get_logger().info(f'Published fake face transform: position={self.head_position}, orientation={self.head_orientation}')
        
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
                    self.translation_step += 0.1
                elif ch == 'y':
                    self.translation_step = max(0.1, self.translation_step - 0.1)
                elif ch == 'g':
                    self.rotation_step += 0.1
                elif ch == 'h':
                    self.rotation_step = max(0.1, self.rotation_step - 0.1)        
                    
                elif ch == 'x':
                    rclpy.shutdown()        
            except:
                termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, old_settings)    
    
    
def main(args=None):
    rclpy.init(args=args)
    node = FakeFacePublisher()
    
    input_thread = threading.Thread(target=read_input, args=(node,))
    input_thread.daemon = True
    input_thread.start()
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()