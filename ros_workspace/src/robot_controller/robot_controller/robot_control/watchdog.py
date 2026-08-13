import traceback
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
import rclpy.time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from .robotiq_gripper import RobotiqGripper as robotiq_gripper
import ur5e_kinematic.ur5e_kinematics as ur5e_kinematics
import utilities.linear_algebra as linalg_utils
import utilities.geometry as geometry_utils
import utilities.control_and_filters as control_and_filters


class UR5eWatchdog(Node):

    def init_joint_states(self):
        # Get joint states
        self.joint_states_global = {}
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        
        while not self.joint_states_global:
            #self.get_logger().info('Waiting for joint states...')
            rclpy.spin_once(self)
        self.get_logger().info('Joint states received.')
        
    def init_velocity_controller(self):
        # Publish velocity commands
        self.velocityCommandReceiverTopic = "watchdog/velocity_commands"
        self.velocityCommandReceiverSub = self.create_subscription(Float64MultiArray, self.velocityCommandReceiverTopic, self.velocity_command_receiver_callback, 10)
        
        self.velocityControllerTopic = "forward_velocity_controller/commands" # All movements should be done w.r.t. "base", not "base_link" or "world"
        self.velocityControllerPub = self.create_publisher(Float64MultiArray, self.velocityControllerTopic, 10)

        while not self.velocityControllerPub.get_subscription_count():
            #self.get_logger().info('Waiting for velocity controller to connect...')
            rclpy.spin_once(self)
            
        self.get_logger().info('Velocity controller connected.')

    def init_tf(self):
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        can_transform = False
        
        while not can_transform:
            try:
                curr_ts = rclpy.time.Time()
                can_transform = self.tfBuffer.can_transform("base", "wrist_3_link", curr_ts) and self.tfBuffer.can_transform("world", "base", curr_ts)
                if can_transform:
                    tf = self.tfBuffer.lookup_transform("world", "base", curr_ts).transform
                    tf_quat = np.array([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])
                    tf_trans = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
                    homog = linalg_utils.quaternion_to_homogeneous_matrix(tf_quat, tf_trans)
                    #sometimes an error occurs in the tf tree, and the transform from world to base_link is identity. In that case, we need to reinitialize the tfBuffer and tfListener until the tf tree is fixed.
                    if np.equal(homog, np.eye(4)).all():
                        self.tfBuffer = Buffer()
                        self.tfListener = TransformListener(self.tfBuffer, self)
                        can_transform = False   
            except Exception as e:
                # time.sleep(0.1) @ToDo maybe sleep a bit 
                self.get_logger().info(e)
                pass
            finally:
                if can_transform:
                    self.get_logger().info('TF tree received.')
                    break
                else:
                    rclpy.spin_once(self)

    def __init__(self, name = "ur5e_watchdog"):
        super().__init__(name)
        
        # Sanity checks at the start of node (The phrases are very unprofessionally taken from my favorite anime, Serial Experiments Lain)
        self.get_logger().info('not a good day to be a dog,')
        
        # Initialize control rate
        self.control_rate = 500 # Hz
        #self.ros_rate = self.create_rate(self.control_rate, self.get_clock())
        
        self.base = "base"
        self.eef = "wrist_3_link"
        self.world = "world"        
        
        self.num_of_total_joints = ur5e_kinematics.NUM_OF_JOINTS
        
        self.init_joint_states()        
        self.init_velocity_controller()        
        self.init_tf()
        
        self.base_to_world_homogeneous = np.array([ # WAS THIS WRONG BEFORE??? THE SIGNS ARE OPPOSITE NOW, LOOK INTO IT
            [-0.7071068, 0.7071068, 0.0, 1.7513],
            [-0.7071068, -0.7071068, 0.0, 0.0955],
            [0.0, 0.0, 1.0, 0.7347],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        self.world_to_base_homogeneous = linalg_utils.reverse_homogeneous_matrix(self.base_to_world_homogeneous)
        
        self.first_movement = True        
        self.prev_velocities = np.zeros(6)
                        
        self.constraint_functions = []  # List to hold registered constraint functions
        self.constraint_register()  # Call the method to register constraints
                                                                        
        # Sanity check at the end of node. If both of these are printed, then the node is probably working properly.
        self.get_logger().info('but again, when ever is')  


    def joint_state_callback(self, msg):
        # Why does UR5e always send joint states in an order other than 012345? It was 210345 in ROS1 and now this.
        self.joint_states_global["pos"] = np.array([
                                            msg.position[5],
                                            msg.position[0],
                                            msg.position[1],
                                            msg.position[2],
                                            msg.position[3],
                                            msg.position[4]])
        self.joint_states_global["vels"] = np.array([
                                            msg.velocity[5],
                                            msg.velocity[0],
                                            msg.velocity[1],
                                            msg.velocity[2],
                                            msg.velocity[3],
                                            msg.velocity[4]])
        self.joint_states_global["effs"] = np.array([
                                            msg.effort[5],
                                            msg.effort[0],
                                            msg.effort[1],
                                            msg.effort[2],
                                            msg.effort[3],
                                            msg.effort[4]])         
    
    def velocity_command_receiver_callback(self, msg):
        vels = np.array(msg.data)
        
        for constraint_func in self.constraint_functions:
            vels = constraint_func(vels)
            
        self.publish_velocity_command(vels)


    def get_current_coordinate(self):
        current_coordinate = np.zeros(3)        
        try:
            current_coordinate = ur5e_kinematics.get_ee_position(self.joint_states_global["pos"])
        except:
            self.get_logger().info(f"Error in getting current coordinate: {traceback.format_exc()}")
        return current_coordinate

    def get_current_orientation(self):
        orientation = ur5e_kinematics.get_ee_orientation(self.joint_states_global["pos"])
        return orientation
       
    def get_joint_pos(self):
        return self.joint_states_global["pos"]
          
    def get_jacobian_matrix(self):
        # get_jacobian is a function that returns the jacobian matrix of the robot at the current joint states, it was calculated using matlab.
        jacobian = ur5e_kinematics.get_jacobian(self.joint_states_global["pos"].tolist())
        return jacobian
                        
    def get_inverse_jacobian(self):
        # get_jacobian is a function that returns the jacobian matrix of the robot at the current joint states, it was calculated using matlab.
        jacobian = ur5e_kinematics.get_jacobian(self.joint_states_global["pos"].tolist())
        invj = np.linalg.pinv(jacobian, rcond=1e-15)
        return invj
        
    
    def constraint_register(self):
        '''
            Use this function to register all the constraints that you want to apply to the robot. This function is called in the constructor of the class.
        '''
        #self.constraint_functions.append(self.height_constraint)
        self.constraint_functions.append(self.joint_limit_constraint)
        
    def height_constraint(self, commanded_velocity):
        max_z = 0.85
        min_z = 0.2
        
        caution_distance = 0.1
        
        cartesian_velocity = self.get_jacobian_matrix() @ commanded_velocity
        
        current_z = self.get_current_coordinate()[2]
        commanded_z = current_z + cartesian_velocity[2] * (1 / self.control_rate)
        
        if cartesian_velocity[2] > 0 and commanded_z > max_z - caution_distance:
            #self.get_logger().info(f"Height constraint activated: commanded_z ({commanded_z}) > max_z - caution_distance ({max_z - caution_distance})")
            if commanded_z > max_z:
                #self.get_logger().info(f"Height constraint activated: commanded_z ({commanded_z}) > max_z ({max_z})")
                cartesian_velocity[2] = 0
            else:
                cartesian_velocity[2] = ((max_z - commanded_z) / (caution_distance)) * cartesian_velocity[2]
        
        elif cartesian_velocity[2] < 0 and commanded_z < min_z + caution_distance:
            #self.get_logger().info(f"Height constraint activated: commanded_z ({commanded_z}) < min_z + caution_distance ({min_z + caution_distance})")
            if commanded_z < min_z:
                #self.get_logger().info(f"Height constraint activated: commanded_z ({commanded_z}) < min_z ({min_z})")
                cartesian_velocity[2] = 0
            else:
                cartesian_velocity[2] = ((commanded_z - min_z) / (caution_distance)) * cartesian_velocity[2]
                
        else:
            return commanded_velocity
        
        altered_velocity = self.get_inverse_jacobian() @ cartesian_velocity
        return altered_velocity
       
    def joint_limit_constraint(self, commanded_velocity):
        joint_limits = ur5e_kinematics.JOINT_LIMITS
        caution_angle = 0.15  # radians
        
        current_joint_positions = self.joint_states_global["pos"]
        commanded_joint_positions = current_joint_positions + commanded_velocity * (1 / self.control_rate)
        
        altered_velocity = np.copy(commanded_velocity)
        
        for i, command in enumerate(commanded_velocity):
            if command > 0 and commanded_joint_positions[i] > joint_limits[i][1] - caution_angle:
                #self.get_logger().info(f"Joint limit constraint activated: Joint {i} commanded position exceeds cautionary limit.")
                if commanded_joint_positions[i] > joint_limits[i][1]:
                    self.get_logger().info(f"Joint limit constraint activated: Joint {i} commanded position exceeds upper limit.")
                    altered_velocity[i] = 0
                else:
                    altered_velocity[i] = ((joint_limits[i][1] - (commanded_joint_positions[i])) / caution_angle) * command
                    
            elif command < 0 and commanded_joint_positions[i] < joint_limits[i][0] + caution_angle:
                #self.get_logger().info(f"Joint limit constraint activated: Joint {i} commanded position exceeds cautionary limit.")
                if commanded_joint_positions[i] < joint_limits[i][0]:
                    self.get_logger().info(f"Joint limit constraint activated: Joint {i} commanded position exceeds lower limit.")
                    altered_velocity[i] = 0
                else:
                    altered_velocity[i] = ((commanded_joint_positions[i] - joint_limits[i][0]) / caution_angle) * command
                    
        return altered_velocity
                    

    def publish_velocity_command(self, vels):
        if type(vels) is not np.ndarray:
            vels = np.array(vels)
         
        vel_msg = Float64MultiArray()
        vel_msg.data = vels
        self.velocityControllerPub.publish(vel_msg)
        
    def stop_movement(self):
        vel_msg = Float64MultiArray()
        vel_msg.data = np.array([0, 0, 0, 0, 0, 0])
        self.velocityControllerPub.publish(vel_msg)
        self.first_movement = True
        self.get_logger().info("Movement stopped.")
    
    def shutdown_controller(self):
        self.get_logger().info(f"Stopping {self.get_name()}...")
        
        self.stop_movement()
        self.get_logger().info('Shutting down watchdog...')
        self.destroy_node()



def main(args=None):
    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)    
        watchdog = UR5eWatchdog()
        
        rclpy.spin(watchdog)        

    except KeyboardInterrupt:
        watchdog.stop_movement()
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:    
        print("Error in main: ", traceback.format_exc())   
        pass
    
    watchdog.shutdown_controller()
    
    rclpy.try_shutdown()
    
    print("\n I will take care of yourself, but you do too \n")