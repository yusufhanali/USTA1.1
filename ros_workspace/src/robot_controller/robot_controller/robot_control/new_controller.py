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

class GripperController:
    def __init__(self, host, port):
        self.host = host # The robot's computer address
        self.port = port # The gripper's port number
        self.gripper = None
        self._init_gripper()

    def _init_gripper(self):
        print("Creating gripper...")
        self.gripper = robotiq_gripper()
        print("Connecting to gripper...")
        self.gripper.connect(self.host, self.port)
        print("Activating gripper...")
        self.gripper.activate(auto_calibrate=False)
        
        self.close()

    def open(self, speed=255, force=255):
        self.gripper.move_and_wait_for_pos(0, speed, force)
        # self.pub.publish(String("0.0"))

    def close(self, speed=255, force=255):
        self.gripper.move_and_wait_for_pos(255, speed, force)
        # self.pub.publish(String("0.7"))

    def close_async(self, speed=255, force=255):
        self.gripper.move(255, speed, force)
        
    def open_async(self, speed=255, force=255):
        self.gripper.move(0, speed, force)

    def move_gripper(self, pos, speed=255, force=255):
        pos = 0 if pos < 0 else (255 if pos > 255 else pos)
        self.gripper.move_and_wait_for_pos(pos, speed, force)
        
    def current_pos(self):        
        return self.gripper.get_current_position()

class CubicSplineController():    
    def rotation_vector(self, desired_orientation):
        current_orientation = self.robot.get_current_orientation()
        desired_orientation_matrix = linalg_utils.quaternion_to_rotation_matrix(desired_orientation) if len(desired_orientation) == 4 else desired_orientation
        current_to_desired = desired_orientation_matrix @ current_orientation.T
        rotvec = linalg_utils.rotation_matrix_to_rotvec(current_to_desired)
        return rotvec

    def __init__(self, robot, start_point, end_point, start_derivative, end_derivative, desired_orientation = None, speed = 0.05, control_rate = 500):
        self.robot = robot
        self.start_point = start_point
        self.end_point = end_point
        self.start_derivative = start_derivative
        self.end_derivative = end_derivative
        self.desired_orientation = desired_orientation
        self.speed = speed if speed < 0.8 else 0.8
        self.control_rate = control_rate  # Hz

        self.linear_scaler = control_and_filters.WedgeShapedScaler(min_input=0.0, max_input=1.0, entry_distance=0.15 + self.speed/4, exit_distance=0.2 + self.speed/4, peak_output=0.5 + self.speed/2)
        #self.rotational_scaler = control_and_filters.WedgeShapedScaler(min_input=0.0, max_input=1.0, entry_distance=0.3, exit_distance=0.3, peak_output=10/7)

        self.t = 0.0

        self.x_coordinate, \
        self.y_coordinate, \
        self.z_coordinate, \
        self.x_derivative, \
        self.y_derivative, \
        self.z_derivative = self.get_cubic_spline_equations()
        self.robot.publish_marker_array(self.trajectory_MarkerArray())
        #self.robot.publish_marker_array(self.derivatives_MarkerArray())
        
        self.trajectory_length = self.calculate_remaining_trajectory_length()
        
        self.initial_time_estimate = self.calculate_remaining_time(self.trajectory_length, self.speed)
        self.estimated_total_time = self.initial_time_estimate
        self.robot.get_logger().info(f"Estimated time to complete trajectory: {self.estimated_total_time}")
        
        if desired_orientation is not None:
            rotvec = self.rotation_vector(desired_orientation)
            self.robot.get_logger().info(f"Initial rotation vector to desired orientation: {rotvec}")
            desired_rot_speed = rotvec / self.estimated_total_time
            self.robot.get_logger().info(f"Initial desired rotational speed: {desired_rot_speed}")
        else:
            desired_rot_speed = np.zeros(3)
        self.desired_rot_speed = desired_rot_speed
        
        self.current_error = 0.0
        
        self.desired_log = [] #TODO
        self.sent_log = []
                        
    def get_cubic_spline_equations(self, start_point=None, end_point=None, start_derivative=None, end_derivative=None):
        if start_point is None:
            start_point = self.start_point
        if end_point is None:
            end_point = self.end_point
        if start_derivative is None:
            start_derivative = self.start_derivative
        if end_derivative is None:
            end_derivative = self.end_derivative
        
        #self.robot.get_logger().info(f"Calculating cubic spline equations, start point: {start_point} - end point: {end_point} - start derivative: {start_derivative} - end derivative: {end_derivative}")
        x_coeff, y_coeff, z_coeff = geometry_utils.cubic_spline(start_pos=start_point, start_derivative=start_derivative, end_pos=end_point, end_derivative=end_derivative)
    
        x_coordinates = lambda t: x_coeff[0]*t**3 + x_coeff[1]*t**2 + x_coeff[2]*t + x_coeff[3]
        y_coordinates = lambda t: y_coeff[0]*t**3 + y_coeff[1]*t**2 + y_coeff[2]*t + y_coeff[3]
        z_coordinates = lambda t: z_coeff[0]*t**3 + z_coeff[1]*t**2 + z_coeff[2]*t + z_coeff[3]
    
        x_derivatives = lambda t: 3*x_coeff[0]*t**2 + 2*x_coeff[1]*t + x_coeff[2]
        y_derivatives = lambda t: 3*y_coeff[0]*t**2 + 2*y_coeff[1]*t + y_coeff[2]
        z_derivatives = lambda t: 3*z_coeff[0]*t**2 + 2*z_coeff[1]*t + z_coeff[2]
    
        return x_coordinates, y_coordinates, z_coordinates, x_derivatives, y_derivatives, z_derivatives
    
    def calculate_remaining_trajectory_length(self, num_points=400, start_t=None, x_coordinate=None, y_coordinate=None, z_coordinate=None):
        if start_t is None:
            start_t = self.t
        if x_coordinate is None:
            x_coordinate = self.x_coordinate
        if y_coordinate is None:
            y_coordinate = self.y_coordinate
        if z_coordinate is None:
            z_coordinate = self.z_coordinate

        length = 0.0
        prev_point = np.array([x_coordinate(start_t), y_coordinate(start_t), z_coordinate(start_t)])

        for i in range(1, num_points):
            t = start_t + (1 - start_t) * (i / num_points)
            curr_point = np.array([x_coordinate(t), y_coordinate(t), z_coordinate(t)])
            length += np.linalg.norm(curr_point - prev_point)
            prev_point = curr_point

        #self.robot.get_logger().info(f"Calculated remaining trajectory length: {length}")

        return length

    def calculate_remaining_time(self, length, speed):        
        return length / speed if speed > 0 else 5

    def refresh_derivatives(self):        
        start_point = self.robot.get_current_coordinate()
        eef_velocity = self.robot.get_current_eef_velocity()
        start_derivative = np.array([eef_velocity[0], eef_velocity[1], eef_velocity[2]])
        
        here_to_end_equations = self.get_cubic_spline_equations(start_point=start_point, end_point=self.end_point, start_derivative=start_derivative, end_derivative=self.end_derivative)
        self.robot.publish_marker_array(self.trajectory_MarkerArray(t=0.0 ,x_coordinate=here_to_end_equations[0], y_coordinate=here_to_end_equations[1], z_coordinate=here_to_end_equations[2], color=(1.0, 0.0, 0.0, 1.0), id_offset=600, draw_fully=True))
                     
        self.t = 0.0
                                
        self.x_coordinate = here_to_end_equations[0]
        self.y_coordinate = here_to_end_equations[1]
        self.z_coordinate = here_to_end_equations[2]
        
        self.x_derivative = here_to_end_equations[3]
        self.y_derivative = here_to_end_equations[4]
        self.z_derivative = here_to_end_equations[5]
        
        if self.desired_orientation is not None:
            rotvec = self.rotation_vector(self.desired_orientation)
            desired_rot_speed = rotvec / self.estimated_total_time
            self.desired_rot_speed = desired_rot_speed
        
    def get_cubic_velocities(self, speed_multiplier = 1.0, max_error = 0.005):        
        if self.desired_orientation is not None:
            rotvec = self.rotation_vector(self.desired_orientation)
            rotation_error_norm = np.linalg.norm(rotvec)
            if rotation_error_norm < 0.005:
                self.desired_rot_speed = np.zeros(3)
                #self.robot.get_logger().info("Desired rotation reached.")
        
        safety_multiplier = 1.0
                       
        if self.current_error > max_error:
            self.robot.get_logger().info(f"Current Coordinate: {self.robot.get_current_coordinate()} - Desired Coordinate: {[self.x_coordinate(self.t), self.y_coordinate(self.t), self.z_coordinate(self.t)]} - Current Error: {self.current_error} - t: {self.t}")
            self.refresh_derivatives()
            safety_multiplier = 0.5
                    
        velocity_desired = np.array([self.x_derivative(self.t), self.y_derivative(self.t), self.z_derivative(self.t)])
        self.desired_log.append(np.concatenate([velocity_desired, np.array([np.linalg.norm(velocity_desired)])]))

        desired_linear_speed = (velocity_desired / np.linalg.norm(velocity_desired)) * self.speed
        
        desired_speed = np.concatenate((desired_linear_speed, self.desired_rot_speed))
        
        desired_speed[:3] *= self.linear_scaler.scale(self.t) + (1 - self.linear_scaler.peak_output)
        #desired_speed[3:] *= self.rotational_scaler.scale(self.t)
        desired_speed *= speed_multiplier * safety_multiplier
        self.sent_log.append(np.concatenate([desired_speed[:3], np.array([np.linalg.norm(desired_speed[:3]), np.linalg.norm(velocity_desired)])]))
        
        curr_speed = np.linalg.norm(desired_speed[:3])
        
        #self.robot.get_logger().info(f"norm_of_desired_linear_speed: {np.linalg.norm(desired_speed[:3])} - speed_multiplier: {speed_multiplier} - t: {self.t}")
                
        self.current_error = np.linalg.norm(self.robot.get_current_coordinate() - np.array([self.x_coordinate(self.t), self.y_coordinate(self.t), self.z_coordinate(self.t)])) 
        self.t += (curr_speed) / (np.linalg.norm(velocity_desired) * self.control_rate)
                
        return desired_speed
    
    def trajectory_MarkerArray(self, num_points=200, t=None, x_coordinate=None, y_coordinate=None, z_coordinate=None, color=(1.0, 1.0, 0.0, 1.0), id_offset=0, draw_fully=False):
        if x_coordinate is None:
            x_coordinate = self.x_coordinate
        if y_coordinate is None:
            y_coordinate = self.y_coordinate
        if z_coordinate is None:
            z_coordinate = self.z_coordinate
        if t is None:
            t = self.t
                
        marker_array = MarkerArray()
        num_points_max = num_points
        if not draw_fully:
            num_points = int((1-t) * num_points) if t < 1 else num_points
        counter = 0
        for i in range(num_points_max-num_points):
            dummy_marker = Marker()
            dummy_marker.action = Marker.DELETE
            dummy_marker.header.frame_id = self.robot.base
            dummy_marker.ns = "trajectory"
            dummy_marker.id = counter + id_offset
            counter += 1
            marker_array.markers.append(dummy_marker)
        for i in np.linspace(t, 1, num_points):
            point_marker = Marker()
            point_marker.header.frame_id = self.robot.base
            #point_marker.header.stamp = self.robot.get_clock().now().to_msg()
            point_marker.ns = "trajectory"
            point_marker.id = counter + id_offset
            counter += 1
            point_marker.type = Marker.SPHERE
            point_marker.action = Marker.ADD
            point_marker.scale.x = 0.01  # Diameter in x
            point_marker.scale.y = 0.01  # Diameter in y
            point_marker.scale.z = 0.01  # Diameter in z
            point_marker.color.r = color[0]
            point_marker.color.g = color[1]
            point_marker.color.b = color[2]
            point_marker.color.a = color[3]

            # Set the ball position
            point_marker.pose.position = Point(x=x_coordinate(i), y=y_coordinate(i), z=z_coordinate(i))
            point_marker.pose.orientation.w = 1.0  # No rotation

            marker_array.markers.append(point_marker)
        
        return marker_array

    def derivatives_MarkerArray(self, num_points=200):
        marker_array = MarkerArray()
        num_points_max = num_points
        num_points = int((1-self.t) * num_points) if self.t < 1 else num_points
        counter = 0
        for i in range(num_points_max-num_points):
            dummy_marker = Marker()
            dummy_marker.action = Marker.DELETE
            dummy_marker.header.frame_id = self.robot.base
            dummy_marker.ns = "derivatives"
            dummy_marker.id = counter
            counter += 1
            marker_array.markers.append(dummy_marker)
        for i in np.linspace(self.t, 1, num_points):
            arrow_marker = Marker()
            arrow_marker.header.frame_id = self.robot.base
            #arrow_marker.header.stamp = self.robot.get_clock().now().to_msg()
            arrow_marker.ns = "derivatives"
            arrow_marker.id = counter
            counter += 1
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.scale.x = 0.02  # Arrow shaft diameter
            arrow_marker.scale.y = 0.04  # Arrow head diameter
            arrow_marker.scale.z = 0.04  # Arrow head length
            arrow_marker.color.r = 0.0
            arrow_marker.color.g = 1.0
            arrow_marker.color.b = 1.0
            arrow_marker.color.a = 1.0

            arrow_marker.points = [
                Point(x=self.x_coordinate(i), y=self.y_coordinate(i), z=self.z_coordinate(i)),
                Point(x=self.x_coordinate(i) + self.x_derivative(i), y=self.y_coordinate(i) + self.y_derivative(i), z=self.z_coordinate(i) + self.z_derivative(i)),
            ]

            marker_array.markers.append(arrow_marker)
        
        return marker_array


class NewController(Node):

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
        self.velocityControllerTopic = "watchdog/velocity_commands" # All movements should be done w.r.t. "base", not "base_link" or "world"
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

    def init_gripper(self):
        # Initialize gripper
        self.robot_ip = "192.168.1.102"
        self.gripper_port = 63352
        self.gripper = GripperController(self.robot_ip, self.gripper_port)
        
        #self.gripper.open()
        self.get_logger().info('Gripper opened.')
        #self.gripper.close()
        self.get_logger().info('Gripper closed, gripper ready to go.')

    def __init__(self, name = "new_controller"):
        super().__init__(name)
        
        # Sanity checks at the start of node (The phrases are very unprofessionally taken from my favorite anime, Serial Experiments Lain)
        self.get_logger().info('present day,')
        
        # Initialize control rate
        self.control_rate = 500 # Hz
        self.ros_rate = self.create_rate(self.control_rate, self.get_clock())
        
        self.base = "base"
        self.eef = "wrist_3_link"
        self.world = "world"        
        
        self.num_of_total_joints = ur5e_kinematics.NUM_OF_JOINTS
        
        self.init_joint_states()        
        self.init_velocity_controller()        
        self.init_gripper()
        self.init_tf()
        
        self.base_to_world_homogeneous = np.array([ # WAS THIS WRONG BEFORE??? THE SIGNS ARE OPPOSITE NOW, LOOK INTO IT
            [-0.7071068, 0.7071068, 0.0, 1.7513],
            [-0.7071068, -0.7071068, 0.0, 0.0955],
            [0.0, 0.0, 1.0, 0.7347],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        self.world_to_base_homogeneous = linalg_utils.reverse_homogeneous_matrix(self.base_to_world_homogeneous)
        
        self.base_to_x_towards_board = linalg_utils.rot_z_matrix(np.pi/4)
        self.x_towards_board_to_base = linalg_utils.rot_z_matrix(-np.pi/4)
        
        self.first_movement = True        
        self.prev_velocities = np.zeros(6)
        
        self.home_pos = np.array([-0.8, -1.73, -2.1,  1.1,  1.52,  3.16])
        self.gripper_length = 0.17
        
        self.marker_array_publisher = self.create_publisher(MarkerArray, "vis_marker_array", 10)
        self.marker_publisher = self.create_publisher(Marker, "vis_marker", 10)
        
        self.speed = 0.05
                
        self.movement_stop_threshold = 0.007
        
        self.joint_velocity_additive_modifier = np.zeros(6)
                                                
        # Sanity check at the end of node. If both of these are printed, then the node is probably working properly.
        self.get_logger().info('present time.')  

    def init_log_buffers(self):
        self.joint_velocities_sent = []
        self.joint_velocities_real = []
        self.log_joint_velocities = False
        
        self.end_effector_poses = []
        self.log_end_effector_poses = False
        
        self.freq_start_time = time.time()
        self.freqs = []
        self.log_frequencies = False

    def change_control_rate(self, new_rate):
        self.control_rate = new_rate
        self.ros_rate = self.create_rate(self.control_rate, self.get_clock())
        self.get_logger().info(f"Control rate changed to: {self.control_rate} Hz")


    def get_current_coordinate(self):
        current_coordinate = np.zeros(3)        
        try:
            current_coordinate = ur5e_kinematics.get_ee_position(self.joint_states_global["pos"])
        except:
            self.get_logger().info(f"Error in getting current coordinate: {traceback.format_exc()}")
        return current_coordinate

    def get_current_eef_velocity(self):
        current_velocity = np.zeros(6)        
        try:
            current_velocity = self.get_jacobian_matrix() @ self.joint_states_global["vels"]
        except:
            self.get_logger().error(f"Error in getting current eef velocity: {traceback.format_exc()}")
        return current_velocity

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
        
    def filter_joint_velocities(self, joint_velocities):

        if self.first_movement:
            self.prev_velocities = joint_velocities
            self.first_movement = False
            return joint_velocities
                
        max_speed_diffs = np.array([0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0035])
        critical_speed_diffs = np.array([0.007, 0.007, 0.007, 0.007, 0.007, 0.007])
        
        filter_weight = 0.9
                
        for j in range(ur5e_kinematics.NUM_OF_JOINTS):
            change = joint_velocities[j] - self.prev_velocities[j]
            if abs(change) > max_speed_diffs[j]:
                joint_velocities[j] = self.prev_velocities[j]*filter_weight + joint_velocities[j]*(1-filter_weight)
                #joint_velocities[j] = self.prev_velocities[j] + np.sign(change) * max_speed_diffs[j]
                if abs(joint_velocities[j] - self.prev_velocities[j]) > critical_speed_diffs[j]:
                    joint_velocities[j] = self.prev_velocities[j] + np.sign(change) * critical_speed_diffs[j] 
        
        self.prev_velocities = joint_velocities
        
        return joint_velocities
    
    def publish_velocity_command(self, vels):
        if type(vels) is not np.ndarray:
            vels = np.array(vels)
         
        vels += self.joint_velocity_additive_modifier
         
        vels = self.filter_joint_velocities(vels)      

        vel_msg = Float64MultiArray()
        vel_msg.data = vels
        self.velocityControllerPub.publish(vel_msg)
        
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
    
        
    def open_gripper(self, speed=255, force=255):
        self.gripper.open(speed, force)
        self.get_logger().info("Gripper opened.")
        
    def close_gripper(self, speed=255, force=255):
        self.gripper.close(speed, force)
        self.get_logger().info("Gripper closed.")
        
    def move_gripper(self, pos, speed=255, force=255):
        self.gripper.move_gripper(pos, speed, force)
        self.get_logger().info(f"Gripper moved to position: {pos}.")

    def stop_movement(self):
        vel_msg = Float64MultiArray()
        vel_msg.data = np.array([0, 0, 0, 0, 0, 0])
        self.velocityControllerPub.publish(vel_msg)
        self.first_movement = True
        self.get_logger().info("Movement stopped.")
       
    def go_to_joint_pos(self, desired_joint_positions, speed=None):
        if speed is None:
            speed = self.speed

        trajectory = desired_joint_positions - self.joint_states_global["pos"]      
        norm = np.linalg.norm(desired_joint_positions - self.joint_states_global["pos"])        
        delta_t = norm/speed
                
        velocity_command = trajectory / delta_t
        self.get_logger().info(f"desired_joint_positions: {desired_joint_positions}")
        self.get_logger().info(f"velocity_command: {velocity_command}")
            
        self.publish_velocity_command(velocity_command)
        
        start_time = time.time()
        
        position_error = np.linalg.norm(desired_joint_positions - self.joint_states_global["pos"])
        while position_error > 0.0005 and time.time() - start_time < delta_t + 0.1:
            position_error = np.linalg.norm(desired_joint_positions - self.joint_states_global["pos"])
            
        self.stop_movement() 
        
        self.get_logger().info(f"Position reached, current position: {self.joint_states_global['pos']}")  
            
    def go_to_home_pos(self, speed=None):
        if speed is None:
            speed = self.speed
        self.go_to_joint_pos(self.home_pos, speed=speed)
    
    def rotation_vector(self, desired_orientation):
        current_orientation = self.get_current_orientation()
        desired_orientation_matrix = linalg_utils.quaternion_to_rotation_matrix(desired_orientation) if len(desired_orientation) == 4 else desired_orientation
        current_to_desired = desired_orientation_matrix @ current_orientation.T
        rotvec = linalg_utils.rotation_matrix_to_rotvec(current_to_desired)
        return rotvec
  
    def go_to_pose_in_base(self, desired_coordinate = None, desired_orientation=None, speed=None):                
        if desired_coordinate is None and desired_orientation is None:
            self.get_logger().info("No desired coordinate or orientation provided.")
            return
        
        if speed is None:
            speed = self.speed
        
        if desired_coordinate is not None:
            linvec = desired_coordinate - self.get_current_coordinate()
            dist = np.linalg.norm(linvec)
            est_time = dist / speed if speed > 0 else 5
            desired_lin_speed_in_base = (linvec / np.linalg.norm(linvec)) * speed
        else:
            self.get_logger().info("No desired coordinate, keeping current position.")
            est_time = 5
            desired_lin_speed_in_base = np.zeros(3)
        
        if desired_orientation is not None:
            rotvec = self.rotation_vector(desired_orientation)
            rot_speed = np.linalg.norm(rotvec)/est_time if desired_coordinate is not None else 0.03
            desired_rot_speed_in_base = (rotvec / np.linalg.norm(rotvec)) * rot_speed
            self.get_logger().info(f"Desired Rot Speed: {desired_rot_speed_in_base}")
        else:
            self.get_logger().info("No desired orientation, keeping current orientation.")
            desired_rot_speed_in_base = np.zeros(3)
            
        velocity_desired = np.concatenate((desired_lin_speed_in_base, desired_rot_speed_in_base))    
        
        max_error = 0.001                    
        position_error = 99999
        error_increase_count = 0
        while rclpy.ok() and position_error > max_error:
                            
            pinv_jacobian = self.get_inverse_jacobian()
            velocity_command = pinv_jacobian @ velocity_desired     
                                
            self.publish_velocity_command(velocity_command) 
            
            self.ros_rate.sleep()
            
            current_coordinate = self.get_current_coordinate()
            curr_error = np.linalg.norm(desired_coordinate - current_coordinate)
            if position_error < curr_error:
                error_increase_count += 1
                if error_increase_count > 10:
                    self.get_logger().info("Error increase count exceeded, stopping movement.")
                    break
            position_error = curr_error

        self.stop_movement()
        self.get_logger().info(f"Position reached, current position: {current_coordinate}")

    
    def create_cubic_spline_controller(self, start_point, end_point, start_derivative, end_derivative, desired_orientation=None, speed=None):
        if speed is None:
            speed = self.speed
        
        cubic_spline_controller = CubicSplineController(
            robot=self,
            start_point=start_point,
            end_point=end_point,
            start_derivative=start_derivative,
            end_derivative=end_derivative,
            desired_orientation=desired_orientation,
            speed=speed,
            control_rate=self.control_rate
        )
        
        return cubic_spline_controller

    def go_to_pose_in_base_with_cubic_spline(self, desired_coordinate=None, start_derivative=None, end_derivative=None, desired_orientation=None, speed=None, max_error=0.005):
        if speed is None:
            speed = self.speed

        cubic_spline_controller = self.create_cubic_spline_controller(
            start_point=self.get_current_coordinate(),
            end_point=desired_coordinate,
            start_derivative=start_derivative,
            end_derivative=end_derivative,
            desired_orientation=desired_orientation,
            speed=speed
        )
        
        current_error = np.linalg.norm(desired_coordinate - self.get_current_coordinate())           
        while rclpy.ok() and current_error > max_error and cubic_spline_controller.t < 1.0:            
            velocity_command = cubic_spline_controller.get_cubic_velocities()
            
            current_coordinate = self.get_current_coordinate()
            self.publish_arrow(current_coordinate, current_coordinate + velocity_command[:3])

            pinv_jacobian = self.get_inverse_jacobian()
            velocity_command = pinv_jacobian @ velocity_command
            
            self.publish_velocity_command(velocity_command) 
            
            current_error = np.linalg.norm(desired_coordinate - self.get_current_coordinate())
            self.ros_rate.sleep()
            
        self.stop_movement()
        self.get_logger().info(f"Current t: {cubic_spline_controller.t} Current position: {self.get_current_coordinate()} - Desired position: {desired_coordinate}")
        self.get_logger().info(f"Error: {np.linalg.norm(desired_coordinate - self.get_current_coordinate())}")
                
        plt.figure()
        plt.plot(np.array(cubic_spline_controller.desired_log))
        plt.title("Desired Velocities Over Time")
        plt.legend(["X Velocity", "Y Velocity", "Z Velocity", "Norm of Desired Linear Velocity"])
        plt.xlabel("Time Step")
        plt.ylabel("Desired Velocity (m/s)")
        plt.figure()
        plt.plot(np.array(cubic_spline_controller.sent_log))
        plt.title("Sent Velocities Over Time")
        plt.legend(["X Velocity", "Y Velocity", "Z Velocity", "Norm of Sent Linear Velocity", "Norm of Sampled Linear Velocity"])
        plt.xlabel("Time Step")
        plt.ylabel("Sent Velocity (m/s)")
        plt.show(block=True)
        
        
    def publish_marker(self, marker):
        if not isinstance(marker, Marker):
            self.get_logger().info("Error: marker is not of type Marker.")
            return
        self.marker_publisher.publish(marker)
    
    def publish_marker_array(self, marker_array):
        if not isinstance(marker_array, MarkerArray):
            self.get_logger().info("Error: marker_array is not of type MarkerArray.")
            return
        self.marker_array_publisher.publish(marker_array)
  
    def publish_arrow(self, start_pos = np.zeros(3), end_pos = np.zeros(3), frame = "base", id = 0, color = (0.0, 0.0, 1.0, 1.0)):
        # Publish an arrow from start_pos to curr_target
        arrow_marker = Marker()
        arrow_marker.header.frame_id = frame
        arrow_marker.header.stamp = self.get_clock().now().to_msg()
        arrow_marker.ns = "arrow"
        arrow_marker.id = id
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        arrow_marker.scale.x = 0.02  # Arrow shaft diameter
        arrow_marker.scale.y = 0.04  # Arrow head diameter
        arrow_marker.scale.z = 0.04  # Arrow head length
        arrow_marker.color.r = color[0]
        arrow_marker.color.g = color[1]
        arrow_marker.color.b = color[2]
        arrow_marker.color.a = color[3]

        # Set the arrow start and end points
        arrow_marker.points = [
            Point(x=start_pos[0], y=start_pos[1], z=start_pos[2]),
            Point(x=end_pos[0], y=end_pos[1], z=end_pos[2]),
        ]

        # Publish the arrow marker
        self.publish_marker(arrow_marker)

    def publish_ball(self, position = np.zeros(3), radius=0.02, marker_id=0, frame="base", color=(0.0, 1.0, 0.0, 1.0)):
        ball_marker = Marker()
        ball_marker.header.frame_id = frame
        ball_marker.header.stamp = self.get_clock().now().to_msg()
        ball_marker.ns = "ball"
        ball_marker.id = marker_id
        ball_marker.type = Marker.SPHERE
        ball_marker.action = Marker.ADD
        ball_marker.scale.x = radius * 2  # Diameter in x
        ball_marker.scale.y = radius * 2  # Diameter in y
        ball_marker.scale.z = radius * 2  # Diameter in z
        ball_marker.color.r = color[0]
        ball_marker.color.g = color[1]
        ball_marker.color.b = color[2]
        ball_marker.color.a = color[3]

        # Set the ball position
        ball_marker.pose.position = Point(x=position[0], y=position[1], z=position[2])
        ball_marker.pose.orientation.w = 1.0  # No rotation

        # Publish the ball marker
        self.publish_marker(ball_marker)
 
    def publish_marker_array_from_points(self, points, marker_type=Marker.SPHERE, radius=0.01, frame="base", color=(1.0, 0.0, 0.0, 1.0)):
        marker_array = MarkerArray()
        for i, point in enumerate(points):
            marker = Marker()
            marker.header.frame_id = frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "points"
            marker.id = i
            marker.type = marker_type
            marker.action = Marker.ADD
            marker.scale.x = radius * 2  # Diameter in x
            marker.scale.y = radius * 2  # Diameter in y
            marker.scale.z = radius * 2  # Diameter in z
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            # Set the ball position
            marker.pose.position = Point(x=point[0], y=point[1], z=point[2])
            marker.pose.orientation.w = 1.0  # No rotation

            marker_array.markers.append(marker)

        self.publish_marker_array(marker_array) 
 
    def prepare_log_plots(self):
        self.figure_index = 0

        if hasattr(self, 'freqs') and len(self.freqs) > 0:    
            self.freqs = np.array(self.freqs[10:])
            plt.figure(self.figure_index)
            self.figure_index += 1
            plt.plot(self.freqs)
            plt.title("Loop Frequency Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Frequency (Hz)")    
            # Add statistics text to the plot
            average_freq = np.mean(self.freqs)
            standard_deviation_freq = np.std(self.freqs)
            outliers_freq = self.freqs[np.abs(self.freqs - average_freq) > 2 * standard_deviation_freq]    
            stats_text = (
                f"Amount of frequency measurements: {len(self.freqs)}\n"
                f"Average loop frequency: {average_freq:.2f} Hz\n"
                f"Standard deviation: {standard_deviation_freq:.2f} Hz\n"
                f"Amount of outliers: {len(outliers_freq)}\n"
                f"Percentage of outliers: {len(outliers_freq) / len(self.freqs) * 100:.2f}%\n"
                f"Max frequency: {np.max(self.freqs):.2f} Hz\n"
                f"Min frequency: {np.min(self.freqs):.2f} Hz"
            )    
            plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            #plt.show(block=False)
            
        if hasattr(self, 'end_effector_poses') and len(self.end_effector_poses) > 0:
            self.end_effector_poses = np.array(self.end_effector_poses)
            positions = self.end_effector_poses
            #positions = np.array([pose[0] for pose in self.end_effector_poses])
            #orientations = np.array([pose[1] for pose in self.end_effector_poses])
            
            plt.figure(self.figure_index)
            self.figure_index += 1
            plt.plot(positions[:, 0], label="X")        
            plt.plot(positions[:, 1], label="Y")
            plt.plot(positions[:, 2], label="Z")
            plt.title("End Effector Position Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Position (m)")
            plt.legend()
            stats_text = (
                f"Amount of position measurements: {len(positions)}\n"
                f"Max X position: {np.max(positions[:, 0]):.4f} m\n"
                f"Min X position: {np.min(positions[:, 0]):.4f} m\n"
                f"Max Y position: {np.max(positions[:, 1]):.4f} m\n"
                f"Min Y position: {np.min(positions[:, 1]):.4f} m\n"
                f"Max Z position: {np.max(positions[:, 2]):.4f} m\n"
                f"Min Z position: {np.min(positions[:, 2]):.4f} m"
            )
            plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            #plt.show(block=False)

        if hasattr(self, 'joint_velocities_sent') and len(self.joint_velocities_sent) > 0:        
            self.joint_velocities_sent = np.array(self.joint_velocities_sent)
            self.joint_velocities_real = np.array(self.joint_velocities_real)
            for i in range(self.joint_velocities_sent.shape[1]):
                plt.figure(self.figure_index)
                self.figure_index += 1
                plt.subplot(2, 1, 1)
                plt.plot(self.joint_velocities_sent[:, i], label=f"Sent Joint {i+1}")
                plt.title("Sent Joint Velocities Over Time")
                plt.xlabel("Time Step")
                plt.ylabel("Velocity (rad/s)")
                plt.legend()
                plt.subplot(2, 1, 2)
                plt.plot(self.joint_velocities_real[:, i], label=f"Real Joint {i+1}", color='orange')
                plt.title("Real Joint Velocities Over Time")
                plt.xlabel("Time Step")
                plt.ylabel("Velocity (rad/s)")
                plt.legend()
                plt.tight_layout()
                #plt.show(block=False)

    def show_log_plots(self):
        self.prepare_log_plots()
        
        if self.figure_index > 0:
            plt.show()
        
    def start_controller(self, speed=0.3, go_home=True):
        self.speed = speed
        self.init_log_buffers()
        
        self.get_logger().info(f"Controller started with speed: {self.speed}")
        
        if go_home:
            self.get_logger().info("Going to home position...")
            self.go_to_home_pos(speed=self.speed)
            self.get_logger().info("Home position reached.")
    
    
    def shutdown_controller(self):
        self.get_logger().info(f"Stopping {self.get_name()} controller...")
        self.get_logger().info(f"Final position: {self.get_current_coordinate()}")
        
        self.stop_movement()
        self.show_log_plots()
        self.get_logger().info('Shutting down node...')
        self.destroy_node()


def spin_thread(node, executor=None):
    
    try:
        rclpy.spin(node, executor=executor)
    except Exception as e:
        print("Error in spin_thread: ", traceback.format_exc())


def main(args=None):

    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)    
        new_controller = NewController()
        
        controller_spin_thread = threading.Thread(target=spin_thread, args=(new_controller,))
        controller_spin_thread.start()
        
        new_controller.start_controller(speed=0.3)
        
        speed_input = float(input("Enter speed for cubic spline movement (e.g., 0.05): "))
        
        new_controller.go_to_pose_in_base_with_cubic_spline(
            desired_coordinate = np.array([0.713, -0.155, 0.24]),
            start_derivative = np.array([1.0, -1.0, 1.0]),
            end_derivative = np.array([0.0, 0.0, -1.0]),
            desired_orientation = np.array([0.92388, 0.38268, 0.0, 0.0]),
            speed=speed_input
        )

        new_controller.stop_movement()
    except KeyboardInterrupt:
        new_controller.stop_movement()
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:    
        print("Error in main: ", traceback.format_exc())   
        pass
    
    new_controller.shutdown_controller()
    
    rclpy.try_shutdown()
    controller_spin_thread.join()
    
    print("\n take care of yourself \n")