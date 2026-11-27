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

from robot_control.new_controller import spin_thread
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

    def init_camera_parameters(self):
        self.camera_position = self.tfBuffer.lookup_transform("wrist_3_link", "wrist_yifan_camera_link", rclpy.time.Time()).transform.translation
        # lens is offset from the camera's geometrical center at the x axis by +17.5 mm
        self.camera_position.x -= 0.0175   

    def init_temp_stuff(self):
        pass
    
    def init_wrist1_height_stuff(self):
        self.wrist1_min_height = 0.15
        self.wrist1_caution_distance = 0.05
        
        self.wrist1_caution_scaler = control_and_filters.LinearScaler(min_input=self.wrist1_min_height, max_input=self.wrist1_min_height+self.wrist1_caution_distance, min_output=0.1, max_output=1.0)
    
    def init_singularity_stuff(self):
        self.sigma_caution_threshold = 0.10
        self.sigma_critical_threshold = 0.03
    
    def init_workspace_stuff(self):
        x_min, x_max = -1.0, 0.9
        y_min, y_max = -0.7, 0.7
        z_min, z_max = 0.0, 2.0
        self.boundary_margin = 0.1
        
        self.rotate_45 = R.from_euler('z', 45, degrees=True).as_matrix()
        self.bounds = [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        
    def __init__(self, name = "ur5e_watchdog"):
        super().__init__(name)
        
        # Sanity checks at the start of node (The phrases are very unprofessionally taken from my favorite anime, Serial Experiments Lain)
        self.get_logger().info('not a good day to be a dog,')
        
        # Initialize control rate
        self.control_rate = 500 # Hz
        self.ros_rate = self.create_rate(self.control_rate, self.get_clock())
        
        self.current_jacobian = np.eye(6)
        self.current_jacobian_inverse = np.eye(6)
        
        self.base = "base"
        self.eef = "wrist_3_link"
        self.world = "world"        
        
        self.num_of_total_joints = ur5e_kinematics.NUM_OF_JOINTS
        
        self.init_joint_states()        
        self.init_velocity_controller()        
        self.init_tf()
        self.init_camera_parameters()
        self.init_workspace_stuff()
        self.init_singularity_stuff()
        self.init_wrist1_height_stuff()
        self.init_temp_stuff()
        self.init_log_buffers()
        
        self.base_to_world_homogeneous = np.array([ # WAS THIS WRONG BEFORE??? THE SIGNS ARE OPPOSITE NOW, LOOK INTO IT
            [-0.7071068, 0.7071068, 0.0, 1.7513],
            [-0.7071068, -0.7071068, 0.0, 0.0955],
            [0.0, 0.0, 1.0, 0.7347],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        self.world_to_base_homogeneous = linalg_utils.reverse_homogeneous_matrix(self.base_to_world_homogeneous)
        
        self.prev_velocities = np.zeros(6)
        self.current_velocity_command = np.zeros(6)
        self.current_position = np.zeros(3)
                        
        self.constraint_functions = []  # List to hold registered constraint functions
        self.constraint_register()  # Call the method to register constraints
                                                                        
        # Sanity check at the end of node. If both of these are printed, then the node is probably working properly.
        self.get_logger().info('but again, when ever is')

    def init_log_buffers(self):
        self.joint_velocities_real = []
        self.joint_velocities_sent = []
        self.log_joint_velocities = False
        

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
        
        self.current_velocity_command = vels


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
        self.constraint_functions.append(self.wrist1_height_constraint)
        self.get_logger().info("Registered wrist1_height_constraint")
        self.constraint_functions.append(self.workspace_constraint)
        self.get_logger().info("Registered workspace_constraint")
        self.constraint_functions.append(self.joint_velocity_constraint)
        self.get_logger().info("Registered joint_velocity_constraint")
        self.constraint_functions.append(self.joint_limit_constraint)
        self.get_logger().info("Registered joint_limit_constraint")
        self.constraint_functions.append(self.singularity_constraint)
        self.get_logger().info("Registered singularity_constraint")
        self.constraint_functions.append(self.camera_collision_constraint)
        self.get_logger().info("Registered camera_collision_constraint")
        
    def wrist1_height_constraint(self, commanded_velocity):
        min_height = self.wrist1_min_height        
        caution_distance = self.wrist1_caution_distance
        
        # Get the current position of the wrist_1_link in the base frame
        wrist1_position = ur5e_kinematics.wrist1_to_base_transformation(self.joint_states_global["pos"]) @ np.array([0, 0, 0, 1]).T
        wrist1_height = wrist1_position[2]  # z-coordinate
        
        cartesian_velocity = self.current_jacobian @ commanded_velocity
        z_velocity = cartesian_velocity[2]
        
        projected_wrist1_height = wrist1_height + z_velocity * 0.005
        
        if projected_wrist1_height > min_height + caution_distance:
            return commanded_velocity  # No constraint needed
        
        if projected_wrist1_height < min_height and z_velocity < 0:
            self.get_logger().warn(f"Wrist1 height constraint activated: Projected wrist1 height ({projected_wrist1_height:.3f}) below minimum ({min_height}). Stopping downward motion.")
            cartesian_velocity[2] = 0  # Stop downward motion
            altered_velocity = self.current_jacobian_inverse @ cartesian_velocity
            return altered_velocity
            
        scale = self.wrist1_caution_scaler.scale(projected_wrist1_height)
        cartesian_velocity[2] *= scale  # Scale downward motion
        altered_velocity = self.current_jacobian_inverse @ cartesian_velocity
        self.get_logger().info(f"Wrist1 height constraint activated: Projected wrist1 height ({projected_wrist1_height:.3f}) within caution zone. Scaling downward motion by {scale:.2f}.")
        return altered_velocity
            
        
       
    def joint_velocity_constraint(self, commanded_velocity):
        altered_velocity = np.copy(commanded_velocity)
        
        for i, command in enumerate(commanded_velocity):
            if abs(command) > ur5e_kinematics.JOINT_VELOCITY_LIMITS[i]:
                self.get_logger().info(f"Joint velocity constraint activated: Joint {i} commanded velocity ({command}) exceeds limit ({ur5e_kinematics.JOINT_VELOCITY_LIMITS[i]}).")
                altered_velocity[i] = np.sign(command) * ur5e_kinematics.JOINT_VELOCITY_LIMITS[i]
                
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

    def workspace_constraint(self, commanded_velocity):
        rotated_position = self.rotate_45 @ self.current_position
                
        cartesian_velocity = self.current_jacobian @ commanded_velocity
        v_linear_base = cartesian_velocity[:3]
        rotated_velocity = self.rotate_45 @ v_linear_base
        
        altered = False
        
        for i in range(3):
            pos_i = rotated_position[i]
            vel_i = rotated_velocity[i]
            min_b, max_b = self.bounds[i]
            
            # Moving towards minimum boundary
            if vel_i < 0:
                if pos_i < min_b:
                    rotated_velocity[i] = 0.0
                    altered = True
                elif pos_i < min_b + self.boundary_margin:
                    scale = (pos_i - min_b) / self.boundary_margin
                    rotated_velocity[i] *= np.clip(scale, 0.0, 1.0)
                    altered = True
                    
            # Moving towards maximum boundary
            elif vel_i > 0:
                if pos_i > max_b:
                    rotated_velocity[i] = 0.0
                    altered = True
                elif pos_i > max_b - self.boundary_margin:
                    scale = (max_b - pos_i) / self.boundary_margin
                    rotated_velocity[i] *= np.clip(scale, 0.0, 1.0)
                    altered = True

        # 5. If altered, rotate back to base frame and solve inverse kinematics
        if altered:
            cartesian_velocity[:3] = self.rotate_45.T @ rotated_velocity
            altered_velocity = self.current_jacobian_inverse @ cartesian_velocity
            return altered_velocity
            
        return commanded_velocity

    def singularity_constraint(self, commanded_velocity):
        sigma_values = np.linalg.svd(self.current_jacobian, compute_uv=False)
        sigma_current = sigma_values[-1]
        
        # 1. Define speed limits (rad/s)
        limit_scale_max = 1.0
        limit_scale_min = 0.05
        
        # 2. Smoothly calculate the speed ceiling based on sigma
        if sigma_current >= self.sigma_caution_threshold:
            allowed_limit_scale = limit_scale_max
        elif sigma_current <= self.sigma_critical_threshold:
            allowed_limit_scale = limit_scale_min
        else:
            # Smooth interpolation between v_max_near and v_max_normal
            ratio = (sigma_current - self.sigma_critical_threshold) / (self.sigma_caution_threshold - self.sigma_critical_threshold)
            allowed_limit_scale = limit_scale_min + ratio * (limit_scale_max - limit_scale_min)
        # 3. Check if motion is escaping
        dt = 0.02
        q_next = self.get_joint_pos() + commanded_velocity * dt
        sigma_next = np.linalg.svd(ur5e_kinematics.get_jacobian(q_next.tolist()), compute_uv=False)[-1]
        
        if sigma_next <= sigma_current and sigma_current <= self.sigma_critical_threshold:
            # Moving deeper into critical limit -> absolute stop
            return np.zeros_like(commanded_velocity)
            
        # 4. DIRECTION-PRESERVING SPEED CLAMP:
        # If any joint exceeds allowed_speed, scale the ENTIRE vector down uniformly
        
        applied_scale = 1.0
        for i, speed in enumerate(commanded_velocity):
            if abs(speed) > ur5e_kinematics.JOINT_VELOCITY_LIMITS[i]*allowed_limit_scale:
                applied_scale = min(applied_scale, ur5e_kinematics.JOINT_VELOCITY_LIMITS[i]*allowed_limit_scale / abs(speed))

        return commanded_velocity * applied_scale

    def camera_collision_constraint(self, commanded_velocity):
        # Thresholds (meters)
        d_min = 0.015       # Safety boundary (1.5 cm)
        d_caution = 0.035   # Deceleration buffer (3.5 cm)
        
        # Link collision radii
        R_upper_arm = 0.065
        R_forearm = 0.055
        R_cam = 0.035
        
        current_q = self.joint_states_global["pos"]
        
        # Camera sample points on wrist_3_link
        cam_local_points = np.array([
            [self.camera_position.x - 0.035, self.camera_position.y, self.camera_position.z, 1.0],  # Left
            [self.camera_position.x,         self.camera_position.y, self.camera_position.z, 1.0],  # Center
            [self.camera_position.x + 0.035, self.camera_position.y, self.camera_position.z, 1.0],  # Right
        ])
        
        def calc_min_clearance(q_val):            
            tfs = ur5e_kinematics.get_link_transforms(q_val)
            P_shoulder = tfs[1][:3, 3]
            P_elbow = tfs[2][:3, 3]
            P_wrist1 = tfs[3][:3, 3]
            
            min_d = float('inf')
            for pt in cam_local_points:
                P_cam = (tfs[6] @ pt)[:3]

                d_forearm = geometry_utils.point_to_line_distance(P_cam, P_elbow, P_wrist1) - (R_forearm + R_cam)
                d_upper = geometry_utils.point_to_line_distance(P_cam, P_shoulder, P_elbow) - (R_upper_arm + R_cam)
                d_table = (P_cam[2] - 0.00) - R_cam  # Table at z = 0.00
                
                min_d = min(min_d, d_forearm, d_upper, d_table)
            return min_d

        def calc_clearance_gradient(q_val, d0, eps=1e-4):
            grad = np.zeros(6)
            for i in range(6):
                q_step = np.copy(q_val)
                q_step[i] += eps
                grad[i] = (calc_min_clearance(q_step) - d0) / eps
            return grad

        curr_dist = calc_min_clearance(current_q)

        # If well clear of the caution zone, pass command straight through
        if curr_dist >= d_min + d_caution:
            return commanded_velocity

        # Compute normal direction pointing AWAY from obstacle
        grad = calc_clearance_gradient(current_q, curr_dist)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            return commanded_velocity
        n = grad / grad_norm

        # Decompose commanded velocity into normal and tangential components
        v_n = np.dot(commanded_velocity, n)
        q_dot_tan = commanded_velocity - v_n * n  # Motion parallel to obstacle surface

        # -----------------------------------------------------------------
        # Modulate ONLY the normal component (continuous without step jumps)
        # -----------------------------------------------------------------
        if v_n > 0:
            # Commanded velocity is moving AWAY from the obstacle
            if curr_dist <= d_min:
                # If below margin, ensure retreat speed is at least the push-out speed
                v_push = np.clip(0.04 + 3.0 * (d_min - curr_dist), 0.04, 0.15)
                v_n_safe = max(v_n, v_push)
            else:
                v_n_safe = v_n
        else:
            # Commanded velocity is moving TOWARD the obstacle
            if curr_dist <= d_min:
                # Cancel inward velocity and apply smooth proportional push-out
                v_n_safe = np.clip(0.04 + 3.0 * (d_min - curr_dist), 0.04, 0.15)
            else:
                # Inside caution zone: smoothly attenuate inward component to 0 at d_min
                scale = (curr_dist - d_min) / d_caution
                # Quadratic scaling for gentle deceleration
                v_n_safe = (scale ** 2) * v_n

        # Recombine preserved tangential motion with safe normal motion
        altered_velocity = q_dot_tan + v_n_safe * n
        return altered_velocity


    def loop(self):
        self.get_logger().info("Watchdog loop started.")

        while rclpy.ok():
            self.current_jacobian = self.get_jacobian_matrix()
            self.current_jacobian_inverse = self.get_inverse_jacobian()
            self.current_position = self.get_current_coordinate()
                      
            self.constrain_and_publish_velocity_command(self.current_velocity_command.copy())

            self.ros_rate.sleep()


    def constrain_and_publish_velocity_command(self, vels):
        for constraint_func in self.constraint_functions:
            vels = constraint_func(vels)
            
        self.publish_velocity_command(vels)

    def publish_velocity_command(self, vels):
        if type(vels) is not np.ndarray:
            vels = np.array(vels)
        
        if self.log_joint_velocities:
            self.joint_velocities_real.append(self.joint_states_global["vels"])
            self.joint_velocities_sent.append(vels)
         
        vel_msg = Float64MultiArray()
        vel_msg.data = vels
        self.velocityControllerPub.publish(vel_msg)
        
    def stop_movement(self):
        vel_msg = Float64MultiArray()
        vel_msg.data = np.array([0, 0, 0, 0, 0, 0])
        self.velocityControllerPub.publish(vel_msg)
        self.get_logger().info("Movement stopped.")
   
   
    def prepare_log_plots(self):
        self.figure_index = 0
        
        if hasattr(self, 'joint_velocities_sent') and len(self.joint_velocities_sent) > 0:        
                    self.joint_velocities_sent = np.array(self.joint_velocities_sent)
                    self.joint_velocities_real = np.array(self.joint_velocities_real)
                    for i in range(self.joint_velocities_sent.shape[1]):
                        plt.figure(self.figure_index)
                        self.figure_index += 1
                        plt.subplot(2, 1, 1)
                        plt.plot(self.joint_velocities_sent[:, i], label=f"Sent Joint {i+1}")
                        plt.title("Sent Joint Velocities Over Time from Watchdog")
                        plt.xlabel("Time Step")
                        plt.ylabel("Velocity (rad/s)")
                        plt.legend()
                        plt.subplot(2, 1, 2)
                        plt.plot(self.joint_velocities_real[:, i], label=f"Real Joint {i+1}", color='orange')
                        plt.title("Real Joint Velocities Over Time from Watchdog")
                        plt.xlabel("Time Step")
                        plt.ylabel("Velocity (rad/s)")
                        plt.legend()
                        plt.tight_layout()
                        #plt.show(block=False)
   
    def show_log_plots(self):
        self.prepare_log_plots()
        
        if self.figure_index > 0:
            plt.show()

    
    def shutdown_controller(self):
        self.get_logger().info(f"Stopping {self.get_name()}...")
        
        self.stop_movement()
        
        self.show_log_plots()
        
        self.get_logger().info('Shutting down watchdog...')
        self.destroy_node()


def main(args=None):
    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)    
        watchdog = UR5eWatchdog()
        
        watchdog_spin_thread = threading.Thread(target=spin_thread, args=(watchdog,))
        watchdog_spin_thread.start()
        
        watchdog.loop()      

    except KeyboardInterrupt:
        watchdog.stop_movement()
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:    
        print("Error in main: ", traceback.format_exc())   
        pass
    
    watchdog.shutdown_controller()
    
    rclpy.try_shutdown()
    watchdog_spin_thread.join()
    
    print("\n I will take care of yourself, but you do too \n")