import threading
import traceback
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import rclpy
from rclpy.signals import SignalHandlerOptions

from robot_control.new_controller import NewController
import robot_controller.breathing.breathing_src as breathing_src
import utilities.control_and_filters as control_and_filters_utils
import utilities.geometry as geometry_utils
import utilities.linear_algebra as linear_algebra
import ur5e_kinematic.ur5e_kinematics as ur5e_kinematics

class neo_mimic(NewController):
    
    def init_breather(self):

        self.breathe_dict = {}
        self.breathe_dict["control_rate"] = self.control_rate

        human_data_path = "/home/kovan/USTA1.1/ros_workspace/src/robot_controller/robot_controller/breathing/breathe_data.npy"
        f = np.load(human_data_path)               
        self.breathe_dict["f"] = f

        # FAST: period = 2, amplitude = 1.2
        # DEFAULT: period = 4, amplitude = 1.0
        
        self.breathe_period, amplitude = 5.0, 1.0
        freq = 1.0 / self.breathe_period  # Named as beta in the paper
        self.breathe_dict["freq"] = freq
        self.breathe_dict["amplitude"] = amplitude
                
        self.num_of_breathing_joints = 2  # Breathing with joint 1 and 2 (shoulder and elbow)

        self.breathe_controller = breathing_src.Breather(self.breathe_dict, self.joint_states_global["pos"])

    def __init__(self, name = "neo_mimic", do_breathing = True, do_gazing = True):
        super().__init__(name)

        self.use_helmet = True
        self.use_fake_target = False
        self.gaze_target_name = "helmet_face" if self.use_helmet else "exp/head"
        self.gaze_target_name = "gaze_target" if self.use_fake_target else self.gaze_target_name
                    
        self.num_of_breathing_joints = 2  # Breathing with joint 1 and 2 (shoulder and elbow)
        self.num_of_gazing_joints = 4 # All the rest

        self.prev_dist_to_target = 0
        self.gaze_filter = control_and_filters_utils.LinearFilter(alpha=0.05)
        
        self.do_breathing = do_breathing
        self.do_gazing = do_gazing
        
        self.delay = 0.0
        self.start_time = time.time()

    def init_log_buffers(self):        
        super().init_log_buffers()

        self.cx_arr = []
        self.cz_arr = []
        self.log_gaze_corrections = False
        
        self.breath_forwards = []
        self.log_breath_forwards = False
        
        self.human_gaze_arr = []
        self.robot_gaze_arr = []
        self.log_gaze_vectors = True


    def set_start_time(self):
        self.start_time = time.time()

    def set_delay(self, delay):
        self.delay = delay

    def get_target_tf(self):
        if self.delay > 0.0:
            timestamp = self.get_clock().now() - rclpy.duration.Duration(seconds=self.delay)
        else:
            timestamp = rclpy.time.Time()  # Use the current time if no delay is specified
        
        if self.log_gaze_vectors:
            try:
                target_in_world = self.tfBuffer.lookup_transform(self.world, self.gaze_target_name, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
                target_orientation = [target_in_world.transform.rotation.x, target_in_world.transform.rotation.y, target_in_world.transform.rotation.z, target_in_world.transform.rotation.w]
                target_orientation_matrix = linear_algebra.quaternion_to_rotation_matrix(target_orientation)
                target_in_world_gaze_vector = target_orientation_matrix @ np.array([1, 0, 0]).T
                self.human_gaze_arr.append(target_in_world_gaze_vector)
            except Exception as e:
                print("Error in logging human gaze vector: ", traceback.format_exc())
            
        return self.tfBuffer.lookup_transform(self.world, self.gaze_target_name, timestamp, timeout=rclpy.duration.Duration(seconds=1.0))
        

    def set_breathing_gazing(self):        
        try:
            self.breathing_task = np.zeros(3)
            self.target_in_base_position = np.zeros(3)
            self.min_dist_to_target = 1.5
            self.linear_slope = control_and_filters_utils.LinearScaler(1, 1.7, 0.2, 0.5)
            self.slope_smoother = control_and_filters_utils.LinearFilter(alpha=0.1)
            self.forward_smoother = control_and_filters_utils.LinearFilter(alpha=0.3)
            self.target_position_smoother = control_and_filters_utils.VectorLinearFilter(alpha=0.2, dimension=3)
            
            self.gripper.close_async()
            self.go_to_home_pos(0.3)
            
            self.gaze_in_null_sphere = True               
            
            while True:
                try:
                    target_in_world = self.get_target_tf()
                    target_position = [target_in_world.transform.translation.x, target_in_world.transform.translation.y, target_in_world.transform.translation.z]
                    target_orientation = [target_in_world.transform.rotation.x, target_in_world.transform.rotation.y, target_in_world.transform.rotation.z, target_in_world.transform.rotation.w]
                    
                    target_in_world_matrix = linear_algebra.quaternion_to_homogeneous_matrix(target_orientation, target_position)
                    target_in_base = self.world_to_base_homogeneous @ target_in_world_matrix
                    target_in_base_position = np.array([target_in_base[0, 3], target_in_base[1, 3], target_in_base[2, 3]])
                    self.initial_distance_to_target = np.linalg.norm(target_in_base_position)
                    break
                except:
                    pass
                    
            if self.do_breathing:
                self.init_breather()
                self.breathing_start_time = time.time()
                self.breathe_controller.reset()
                
        except Exception as e:
            print("Error in setting breathing and gazing: ", traceback.format_exc())
            self.stop_movement()

    def human_gaze_in_null_sphere(self, target_position, target_orientation, robot_position, null_sphere_radius=0.2):
        origin = np.array(target_position)
        direction = linear_algebra.quaternion_to_rotation_matrix(target_orientation) @ np.array([1, 0, 0]).T
        direction = direction / np.linalg.norm(direction)
        
        center = np.array(robot_position)
        radius = null_sphere_radius
        
        oc = origin - center
        h = np.dot(direction, oc)
        c = np.dot(oc, oc) - radius**2
        
        discriminant_quarter = h**2 - c
    
        if discriminant_quarter < 0:
            return False
            
        sqrt_disc = np.sqrt(discriminant_quarter)
        return (-h - sqrt_disc) >= 0 or (-h + sqrt_disc) >= 0

    def get_gaze_velocities(self, target_position = [0.5, 0.25, 0.4], target_orientation = [0.7071068, 0, 0, 0.7071068], breathing_task = np.zeros(3), waist_ratio = 0.4, w6 = 1.0):
        
        waist = 0 # the base joint, the first one
        wrist1 = 0
        wrist2 = 0
        wrist3 = 0
        
        target_in_w3_pos = np.zeros(3)

        try:
            
            target_to_world_homo = np.eye(4)
            target_to_world_homo[0:3, 0:3] = R.from_quat([target_orientation[0],
                                                target_orientation[1],
                                                target_orientation[2],
                                                target_orientation[3]]).as_matrix()
            target_to_world_homo[0:3, 3] = np.array([target_position[0],
                                        target_position[1],
                                        target_position[2]])     
               
            ee_in_world = self.get_current_coordinate()
            ee_in_world = self.base_to_world_homogeneous @ np.concatenate((ee_in_world, [1]))
            
            pos = np.array([target_position[0] - ee_in_world[0], target_position[1] - ee_in_world[1], target_position[2] - ee_in_world[2]])
            dist_to_target = np.linalg.norm(pos)
            dist_to_target = dist_to_target * 0.1 + self.prev_dist_to_target * 0.9
            self.prev_dist_to_target = dist_to_target
            self.gaze_multiplier = (2.5)*dist_to_target + 1.7 # magic
            if self.do_breathing:
                freq = self.breathe_controller.freq
                self.gaze_multiplier /= freq*3  # Slow down gazing when breathing is fast
            
            instant_velocities = breathing_task
            delta_in_base = instant_velocities[:3]
            #delta_in_base *= 0 # Comment this line to enable gazing
                    
            gazing_velocities = np.zeros(self.num_of_gazing_joints)
            
            try:
                
                target_to_base = self.world_to_base_homogeneous @ target_to_world_homo
                
                #wrist1
                w1_to_base_rot = ur5e_kinematics.wrist1_to_base_rotation(self.joint_states_global["pos"])
                delta_in_w1 = w1_to_base_rot.T @ delta_in_base    
                
                base_to_w1 = ur5e_kinematics.base_to_wrist1_transformation(self.joint_states_global["pos"])            
                target_in_w1 = base_to_w1 @ target_to_base
                target_in_w1_pos = np.array([target_in_w1[0, 3]-delta_in_w1[0], target_in_w1[1, 3]-delta_in_w1[1], 0])                
                target_in_w1_norm = np.linalg.norm(target_in_w1_pos)
                if target_in_w1_norm < 1e-4:
                    target_in_w1_norm = 1e-4
                var1 = 0.0997 / target_in_w1_norm
                var2 = -target_in_w1_pos[1]/target_in_w1_norm
                if var1 >= -1 and var1 <= 1 and var2 >= -1 and var2 <= 1:
                    wrist1 = np.arccos(var1) - np.arccos(var2)
                else:
                    wrist1 = 0
                wrist1 = -wrist1 if target_in_w1_pos[0] > 0 else wrist1
                wrist1 = geometry_utils.angular_wrap(wrist1)
                                
                gazing_velocities[1] = wrist1 * self.gaze_multiplier                
                if abs(gazing_velocities[1]) > 2.5:
                    gazing_velocities[1] = np.sign(gazing_velocities[1]) * 2.5
                elif abs(gazing_velocities[1]) < 0.03:
                    gazing_velocities[1] = 0
                    
                #wrist2
                base_to_w2 = ur5e_kinematics.base_to_wrist2_transformation(self.joint_states_global["pos"])
                target_in_w2 = base_to_w2 @ target_to_base
                target_in_w2_pos = np.array([target_in_w2[0, 3], target_in_w2[1, 3], 0])                
                target_in_w2_norm = np.linalg.norm(target_in_w2_pos)
                if target_in_w2_norm < 1e-4:
                    target_in_w2_norm = 1e-4
                var1 = target_in_w2_pos[1]/target_in_w2_norm
                if var1 >= -1 and var1 <= 1:
                    wrist2 = np.arccos(var1)
                else:
                    wrist2 = 0
                wrist2 = -wrist2 if target_in_w2_pos[0] > 0 else wrist2
                wrist2 = geometry_utils.angular_wrap(wrist2)
                
                gazing_velocities[2] = wrist2 * self.gaze_multiplier * (1 - waist_ratio)
                if abs(gazing_velocities[2]) > 1:
                    gazing_velocities[2] = np.sign(gazing_velocities[2]) * 1
                elif abs(gazing_velocities[2]) < 0.03:
                    gazing_velocities[2] = 0
                #gazing_velocities[2] *= 0.5 # retards wrist2 movement, makes it less aggressive.
                    
                #waist
                base_to_shoulder = ur5e_kinematics.base_to_shoulder_transformation(self.joint_states_global["pos"])
                target_in_shoulder = base_to_shoulder @ target_to_base
                target_in_shoulder_pos = np.array([target_in_shoulder[0, 3], target_in_shoulder[1, 3], 0])                
                target_in_shoulder_norm = np.linalg.norm(target_in_shoulder_pos)
                var1 = target_in_shoulder_pos[0]/target_in_shoulder_norm
                if var1 >= -1 and var1 <= 1:
                    waist = np.arccos(var1)
                else:
                    waist = 0
                waist = waist if target_in_shoulder_pos[1] > 0 else -waist
                waist = geometry_utils.angular_wrap(waist)
                
                gazing_velocities[0] = waist * self.gaze_multiplier * waist_ratio
                if abs(gazing_velocities[0]) > 1:
                    gazing_velocities[0] = np.sign(gazing_velocities[0]) * 1
                elif abs(gazing_velocities[0]) < 0.03:
                    gazing_velocities[0] = 0
                
                #gazing_velocities[0] *= 0.4 # retards waist movement, makes it less aggressive. 
                
                #wrist3 (wrist3 is also eef)
                base_to_w3 = ur5e_kinematics.get_inverse_ee_transformation(self.joint_states_global["pos"])
                target_in_w3 = base_to_w3 @ target_to_base
                target_in_w3_y = np.array([target_in_w3[0, 1], target_in_w3[1, 1], 0])
                target_in_w3_y_norm = np.linalg.norm(target_in_w3_y)
                var1 = target_in_w3_y[1]/target_in_w3_y_norm
                if var1 >= -1 and var1 <= 1:
                    wrist3 = np.arccos(var1)
                else:
                    wrist3 = 0
                wrist3 = -wrist3 if target_in_w3_y[0] > 0 else wrist3
                wrist3 = geometry_utils.angular_wrap(wrist3)
                wrist3 = wrist3 * w6 # w6 is a weight for wrist3, can be tuned to adjust the aggressiveness of wrist3 movement
                
                target_in_w3_pos = np.array([target_in_w3[0, 3], target_in_w3[1, 3], target_in_w3[2, 3]])
                
                gazing_velocities[3] = wrist3 * 4# * self.gaze_multiplier
                if abs(gazing_velocities[3]) > 2.5:
                    gazing_velocities[3] = np.sign(gazing_velocities[3]) * 2.5
                elif abs(gazing_velocities[3]) < 0.03:
                    gazing_velocities[3] = 0
                #gazing_velocities[3] *= 0.5 # retards wrist3 movement, makes it less aggressive.                          
                
                gazing_velocities = self.gaze_filter.filter(gazing_velocities)
                #print("Gaze velocities: ", gazing_velocities, "wrist1: ", wrist1, "wrist2: ", wrist2)

            except Exception as e:
                gazing_velocities = np.zeros(self.num_of_gazing_joints)
                print("Error in getting target position: ", traceback.format_exc())
            finally:
                #gazing_velocities = self.gazing_velocities_filter.filter(gazing_velocities)                
                #print(f"Gazing velocities: {gazing_velocities}\nDistance to target: {dist_to_target}\nGaze multiplier: {self.gaze_multiplier}\n", end="\r")
                
                target_in_base_position = np.array([target_to_base[0, 3], target_to_base[1, 3], target_to_base[2, 3]])
                return gazing_velocities, target_in_base_position
            
        except Exception as e:
            print("Error in getting gaze velocities: ", traceback.format_exc())
            return np.zeros(self.num_of_gazing_joints), np.zeros(3)

    def get_breathe_velocities(self, forward_movement=0.0):
        
        try:
            breathing_velocities = np.zeros(self.num_of_breathing_joints)
            breathing_velocities = self.breathe_controller.step(self.joint_states_global["pos"], forward_movement)
            return breathing_velocities
        except Exception as e:
            print("Error in getting breathe velocities: ", traceback.format_exc())
            return np.zeros(self.num_of_breathing_joints)

    def change_control_rate(self, new_rate):
        if self.do_breathing:
            self.breathe_controller.control_rate = new_rate
        return super().change_control_rate(new_rate)

    def breathe_and_gaze_step(self):
        if self.log_frequencies:
                self.freqs.append(1.0 / (time.time() - self.freq_start_time))
                self.freq_start_time = time.time()
                
        breathe_velocities = np.zeros(self.num_of_breathing_joints)
        gaze_velocities = np.zeros(self.num_of_gazing_joints)
        
        target_in_world = self.get_target_tf()
        target_position = [target_in_world.transform.translation.x, target_in_world.transform.translation.y, target_in_world.transform.translation.z]
        target_position = self.target_position_smoother.filter(target_position)
        target_orientation = [target_in_world.transform.rotation.x, target_in_world.transform.rotation.y, target_in_world.transform.rotation.z, target_in_world.transform.rotation.w]
        
        target_in_world_matrix = linear_algebra.quaternion_to_homogeneous_matrix(target_orientation, target_position)
        target_in_base = self.world_to_base_homogeneous @ target_in_world_matrix
        target_in_base_position = np.array([target_in_base[0, 3], target_in_base[1, 3], target_in_base[2, 3]])
        target_in_base_orientation = linear_algebra.rotation_matrix_to_quaternion(target_in_base[0:3, 0:3])
        target_in_base_gaze_vector = np.array([target_in_base[0, 0], target_in_base[1, 0], target_in_base[2, 0]])
        
        if self.do_breathing:
            target_distance = np.linalg.norm(target_in_base_position)
            #breathing_gazing_controller.breathe_controller.set_frequency(slope_smoother.filter(linear_slope.scale(target_distance)) if target_distance != 0 else linear_slope.min_output)
            #print(f"Target distance: {target_distance:.2f} m, Breathe frequency: {self.breathe_controller.freq:.2f} Hz", end="\r")
            forward = target_distance - self.initial_distance_to_target
            forward = np.clip(forward, -0.6, 0.2)
            forward = self.forward_smoother.filter(forward)
            if self.log_breath_forwards:
                self.breath_forwards.append(forward)
            breathe_velocities, self.breathing_task, cx, cz = self.get_breathe_velocities(forward_movement = forward) #wedge_forward.scale((time.time() - start_time)%10))
            if self.log_gaze_corrections:
                self.cx_arr.append(cx)
                self.cz_arr.append(cz)

        if self.do_gazing:       
                current_ee_position = self.get_current_coordinate()
                current_ee_orientation = self.get_current_orientation()
                    
                target_to_current = current_ee_position - target_in_base_position
                gaze_angle = linear_algebra.angle_between_vectors(target_to_current, target_in_base_gaze_vector)
                
                #null_sphere_radius = 0.4
                #hysteresis_window = 0.1
                #if self.gaze_in_null_sphere:
                #    null_sphere_radius += hysteresis_window
                #
                #sphere_offset_dist = 0.3
                #sphere_amount = 3
                #sphere_amount = (sphere_amount//2) * 2 + 1
                #
                #for i in range(sphere_amount):
                #    center_offset = (i - sphere_amount//2) * sphere_offset_dist                    
                #    eef_x_vector = current_ee_orientation @ np.array([1, 0, 0]).T
                #    center_displacement = eef_x_vector * center_offset
                #    robot_position = current_ee_position + center_displacement
                #    
                #    self.gaze_in_null_sphere = self.human_gaze_in_null_sphere(target_position=target_in_base_position, target_orientation=target_in_base_orientation, robot_position=robot_position, null_sphere_radius=null_sphere_radius)
                #    if self.gaze_in_null_sphere:
                #        break
                #    
                #    #self.publish_ball(position = robot_position, radius=null_sphere_radius, color=(1.0, 0.0, 0.0, 0.5), marker_id=i+2)
                    
                #self.publish_arrow(start_pos = target_in_base_position, end_pos = target_in_base_position + target_in_base_gaze_vector * 2, id=12)
                eef_z_vector = current_ee_orientation @ np.array([0, 0, 1]).T
                if self.log_gaze_vectors:
                    eef_z_vector = self.base_to_world_homogeneous @ np.concatenate((eef_z_vector, [0]))
                    eef_z_vector = eef_z_vector[:3]
                    eef_z_vector[0] = -eef_z_vector[0]
                    self.robot_gaze_arr.append(eef_z_vector)
                #self.publish_arrow(start_pos = current_ee_position, end_pos = current_ee_position + eef_z_vector * 2, id=13)
                
                w6 = 1.0
                
                if True: #not self.gaze_in_null_sphere:
                    middle_point = (current_ee_position + target_in_base_position) / 2
                    dikme_magnitude = np.linalg.norm(current_ee_position - middle_point) * np.tan(gaze_angle)
                    
                    
                    dikme_direction = np.cross((np.cross(target_to_current, target_in_base_gaze_vector)), target_to_current)
                    dikme = (dikme_direction / np.linalg.norm(dikme_direction)) * dikme_magnitude
                    
                    target_position = middle_point + dikme
                    target_position = self.base_to_world_homogeneous @ np.concatenate((target_position, [1]))
                    target_position = target_position[:3]
                    
                    self.publish_ball(position = target_position, frame = self.world)
                    w6 = np.abs(((np.pi/2) - np.abs(gaze_angle))) / (np.pi/2)
                    w6 = np.clip(w6, 0.0, 1.0)

                    
                gaze_velocities, self.target_in_base_position = self.get_gaze_velocities(target_position=target_position,
                                                                                            target_orientation=target_orientation,
                                                                                            breathing_task=np.array([self.breathing_task[0], 0.0, self.breathing_task[1]]),
                                                                                            waist_ratio=0.5,
                                                                                            w6=w6
                                                                                            )
                    
        total_velocities = np.array([gaze_velocities[0], breathe_velocities[0], breathe_velocities[1], gaze_velocities[1] , gaze_velocities[2], gaze_velocities[3]])
        
        if self.log_joint_velocities:                              
            self.joint_velocities_sent.append(total_velocities)
            self.joint_velocities_real.append(self.joint_states_global["vels"])
        if self.log_end_effector_poses:
            self.end_effector_poses.append(current_ee_position)        
        
        return total_velocities


    def prepare_log_plots(self):
        super().prepare_log_plots()
        
        if hasattr(self, 'cx_arr') and len(self.cx_arr) > 0 and hasattr(self, 'cz_arr') and len(self.cz_arr) > 0:    
            plt.figure(self.figure_index)
            self.figure_index += 1
            plt.plot(self.cx_arr, label="Correction X")
            plt.plot(self.cz_arr, label="Correction Z")
            plt.title("Correction Factors Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Correction Value")
            plt.legend()
            #plt.show(block=False)
        
        if hasattr(self, 'breath_forwards') and len(self.breath_forwards) > 0:    
            plt.figure(self.figure_index)
            self.figure_index += 1
            plt.plot(self.breath_forwards, label="Breath Forward Movement")
            plt.title("Breath Forward Movement Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Forward Movement (m)")
            plt.legend()
            #plt.show(block=False)

        if hasattr(self, 'human_gaze_arr') and len(self.human_gaze_arr) > 0 and hasattr(self, 'robot_gaze_arr') and len(self.robot_gaze_arr) > 0:
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, num=self.figure_index)
            self.figure_index += 1
            
            human_gaze_arr = np.array(self.human_gaze_arr)
            robot_gaze_arr = np.array(self.robot_gaze_arr)
            
            # --- Subplot 1: X Components ---
            axs[0].plot(human_gaze_arr[:, 0], label="Human Gaze X")
            axs[0].plot(robot_gaze_arr[:, 0], label="Robot Gaze X", linestyle='--')
            axs[0].set_title("Gaze Vectors Over Time")
            axs[0].set_ylabel("X Component")
            axs[0].legend()
            
            # --- Subplot 2: Y Components ---
            axs[1].plot(human_gaze_arr[:, 1], label="Human Gaze Y")
            axs[1].plot(robot_gaze_arr[:, 1], label="Robot Gaze Y", linestyle='--')
            axs[1].set_ylabel("Y Component")
            axs[1].legend()
            
            # --- Subplot 3: Z Components ---
            axs[2].plot(human_gaze_arr[:, 2], label="Human Gaze Z")
            axs[2].plot(robot_gaze_arr[:, 2], label="Robot Gaze Z", linestyle='--')
            axs[2].set_xlabel("Time Step")
            axs[2].set_ylabel("Z Component")
            axs[2].legend()
            
            # Adjust spacing to prevent labels from overlapping
            plt.tight_layout() 
            
            #plt.show(block=False)

    def start_controller(self, speed=0.3):
        super().start_controller(speed)
        
        self.set_breathing_gazing()
        
        
def spin_thread(node, executor=None):
    
    try:
        rclpy.spin(node, executor=executor)
    except Exception as e:
        print("Error in spin_thread: ", traceback.format_exc())


def main(args=None):

    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)    
        neo_mimic_node = neo_mimic(do_breathing=True, do_gazing=True)
        
        controller_spin_thread = threading.Thread(target=spin_thread, args=(neo_mimic_node,))
        controller_spin_thread.start()
        
        neo_mimic_node.start_controller(speed=0.3)
        
        runtime = input("Enter desired runtime in seconds (default 30): ")
        try:
            runtime = float(runtime)
        except ValueError:
            runtime = 30.0
        target_frequency = input("Enter desired target frequency in Hz (default 1000): ")
        try:
            target_frequency = float(target_frequency)
            if target_frequency <= 0:
                print("Target frequency must be positive. Using default 1000 Hz.")
                target_frequency = 1000.0
        except ValueError:
            target_frequency = 1000.0
        print(f"Running for {runtime} seconds with target frequency of {target_frequency} Hz.")
        neo_mimic_node.change_control_rate(target_frequency)

        delay = 0.0
        try:
            delay = float(input("Enter desired delay in seconds (default 0.0): "))
        except ValueError:
            delay = 0.0
        neo_mimic_node.set_delay(delay)                        
                        
        start_time = time.time()  
        neo_mimic_node.set_start_time()      
        while rclpy.ok() and (time.time() - start_time) < runtime:
                                    
            total_velocities = neo_mimic_node.breathe_and_gaze_step()
            neo_mimic_node.publish_velocity_command(total_velocities)
                                
            neo_mimic_node.ros_rate.sleep()

        neo_mimic_node.stop_movement()
    except KeyboardInterrupt:
        neo_mimic_node.stop_movement()
        print("KeyboardInterrupt received, shutting down main loop...")
    except Exception as e:    
        print("Error in main: ", traceback.format_exc())   
        pass
    
    neo_mimic_node.shutdown_controller()
    
    rclpy.try_shutdown()
    controller_spin_thread.join()

        
    print("\n take care of yourself \n")