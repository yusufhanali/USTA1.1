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

class BreatheAndGazeController(NewController):
    
    def init_breather(self):

        self.breathe_dict = {}
        self.breathe_dict["control_rate"] = self.control_rate

        human_data_path = "/home/kovan/USTA1.1/ros_workspace/src/robot_controller/robot_controller/breathing/breathe_data.npy"
        f = np.load(human_data_path)               
        self.breathe_dict["f"] = f

        # FAST: period = 2, amplitude = 1.2
        # DEFAULT: period = 4, amplitude = 1.0
        
        self.breathe_period, amplitude = 4, 1.0
        freq = 1.0 / self.breathe_period  # Named as beta in the paper
        self.breathe_dict["freq"] = freq
        self.breathe_dict["amplitude"] = amplitude
                
        self.num_of_breathing_joints = 2  # Breathing with joint 1 and 2 (shoulder and elbow)

        self.breathe_controller = breathing_src.Breather(self.breathe_dict, self.joint_states_global["pos"])

    def __init__(self, name = "breathe_and_gazing_controller", do_breathing = True, do_gazing = True):
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
        
        self.home_pos = np.array([-0.8, -1.73, -1.8,  0.5,  1.52,  3.16])

    def init_log_buffers(self):        
        super().init_log_buffers()

        self.cx_arr = []
        self.cz_arr = []
        self.log_gaze_corrections = False
        
        self.breath_forwards = []
        self.log_breath_forwards = False


    def set_breathing_gazing(self, go_home=True):        
        try:
            self.breathing_task = np.zeros(3)
            self.target_in_base_position = np.zeros(3)
            self.min_dist_to_target = 1.5
            self.linear_slope = control_and_filters_utils.LinearScaler(1, 1.7, 0.2, 0.5)
            self.slope_smoother = control_and_filters_utils.LinearFilter(alpha=0.1)
            self.forward_smoother = control_and_filters_utils.LinearFilter(alpha=0.3)
            self.target_position_smoother = control_and_filters_utils.VectorLinearFilter(alpha=0.2, dimension=3)
            
            self.gripper.close_async()
            
            if go_home:
                self.go_to_home_pos(0.3)                    
                    
            if self.do_breathing:
                self.init_breather()
                self.breathing_start_time = time.time()
                self.breathe_controller.reset()
                
        except Exception as e:
            print("Error in setting breathing and gazing: ", traceback.format_exc())
            self.stop_movement()

    def get_gaze_velocities(self, target_position = [0.5, 0.25, 0.4], target_orientation = [0.7071068, 0, 0, 0.7071068], breathing_task = np.zeros(3), waist_ratio = 0.4):
        
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
        
        if self.do_breathing:
            target_distance = np.linalg.norm(self.target_in_base_position)
            #breathing_gazing_controller.breathe_controller.set_frequency(slope_smoother.filter(linear_slope.scale(target_distance)) if target_distance != 0 else linear_slope.min_output)
            #print(f"Target distance: {target_distance:.2f} m, Breathe frequency: {self.breathe_controller.freq:.2f} Hz", end="\r")
            forward = 0.0 if target_distance == 0 or target_distance > self.min_dist_to_target else target_distance-self.min_dist_to_target
            forward = np.clip(forward, -0.7, 0.7)
            forward = self.forward_smoother.filter(forward)
            if self.log_breath_forwards:
                self.breath_forwards.append(forward)
            breathe_velocities, self.breathing_task, cx, cz = self.get_breathe_velocities(forward_movement = forward) #wedge_forward.scale((time.time() - start_time)%10))
            if self.log_gaze_corrections:
                self.cx_arr.append(cx)
                self.cz_arr.append(cz)

        if self.do_gazing:
                target_in_world = self.tfBuffer.lookup_transform(self.world, self.gaze_target_name, rclpy.time.Time())
                target_position = [target_in_world.transform.translation.x, target_in_world.transform.translation.y, target_in_world.transform.translation.z]
                target_position = self.target_position_smoother.filter(target_position)
                target_orientation = [target_in_world.transform.rotation.x, target_in_world.transform.rotation.y, target_in_world.transform.rotation.z, target_in_world.transform.rotation.w]
                gaze_velocities, self.target_in_base_position = self.get_gaze_velocities(target_position=target_position,
                                                                                                           target_orientation=target_orientation,
                                                                                                           breathing_task=np.array([self.breathing_task[0], 0.0, self.breathing_task[1]]),
                                                                                                           waist_ratio=0.5
                                                                                                           )
        
        total_velocities = np.array([gaze_velocities[0], breathe_velocities[0], breathe_velocities[1], gaze_velocities[1] , gaze_velocities[2], gaze_velocities[3]])
        
        if self.log_joint_velocities:                              
            self.joint_velocities_sent.append(total_velocities)
            self.joint_velocities_real.append(self.joint_states_global["vels"])
        if self.log_end_effector_poses:
            self.end_effector_poses.append(self.get_current_coordinate())        
        
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


    def start_controller(self, speed=0.3, go_home=True):
        super().start_controller(speed, go_home)
        
        self.set_breathing_gazing(go_home=go_home)
        
        
def spin_thread(node, executor=None):
    
    try:
        rclpy.spin(node, executor=executor)
    except Exception as e:
        print("Error in spin_thread: ", traceback.format_exc())


def main(args=None):

    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)    
        breathing_gazing_controller = BreatheAndGazeController(do_breathing=True, do_gazing=True)
        
        controller_spin_thread = threading.Thread(target=spin_thread, args=(breathing_gazing_controller,))
        controller_spin_thread.start()
        
        breathing_gazing_controller.start_controller(speed=0.3)
        
        runtime = input("Enter desired runtime in seconds (default 30): ")
        try:
            runtime = float(runtime)
        except ValueError:
            runtime = 30.0
        target_frequency = input("Enter desired target frequency in Hz (default 500): ")
        try:
            target_frequency = float(target_frequency)
            if target_frequency <= 0:
                print("Target frequency must be positive. Using default 500 Hz.")
                target_frequency = 500.0
        except ValueError:
            target_frequency = 500.0
        print(f"Running for {runtime} seconds with target frequency of {target_frequency} Hz.")
        breathing_gazing_controller.change_control_rate(target_frequency)
                        
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < runtime:
                                    
            total_velocities = breathing_gazing_controller.breathe_and_gaze_step()
            breathing_gazing_controller.publish_velocity_command(total_velocities)
                                
            breathing_gazing_controller.ros_rate.sleep()

        breathing_gazing_controller.stop_movement()
    except KeyboardInterrupt:
        breathing_gazing_controller.stop_movement()
        print("KeyboardInterrupt received, shutting down main loop...")
    except Exception as e:    
        print("Error in main: ", traceback.format_exc())   
        pass
    
    breathing_gazing_controller.shutdown_controller()
    
    rclpy.try_shutdown()
    controller_spin_thread.join()

        
    print("\n take care of yourself \n")