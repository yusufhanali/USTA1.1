import threading
import traceback
import time

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.signals import SignalHandlerOptions

from breathing_gazing.breathe_and_gazing import BreatheAndGazeController

from utilities.linear_algebra import transform_to_matrix, reverse_homogeneous_matrix, rotation_matrix_to_rotvec, quaternion_to_rotvec
from utilities.control_and_filters import LinearFilter

class HeadMimicController(BreatheAndGazeController):
    def __init__(self):
        super().__init__()
        
        self.prev_head_position = None
        self.prev_head_orientation = None
        self.prev_linear_velocity = np.zeros(3)
        self.prev_angular_velocity = np.zeros(3)
        self.prev_time = time.time()
        
        self.head_position_smoother = LinearFilter(alpha=0.01)
        self.head_orientation_smoother = LinearFilter(alpha=0.01)        
        self.head_linear_velocity_smoother = LinearFilter(alpha=0.01)
        self.head_angular_velocity_smoother = LinearFilter(alpha=0.01)
                
        self.init_head_pose()
                
        self.linear_velocity_limit = 3
        self.angular_velocity_limit = 3
    
    def init_head_pose(self):
        self.get_logger().info("Initializing head pose buffer...")
        while rclpy.ok() and self.prev_head_position is None:
            try:
                head = self.tfBuffer.lookup_transform(self.world, self.gaze_target_name, rclpy.time.Time())
                head_position = np.array([head.transform.translation.x, head.transform.translation.y, head.transform.translation.z])
                head_orientation = np.array([head.transform.rotation.x, head.transform.rotation.y, head.transform.rotation.z, head.transform.rotation.w])
                head_orientation = quaternion_to_rotvec(head_orientation)
                
                self.prev_head_position = head_position.copy()
                self.prev_head_orientation = head_orientation.copy()
                self.head_position_smoother.set_value(head_position.copy())
                self.head_orientation_smoother.set_value(head_orientation.copy())
            except Exception as e:
                self.get_logger().error(f"Error looking up head transform during initialization: {e}")
                self.init_tf()
                time.sleep(0.1)

        self.get_logger().info(f"Initial head position: {self.prev_head_position}, orientation (rotvec): {self.prev_head_orientation}")

    def init_log_buffers(self):
        super().init_log_buffers()
    
        self.head_linear_velocities = []
        self.head_angular_velocities = []
        self.log_head_velocities = False
        
        self.head_positions = []
        self.head_orientations = []
        self.raw_head_positions = []
        self.raw_head_orientations = []
        self.log_head_poses = False
        
    def get_real_head_velocities(self):
                
        current_time = time.time()
        time_delta = current_time - self.prev_time
        self.prev_time = current_time
        
        linear_velocity = self.prev_linear_velocity
        angular_velocity = self.prev_angular_velocity
        
        try:
            head = self.tfBuffer.lookup_transform(self.world, self.gaze_target_name, rclpy.time.Time())
            head_position = np.array([head.transform.translation.x, head.transform.translation.y, head.transform.translation.z])
            head_orientation = np.array([head.transform.rotation.x, head.transform.rotation.y, head.transform.rotation.z, head.transform.rotation.w])
            head_orientation = quaternion_to_rotvec(head_orientation)
            
            raw_head_position = head_position.copy()
            raw_head_orientation = head_orientation.copy()
            
            head_position = self.head_position_smoother.filter(head_position)
            head_orientation = self.head_orientation_smoother.filter(head_orientation)
        
            linear_displacement = head_position - self.prev_head_position
            angular_displacement = head_orientation - self.prev_head_orientation
                                    
            self.prev_head_position = head_position.copy()
            self.prev_head_orientation = head_orientation.copy()

            linear_velocity = linear_displacement / time_delta
            linear_velocity = self.head_linear_velocity_smoother.filter(linear_velocity)
            linear_velocity = self.world_to_base_homogeneous[:3, :3] @ linear_velocity
            angular_velocity = angular_displacement / time_delta
            angular_velocity = self.head_angular_velocity_smoother.filter(angular_velocity)
            angular_velocity = self.world_to_base_homogeneous[:3, :3] @ angular_velocity
                
        except Exception as e:
            self.get_logger().error(f"Error looking up head transform: {e}")    
        
        if self.log_head_velocities:
            self.head_linear_velocities.append(linear_velocity)
            self.head_angular_velocities.append(angular_velocity)
            
        if self.log_head_poses:
            self.head_positions.append(head_position)
            self.head_orientations.append(head_orientation)
            self.raw_head_positions.append(raw_head_position)
            self.raw_head_orientations.append(raw_head_orientation)
            
        self.prev_linear_velocity = linear_velocity
        self.prev_angular_velocity = angular_velocity
        
        return linear_velocity, angular_velocity
        
    def velocity_loop(self, runtime):
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < runtime:
                                    
            head_linear_velocity, head_angular_velocity = self.get_real_head_velocities()
            head_angular_velocity = np.zeros(3)
            
            combined_velocities = np.concatenate((head_linear_velocity, -head_angular_velocity))
            inv_jacobian = self.get_inverse_jacobian()
            joint_velocities = inv_jacobian @ combined_velocities
            
            self.publish_velocity_command(joint_velocities)
                                
            self.ros_rate.sleep()
    
    def prepare_log_plots(self):
        super().prepare_log_plots()
    
        if len(self.head_linear_velocities) > 0:
            plt.figure(self.figure_index)
            self.figure_index += 1
            self.head_linear_velocities = np.array(self.head_linear_velocities)
            plt.subplot(3, 1, 1)
            plt.plot(self.head_linear_velocities[:, 0], label='Head Linear Velocity X')
            plt.subplot(3, 1, 2)
            plt.plot(self.head_linear_velocities[:, 1], label='Head Linear Velocity Y')
            plt.subplot(3, 1, 3)
            plt.plot(self.head_linear_velocities[:, 2], label='Head Linear Velocity Z')
            plt.suptitle('Head Linear Velocities')
            plt.xlabel('Time Step')
            plt.ylabel('Velocity (m/s)')
            plt.legend()
        
            
        if len(self.head_angular_velocities) > 0:
            plt.figure(self.figure_index)
            self.figure_index += 1
            self.head_angular_velocities = np.array(self.head_angular_velocities)
            plt.subplot(3, 1, 1)
            plt.plot(self.head_angular_velocities[:, 0], label='Head Angular Velocity X')
            plt.subplot(3, 1, 2)
            plt.plot(self.head_angular_velocities[:, 1], label='Head Angular Velocity Y')
            plt.subplot(3, 1, 3)
            plt.plot(self.head_angular_velocities[:, 2], label='Head Angular Velocity Z')
            plt.suptitle('Head Angular Velocities')
            plt.xlabel('Time Step')
            plt.ylabel('Velocity (rad/s)')
            plt.legend()
            
            
        if len(self.head_positions) > 0:
            plt.figure(self.figure_index)
            self.figure_index += 1
            self.head_positions = np.array(self.head_positions)
            self.raw_head_positions = np.array(self.raw_head_positions)
            plt.subplot(3, 1, 1)
            plt.plot(self.head_positions[:, 0], label='Smoothed Head Position X')
            plt.plot(self.raw_head_positions[:, 0], label='Raw Head Position X', alpha=0.5)
            plt.subplot(3, 1, 2)
            plt.plot(self.head_positions[:, 1], label='Smoothed Head Position Y')
            plt.plot(self.raw_head_positions[:, 1], label='Raw Head Position Y', alpha=0.5)
            plt.subplot(3, 1, 3)
            plt.plot(self.head_positions[:, 2], label='Smoothed Head Position Z')
            plt.plot(self.raw_head_positions[:, 2], label='Raw Head Position Z', alpha=0.5)
            plt.suptitle('Smoothed Head Positions')
            plt.xlabel('Time Step')
            plt.ylabel('Position (m)')
            plt.legend()


        if len(self.head_orientations) > 0:
            plt.figure(self.figure_index)
            self.figure_index += 1
            self.head_orientations = np.array(self.head_orientations)
            self.raw_head_orientations = np.array(self.raw_head_orientations)
            plt.subplot(3, 1, 1)
            plt.plot(self.head_orientations[:, 0], label='Smoothed Head Orientation X')
            plt.plot(self.raw_head_orientations[:, 0], label='Raw Head Orientation X', alpha=0.5)
            plt.subplot(3, 1, 2)
            plt.plot(self.head_orientations[:, 1], label='Smoothed Head Orientation Y')
            plt.plot(self.raw_head_orientations[:, 1], label='Raw Head Orientation Y', alpha=0.5)
            plt.subplot(3, 1, 3)
            plt.plot(self.head_orientations[:, 2], label='Smoothed Head Orientation Z')
            plt.plot(self.raw_head_orientations[:, 2], label='Raw Head Orientation Z', alpha=0.5)
            plt.suptitle('Smoothed Head Orientations (Rotvec)')
            plt.xlabel('Time Step')
            plt.ylabel('Orientation (rad)')
            plt.legend()
            
def spin_thread(node, executor=None):
    
    try:
        rclpy.spin(node, executor=executor)
    except Exception as e:
        print("Error in spin_thread: ", traceback.format_exc())


def main(args=None):

    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)    
        head_mimic_controller = HeadMimicController()
        
        controller_spin_thread = threading.Thread(target=spin_thread, args=(head_mimic_controller,))
        controller_spin_thread.start()
        
        head_mimic_controller.start_controller(speed=0.3)
        
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
        head_mimic_controller.change_control_rate(target_frequency)
                        
        head_mimic_controller.velocity_loop(runtime)

        head_mimic_controller.stop_movement()
    except KeyboardInterrupt:
        head_mimic_controller.stop_movement()
        print("KeyboardInterrupt received, shutting down main loop...")
    except Exception as e:    
        print("Error in main: ", traceback.format_exc())   
        pass
    
    head_mimic_controller.get_logger().info('Stopping breathing and gazing controller...')
    head_mimic_controller.get_logger().info(f"Final position: {head_mimic_controller.get_current_coordinate()}")
    
    head_mimic_controller.stop_movement()
    head_mimic_controller.show_log_plots()
    head_mimic_controller.get_logger().info('Shutting down node...')
    head_mimic_controller.destroy_node()
    rclpy.try_shutdown()
    controller_spin_thread.join()

        
    print("\n take care of yourself \n")