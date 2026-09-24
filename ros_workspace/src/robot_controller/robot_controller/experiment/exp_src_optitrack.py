#do it proper
import threading
import time
import traceback
import os
import numpy as np

import rclpy
from rclpy.signals import SignalHandlerOptions

from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float32MultiArray

from .exp_config import *
from robot_control.new_controller import spin_thread
from breathing_gazing.breathe_and_gazing import BreatheAndGazeController
import utilities.linear_algebra as la
import utilities.control_and_filters as cf


class ExperimentController(BreatheAndGazeController):
    def init_interaction_stuff(self):        
        self.handover_orientation = np.array([0.8001031, 0.3314136, 0.1913417, 0.4619398])
        self.handover_position = np.array([0.4, -0.6, 0.5])
    
    def __init__(self, name = "experiment_controller"):
        super().__init__(name)
        
        self.camera_towards_board_quat = np.array(la.rotation_matrix_to_quaternion((la.rot_x_homogeneous_matrix(np.pi) @ la.rot_z_homogeneous_matrix(-np.pi/4))[:3, :3]))
        
        self.init_interaction_stuff()
        
        self.get_logger().info("ExperimentController initialized.")
    
          
    def create_gaze_frame(self, position_in_base):
        """
        Creates a frame at the given position which has a X axis pointing directly towards the robot base, and a Z axis pointing in
        the positive direction of the robot base's Z axis(according to right hand rule) which is parallel to the ground(XY plane of robot base).
        Practical for specifying a frame to gaze at since these choices were made to mimic the helmet's frame while the wearer is looking at the robot base.

        Args:
            position_in_base (_type_): 3D position in the robot base frame where the frame should be created.

        Returns:
            tuple: A tuple containing the position and orientation of the created frame in the world frame.
        """
        frame_position_in_world = self.base_to_world_homogeneous @ np.array([position_in_base[0], position_in_base[1], position_in_base[2], 1.0]).T
        frame_position_in_world = frame_position_in_world[:3]
        
        frame_orientation_y = np.array([0.0, 0.0, 1.0])
        frame_orientation_x = -(position_in_base)/np.linalg.norm(position_in_base)
        frame_orientation_z = np.cross(frame_orientation_x, frame_orientation_y)
        frame_orientation_y = np.cross(frame_orientation_z, frame_orientation_x)
        frame_orientation_y /= np.linalg.norm(frame_orientation_y)
        frame_orientation_z /= np.linalg.norm(frame_orientation_z)
        
        frame_orientation = np.eye(3)
        frame_orientation[:, 0] = frame_orientation_x
        frame_orientation[:, 1] = frame_orientation_y
        frame_orientation[:, 2] = frame_orientation_z
        frame_orientation_in_world = self.base_to_world_homogeneous[:3, :3] @ frame_orientation
               
        #self.publish_arrow(start_pos=frame_position_in_world, end_pos=frame_position_in_world + 0.5 * frame_orientation_in_world[:, 0], frame="world", id=50, color=(1.0, 0.0, 0.0, 1.0))
        #self.publish_arrow(start_pos=frame_position_in_world, end_pos=frame_position_in_world + 0.5 * frame_orientation_in_world[:, 1], frame="world", id=51, color=(0.0, 1.0, 0.0, 1.0))
        #self.publish_arrow(start_pos=frame_position_in_world, end_pos=frame_position_in_world + 0.5 * frame_orientation_in_world[:, 2], frame="world", id=52, color=(0.0, 0.0, 1.0, 1.0))
        #self.publish_ball(frame_position_in_world, frame="world", marker_id=53, color=(1.0, 1.0, 0.0, 1.0))
        
        frame_orientation_in_world = R.from_matrix(frame_orientation_in_world).as_quat()  
    
        return frame_position_in_world, frame_orientation_in_world

    def create_downwards_frame(self, position_in_base):
        """
        Creates a frame at the given position which has a Z axis pointing downwards, and a Y axis pointing away 
        from the robot base but parallel to the ground(XY plane of robot base). Practical for specifying a pickup or dropoff frame since 
        the gripper would be pointing downwards(due to Z) and the camera would not be between the base and the gripper(due to Y).

        Args:
            position_in_base (_type_): 3D position in the robot base frame where the frame should be created.

        Returns:
            tuple: A tuple containing the position and orientation of the created frame in the world frame.
        """
        frame_position_in_world = self.base_to_world_homogeneous @ np.array([position_in_base[0], position_in_base[1], position_in_base[2], 1.0]).T
        frame_position_in_world = frame_position_in_world[:3]
               
        frame_orientation_z = np.array([0.0, 0.0, -1.0])
        frame_orientation_y = position_in_base/np.linalg.norm(position_in_base)
        frame_orientation_x = np.cross(frame_orientation_y, frame_orientation_z)
        frame_orientation_x /= np.linalg.norm(frame_orientation_x)
        frame_orientation_y = np.cross(frame_orientation_z, frame_orientation_x)
        frame_orientation_y /= np.linalg.norm(frame_orientation_y)
        
        frame_orientation = np.eye(3)
        frame_orientation[:, 0] = frame_orientation_x
        frame_orientation[:, 1] = frame_orientation_y
        frame_orientation[:, 2] = frame_orientation_z
        frame_orientation_in_world = self.base_to_world_homogeneous[:3, :3] @ frame_orientation
               
        #self.publish_arrow(start_pos=frame_position_in_world, end_pos=frame_position_in_world + 0.5 * frame_orientation_in_world[:, 0], frame="world", id=64, color=(1.0, 0.0, 0.0, 1.0))
        #self.publish_arrow(start_pos=frame_position_in_world, end_pos=frame_position_in_world + 0.5 * frame_orientation_in_world[:, 1], frame="world", id=65, color=(0.0, 1.0, 0.0, 1.0))
        #self.publish_arrow(start_pos=frame_position_in_world, end_pos=frame_position_in_world + 0.5 * frame_orientation_in_world[:, 2], frame="world", id=66, color=(0.0, 0.0, 1.0, 1.0))
        self.publish_ball(frame_position_in_world, frame="world", marker_id=67, color=(1.0, 1.0, 0.0, 1.0))
        
        frame_orientation_in_world = R.from_matrix(frame_orientation_in_world).as_quat()
        
        return frame_position_in_world, frame_orientation_in_world
    
    
    def gaze_at_point(self, point_in_base):
        gaze_velocities = np.ones(4)
            
        target_position_in_world, target_orientation_in_world = self.create_gaze_frame(point_in_base)
        
        set_joint_positions = np.array([0.0, -np.pi/2, -np.pi/2, -np.pi*3/8, 0.0, -np.pi])
        total_error = np.ones(4)
        gaze_takeover = False
        set_velocity_coefficient = 0.8
        while np.linalg.norm(total_error) > 0.01 and rclpy.ok():
            gaze_velocities, _, total_error = self.get_gaze_velocities(target_position=target_position_in_world,
                                                            target_orientation=target_orientation_in_world,
                                                            waist_ratio=0.35,
                                                            joint_limits=np.array([0.7, 1.0, 1.0, 2.5])
                                                            )
                        
            total_error = np.array(total_error)[:3]
                                    
            total_velocities = np.array([   gaze_velocities[0] / set_velocity_coefficient,
                                            set_joint_positions[1] - self.joint_states_global["pos"][1],
                                            set_joint_positions[2] - self.joint_states_global["pos"][2],
                                            set_joint_positions[3] - self.joint_states_global["pos"][3],
                                            0.0,
                                            set_joint_positions[5] - self.joint_states_global["pos"][5]]) * set_velocity_coefficient
                        
            if not gaze_takeover and total_error[0] <= 0.1 and abs(set_joint_positions[3] - self.joint_states_global["pos"][3]) <= 0.1:
                gaze_takeover = True
            
            if gaze_takeover:
                total_velocities[3] = gaze_velocities[1]
                total_velocities[4] = gaze_velocities[2]
                
            self.publish_velocity_command(total_velocities * 0.6)
                   
            self.ros_rate.sleep()
            
        self.stop_movement()
        self.get_logger().info(f"Gaze completed. Final Total Error: {total_error}, Total Error Norm: {np.linalg.norm(total_error):.2f}")
    
            
    def get_placement_position(self):
        return np.array([0.0, 0.5, 0.1])  # Example placement position in base frame
      
    def is_in_bin(self, position_in_world): #TODO: implement a proper check based on the bin's dimensions and position
        return False
      
    def object_to_pick_up(self):
        """
        Searches the TF tree for markers that are within the defined REACHABLE_DISTANCE and PICK_UP_HEIGHT range, 
        and returns the position of the first suitable object found in world frame.

        Returns:
            marker_position (np.array): The position of the object to pick up in world frame, or None if no suitable object is found.
        """
        frame_name_prefix = "marker_"
        
        try:
            all_frames = self.tfBuffer._getFrameStrings()
            marker_frames = [frame for frame in all_frames if frame.startswith(frame_name_prefix)]
            
            for marker_frame in marker_frames:
                if self.tfBuffer.can_transform("world", marker_frame, rclpy.time.Time()):
                    marker_pose = self.tfBuffer.lookup_transform("world", marker_frame, rclpy.time.Time())
                    marker_position = np.array([marker_pose.transform.translation.x,
                                                marker_pose.transform.translation.y,
                                                marker_pose.transform.translation.z])
                    
                    marker_height = marker_position[2]
                    distance_to_base = np.linalg.norm(self.base_position_in_world[:2] - marker_position[:2])
                    self.get_logger().info(f"Checking marker: {marker_frame}, Position: {marker_position}, Height: {marker_height:.2f}, Distance to Base: {distance_to_base:.2f}")
                    if PICK_UP_MIN_HEIGHT <= marker_height <= PICK_UP_MAX_HEIGHT and distance_to_base <= MAX_REACHABLE_DISTANCE and not self.is_in_bin(marker_position):
                        self.get_logger().info(f"Found object to pick up: {marker_frame} at position {marker_position}")
                        self.publish_ball(marker_position, frame="world", marker_id=99, color=(1.0, 0.0, 0.0, 1.0))
                        return marker_position
                        
        except Exception as e:
            self.get_logger().error(f"Error while fetching frames: {e}")
            return None
        
        self.get_logger().warn("No suitable object found to pick up.")
        return None
    
    
    def pick_up_object(self, object_pose=None, speed=None):
        """
        Picks up an object in the given pose at the given speed.

        Args:
            object_pose (np.array, optional): The position and orientation in the base frame. Defaults to None.
            speed (Float, optional): The speed at which the end effector would be moving. Defaults to None.
        """
        if speed is None:
            speed = self.speed
        
        self.open_gripper()
        
        if object_pose is None:
            object_pose = self.object_to_pick_up()
            
        if len(object_pose) == 7:
            object_position = object_pose[:3]
            object_orientation = object_pose[3:]
        elif len(object_pose) == 3:
            object_position = object_pose
            object_orientation = self.camera_towards_board_quat
        elif len(object_pose) == 16:
            object_position = object_pose[:3, 3]
            object_orientation_matrix = object_pose[:3, :3]
            object_orientation = R.from_matrix(object_orientation_matrix).as_quat()
        else:
            self.get_logger().error(f"Invalid object_pose length: {len(object_pose)}. Expected either:\n3 (xyz),\n7 (xyz + xyzw),\n16 (4x4 homogeneous matrix).")
            return
        
        #rotate around z by 180 degrees if necessary to prevent unnecessary wrist rotation
        current_orientation_matrix = self.get_current_orientation()
        object_orientation_matrix = R.from_quat(object_orientation).as_matrix()
        object_orientation_matrix_rotated =  object_orientation_matrix @ R.from_euler('z', np.pi).as_matrix()
        
        current_to_object_orientation = R.from_matrix(current_orientation_matrix.T @ object_orientation_matrix).as_rotvec()
        current_to_object_orientation_rotated = R.from_matrix(current_orientation_matrix.T @ object_orientation_matrix_rotated).as_rotvec()
        current_to_object_orientation_norm = np.linalg.norm(current_to_object_orientation)
        current_to_object_orientation_rotated_norm = np.linalg.norm(current_to_object_orientation_rotated)
        
        if current_to_object_orientation_rotated_norm < current_to_object_orientation_norm:
            object_orientation = R.from_matrix(object_orientation_matrix_rotated).as_quat()
        
        current_position = self.get_current_coordinate()
        movement_direction = (object_position - current_position) * [1.0, 1.0, 0.0]  # Only consider x and y for direction
        movement_direction /= np.linalg.norm(movement_direction)
        
        self.go_to_pose_in_base_with_cubic_spline(desired_coordinate=object_position,
                                                  start_derivative=movement_direction * 0.3,  # Example start derivative
                                                  end_derivative=[0.0, 0.0, -0.2],  # Example end derivative
                                                  desired_orientation=object_orientation,
                                                  speed=speed)  # xyzw quaternion
        
        self.close_gripper()
        self.get_logger().info("Picked up the object.")
    
    def place_object(self, recepticle_position=np.array([-0.3, 0.3, 0.2]), speed=None):
        if speed is None:
            speed = self.speed
        
        _, dropoff_orientation = self.create_downwards_frame(recepticle_position)        
        dropoff_orientation = R.from_matrix(self.world_to_base_3x3 @ R.from_quat(dropoff_orientation).as_matrix()).as_quat()
        
        self.go_to_pose_in_base_with_cubic_spline(desired_coordinate=recepticle_position,
                                                  start_derivative=[0.0, 0.0, 0.2],
                                                  end_derivative=[0.0, 0.0, -0.2],
                                                  desired_orientation=dropoff_orientation,
                                                  speed=speed)
        self.open_gripper()
        self.get_logger().info("Placed the object.")

    def hand_object_over(self, handover_position=None, handover_orientation=None, speed=None): #TODO: make dynamic
        if speed is None:
            speed = self.speed
        if handover_position is None:
            handover_position = self.handover_position
        if handover_orientation is None:
            _, handover_orientation = self.create_downwards_frame(handover_position)        
            handover_orientation = R.from_matrix(self.world_to_base_3x3 @ R.from_quat(handover_orientation).as_matrix()).as_quat()
        
        eef_position = self.get_current_coordinate()
        eef_position[2] = 0.0
        eef_position /= np.linalg.norm(eef_position)
        
        left_vector = np.array([1.0, 1.0, 0.0])
        up_sign = 1.0
        if np.dot(eef_position, left_vector) < 0:
            up_sign = -1.0
        
        up_vector = np.array([0.0, 0.0, 1.0 * up_sign])
        tangent_to_the_base_circle = np.cross(eef_position, up_vector)
        tangent_to_the_base_circle /= np.linalg.norm(tangent_to_the_base_circle)
        tangent_to_the_base_circle *= 3
        tangent_to_the_base_circle[2] = 2.0
                
        self.go_to_pose_in_base_with_cubic_spline(desired_coordinate=handover_position,
                                                  start_derivative=tangent_to_the_base_circle,
                                                  end_derivative=left_vector * -0.1 * up_sign,
                                                  desired_orientation=handover_orientation,
                                                  speed=speed)
        self.open_gripper()
        self.get_logger().info("Handed over the object.")
        
        self.close_gripper_async()


    def find_and_pick_up_object(self, speed=None):
        if speed is None:
            speed = self.speed
            
        object_position = self.object_to_pick_up()
        if object_position is None:
            self.get_logger().warn("Can't find and pick up object because no suitable object was found.")
            return
        
        object_position_in_base = (self.world_to_base_homogeneous @ np.array([object_position[0], object_position[1], object_position[2], 1.0]).T)[:3]
        pickup_orientation = self.create_downwards_frame(object_position_in_base)[1]
        pickup_orientation = R.from_matrix(self.world_to_base_3x3 @ R.from_quat(pickup_orientation).as_matrix()).as_quat()
        pickup_pose = np.concatenate((object_position_in_base, pickup_orientation))
        
        self.pick_up_object(object_pose=pickup_pose, speed=speed)
        
class apf_generator:
    def __init__(self, apf_source_name: str = "", apf_constant: float = 0.01, threshold_distance: float = 0.50, max_force: float = 0.1):
        self.apf_source_name = apf_source_name
        self.apf_constant = apf_constant
        self.threshold_distance = threshold_distance
        self.max_force = max_force

        self.apf_source_position_filter = cf.LinearFilter(alpha = 0.5)
        self.apf_target_position_filter = cf.LinearFilter(alpha = 0.5)
        
    def generate_apf_force(self, apf_source_position_in_base, apf_target_position_in_base): 
        apf_force = np.zeros(3)
               
        if apf_source_position_in_base is not None and apf_target_position_in_base is not None:
            apf_source_position_in_base = self.apf_source_position_filter.filter(apf_source_position_in_base)
            apf_target_position_in_base = self.apf_target_position_filter.filter(apf_target_position_in_base)
            
            distance = np.linalg.norm(apf_source_position_in_base - apf_target_position_in_base)
            
            if distance < self.threshold_distance:    
                apf_force_magnitude = self.apf_constant / (distance * distance)  # F = Z / r^2
                apf_force_magnitude = min(apf_force_magnitude, self.max_force)  # Limit the force magnitude
                apf_force_direction = (apf_target_position_in_base - apf_source_position_in_base) / distance  # unit vector
                apf_force = apf_force_magnitude * apf_force_direction  # F vector

        return apf_force
    
    def get_apf_force_from_controller(self, controller: ExperimentController):
        if controller.tfBuffer.can_transform("world", self.apf_source_name, rclpy.time.Time()):
            apf_source_pose = controller.tfBuffer.lookup_transform("world", self.apf_source_name, rclpy.time.Time())
            apf_source_position = np.array([apf_source_pose.transform.translation.x,
                                            apf_source_pose.transform.translation.y,
                                            apf_source_pose.transform.translation.z,
                                            1.0]).T
            apf_source_position_in_base = (controller.world_to_base_homogeneous @ apf_source_position)[:3]
            
            apf_target_position_in_base = controller.get_current_coordinate()
            
            return self.generate_apf_force(apf_source_position_in_base, apf_target_position_in_base)
        else:
            return np.zeros(3)
                
def observer(experiment_controller: ExperimentController = None, frequency: int = 30):    
    human_apf_generator = apf_generator(apf_source_name = "rigid_body_9", apf_constant = 0.005, threshold_distance = 0.40, max_force = 1.5)

    while rclpy.ok():
        loop_start_time = time.time()
        
        apf_force = human_apf_generator.get_apf_force_from_controller(experiment_controller) 
        
        apf_velocity_command = np.concatenate((apf_force, np.zeros(3)))  # Assuming no rotational component
        apf_joint_velocity_command = experiment_controller.get_inverse_jacobian() @ apf_velocity_command
            
        experiment_controller.joint_velocity_additive_modifier = apf_joint_velocity_command

        elapsed_time = time.time() - loop_start_time
        #print(f"Maximum allowed time: {1.0 / frequency * 1000:.2f} ms, Elapsed time in ms: {elapsed_time * 1000:.2f}", end="\r")
        sleep_time = max(0, (1.0 / frequency) - elapsed_time)
        time.sleep(sleep_time)
         
def machine(experiment_controller: ExperimentController = None):  
    state = 0  
    while rclpy.ok():
        if state == 1: # basically if in grasping mode
            #TODO: get object position
            experiment_controller.pick_up_object(experiment_controller.grasp_pose)
                
            place_or_handover = input("Do you want to place the object or hand it over? (place/handover): ").strip().lower()
            if place_or_handover == "handover":
                handover_position = np.array([-0.14, -0.65, 0.02])  # Example handover position in base frame          
                experiment_controller.hand_object_over(handover_position=handover_position)
            else:
                receptacle_position = np.array([-0.65, 0.09, 0.08])  # Example receptacle position in base frame
                experiment_controller.place_object(recepticle_position=receptacle_position)
                
            experiment_controller.set_breathing_gazing(go_home=True, go_home_speed=0.5)
            state = 0
            
        elif state == 0: # basically if in breathing and gazing mode           
            
            experiment_controller.find_and_pick_up_object()
            
            return
            
            breathe_and_gaze_velocities = experiment_controller.breathe_and_gaze_step()
            experiment_controller.publish_velocity_command(breathe_and_gaze_velocities)
            experiment_controller.ros_rate.sleep()
    
    
def main(args=None):
    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)    
        experiment_controller = ExperimentController()
        
        controller_spin_thread = threading.Thread(target=spin_thread, args=(experiment_controller,))
        controller_spin_thread.start()
        
        observer_thread = threading.Thread(target=observer, args=(experiment_controller, ))
        observer_thread.start()
        
        print("Starting main.")
                   
        #experiment_controller.start_controller(speed=0.2, go_home=True)
                         
        machine(experiment_controller)
        
        experiment_controller.stop_movement()
    except KeyboardInterrupt:
        experiment_controller.stop_movement()
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:    
        print("Error in main: ", traceback.format_exc())   
        pass
    
    experiment_controller.shutdown_controller()

    rclpy.try_shutdown()
    controller_spin_thread.join()
    observer_thread.join()
    
    os.system("cowsay ' take care of yourself '")