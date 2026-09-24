#do it proper
import threading
import time
import traceback
import os

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
    def init_clicked_point_stuff(self):
        self.clicked_point_subscriber = self.create_subscription(PointStamped, "/clicked_point_coordinates", self.clicked_point_callback, 10)
        
        self.clicked_event = threading.Event()
        self.clicked_event.clear()
        self.last_clicked_point_in_base = np.array([-1.0, -1.0, -1.0])
        self.last_clicked_point_in_sent_frame = np.array([-1.0, -1.0, -1.0])
        
        while not self.tfBuffer.can_transform(self.base, "overhead_hri_camera_link", rclpy.time.Time()):
            self.init_tf()
        while rclpy.ok():
            try:
                self.hri_overhead_to_base_matrix = self.tfBuffer.lookup_transform(self.base, "overhead_hri_camera_link", rclpy.time.Time())
                self.hri_overhead_to_base_matrix = la.tf_transform_to_homogeneous_matrix(self.hri_overhead_to_base_matrix.transform)
                break
            except:
                pass
            
        self.grasp_signal_publisher = self.create_publisher(Bool, "/grasp_signal", 10)
        
        self.grasp_event = threading.Event()
        self.grasp_event.clear()
        self.grasp_pose = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0])
        self.grasp_pose_subscriber = self.create_subscription(Float32MultiArray, "/grasp_pose", self.grasp_pose_callback, 10)
    
    def init_interaction_stuff(self):        
        self.handover_orientation = np.array([0.8001031, 0.3314136, 0.1913417, 0.4619398])
        self.handover_position = np.array([0.4, -0.6, 0.5])
    
    def __init__(self, name = "experiment_controller"):
        super().__init__(name)
        
        self.camera_towards_board_quat = np.array(la.rotation_matrix_to_quaternion((la.rot_x_homogeneous_matrix(np.pi) @ la.rot_z_homogeneous_matrix(-np.pi/4))[:3, :3]))
        
        self.init_clicked_point_stuff()
        self.init_interaction_stuff()
        
        self.get_logger().info("ExperimentController initialized.")

    def clicked_point_callback(self, msg):
        clicked_point = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0]).T
        self.last_clicked_point_in_sent_frame = clicked_point.T[:3]
        base_frame = msg.header.frame_id
        
        self.get_logger().info(f"Received clicked point in frame '{base_frame}': ({clicked_point[0]:.3f}, {clicked_point[1]:.3f}, {clicked_point[2]:.3f})")
        
        to_base_matrix = self.tfBuffer.lookup_transform(self.base, base_frame, rclpy.time.Time())
        to_base_matrix = la.tf_transform_to_homogeneous_matrix(to_base_matrix.transform)
            
        clicked_point_in_base = to_base_matrix @ clicked_point
        clicked_point_in_base = clicked_point_in_base[:3]
        
        self.last_clicked_point_in_base = clicked_point_in_base
        self.clicked_event.set()
        
        self.publish_ball(clicked_point_in_base, frame=self.base)
        
    def grasp_pose_callback(self, msg):
        if len(msg.data) != 7:
            self.get_logger().error(f"Received grasp pose with incorrect length: {len(msg.data)}. Expected 7 (xyz + xyzw).")
            return
        
        self.get_logger().info(f"Received grasp pose: {msg.data}")
        self.grasp_pose = np.array(msg.data)  
        self.grasp_event.set() 

        
    def create_object_frame(self, object_position_in_base):  
        target_position_in_world = self.base_to_world_homogeneous @ np.array([object_position_in_base[0], object_position_in_base[1], object_position_in_base[2], 1.0]).T
        target_position_in_world = target_position_in_world[:3]
        
        target_orientation_y = np.array([0.0, 0.0, 1.0])
        target_orientation_x = -(object_position_in_base)/np.linalg.norm(object_position_in_base)
        target_orientation_z = np.cross(target_orientation_x, target_orientation_y)
        target_orientation_y = np.cross(target_orientation_z, target_orientation_x)
        target_orientation_y /= np.linalg.norm(target_orientation_y)
        target_orientation_z /= np.linalg.norm(target_orientation_z)
        
        target_orientation = np.eye(3)
        target_orientation[:, 0] = target_orientation_x
        target_orientation[:, 1] = target_orientation_y
        target_orientation[:, 2] = target_orientation_z
        target_orientation_in_world = self.base_to_world_homogeneous[:3, :3] @ target_orientation
               
        #self.publish_arrow(start_pos=target_position_in_world, end_pos=target_position_in_world + 0.5 * target_orientation_in_world[:, 0], frame="world", id=0, color=(1.0, 0.0, 0.0, 1.0))
        #self.publish_arrow(start_pos=target_position_in_world, end_pos=target_position_in_world + 0.5 * target_orientation_in_world[:, 1], frame="world", id=1, color=(0.0, 1.0, 0.0, 1.0))
        #self.publish_arrow(start_pos=target_position_in_world, end_pos=target_position_in_world + 0.5 * target_orientation_in_world[:, 2], frame="world", id=2, color=(0.0, 0.0, 1.0, 1.0))
        self.publish_ball(target_position_in_world, frame="world", marker_id=3, color=(1.0, 1.0, 0.0, 1.0))
        
        target_orientation_in_world = R.from_matrix(target_orientation_in_world).as_quat()  
    
        return target_position_in_world, target_orientation_in_world

    def create_dropoff_frame(self, dropoff_position_in_base):  
        dropoff_position_in_world = self.base_to_world_homogeneous @ np.array([dropoff_position_in_base[0], dropoff_position_in_base[1], dropoff_position_in_base[2], 1.0]).T
        dropoff_position_in_world = dropoff_position_in_world[:3]
               
        dropoff_orientation_z = np.array([0.0, 0.0, -1.0])
        dropoff_orientation_y = dropoff_position_in_base/np.linalg.norm(dropoff_position_in_base)
        dropoff_orientation_x = np.cross(dropoff_orientation_y, dropoff_orientation_z)
        dropoff_orientation_x /= np.linalg.norm(dropoff_orientation_x)
        dropoff_orientation_y = np.cross(dropoff_orientation_z, dropoff_orientation_x)
        dropoff_orientation_y /= np.linalg.norm(dropoff_orientation_y)
        
        dropoff_orientation = np.eye(3)
        dropoff_orientation[:, 0] = dropoff_orientation_x
        dropoff_orientation[:, 1] = dropoff_orientation_y
        dropoff_orientation[:, 2] = dropoff_orientation_z
        dropoff_orientation_in_world = self.base_to_world_homogeneous[:3, :3] @ dropoff_orientation
               
        #self.publish_arrow(start_pos=dropoff_position_in_world, end_pos=dropoff_position_in_world + 0.5 * dropoff_orientation_in_world[:, 0], frame="world", id=4, color=(1.0, 0.0, 0.0, 1.0))
        #self.publish_arrow(start_pos=dropoff_position_in_world, end_pos=dropoff_position_in_world + 0.5 * dropoff_orientation_in_world[:, 1], frame="world", id=5, color=(0.0, 1.0, 0.0, 1.0))
        #self.publish_arrow(start_pos=dropoff_position_in_world, end_pos=dropoff_position_in_world + 0.5 * dropoff_orientation_in_world[:, 2], frame="world", id=6, color=(0.0, 0.0, 1.0, 1.0))
        
        dropoff_orientation_in_world = R.from_matrix(dropoff_orientation_in_world).as_quat()
        
        return dropoff_position_in_world, dropoff_orientation_in_world
    
    def gaze_at_object(self, object_position_in_base):
        gaze_velocities = np.ones(4)
            
        target_position_in_world, target_orientation_in_world = self.create_object_frame(object_position_in_base)
                
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
        
        self.grasp_signal_publisher.publish(Bool(data=True))
        self.get_logger().info("Grasp signal published.")
        
    
    def get_placement_position(self):
        return np.array([0.0, 0.5, 0.1])  # Example placement position in base frame
      
    def object_to_pick_up(self):
        return np.array([0.5, 0.0, 0.1])  # Example object position in base frame
    
    def pick_up_object(self, object_pose=None, speed=None):
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
        
        _, dropoff_orientation = self.create_dropoff_frame(recepticle_position)        
        dropoff_orientation = R.from_matrix(self.world_to_base_3x3 @ R.from_quat(dropoff_orientation).as_matrix()).as_quat()
        
        self.go_to_pose_in_base_with_cubic_spline(desired_coordinate=recepticle_position,
                                                  start_derivative=[0.0, 0.0, 0.2],
                                                  end_derivative=[0.0, 0.0, -0.2],
                                                  desired_orientation=dropoff_orientation,
                                                  speed=speed)
        self.open_gripper()
        self.get_logger().info("Placed the object.")

    def hand_object_over(self, handover_position=None, handover_orientation=None, speed=None):
        if speed is None:
            speed = self.speed
        if handover_position is None:
            handover_position = self.handover_position
        if handover_orientation is None:
            _, handover_orientation = self.create_dropoff_frame(handover_position)        
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
    while rclpy.ok():
        if experiment_controller.clicked_event.is_set():
            experiment_controller.clicked_event.wait()
            clicked_point = experiment_controller.last_clicked_point_in_base
            experiment_controller.clicked_event.clear()
            
            experiment_controller.gaze_at_object(clicked_point)
            
            experiment_controller.grasp_event.wait()
            experiment_controller.grasp_event.clear()
            
            if not experiment_controller.grasp_pose.any() or np.all(experiment_controller.grasp_pose == -1.0):
                experiment_controller.get_logger().error("No valid grasp pose found.")
            else:
                experiment_controller.pick_up_object(experiment_controller.grasp_pose)
                
                place_or_handover = input("Do you want to place the object or hand it over? (place/handover): ").strip().lower()
                if place_or_handover == "handover":
                    handover_position = np.array([-0.14, -0.65, 0.02])  # Example handover position in base frame          
                    experiment_controller.hand_object_over(handover_position=handover_position)
                else:
                    receptacle_position = np.array([-0.65, 0.09, 0.08])  # Example receptacle position in base frame
                    experiment_controller.place_object(recepticle_position=receptacle_position)
                    
                experiment_controller.set_breathing_gazing(go_home=True, go_home_speed=0.5)
        else: # basically if in breathing and gazing mode           
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
                   
        experiment_controller.start_controller(speed=0.2, go_home=True)  # Start the controller without going to home position
                         
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