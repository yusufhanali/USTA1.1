#do it proper
import threading
import time
import traceback

import rclpy
from rclpy.signals import SignalHandlerOptions

from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float32MultiArray

from .exp_config import *
from robot_control.new_controller import spin_thread
from breathing_gazing.breathe_and_gazing import BreatheAndGazeController
import utilities.linear_algebra as la


class ExperimentController(BreatheAndGazeController):
    def init_clicked_point_stuff(self):
        self.clicked_point_subscriber = self.create_subscription(PointStamped, "/clicked_point_coordinates", self.clicked_point_callback, 10)
        
        self.clicked_event = threading.Event()
        self.clicked_event.clear()
        self.last_clicked_point_in_base = np.array([-1.0, -1.0, -1.0])
        
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
            
    def __init__(self, name = "experiment_controller"):
        super().__init__(name)
        
        self.camera_towards_board_quat = np.array(la.rotation_matrix_to_quaternion((la.rot_x_homogeneous_matrix(np.pi) @ la.rot_z_homogeneous_matrix(-np.pi/4))[:3, :3]))
        
        self.init_clicked_point_stuff()
        
        self.get_logger().info("ExperimentController initialized.")

    def clicked_point_callback(self, msg):
        clicked_point = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0]).T
        base_frame = msg.header.frame_id
        
        self.get_logger().info(f"Received clicked point in frame '{base_frame}': ({clicked_point[0]:.3f}, {clicked_point[1]:.3f}, {clicked_point[2]:.3f})")
        
        if base_frame == "overhead_hri_camera_link":
            to_base_matrix = self.hri_overhead_to_base_matrix
        else:
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
               
        self.publish_arrow(start_pos=target_position_in_world, end_pos=target_position_in_world + 0.5 * target_orientation_in_world[:, 0], frame="world", id=0, color=(1.0, 0.0, 0.0, 1.0))
        self.publish_arrow(start_pos=target_position_in_world, end_pos=target_position_in_world + 0.5 * target_orientation_in_world[:, 1], frame="world", id=1, color=(0.0, 1.0, 0.0, 1.0))
        self.publish_arrow(start_pos=target_position_in_world, end_pos=target_position_in_world + 0.5 * target_orientation_in_world[:, 2], frame="world", id=2, color=(0.0, 0.0, 1.0, 1.0))
        self.publish_ball(target_position_in_world, frame="world", marker_id=3, color=(1.0, 1.0, 0.0, 1.0))
        
        target_orientation_in_world = R.from_matrix(target_orientation_in_world).as_quat()  
    
        return target_position_in_world, target_orientation_in_world
    
    def gaze_at_object(self, object_position_in_base):
        gaze_velocities = np.ones(4)
            
        target_position_in_world, target_orientation_in_world = self.create_object_frame(object_position_in_base)
                
        set_joint_positions = np.array([0.0, -np.pi/2, -np.pi/2, 0.0, 0.0, np.pi])
        total_error = np.ones(4)
        while np.linalg.norm(total_error) > 0.02 and rclpy.ok():
            gaze_velocities, _, total_error = self.get_gaze_velocities(target_position=target_position_in_world,
                                                            target_orientation=target_orientation_in_world,
                                                            waist_ratio=0.35,
                                                            joint_limits=np.array([0.7, 1.0, 1.0, 2.5])
                                                            )
                        
            total_error = np.array(total_error)[:3]
                                    
            set_velocities = np.array([0.0,
                                        set_joint_positions[1] - self.joint_states_global["pos"][1],
                                        set_joint_positions[2] - self.joint_states_global["pos"][2],
                                        0.0,
                                        0.0,
                                        set_joint_positions[5] - self.joint_states_global["pos"][5]]) * 0.5
            
            total_velocities = np.array([gaze_velocities[0], set_velocities[1], set_velocities[2], gaze_velocities[1], gaze_velocities[2], set_velocities[5]])
            if total_error[0] > 0.2:
                total_velocities[3] = 0.0
                total_velocities[4] = 0.0
                        
            print(f"Total Velocities: {total_velocities}", end='\r')
                        
            self.publish_velocity_command(total_velocities)
            
            self.ros_rate.sleep()
            
        self.stop_movement()
        self.get_logger().info(f"Gaze completed. Final Total Error: {total_error}, Total Error Norm: {np.linalg.norm(total_error):.2f}")
        
        self.grasp_signal_publisher.publish(Bool(data=True))
        self.get_logger().info("Grasp signal published.")
        
    
    
    def get_placement_position(self):
        return np.array([0.0, 0.5, 0.1])  # Example placement position in base frame
      
    def object_to_pick_up(self):
        return np.array([0.5, 0.0, 0.1])  # Example object position in base frame
    
    def pick_up_object(self, object_pose=None):
        self.open_gripper()
        
        if object_pose is None:
            object_pose = self.object_to_pick_up()
            
        if len(object_pose) == 7:
            object_position = object_pose[:3]
            object_orientation = object_pose[3:]
        elif len(object_pose) == 3:
            object_position = object_pose
            object_orientation = self.camera_towards_board_quat
        else:
            self.get_logger().error(f"Invalid object_pose length: {len(object_pose)}. Expected 3 (xyz) or 7 (xyz + xyzw).")
            return
        
        current_position = self.get_current_coordinate()
        movement_direction = object_position - current_position
        movement_direction /= np.linalg.norm(movement_direction)
        
        self.go_to_pose_in_base_with_cubic_spline(desired_coordinate=object_position,
                                                  start_derivative=movement_direction * 0.1,  # Example start derivative
                                                  end_derivative=movement_direction * 0.1,  # Example end derivative
                                                  desired_orientation=object_orientation,
                                                  speed=0.1)  # xyzw quaternion
        
        self.close_gripper()
        self.get_logger().info("Picked up the object.")
    
    
def main(args=None):
    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)    
        experiment_controller = ExperimentController()
        
        controller_spin_thread = threading.Thread(target=spin_thread, args=(experiment_controller,))
        controller_spin_thread.start()
        
        print("Starting main.")
                         
        experiment_controller.start_controller(speed=0.15, go_home=True)
        
        while rclpy.ok():
            experiment_controller.clicked_event.wait()
            clicked_point = experiment_controller.last_clicked_point_in_base
            experiment_controller.clicked_event.clear()
            
            experiment_controller.gaze_at_object(clicked_point)
            
            experiment_controller.grasp_event.wait()
            experiment_controller.grasp_event.clear()
            
            experiment_controller.pick_up_object(experiment_controller.grasp_pose)
        
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
    
    print("\n take care of yourself \n")