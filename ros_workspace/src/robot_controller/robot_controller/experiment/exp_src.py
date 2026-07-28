#do it proper
import threading
import time
import traceback

import rclpy
from rclpy.signals import SignalHandlerOptions

from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PointStamped

from .exp_config import *
from robot_control.new_controller import spin_thread
from breathing_gazing.breathe_and_gazing import BreatheAndGazeController
import utilities.linear_algebra as la


class ExperimentController(BreatheAndGazeController):
    def init_clicked_point_stuff(self):
        self.clicked_point_subscriber = self.create_subscription(PointStamped, "/clicked_point_coordinates", self.clicked_point_callback, 10)
        
        while True:
            try:
                self.hri_overhead_to_base_matrix = self.tfBuffer.lookup_transform(self.base, "overhead_hri_camera_link", rclpy.time.Time())
                self.hri_overhead_to_base_matrix = la.tf_transform_to_homogeneous_matrix(self.hri_overhead_to_base_matrix.transform)
                break
            except:
                pass
    def __init__(self, name = "experiment_controller"):
        super().__init__(name)
        
        self.camera_towards_board_quat = np.array(la.rotation_matrix_to_quaternion((la.rot_x_homogeneous_matrix(np.pi) @ la.rot_z_homogeneous_matrix(-np.pi/4))[:3, :3]))
        
        self.init_clicked_point_stuff()

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
        
        self.publish_ball(clicked_point_in_base, frame=self.base)
    
    def get_placement_position(self):
        return np.array([0.0, 0.5, 0.1])  # Example placement position in base frame
      
    def object_to_pick_up(self):
        return np.array([0.5, 0.0, 0.1])  # Example object position in base frame
    
    def pick_up_object(self):
        self.open_gripper()
        
        object_position = self.object_to_pick_up()
        
        self.go_to_pose_in_base_with_cubic_spline(desired_coordinate=object_position,
                                                  start_derivative=np.array([1, 0, 0]),  # Example start derivative
                                                  end_derivative=np.array([0, 0, -1]),  # Example end derivative
                                                  desired_orientation=self.camera_towards_board_quat)  # xyzw quaternion
        
        self.close_gripper()
        print("Picked up the object.")
    
    
def main(args=None):
    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)    
        experiment_controller = ExperimentController()
        
        controller_spin_thread = threading.Thread(target=spin_thread, args=(experiment_controller,))
        controller_spin_thread.start()
        
        print("Starting main.")
                         
        experiment_controller.speed = 0.3
        
        experiment_controller.start_controller(go_home=False)
        
        while rclpy.ok():
            time.sleep(0.01)

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