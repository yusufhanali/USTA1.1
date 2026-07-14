#do it properly

import threading
import time
import traceback

import rclpy
from rclpy.signals import SignalHandlerOptions

from scipy.spatial.transform import Rotation as R

from .exp_config import *
from robot_control.new_controller import spin_thread
from breathing_gazing.breathe_and_gazing import BreatheAndGazeController
import utilities.linear_algebra as la


class ExperimentController(BreatheAndGazeController):
    def __init__(self, name = "experiment_controller"):
        super().__init__(name)
        
        #self.camera_towards_board_quat = np.array([0.9238795, 0.3826834, 0, 0])
        self.camera_towards_board_quat = np.array(la.rotation_matrix_to_quaternion((la.rot_x_homogeneous_matrix(np.pi) @ la.rot_z_homogeneous_matrix(-np.pi/4))[:3, :3]))

    
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
        experiment_controller.start_controller()
        
        experiment_controller.pick_up_object()

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