#do it properly

from enum import Enum
import threading
import time
import traceback
from .exp_config import *
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.signals import SignalHandlerOptions
from breathing_gazing.breathe_and_gazing import BreatheAndGazeController
from utilities.geometry import three_point_cubic_spline
import ur5e_kinematic.ur5e_kinematics as ur5e_kinematics

class RobotState(Enum):
    START = 0
    PICKING = 1
    PLACING = 2
    BREATHING_GAZING = 3
    END = 4

already_picked_up = []

def is_in_pick_up_area(point):
    if point[2] > PICK_UP_MIN_HEIGHT and point[2] < PICK_UP_MAX_HEIGHT:
        if point[0] > PICK_UP_AREA[0][0] and point[0] < PICK_UP_AREA[1][0]:
            if point[1] > PICK_UP_AREA[0][1] and point[1] < PICK_UP_AREA[2][1]:
                return True
    return False

class ExperimentController(BreatheAndGazeController):
    def __init__(self, name = "experiment_controller"):
        super().__init__(name)

    def get_all_markers(self):
        
        markers = []
        
        try:            
            for i in range(100):
                if self.tfBuffer.can_transform(self.world, f"marker_{i}", rclpy.time.Time(seconds=0)):
                    object_tf = self.tfBuffer.lookup_transform(self.world, f"marker_{i}", rclpy.time.Time(seconds=0))
                    object_position = np.array([object_tf.transform.translation.x, object_tf.transform.translation.y, object_tf.transform.translation.z])
                    self.get_logger().info(f"Marker {i} position: {object_position}")
                    markers.append(object_position)                    
        except Exception as e:
            print("Error in getting all markers:", traceback.format_exc())
            return markers
        return markers
    
    def get_placement_position(self): # in world frame
        
        try:
            markers = self.get_all_markers()
            
            for dock in PLACEMENT_AREA:
                clear = True
                for marker in markers:
                    marker = marker[:2]
                    print("Marker test: ", marker, " - ", dock)
                    if np.linalg.norm(marker - dock) < 0.05:
                        clear = False
                        self.get_logger().info(f"Marker is too close to the dock: {marker} - {dock}")
                        break
                
                if clear:
                    return dock
            self.get_logger().info("No clear placement position found.")
            return None
        except Exception as e:
            self.get_logger().error(f"Error in getting placement position: {traceback.format_exc()}")
            return None
      
    def object_to_pick_up(self):

        markers = self.get_all_markers()

        try: 
            for marker in markers:
                already = False
                object_position = marker
                self.get_logger().info(f"Object position for marker: {object_position}")
                if is_in_pick_up_area(object_position):
                    object_position = self.world_to_base_homogeneous @ np.concatenate((object_position, [1]))
                    object_position = object_position[:3]
                    self.get_logger().info(f"Object position in base frame: {object_position}")
                    for obj_pos in already_picked_up:
                        if object_position[0] == obj_pos[0] and object_position[1] == obj_pos[1]:
                            self.get_logger().info(f"Already picked up object: {object_position}")
                            already = True
                            break
                    if already:
                        continue
                    self.get_logger().info(f"Object to pick up: {object_position}")
                    return object_position
        except Exception as e:
            self.get_logger().error(f"Error in getting object to pick up: {traceback.format_exc()}")
            return None
        self.get_logger().info("No object to pick up.")
        return None
    
def spin_thread(node, executor=None):
    
    try:
        rclpy.spin(node, executor=executor)
    except Exception as e:
        print("Error in spin_thread: ", traceback.format_exc())

def robot_state_machine(controller):
    global shutdown_flag
    
    velocity_command = np.zeros(controller.num_of_total_joints)
    velocity_command_modifier = np.zeros(controller.num_of_total_joints)
    
    # Set flags for breathing and gazing
    do_breathing = True
    do_gazing = True
            
    inf_breathe = False
    
    controller.breathing_start_time = time.time()
    
    gaze_breathe_duration = 10
    
    freq_test_time = time.time()
    
    while rclpy.ok() and not shutdown_flag:

        controller.delta_t = 1/(time.time() - freq_test_time)
        freq_test_time = time.time()
                    
        if observer.get_stop():
            controller.stop_movement()
            controller.state = RobotState.END
                    
        if controller.state == RobotState.START:
            
            pose_diff = np.linalg.norm(controller.joint_states_global["pos"] - controller.home_pos)
            print("Pose Diff: ", pose_diff)
            if pose_diff > 0.1:
                controller.go_to_home_pos(speed=controller.speed)
            controller.set_breathing_gazing()
            controller.state = RobotState.BREATHING_GAZING
            continue
            
        elif controller.state == RobotState.BREATHING_GAZING:
                            
            breathing_velocities = np.zeros(controller.num_of_breathing_joints)
            breathing_task = np.zeros(3)
            if do_breathing:
                breathing_velocities, breathing_task = controller.breathe_controller.step(controller.joint_states_global["pos"], controller.joint_states_global["vels"], ur5e_kinematics.get_jacobian)
                            
            gazing_velocities = np.zeros(controller.num_of_total_joints - controller.num_of_breathing_joints)
            if do_gazing: 
                gazing_velocities = controller.get_gaze_velocities(breathing_task) 
                                            
            breathing_velocities = controller.breathe_velocity_filter.filter(breathing_velocities)
            velocity_command = np.concatenate((breathing_velocities, gazing_velocities))
         
            if not inf_breathe and time.time() - controller.breathing_start_time > gaze_breathe_duration:
                controller.state = RobotState.PICKING
                controller.stop_movement()
                object_to_pick = controller.object_to_pick_up()
                if object_to_pick is not None:
                    if not controller.set_pick_up_cubic_variables(object_to_pick, "pick_pose", raise_eff=False):
                        controller.state = RobotState.END
                else:
                    controller.state = RobotState.END
                continue
            
        elif controller.state == RobotState.PICKING:
            
            if controller.t >= 0 and controller.t <= 1 and np.linalg.norm(controller.get_current_coordinate() - controller.end_pos) > controller.movement_stop_threshold and (controller.t < 0.9 or abs(controller.get_current_coordinate()[2] - controller.end_pos[2]) > controller.movement_stop_threshold):
                velocity_command = controller.get_derivative_velocities()
            elif controller.t < 0:
                print("SOMETHING WENT EXTREMELY WRONG.")
                controller.stop_movement()
                controller.state = RobotState.END
                continue
            else:                
                print("controller.t =", controller.t)
                print("controller.movement_stop_threshold =", controller.movement_stop_threshold)
                print("abs(controller.get_current_coordinate()[2] - controller.end_pos[2]) =", abs(controller.get_current_coordinate()[2] - controller.end_pos[2]))
                controller.stop_movement()
                controller.state = RobotState.PLACING
                print("Picking done.")
                print("Current coordinate: ", controller.get_current_coordinate())
                print("End coordinate: ", controller.end_pos)
                print("Current - end coordinate: ", np.linalg.norm(controller.get_current_coordinate() - controller.end_pos))
                if not controller.set_place_cubic_variables():
                    controller.state = RobotState.BREATHING_GAZING
                controller.gripper.close()
                continue
            
        elif controller.state == RobotState.PLACING:
            
            if controller.t >= 0 and controller.t <= 1 and np.linalg.norm(controller.get_current_coordinate() - controller.end_pos) > controller.movement_stop_threshold and (controller.t < 0.9 or abs(controller.get_current_coordinate()[2] - controller.end_pos[2]) > controller.movement_stop_threshold):
                velocity_command = controller.get_derivative_velocities()
            elif controller.t < 0:
                print("SOMETHING WENT EXTREMELY WRONG.")
                controller.stop_movement()
                controller.state = RobotState.END
                continue
            else:
                print("controller.t =", controller.t)
                print("controller.movement_stop_threshold =", controller.movement_stop_threshold)
                print("abs(controller.get_current_coordinate()[2] - controller.end_pos[2]) =", abs(controller.get_current_coordinate()[2] - controller.end_pos[2]))
                controller.stop_movement()
                print("Placing done.")
                print("Current coordinate: ", controller.get_current_coordinate())
                print("End coordinate: ", controller.end_pos)
                print("Current - end coordinate: ", np.linalg.norm(controller.get_current_coordinate() - controller.end_pos))
                controller.gripper.open()
                
                object_to_pick = controller.object_to_pick_up()
                if object_to_pick is not None and controller.set_pick_up_cubic_variables(object_to_pick, "pick_pose", raise_eff=True):
                    controller.state = RobotState.PICKING
                else:                    
                    controller.set_breathing_gazing()
                    controller.state = RobotState.BREATHING_GAZING
                    inf_breathe = True
                    
                continue
            
        elif controller.state == RobotState.END:
            print("STATE = END")
            controller.stop_movement()
            break
        
        velocity_command += velocity_command_modifier
        controller.publishVelocityCommand(velocity_command)
            
        controller.ros_rate.sleep()
      
    try:  
        controller.stop_movement()
        #controller.go_to_home_pos(speed=controller.speed)
    except:
        print("Error in stopping movement: ", traceback.format_exc())
        pass
  
def observer(controller, min_distance=0.9, max_distance=1.9):
    global shutdown_flag
    
    min_multiplier=0.5
    max_multiplier=1.1
    
    prev_multiplier=min_multiplier
    
    min_frequency=0.25
    max_frequency=0.50
    
    prev_frequency=min_frequency
    
    distance = 0.0
    prev_distance = distance
        
    try:
        
        while controller.tfBuffer.can_transform(controller.eef, controller.tf_name, rclpy.time.Time(seconds=0)) == False:
            print("Waiting for transform...")
            time.sleep(0.1)
        
        while rclpy.ok() and not shutdown_flag:
            
            tf = controller.tfBuffer.lookup_transform("x_towards_board", controller.tf_name, rclpy.time.Time(seconds=0))
            distance = abs(tf.transform.translation.x)
                        
            if distance < min_distance:
                speed_multiplier = min_multiplier
                freq = min_frequency
            elif distance > max_distance:
                speed_multiplier = max_multiplier
                freq = max_frequency
            else:
                speed_multiplier = ((distance - min_distance) / (max_distance - min_distance)) * (max_multiplier - min_multiplier) + min_multiplier
                freq = ((distance - min_distance) / (max_distance - min_distance)) * (max_frequency - min_frequency) + min_frequency
            
            speed_multiplier = 0.9* prev_multiplier + 0.1 * speed_multiplier
            freq = 0.9* prev_frequency + 0.1 * freq
            distance = 0.9 * prev_distance + 0.1 * distance
            
            prev_multiplier = speed_multiplier
            prev_frequency = freq
            
            controller.speed_multiplier = speed_multiplier
            controller.breathe_controller.set_frequency(freq)
            controller.breathe_frequency = freq
                        
            time.sleep(0.01) 
    except KeyboardInterrupt:
        print("KeyboardInterrupt received in observer, shutting down...")
        shutdown_flag = True
    except Exception as e:
        print("Error in observer: ", traceback.format_exc())
        pass

def main(args=None):
    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)    
        experiment_controller = ExperimentController()
        
        controller_spin_thread = threading.Thread(target=spin_thread, args=(experiment_controller,))
        controller_spin_thread.start()
        
        #controller_observer_thread = threading.Thread(target=observer, args=(experiment_controller,))
        #controller_observer_thread.start()
        
        print("Starting main.")
                         
        experiment_controller.speed = 0.3
        
        experiment_controller.object_to_pick_up()

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