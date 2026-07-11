#ROS1

import numpy as np
import time
import rospy
import moveit_commander
import robotiq_gripper
import tf2_ros
from scipy.spatial.transform import Rotation as R
# from simple_pid import PID
from Bezier import Bezier
from matplotlib import pyplot as plt
from enum import Enum
import argparse
import subprocess

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from std_msgs.msg import Float64
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import breathing
import gazing

from utils import get_head_pose

import os 

joint1des = []
joint1real = []
joint1torq = []
joint2des = []
joint2real = []
joint2torq = []

timestep = []

start_recording = False

step_len_glob = 0.0002

last_gaze_target = [1, 1, 0]
use_helmet = True

picked_up_cylinder_poses = []

joint_states_global = {}
def js_callback(data):
    global joint_states_global
    joint_states_global["pos"] = np.array([data.position[2], 
                                  data.position[1], 
                                  data.position[0], 
                                  data.position[3], 
                                  data.position[4], 
                                  data.position[5]])
    
    joint_states_global["vels"] = np.array([data.velocity[2], 
                                  data.velocity[1], 
                                  data.velocity[0], 
                                  data.velocity[3], 
                                  data.velocity[4], 
                                  data.velocity[5]])
    
    joint_states_global["eff"] = np.array([data.effort[2],
                                  data.effort[1],
                                  data.effort[0],
                                  data.effort[3],
                                  data.effort[4],
                                  data.effort[5]])
    
class GripperController:
    def __init__(self, host, port):
        self.host = host # The robot's computer address
        self.port = port # The gripper's port number
        self.gripper = None
        self._init_gripper()

    def _init_gripper(self):
        print("Creating gripper...")
        self.gripper = robotiq_gripper.RobotiqGripper()
        print("Connecting to gripper...")
        self.gripper.connect(self.host, self.port)
        print("Activating gripper...")
        self.gripper.activate(auto_calibrate=False)
        
        self.close()

    def open(self, speed=255, force=255):
        self.gripper.move_and_wait_for_pos(0, speed, force)
        # self.pub.publish(String("0.0"))

    def close(self, speed=255, force=255):
        self.gripper.move_and_wait_for_pos(255, speed, force)
        # self.pub.publish(String("0.7"))

    def close_async(self, speed=255, force=255):
        self.gripper.move(255, speed, force)

    def move_gripper(self, pos, speed=255, force=255):
        pos = 0 if pos < 0 else (255 if pos > 255 else pos)
        self.gripper.move_and_wait_for_pos(pos, speed, force)
        
    def current_pos(self):        
        return self.gripper.get_current_position()
    
    def is_holding(self):
        prev_pos = self.gripper.get_current_position()
        if prev_pos < 50:
            return False
        #else:
        #    print("WHY ARE WE HERE, JUST TO SUFFER?!")
        self.gripper.move_and_wait_for_pos(prev_pos+1,255,255)
        if (abs(self.gripper.get_current_position()-prev_pos)==0) and (self.gripper.get_current_position() < 100):
            return True
        else:
            return False
    
class exp_speed(Enum):
    Fast = 1
    Slow = 2
    Adaptive = 3
    
speed_fast = 0.5
speed_slow = 0.1
    
class exp_state(Enum):
    START = 1
    INTERACTION = 2
    GAZE_AND_BREATHE = 3
    PICKANDPLACE = 4  
    
state = exp_state.START
session_speed = exp_speed.Adaptive
        
def object_placed_callback(msg):
    global state
    
    if msg.data:
        state = exp_state.PICKANDPLACE
    
des_home = []
    
def attention_person_callback(msg):    
    global state
    global des_home
    
    if msg.data:
        state = exp_state.GAZE_AND_BREATHE
        des_home = home_pos_0
        
def end_experiment_callback(msg):
    global end_experiment
    
    if msg.data:
        end_experiment = True
    
def stop_movement():
    velocity_command = np.zeros(6)
    publish_vel(velocity_command)
    
def get_session_parameters(session_speed):
    if session_speed == exp_speed.Fast:
        return 2, 1.2
    elif session_speed == exp_speed.Slow:
        return 4, 1.0
    elif session_speed == exp_speed.Adaptive:
        return 2, 1.2 
    else:
        return 4, 1.0

def go_to_joint_pos(desired_joint_positions, speed=0.05):
    
    trajectory = desired_joint_positions - joint_states_global["pos"]      
    norm = np.linalg.norm(desired_joint_positions - joint_states_global["pos"])        
    delta_t = norm/speed

    try:
        gazing_target = get_head_pose(tf_buffer, use_helmet)
        distance_min, distance_max = 1.0 , 3.0
        distance = np.sqrt(gazing_target[0]**2 + gazing_target[1]**2 + gazing_target[2]**2)
            
        t_min, t_max = 0.7, 2.5
        mapping_t = lambda x: (t_max - t_min) * (x - distance_min) / (distance_max - distance_min) + t_min
        delta_t = mapping_t(distance)
    except:
        delta_t = delta_t
            
    velocity_command = trajectory / delta_t
    print("desired_joint_positions: ", desired_joint_positions)
    print("velocity_command: ", velocity_command)
        
    publish_vel(velocity_command)
    
    start_time = time.time()
    
    while norm > 0.001 and time.time() - start_time < delta_t + 0.1:
        norm = np.linalg.norm(desired_joint_positions - joint_states_global["pos"])
        
    stop_movement() 
    
    print("Position reached, current position: ", joint_states_global["pos"])  

bezier_speed = []
def get_speed_bezier(ind=0,min_Speed=0.01,max_Speed=0.05,amount_steps=2,calc=False):
    global bezier_speed
            
    if calc:    
        bezier_speed = []
        proportion_bezier = 0.95
        steps_to_calc = int(amount_steps*proportion_bezier)
        steps = np.arange(0,1,0.01)
        max_multiplier = 1.05
        points = np.array([[0,min_Speed],[0,max_Speed*max_multiplier],[steps_to_calc,max_Speed*max_multiplier],[steps_to_calc,min_Speed]])
        bezier_curve = Bezier.Curve(steps,points)      
        bezier_speed = np.interp(np.arange(0,int(steps_to_calc/2),1),bezier_curve[:,0],bezier_curve[:,1])
        bezier_speed = np.concatenate((bezier_speed,np.full((amount_steps-2*(int(steps_to_calc/2))),bezier_speed[-1])))
        bezier_speed = np.concatenate((bezier_speed,np.interp(np.arange(int(steps_to_calc/2),steps_to_calc,1),bezier_curve[:,0],bezier_curve[:,1])))          
        
        if len(bezier_speed)<amount_steps:
            bezier_speed = np.concatenate((bezier_speed,np.full(amount_steps-len(bezier_speed),bezier_speed[-1])))
                
        return min_Speed      
    else:        
        return bezier_speed[ind]
                    
def get_speed_adaptive(prev_speed):
    
    global speed_fast
    global speed_slow
    
    max_change = 0.005
    
    gazing_target = get_head_pose(tf_buffer, use_helmet)                        
        
    if gazing_target is not None and session_speed == exp_speed.Adaptive:
        distance_min, distance_max = 1.0, 3.0
        
        distance = np.sqrt(gazing_target[0]**2 + gazing_target[1]**2 + gazing_target[2]**2)
        
        speed_min, speed_max = speed_slow, speed_fast
        mapping_speed = lambda x: (speed_max - speed_min) * (x - distance_min) / (distance_max - distance_min) + speed_min
        
        new_speed = mapping_speed(distance)
        
        if abs(new_speed - prev_speed) < max_change:
            return new_speed
        else:
            return prev_speed + np.sign(new_speed-prev_speed)*max_change
            
    else:
        return spid

def create_straight_trajectory(desired_pos):
    
    if desired_pos is None:
        return []
    
    current_pos = np.zeros(3)
        
    transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
    current_pos[0] = transformation.transform.translation.x
    current_pos[1] = transformation.transform.translation.y
    current_pos[2] = transformation.transform.translation.z 
        
    step_len = step_len_glob
    trajectory_vector = desired_pos-current_pos
    trajectory_len = np.linalg.norm(trajectory_vector)
    step_vector = (trajectory_vector/trajectory_len) * step_len
    
    current_step = current_pos
    trajectory = [current_step] 
    
    for i in range(int(trajectory_len/step_len)):
        current_step = current_step+step_vector
        trajectory.append(current_step)
        
    """    
    dist = np.linalg.norm(current_pos-desired_pos)        
            
    steps = np.linspace(0,1,num=int(dist/step_len),endpoint=False)

    straight_x = np.interp(steps,[0,1],[current_pos[0], desired_pos[0]])
    straight_y = np.interp(steps,[0,1],[current_pos[1], desired_pos[1]])
    straight_z = np.interp(steps,[0,1],[current_pos[2], desired_pos[2]])
        
       
    for i in range(len(straight_x)):        
        trajectory.append([straight_x[i], straight_y[i], straight_z[i]])"""
    
    return trajectory

def go_to_world_pos(desired_pos, max_speed=0.05, min_speed=0.015):
        
    if desired_pos is None:
        return
    
    trajectory = create_straight_trajectory(desired_pos)
    
    follow_trajectory(trajectory, max_speed, min_speed)
 
def follow_trajectory(trajectory, max_speed=0.05, min_speed=0.015):
    
    global start_recording
    
    if len(trajectory) == 0:
        return
    
    current_pos = np.zeros(3)        
    transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
    current_pos[0] = transformation.transform.translation.x
    current_pos[1] = transformation.transform.translation.y
    current_pos[2] = transformation.transform.translation.z 
    
    dist = np.linalg.norm(np.array(trajectory[0])-np.array(trajectory[-1]))    
    amount_points = len(trajectory)
    
    pos_index = 0
        
    min_error = max_speed/25
    max_error = min_error*(1.4)    

    speed_of_curr_step = max_speed  
                
    ease_parabola = lambda x: -40*x*x + 12*x + 0.1
    min_error = speed_fast/25
    max_error = min_error*(1.4)
    
    curr_time = time.time()
        
    while not rospy.is_shutdown() and pos_index < amount_points-1 and state == exp_state.PICKANDPLACE:

        transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
        current_pos[0] = transformation.transform.translation.x
        current_pos[1] = transformation.transform.translation.y
        current_pos[2] = transformation.transform.translation.z
        
        trajectory_desired_pos = trajectory[pos_index+1] - current_pos
        trajectory_desired_full = np.concatenate((trajectory_desired_pos, np.zeros(3))) # May need to change from zeros to actual values while creating sphere
        
        #print("trajdesfull: ", trajectory_desired_full)
        jacobian = group.get_jacobian_matrix(joint_states_global["pos"].tolist())   
        speed_of_curr_step = max_speed if not session_speed == exp_speed.Adaptive else get_speed_adaptive(speed_of_curr_step)
        
        if dist > 0.01:
            if pos_index/amount_points <= 0.15:
                temp = speed_of_curr_step * ease_parabola(pos_index/amount_points)
            elif pos_index/amount_points >= 0.85:
                temp = speed_of_curr_step * ease_parabola((pos_index/amount_points-0.7))
            else:
                temp = speed_of_curr_step
        else:
            temp = speed_of_curr_step
        velocity_desired = ( trajectory_desired_full / np.linalg.norm(trajectory_desired_full) ) * temp
                
        pos_index += 1
             
        pinv_jacobian = np.linalg.pinv(jacobian, rcond=1e-15)  # Take psuedo inverse of Jacobian, rcond may be tuned not to get closer to singularities.

        velocity_command = pinv_jacobian @ velocity_desired
                
        #print("VC: ", velocity_command)        
        #print("Position Error: ", err, file=open("error.txt", "a"))        
        
        # Publish joint vels to robot
        publish_vel(velocity_command)   
        
        transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
        current_pos[0] = transformation.transform.translation.x
        current_pos[1] = transformation.transform.translation.y
        current_pos[2] = transformation.transform.translation.z 
        
        prev_err = 9999999999
        
        while min_error < np.linalg.norm(current_pos-trajectory[pos_index]) < max_error and not rospy.is_shutdown():
            transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
            current_pos[0] = transformation.transform.translation.x
            current_pos[1] = transformation.transform.translation.y
            current_pos[2] = transformation.transform.translation.z
            
            if np.linalg.norm(current_pos-trajectory[pos_index]) - prev_err > 0.00007:
                trajectory_desired_pos = trajectory[pos_index] - current_pos
                trajectory_desired_full = np.concatenate((trajectory_desired_pos, np.zeros(3))) 
                velocity_desired = ( trajectory_desired_full / np.linalg.norm(trajectory_desired_full) ) * temp
                jacobian = group.get_jacobian_matrix(joint_states_global["pos"].tolist())        
                pinv_jacobian = np.linalg.pinv(jacobian, rcond=1e-15)
                velocity_command = pinv_jacobian @ velocity_desired
                publish_vel(velocity_command)
                print("ERROR CORRECTION")
                        
            prev_err = np.linalg.norm(current_pos-trajectory[pos_index])
            print("ERR", np.linalg.norm(current_pos-trajectory[pos_index]), file=open("error.txt", "a"))            
            
        if np.linalg.norm(current_pos-trajectory[pos_index]) >= max_error:
            pos_index -= 1
          
        if start_recording:              
            timestep.append(1/(time.time() - curr_time))
        curr_time = time.time()
        
    stop_movement()
       
def take_orientation(desired_rot, speed=0.05, delta_t_to_start_rot=7, wrt_base=False):
    if desired_rot is None:
        return
        
    base_to_world_z = 0.7520603
    z_diff_const = 0.8168736 if not wrt_base else 0.0
    current_rot = np.array([joint_states_global["pos"][1] + joint_states_global["pos"][2] + joint_states_global["pos"][3], 
                            joint_states_global["pos"][4],
                            joint_states_global["pos"][0] - joint_states_global["pos"][5] + z_diff_const])
        
    print("Current rot: ", current_rot)
    print("Desired rot: ", desired_rot)                                         
                                                 
    rotation_to_start = desired_rot - current_rot
    
    for i in range(len(rotation_to_start)):
        if rotation_to_start[i] > np.pi:
            rotation_to_start[i] -= 2*np.pi
        elif rotation_to_start[i] < -np.pi:
            rotation_to_start[i] += 2*np.pi
    
    if np.linalg.norm(rotation_to_start) < 0.01:
        print("ORIENTATION CLOSE ENOUGH")
        return
    
    print("ORIENTING", np.linalg.norm(rotation_to_start), "ROTATION: ", rotation_to_start)
    
    desired_joint_positions = joint_states_global["pos"] + np.concatenate((np.zeros(3), rotation_to_start))    
                        
    angular_velocity_to_start = rotation_to_start / delta_t_to_start_rot   
    
    velocity_command = np.concatenate((np.zeros(3), angular_velocity_to_start))
    velocity_command[5] *= -1
    
    publish_vel(velocity_command) 
    
    start_time = time.time()
    
    norm = np.linalg.norm(desired_joint_positions - joint_states_global["pos"])
    while norm > 0.001 and time.time() - start_time < delta_t_to_start_rot and state == exp_state.PICKANDPLACE:
        norm = np.linalg.norm(desired_joint_positions - joint_states_global["pos"])            
        
    stop_movement()    
    print("Rotation reached")

def turn_eef(rotation, speed=0.05, delta_t_to_start_rot=7):    
    # rotation is given in x y z order    
    if rotation is None:
        return                                                 
    
    for i in range(len(rotation)):
        if rotation[i] > np.pi:
            rotation[i] -= 2*np.pi
        elif rotation[i] < -np.pi:
            rotation[i] += 2*np.pi
    
    desired_joint_positions = joint_states_global["pos"] + np.array([0, 0, 0, rotation[0], rotation[1], rotation[2]])   
                        
    angular_velocity_to_start = rotation / delta_t_to_start_rot   
    
    velocity_command = np.array([0, 0, 0, angular_velocity_to_start[0], angular_velocity_to_start[1], angular_velocity_to_start[2]])
    velocity_command[5] *= 1
    
    publish_vel(velocity_command) 
    
    start_time = time.time()
    
    norm = np.linalg.norm(desired_joint_positions - joint_states_global["pos"])
    while norm > 0.001 and time.time() - start_time < delta_t_to_start_rot:
        norm = np.linalg.norm(desired_joint_positions - joint_states_global["pos"])            
        
    stop_movement()
    
    print("Rotation reached")

def make_parabola(start_pos, mid_pos, end_pos, amt_points=100):
    
    x_axis = (end_pos - start_pos) / np.linalg.norm(end_pos - start_pos)
    z_axis = np.cross(x_axis, mid_pos - start_pos)
    z_norm = np.linalg.norm(z_axis)
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    y_axis = y_axis / y_norm
    orig_axis = start_pos #not an axis, dont know why i called it that - sometimes somethings just happen and all you can do is accept and move on
    
    x_axis = np.concatenate((x_axis, [0]))
    y_axis = np.concatenate((y_axis, [0]))
    z_axis = np.concatenate((z_axis, [0]))
    orig_axis = np.concatenate((orig_axis, [1]))
    
    transform_matrix = np.array([x_axis, y_axis, z_axis, orig_axis])
    transform_matrix = np.transpose(transform_matrix)
    
    print("Transform Matrix: ", transform_matrix)
    
    # -------------------------------
    
    vec_ie = end_pos - start_pos
    vec_im = mid_pos - start_pos
    
    norm_ie = np.linalg.norm(vec_ie)
    norm_im = np.linalg.norm(vec_im)
    
    cosalfa = np.dot(vec_ie, vec_im) / (norm_ie * norm_im)
    sinalfa = np.sqrt(1 - cosalfa**2)
        
    vec_xm = cosalfa*norm_im
    vec_ym = sinalfa*norm_im
    
    a = vec_ym/(vec_xm*vec_xm - vec_xm*norm_ie)
    b = -a*norm_ie
        
    steps = np.linspace(0,1,num=amt_points,endpoint=False)
    sample_points_x = np.interp(steps,[0,1],[0,norm_ie])    
    
    print("Sample Points: ", sample_points_x)
    sample_points_y = []
    
    for i in range(len(sample_points_x)):        
        sample_points_y.append(a*sample_points_x[i]**2 + b*sample_points_x[i])
        
    traj_points = []
        
    for i in range(len(sample_points_x)):        
        traj_points.append((transform_matrix @ np.array([sample_points_x[i], sample_points_y[i], 0, 1]))[0:3])
        print("Traj: ", traj_points[-1])
    
    return traj_points

def follow_parabola(desired_pos, mid_point, speed=0.05):
    
    current_pos = np.zeros(3)
    transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
    current_pos[0] = transformation.transform.translation.x
    current_pos[1] = transformation.transform.translation.y
    current_pos[2] = transformation.transform.translation.z
    
    samp = float((np.linalg.norm(current_pos - mid_point) + np.linalg.norm(desired_pos-mid_point)))*700
    
    trajectory = make_parabola(current_pos, mid_point, desired_pos, int(samp))
    follow_trajectory(trajectory, max_speed=speed)
    
def pick_up_object(object, speed=0.05):
    
    if state == exp_state.PICKANDPLACE:
        gripper.open()
    
    #rotate gripper downwards
    desired_rot = np.array([-np.pi/2, -np.pi/2, -np.pi/4]) 
    take_orientation(desired_rot, speed, 2, wrt_base=True)
    
    #go on top of the object (but a little higher than the object)
    obj_tf_world = tf_buffer.lookup_transform(world, f"{object}", rospy.Time())    
    desired_pos = np.array([obj_tf_world.transform.translation.x+0.0025, obj_tf_world.transform.translation.y+0.025, obj_tf_world.transform.translation.z + 0.04])
    go_to_world_pos(desired_pos, max_speed=speed)

    #lower the gripper
    desired_pos[2] -= 0.08
    desired_pos[0] += desired_pos[1]/24
    desired_pos[1] += -(desired_pos[0]-1.03)/30
    go_to_world_pos(desired_pos, max_speed=speed) 
    
    #grip
    if state == exp_state.PICKANDPLACE:
        gripper.close()
        
    #raise gripper
    desired_pos[2] += 0.15
    go_to_world_pos(desired_pos, max_speed=speed)
    
def pick_up_pos(pos, speed=0.05):
    
    if state == exp_state.PICKANDPLACE:
        gripper.open()
    
    pt_stamped = PointStamped()
    pt_stamped.header.frame_id = world
    pt_stamped.point.x = pos[0]
    pt_stamped.point.y = pos[1]
    pt_stamped.point.z = pos[2]
    
    cylinder_pos_publisher.publish(pt_stamped)
    
    #rotate gripper downwards
    desired_rot = np.array([-np.pi/2, -np.pi/2, -np.pi/4]) 
    take_orientation(desired_rot, speed, 2, wrt_base=True)
        
    #go on top of the object(but a little higher than the object)
    desired_pos = np.array([pos[0]+0.0025, pos[1]+0.025, pos[2]+0.08])
    go_to_world_pos(desired_pos, max_speed=speed)
    
    #lower the gripper
    desired_pos[2] -= 0.12
    desired_pos[0] += desired_pos[1]/24
    desired_pos[1] += -(desired_pos[0]-1.03)/30
    go_to_world_pos(desired_pos, max_speed= speed if speed<0.3 else 0.3)    
    
    #grip
    if state == exp_state.PICKANDPLACE:
        gripper.close()
        
    #raise gripper
    desired_pos[2] += 0.15
    go_to_world_pos(desired_pos, max_speed=speed)
        
def pick_up_pos_parabolic(pos, speed=0.05):
    
    if state == exp_state.PICKANDPLACE:
        gripper.open()
    
    pt_stamped = PointStamped()
    pt_stamped.header.frame_id = world
    pt_stamped.point.x = pos[0]
    pt_stamped.point.y = pos[1]
    pt_stamped.point.z = pos[2]
    
    cylinder_pos_publisher.publish(pt_stamped)
    
    #rotate gripper downwards
    desired_rot = np.array([-np.pi/2, -np.pi/2, -np.pi/4]) 
    take_orientation(desired_rot, speed, 2, wrt_base=True)
        
    current_pos = np.zeros(3)
    transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
    current_pos[0] = transformation.transform.translation.x
    current_pos[1] = transformation.transform.translation.y
    current_pos[2] = transformation.transform.translation.z
        
    #go on top of the object(but a little higher than the object)
    desired_pos = np.array([pos[0]+0.0025, pos[1]+0.03, pos[2]+0.08])
    mid_pt = ((current_pos + desired_pos) / 2) + np.array([0,0,exaggeration_const])
    follow_parabola(desired_pos, mid_pt, speed) # PARABOLIC
    
    #lower the gripper
    desired_pos[2] -= 0.12
    desired_pos[0] += desired_pos[1]/24
    desired_pos[1] += -(desired_pos[0]-1.03)/30    
    go_to_world_pos(desired_pos, max_speed= speed if speed<0.3 else 0.3)  
    
    #grip
    if state == exp_state.PICKANDPLACE:
        gripper.close()
        
    #raise gripper
    desired_pos[2] += 0.15
    go_to_world_pos(desired_pos, max_speed=speed)
    
def place_object(destination, speed=0.05):
    
    #rotate gripper downwards
    desired_rot = np.array([-np.pi/2, -np.pi/2, -np.pi/4]) 
    take_orientation(desired_rot, spid, 2, wrt_base=True)
        
    #go to drop position
    goto = np.array([destination[0], destination[1], destination[2]])
    goto[2] += 0.04
    go_to_world_pos(goto, max_speed=speed)
    
    #lower the gripper
    goto[2] -= 0.1
    go_to_world_pos(goto, max_speed=speed if speed<0.3 else 0.3)
    
    #drop
    if state == exp_state.PICKANDPLACE:
        gripper.open()
        
    #raise gripper
    goto[2] += 0.11
    go_to_world_pos(goto, max_speed=speed)
  
def place_object_parabolic(destination, speed=0.05):
    
    #rotate gripper downwards
    desired_rot = np.array([-np.pi/2, -np.pi/2, -np.pi/4]) 
    take_orientation(desired_rot, spid, 2, wrt_base=True)
        
    #go to drop position
    goto = np.array([destination[0], destination[1], destination[2]])        
    goto[2] += 0.04
    
    current_pos = np.zeros(3)
    transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
    current_pos[0] = transformation.transform.translation.x
    current_pos[1] = transformation.transform.translation.y
    current_pos[2] = transformation.transform.translation.z
        
    mid_pt = ((current_pos + goto) / 2) + np.array([0,0,exaggeration_const])
    follow_parabola(goto, mid_pt, speed) # PARABOLIC    
    
    #lower the gripper
    goto[2] -= 0.1
    go_to_world_pos(goto, max_speed=speed if speed<0.3 else 0.3)
    
    #drop
    if state == exp_state.PICKANDPLACE:
        gripper.open()
        
    #raise gripper
    goto[2] += 0.11
    go_to_world_pos(goto, max_speed=speed)
        
def breathe_and_gaze(do_breathing=True, do_gazing=True):    
    gripper.close_async()
    
    global session_speed
    global end_experiment
    global state
        
    start_time = time.time()
    
    breathe_controller.reset()    
        
    joint1des.append(0)
    joint2des.append(0)
        
    while (not rospy.is_shutdown()) and (state == exp_state.GAZE_AND_BREATHE) and not end_experiment:      
        
        if (time.time() - start_time) >= 10:
            end_experiment = True
            break
            
        breathing_velocities = np.zeros(num_of_breathing_joints)
        if do_breathing:
            breathing_velocities = breathe_controller.step(joint_states_global["pos"],
                                                    joint_states_global["vels"],
                                                    group.get_jacobian_matrix)
        
        if do_gazing:
            # Calculate head transformation matrix
            # !!! Change "base_link" with "world" if the gazing is in the world frame !!!
            transformation = tf_buffer.lookup_transform("base_link", "wrist_1_link", rospy.Time())            
            
            r = R.from_quat(np.array([transformation.transform.rotation.x,
                                    transformation.transform.rotation.y,
                                    transformation.transform.rotation.z,
                                    transformation.transform.rotation.w]))
            r = r.as_matrix()
            r = np.vstack((r, [0,0,0]))
            r = np.hstack((r, np.array([[transformation.transform.translation.x,
                                    transformation.transform.translation.y,
                                    transformation.transform.translation.z,
                                    1.0]]).T))
            
            if gazing_breathing_compensation and do_breathing:
                # Compensate the gazing with breathing
                lookahead_into_the_future = 0.27  # amt of time we look ahead into the future in seconds
                r_estimated = gazing_controller.get_head_position(joint_states_global["pos"][:4] + np.concatenate((breathing_velocities, [0])) * lookahead_into_the_future)
            else:
                r_estimated = r
            
            # gazing_target = [0, 1, 0]  # !!! Change this wrt. the gazing target, gazing in base_link frame !!!
            
            if is_head_fake:            
                gazing_target = [fake_head_pos_x[index], fake_head_pos_y[index], fake_head_pos_z[index]]
                fake_pos = PointStamped()
                fake_pos.header.frame_id = world
                fake_pos.header.stamp = rospy.Time.now()
                fake_pos.point.x = gazing_target[0]
                fake_pos.point.y = gazing_target[1]
                fake_pos.point.z = gazing_target[2]
                fake_head_pos_publisher.publish(fake_pos)
            else:
                gazing_target = get_head_pose(tf_buffer, use_helmet) 
                    
            gazing_velocities = gazing_controller.step(gazing_target, r_estimated, joint_states_global["pos"])
                        
            if gazing_target is not None and session_speed == exp_speed.Adaptive:
                distance_min, distance_max = 1.0 , 3.0
                
                distance = np.sqrt(gazing_target[0]**2 + gazing_target[1]**2 + gazing_target[2]**2)
                
                frequency_min, frequency_max = 0.1, 0.8
                mapping_frequency = lambda x: (frequency_max - frequency_min) * (x - distance_min) / (distance_max - distance_min) + frequency_min
                new_freq = mapping_frequency(distance)
                
                if abs(new_freq - breathe_controller.freq) > min_delta_freq:
                    if breathe_controller.filter:
                        breathe_controller.desired_freq = new_freq
                    else:
                        breathe_controller.freq = new_freq   
                    
                amplitude_min, amplitude_max = 0.6, 1.4
                mapping_amplitude = lambda x: (amplitude_max - amplitude_min) * (x - distance_min) / (distance_max - distance_min) + amplitude_min
                new_amplitude = mapping_amplitude(distance)
                
                if abs(new_amplitude - breathe_controller.amplitude) > min_delta_amplitude:
                    breathe_controller.amplitude = new_amplitude
                    
                pid_min = (2.0, 0, 0)
                pid_max = (4.0, 0, 0)
                
                mapping_kp = lambda x: (pid_max[0] - pid_min[0]) * (x - distance_min) / (distance_max - distance_min) + pid_min[0]
                mapping_ki = lambda x: (pid_max[2] - pid_min[2]) * (x - distance_min) / (distance_max - distance_min) + pid_min[2]
                
                param = mapping_kp(distance), 0, mapping_ki(distance)
                gazing_controller.update_pid(param)            
                
        # Publish joint vels to robot
        if do_breathing and not do_gazing:
            velocity_command = np.concatenate((breathing_velocities, [0]*(num_of_total_joints - num_of_breathing_joints)))
        elif do_gazing and not do_breathing: 
            velocity_command = np.concatenate(([0]*3, gazing_velocities))
        else: 
            velocity_command = np.concatenate((breathing_velocities[:3], gazing_velocities))
                   
        if is_head_fake:
            index += 1
            if index == sample_count:
                break

                                                    
        publish_vel(velocity_command)
        
        ros_rate.sleep()
        
    stop_movement()
    
    gaze_time = time.time() - start_time
    gaze_msg = Float64(gaze_time)
    gaze_time_publisher.publish(gaze_msg)
      
def is_in_docks(pos):
    
    docks_x1 = 0.94
    docks_x2 = 1.44
    docks_y1 = 0.29
    docks_y2 = 0.66
    docks_z = 0.8
    
    if pos[0] < docks_x2 and pos[0] > docks_x1 and pos[1] < docks_y2 and pos[1] > docks_y1 and pos[2] < docks_z:
        return True
    else: 
        return False
    return False

def is_in_exparea(pos):
    
    expare_x1 = 1.03
    expare_x2 = 1.44
    expare_y1 = -0.45
    expare_y2 = 0.00
    expare_z = 0.8
    
    if pos[0] < expare_x2 and pos[0] > expare_x1 and pos[1] < expare_y2 and pos[1] > expare_y1 and pos[2] < expare_z:
        return True
    else: 
        return False
    return False

def is_gripper_holding():
    
    return gripper.is_holding()

def did_i_pick_it_up(pos):
    
    pos = np.array(pos)
    
    for p in picked_up_cylinder_poses:
        p_np = np.array(p)
        if np.linalg.norm(pos - p_np) < 0.0001:
            return True
        
    return False

def stretch():    
    #tehehhehe
    
    desired_rot = np.array([-np.pi/2, -np.pi/2, -np.pi/4]) 
    take_orientation(desired_rot, spid, 2, wrt_base=True)
    
    current_pos = np.zeros(3)
    transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
    current_pos[0] = transformation.transform.translation.x
    current_pos[1] = transformation.transform.translation.y
    current_pos[2] = transformation.transform.translation.z
    
    goto = current_pos + np.array([0.15,0,0.10])
    mid_pt = goto + np.array([-0.01,0,0.004])
    follow_parabola(goto, mid_pt, spid/4)
    
    current_pos = np.zeros(3)
    transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
    current_pos[0] = transformation.transform.translation.x
    current_pos[1] = transformation.transform.translation.y
    current_pos[2] = transformation.transform.translation.z
    
    goto = dock_1_xyz + np.array([0,0,-0.04])
    mid_pt = ((current_pos + goto) / 2) + np.array([0,0,0.3])
    follow_parabola(goto, mid_pt, spid/4)
    
    gripper.open()
    
def irkil():        
    global state
        
    state = exp_state.PICKANDPLACE
        
    current_pos = np.zeros(3)
    transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
    current_pos[0] = transformation.transform.translation.x
    current_pos[1] = transformation.transform.translation.y
    current_pos[2] = transformation.transform.translation.z
    
    goto = current_pos + np.array([0.15,0,0.05])
    mid_pt = (goto + current_pos)/2 + np.array([0,0,0.02])
    follow_parabola(goto, mid_pt, spid/2)
    
    prev_curr = current_pos
    current_pos = np.zeros(3)
    transformation = tf_buffer.lookup_transform(world, eef, rospy.Time())
    current_pos[0] = transformation.transform.translation.x
    current_pos[1] = transformation.transform.translation.y
    current_pos[2] = transformation.transform.translation.z
    mid_pt = (prev_curr + current_pos)/2 + np.array([0,0,0.02])
    
    time.sleep(1.5)
    
    follow_parabola(prev_curr, mid_pt, 0.01)
    
    state = exp_state.GAZE_AND_BREATHE
    
def publish_vel(velocity_command):
    
    global start_recording
    
    velocity_command = np.array(velocity_command)
    
    cons = 1
    filtered_vel = velocity_command * cons + joint_states_global["vels"] * (1-cons)
    
    vel_msg = Float64MultiArray()
    vel_msg.data = filtered_vel.tolist()
    
    if start_recording:
        joint1real.append(joint_states_global["vels"][1])
        joint2real.append(joint_states_global["vels"][2])
        joint1des.append(vel_msg.data[1])
        joint2des.append(vel_msg.data[2])
        joint1torq.append(joint_states_global["eff"][1])
        joint2torq.append(joint_states_global["eff"][2])
    
    joint_vel_publisher.publish(vel_msg)
     

if __name__ == "__main__":
    
    # The below statement is supposed to emulate an init function, but since all variables will be accessed by other functions and need to be global,
    # i have put it in a always true if statement in the global scope instead of declaring all variables as global in a real function for convenience.
    # the reason i did this was that i did not want to put these clunky initializations in the main loop and make the code as intuitive as possible.
    # also i wanted to be able to collapse it in the editor for better readability.
    if True or "def"=="init()":
        
        print("Initializing main controller...")
        
        rospy.init_node("hri4cobot_maincontroller")
        group = moveit_commander.MoveGroupCommander("manipulator")  # move_group for getting jacobian, any other jacobian lib is fine
        
        # Get joint states
        rospy.Subscriber("joint_states", JointState, js_callback)
        rospy.Subscriber("object_placed", Bool, object_placed_callback)
        rospy.Subscriber("attention_human", Bool, attention_person_callback)
        rospy.Subscriber("end_exp", Bool, end_experiment_callback)
        
        gaze_time_publisher = rospy.Publisher("gaze_time", Float64, queue_size=10)
        cylinder_pos_publisher = rospy.Publisher("cylinder_pos", PointStamped, queue_size=10)
        
        tf_buffer = tf2_ros.Buffer() 
        tf_listener = tf2_ros.TransformListener(tf_buffer, buff_size=120)
        
        # Find the topic for ros_control to take velocity command
        joint_vel_command_topic = "joint_group_vel_controller/command"
        joint_vel_publisher = rospy.Publisher(joint_vel_command_topic, Float64MultiArray, queue_size=1)
        while not joint_vel_publisher.get_num_connections(): rospy.sleep(0.1)
        
        # If we are using fake head pose, we need to publish it to rviz so it is visualized
        fake_head_pos_publisher = rospy.Publisher("fake_head_pos", PointStamped, queue_size=100)
        while not fake_head_pos_publisher.get_num_connections(): rospy.sleep(0.1)

        control_rate = 200  # Frequency for main control loop
        ros_rate = rospy.Rate(control_rate)  
        
        # Set breathing parameters 
        breathe_dict = {}
        breathe_dict["control_rate"] = control_rate

        # Set breathe vector direction in base_link frame
        breathe_vec = np.array([0., 0., 1.])  # named as r in the paper
        breathe_vec = breathe_vec / np.linalg.norm(breathe_vec) if np.all(breathe_vec > 0) else breathe_vec
        breathe_dict["breathe_vec"] = breathe_vec

        # Human breathing data based breathing velocity profile function f(·)
        human_data_path = "/home/kovan4/HRI4Cobot/breathe_data.npy"
        f = np.load(human_data_path)               
        breathe_dict["f"] = f

        # FAST: period = 2, amplitude = 1.2
        # DEFAULT: period = 4, amplitude = 1.0

        session_speed = exp_speed.Adaptive
        
        period, amplitude = get_session_parameters(session_speed)
        freq = 1 / period  # Named as beta in the paper
        breathe_dict["freq"] = freq
        breathe_dict["amplitude"] = amplitude

        num_of_total_joints = 6  # UR5 has 6 joints
        # Use at least 3 joints for breathing, below that, it is not effective
        num_of_breathing_joints = 3  # Num of body joints starting from base towards to end effector
        breathe_dict["num_of_joints"] = num_of_breathing_joints

        breathe_dict["compensation"] = False
        breathe_dict["filter"] = True
        breathe_controller = breathing.Breather(breathe_dict)

        # Set gazing parameters
        # Change the kp, kd, ki values for better performance
        gaze_dict = {}
        gaze_dict["kp"] = 4.0
        gaze_dict["kd"] = 0.0
        gaze_dict["ki"] = 0.0
        # Initial guesses are the initial joint values of the robot starting from the initial head joint
        # In ur wrist_1, wrist_2, wrist_3 are the head joints
        gaze_dict["initial_guesses"] = [-3.14, -1.57, -3.14]  # Decide accounting your robots initial joint values and gazing area

        gazing_controller = gazing.Gazer(gaze_dict)
        
        gripper = GripperController("10.0.0.2", 63352)
        
        gazing_breathing_compensation = True
                    
        eef = "tool0_controller"
        world = "world"
        
        end_experiment = False
        os.environ["QT_API"] = "pyqt5"
        
        print("Main controller initialized.")

    parser = argparse.ArgumentParser(description='HRI4Cobot')
    parser.add_argument('--session_no', type=str, default='1', help='Session number')
    parser.add_argument('--exp_id', type=str, default='100', help='Experiment ID')
    args = parser.parse_args()

    exaggeration_const = 0.12

    once = True

    # The same thing as the above statement basically, but for the main loop.
    if True or "def"=="main()":    
            
        # print("Initial position: ", joint_states_global["pos"])
    
        # exit()
        
        """data_collector_process = subprocess.Popen(
            f"python3 /home/kovan4/HRI4Cobot/araz_yusuf_exp_data_collection.py --session_no {args.session_no} --exp_id {args.exp_id} --use_helmet {use_helmet}",
            shell=True,
        )"""
                   
        sample_count = 2500  
        index = 0    
        is_head_fake = False        
        if is_head_fake:
            # If we are using fake head pose, we need to publish it to rviz so it is visualized
            fake_head_pos_publisher = rospy.Publisher("fake_head_pos", PointStamped, queue_size=100)
            while not fake_head_pos_publisher.get_num_connections(): rospy.sleep(0.1)
            
            fake_head_pos_x = np.load("fake_head_pos/fake_head_x.npy")
            fake_head_pos_y = np.load("fake_head_pos/fake_head_y.npy")
            fake_head_pos_z = np.load("fake_head_pos/fake_head_z.npy")        
            breathe_controller.freq = 0.40642
        
        min_delta_freq = 0.002
        min_delta_amplitude = 0.005
        
        global_start_time = time.time()
        
        start_pos = [2.19, -1.68, 2.27, -3.81, -0.46, -np.pi]
        home_pos_0 = [2.33, -1.8, 2.09, -3.4, -0.6, -np.pi]
        home_pos = [2.33, -1.8, 2.09, -3.4, -1.56, -np.pi]
        interaction_pos = [2.30, -0.48, 0.71, -3.5, -1.47, -np.pi]
        home_gaze = [2.33, -1.8, 2.1125, -2.44, -2.30, -np.pi]
            
        home_xyz = [1.11, 0.0, 1.23]
        
        dock_1_j = [ 1.27256596, -1.14775629,  1.66539604, -2.09093918, -1.57030422, -4.22580916]
        dock_2_j = [ 1.46427977, -0.93801411,  1.31339771, -1.94859995, -1.57079584, -4.03415615]
        dock_3_j = [ 1.61023927, -0.635372,    0.75709278, -1.6950294,  -1.57114822, -3.88808614]
        dock_4_j = [ 1.40098095, -1.38406041,  1.98595554, -2.17528643, -1.57067615, -4.09737498]
        dock_5_j = [ 1.60806429, -1.14265539,  1.63513261, -2.06585898, -1.57116825, -3.89040906]
        dock_6_j = [ 1.75086617, -0.86059238,  1.15950948 ,-1.87214341, -1.57149631, -3.74755437]       
        
        docks_joint = [dock_1_j,dock_2_j,dock_3_j,dock_4_j,dock_5_j, dock_6_j]
        
        dock_1_xyz = [ 1.3647, 0.58646, 0.83]
        dock_2_xyz = [ 1.2174, 0.58511, 0.83]
        dock_3_xyz = [ 1.0699, 0.58344, 0.83]
        dock_4_xyz = [ 1.3668, 0.43845, 0.83]
        dock_5_xyz = [ 1.2204, 0.43690, 0.83]
        dock_6_xyz = [ 1.0669, 0.43423, 0.83] 
        
        docks_xyz = [dock_1_xyz, dock_1_xyz, dock_2_xyz, dock_3_xyz, dock_4_xyz, dock_5_xyz, dock_6_xyz]
        docks_clear = [False, True, True, True, True, True, True]      
                
        gripper.close()
        go_to_joint_pos(home_pos, 0.5)
                        
        session_no = int(args.session_no)
        exp_id = int(args.exp_id)
        
        if session_no == 1:
            session_speed = exp_speed.Slow
        elif session_no == 2:
            session_speed = exp_speed.Fast
        else:
            session_speed = exp_speed.Adaptive
        
        state = exp_state.PICKANDPLACE
                
        spid = speed_fast if session_speed == exp_speed.Fast else speed_slow if session_speed == exp_speed.Slow else (speed_fast+speed_slow)/2
           
        available_cylinders = 0
           
        for i in range(1,7):
                                        
            try:
                cylinder_tf = tf_buffer.lookup_transform(world, f"cylinder_{i}", rospy.Time())
            except:
                print("CYLINDER", i, "NOT HERE")
                continue
            
            cylinder_pos = [cylinder_tf.transform.translation.x, cylinder_tf.transform.translation.y, cylinder_tf.transform.translation.z]    
                        
            if is_in_exparea(cylinder_pos) and not is_in_docks(cylinder_pos):
                         
                available_cylinders += 1        
                pt_stamped = PointStamped()
                pt_stamped.header.frame_id = world
                pt_stamped.point.x = cylinder_pos[0]
                pt_stamped.point.y = cylinder_pos[1]
                pt_stamped.point.z = cylinder_pos[2]
                
                cylinder_pos_publisher.publish(pt_stamped)
        
        if available_cylinders < 1:
            state = exp_state.GAZE_AND_BREATHE
            des_home = home_pos
                
        time.sleep(1)                
             
        start_recording = True
                
        while not rospy.is_shutdown() and not end_experiment:

            if state == exp_state.PICKANDPLACE:
                
                available_cylinders = 0
                
                if is_gripper_holding():
                                
                    dock_to_go = 0                        
                        
                    for j in range(len(docks_clear)):
                        if docks_clear[j]:
                            dock_to_go = j
                            break
                            
                    if dock_to_go > 6 or dock_to_go < 1:
                        print("ALL DOCKS OCCUPIED", dock_to_go)
                        state = exp_state.GAZE_AND_BREATHE
                        des_home = home_pos
                    else:
                        place_object_parabolic(docks_xyz[dock_to_go], speed=spid)    
                        if state == exp_state.PICKANDPLACE and not is_gripper_holding():                    
                            docks_clear[dock_to_go] = False

                                                          
                for i in range(1,7):
                    
                    print("TRYING FOR CYLINDER", i)
                    
                    try:
                        cylinder_tf = tf_buffer.lookup_transform(world, f"cylinder_{i}", rospy.Time())
                    except:
                        print("CYLINDER", i, "NOT HERE")
                        if i == 6:
                            state = exp_state.GAZE_AND_BREATHE
                            des_home = home_pos_0
                            print("ALL CYLINDERS NOT HERE")
                            break
                        continue
                    
                    cylinder_pos = [cylinder_tf.transform.translation.x, cylinder_tf.transform.translation.y, cylinder_tf.transform.translation.z]     
                    cy_pos = cylinder_pos
                                
                    if is_in_exparea(cylinder_pos) and not is_in_docks(cylinder_pos) and not did_i_pick_it_up(cylinder_pos):
                        
                        available_cylinders += 1
                        
                        pick_up_pos(cylinder_pos,speed=spid) 
                
                        dock_to_go = 0                        
                        
                        for j in range(len(docks_clear)):
                            if docks_clear[j]:
                                dock_to_go = j
                                break
                                
                        if dock_to_go > 6 or dock_to_go < 1:
                            print("ALL DOCKS OCCUPIED", dock_to_go)
                            state = exp_state.GAZE_AND_BREATHE
                            des_home = home_pos
                            break
                        
                        place_object(docks_xyz[dock_to_go], speed=spid)
                        
                        if state == exp_state.PICKANDPLACE and not is_gripper_holding():              
                            docks_clear[dock_to_go] = False
                            picked_up_cylinder_poses.append(cy_pos)
                        
                        break
                    
                    
                    all_full = True
                    for k in docks_clear:
                        if k:
                            all_full = False
                            break
                    if all_full:
                        state = exp_state.GAZE_AND_BREATHE
                        des_home = home_pos
                    
                if available_cylinders < 1:
                    state = exp_state.GAZE_AND_BREATHE
                    des_home = home_pos
                    continue
            
            elif state == exp_state.GAZE_AND_BREATHE:
                        
                go_to_joint_pos(des_home, spid)
                
                breathe_and_gaze(True, True)
        
        #data_collector_process.kill()    
                                    
        stop_movement()
                
        go_to_joint_pos(home_pos, 0.5)
                
        stop_movement()
        
        time_data = False
        actuator_data = True
        
        if actuator_data:
            fig, axs = plt.subplots(2)
            axs[0].plot(joint1des, label="joint1 desired")
            axs[0].plot(joint1real, label="joint1 actual")
            axs[0].plot(joint1torq, label="joint1 torque")
            axs[0].set_title("Joint1 data at step_len=" + str(step_len_glob*1000) + "mm")
            axs[0].legend()
            axs[1].plot(joint2des, label="joint2 desired")
            axs[1].plot(joint2real, label="joint2 actual")
            axs[1].plot(joint2torq, label="joint2 torque")
            axs[1].set_title("Joint2 data at step_len=" + str(step_len_glob*1000) + "mm")
            axs[1].legend()
            
           #with open("/home/kovan4/HRI4Cobot/noise_effort_analysis/joint1des_at_steplen_" + str(step_len_glob*1000) + "mm" + ".npy", "wb") as f:
           #    np.save(f, joint1des)
           #with open("/home/kovan4/HRI4Cobot/noise_effort_analysis/joint1real_at_steplen_" + str(step_len_glob*1000) + "mm" + ".npy", "wb") as f:
           #    np.save(f, joint1real)
           #with open("/home/kovan4/HRI4Cobot/noise_effort_analysis/joint1torq_at_steplen_" + str(step_len_glob*1000) + "mm" + ".npy", "wb") as f:
           #    np.save(f, joint1torq)
           #with open("/home/kovan4/HRI4Cobot/noise_effort_analysis/joint2des_at_steplen_" + str(step_len_glob*1000) + "mm" + ".npy", "wb") as f:
           #    np.save(f, joint2des)
           #with open("/home/kovan4/HRI4Cobot/noise_effort_analysis/joint2real_at_steplen_" + str(step_len_glob*1000) + "mm" + ".npy", "wb") as f:
           #    np.save(f, joint2real)
           #with open("/home/kovan4/HRI4Cobot/noise_effort_analysis/joint2torq_at_steplen_" + str(step_len_glob*1000) + "mm" + ".npy", "wb") as f:
           #    np.save(f, joint2torq)
            
            #Don't forget to increase subplot amt
            #axs[2].plot(timestep, label="control rate in Hz")
            #axs[2].set_title("Control Rate")
            #axs[2].legend()
        
        if time_data and not actuator_data:
            plt.plot(timestep)
        
        plt.show()
                        
        rospy.sleep(1.0)    