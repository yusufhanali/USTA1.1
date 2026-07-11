import numpy as np

import ur5e_kinematic.ur5e_kinematics as ur5e_kinematics
import utilities.control_and_filters as control_and_filters

from utilities.control_and_filters import PIDController

class Breather:
    
    def __init__(self, breathe_dict, joint_positions) -> None:
        assert "control_rate" in breathe_dict.keys(), "Control rate is not added into the breahte_dict."
        self.control_rate = breathe_dict["control_rate"]

        self.breathe_vec = np.array([0.0, 1.0])  # Breathing in just x and z axis
        
        assert "f" in breathe_dict.keys(), "f function for breathing velocity profile does not exist in breathe dictionary."
        self.f = breathe_dict["f"]      

        try:
            self.freq = breathe_dict["freq"]
        except:
            self.freq = 0.25
            print(f"freq is not given in the breathe dict. It is set to {self.freq}")

        # Give index for each data point to interpolate in the main loop
        self.num_of_vel_pts = self.f.shape[0]
        self.indices = np.linspace(1, self.num_of_vel_pts, self.num_of_vel_pts)

        try:
            self.amplitude = breathe_dict["amplitude"]
        except:
            self.amplitude = 2.0
            print(f"amplitude is not given in the breathe dict. It is set to {self.amplitude}")

        self.loop_index = 0
        self.breathe_count = 0
                
        self.initial_wrist_position = ur5e_kinematics.shoulder_to_base_translation_x_z(joint_positions)
        self.difference_x_filter = control_and_filters.LinearFilter(alpha=0.1)
        self.correction_factor_x = 8
        self.correction_z = 0.0
        self.z_PID = PIDController(kp=0.20, ki=0.15, kd=0.0, dt=1/self.control_rate)

    def step_breathe(self, joint_positions, forward_movement=0.0):
        
        # interpolate velocity from f function (human data)
        velocity_magnitude = np.interp(self.loop_index, self.indices, self.f)
        
        difference_x = (self.initial_wrist_position[0] + forward_movement) - ur5e_kinematics.shoulder_to_base_translation_x_z(joint_positions)[0]
        difference_x = self.difference_x_filter.filter(difference_x)
        
        velocity_task = self.breathe_vec * velocity_magnitude * self.amplitude * self.freq
        velocity_task[1] += self.correction_z  # Apply correction in z direction
        velocity_task[0] += difference_x*self.correction_factor_x  # Apply correction in x direction
                
        jacobian = ur5e_kinematics.shoulder_to_base_jacobian(joint_positions)
        inv_jacobian = np.linalg.pinv(jacobian, rcond=1e-15)
        velocity_command = inv_jacobian @ velocity_task # Velocities just for joints 1 and 2 (shoulder and elbow)
        
        return velocity_command, velocity_task, difference_x
    
    def reset(self):
        self.loop_index = 0
        self.breathe_count = 0
    
    def step(self, joint_positions, forward_movement=0.0):
        
        velocity_command, velocity_task, corr = self.step_breathe(joint_positions, forward_movement)

        self.loop_index = self.loop_index + (self.num_of_vel_pts/self.control_rate)*self.freq 
        self.loop_index = self.loop_index % self.num_of_vel_pts
        if int(self.loop_index) == 0:
            if self.breathe_count == 0:
                self.initial_wrist_position = ur5e_kinematics.shoulder_to_base_translation_x_z(joint_positions)
            else:
                self.correction_z = self.z_PID.compute(setpoint=ur5e_kinematics.shoulder_to_base_translation_x_z(joint_positions)[1], measured_value= self.initial_wrist_position[1])
                #print(f"Breath number {self.breathe_count}, correction in z direction: {self.correction_z:.4f} m/s")
            self.breathe_count += 1
            self.loop_index = 1
            # print(f"Breathe count: {self.breathe_count}")

        return velocity_command, velocity_task, corr, self.correction_z
    
    def set_frequency(self, freq):
        self.freq = freq
        
    def set_amplitude(self, amplitude):
        self.amplitude = amplitude