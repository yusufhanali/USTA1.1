import numpy as np


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        self.integral = 0.0
        self.previous_error = 0.0
        
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        
    def change_params(self, kp=None, ki=None, kd=None, dt=None):
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
        if dt is not None:
            self.dt = dt

    def compute(self, setpoint, measured_value):
        error = measured_value - setpoint
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt if self.dt > 0 else 0.0
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class LinearFilter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.filtered_value = 0.0
        
    def set_value(self, value):
        self.filtered_value = value

    def filter(self, x):
        self.filtered_value = self.alpha * x + (1 - self.alpha) * self.filtered_value
        return self.filtered_value

class VectorLinearFilter:
    def __init__(self, alpha=0.1, dimension=3):
        self.alpha = alpha
        self.dimension = dimension
        self.reset()

    def reset(self):
        self.filtered_value = np.zeros(self.dimension)

    def filter(self, x):
        self.filtered_value = self.alpha * np.array(x) + (1 - self.alpha) * self.filtered_value
        return self.filtered_value

class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = []
        self.sum = 0.0
        
    def reset(self):
        self.values = []
        self.sum = 0.0
        
    def filter(self, x):
        self.values.append(x)
        self.sum += x
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return self.sum / len(self.values)

class LinearScaler:
    def __init__(self, min_input=0.0, max_input=1.0, min_output=0.0, max_output=1.0):
        self.min_input = min_input
        self.max_input = max_input
        self.min_output = min_output
        self.max_output = max_output
        
        self.input_range = max_input - min_input
        self.output_range = max_output - min_output
        
    def scale(self, value):
        if value < self.min_input:
            return self.min_output
        elif value > self.max_input:
            return self.max_output
        else:
            return ((value - self.min_input) / (self.input_range)) * (self.output_range) + self.min_output
    
class WedgeShapedScaler:
    def __init__(self, min_input=0.0, max_input=1.0, entry_distance=0.2, exit_distance=0.2, peak_output=1.0):
        self.min_input = min_input
        self.max_input = max_input
        self.entry_distance = entry_distance
        self.exit_distance = exit_distance
        self.peak_output = peak_output
        self.plateau_start = min_input + entry_distance
        self.plateau_end = max_input - exit_distance

    def scale(self, value):
        if value < self.min_input:
            return 0.0
        elif value > self.max_input:
            return 0.0
        elif value <= self.plateau_start:
            return ((value - self.min_input) / (self.plateau_start - self.min_input)) * self.peak_output
        elif value <= self.plateau_end:
            return self.peak_output
        else:
            return ((self.max_input - value) / (self.max_input - self.plateau_end)) * self.peak_output
        
def visualize_scaler(scaler, num_points=100):
    import matplotlib.pyplot as plt

    inputs = np.linspace(scaler.min_input - 0.1, scaler.max_input + 0.1, num_points)
    outputs = [scaler.scale(x) for x in inputs]

    plt.plot(inputs, outputs)
    plt.title('Scaler Visualization')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    scaler = WedgeShapedScaler(min_input=0.0, max_input=1.0, entry_distance=0.2, exit_distance=0.2, peak_output=1.0)
    visualize_scaler(scaler)