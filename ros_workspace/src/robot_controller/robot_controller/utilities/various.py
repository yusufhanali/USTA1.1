import os
import time
import numpy as np

class CustomTimer():    
    def __init__(self):
        self.alarms = []
        self.trigger_times = []
        
    def register(self, time_offset, true_period, disposable, function, *args, **kwargs):
        """
        Register a function to be called at a specified period.
        
        Parameters:
            period (float): Time in seconds between calls.
            disposable (bool): If True, the function will be removed after the first call.
            function (callable): The function to call.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """
        self.alarms.append((time_offset, true_period, disposable, function, args, kwargs))
        self.trigger_times.append(time.time() + time_offset)
        
    def step(self):
        """
        Check if any registered functions should be called based on the current time.
        """
        garbage = []
        current_time = time.time()
        for i, (time_offset, true_period, disposable, function, args, kwargs) in enumerate(self.alarms):
            if current_time >= self.trigger_times[i]:
                function(*args, **kwargs)
                if disposable:
                    garbage.append(i)
                else:
                    self.trigger_times[i] += true_period
        # Remove disposable alarms
        for i in reversed(garbage):
            del self.alarms[i]
            del self.trigger_times[i]

def order_points_counter_clockwise(points):
    """
    Orders the points in counter-clockwise order.

    Parameters:
        points (array-like): List of points representing a polygon (N, 2 or 3).

    Returns:
        List of points ordered counter-clockwise.
    """
    # Convert to a NumPy array for convenience
    points = np.array(points)

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the angle of each point with respect to the centroid
    # `np.arctan2` gives angles in radians from -π to π
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points based on their angles
    sorted_indices = np.argsort(angles)

    # Return the points in the sorted order
    return points[sorted_indices]

def match_by_timestamp(tuple1, tuple2, time_diff_max):
    """
    Matches rows from two uniformly sampled timestamp arrays using direct arithmetic.
    It is assumed that for both arrays:
    - The first column is the timestamp.
    - The timestamps are uniformly sampled.
    - The timestamps are sorted in ascending order.

    Each tuple is of the form (data, period) where:
      - data is a np.array sorted by timestamp (first column is timestamp).
      - period is the constant time difference (in milliseconds) between consecutive rows.

    The function determines the lower frequency array (with a larger period) as the reference.
    For each row in the reference array, it computes the index in the other array based
    on the known period and clamps the index to the valid range.

    Parameters:
        tuple1: (np.array, float)
        tuple2: (np.array, float)
        time_diff_max: float
            Maximum allowed timestamp difference (in ms) for a valid match.

    Returns:
        A tuple (indices1, indices2) where indices correspond to the rows in the original
        arrays that are considered a match.
    """
    arr1, period1 = tuple1
    arr2, period2 = tuple2

    # Choose the array with the larger period as the reference (lower frequency)
    if period1 >= period2:
        lower_arr, lower_period, lower_label = arr1, period1, 'tuple1'
        higher_arr, higher_period = arr2, period2
    else:
        lower_arr, lower_period, lower_label = arr2, period2, 'tuple2'
        higher_arr, higher_period = arr1, period1

    indices_lower = []
    indices_higher = []
    t0_higher = higher_arr[0, 0]  # first timestamp of the higher frequency array

    for i, row in enumerate(lower_arr):
        t_lower = row[0]
        # Compute the expected index in the higher frequency array
        j = int(round((t_lower - t0_higher) / higher_period))
        # Clamp j to valid range [0, len(higher_arr)-1]
        j = max(0, min(j, len(higher_arr) - 1))
        candidate_ts = higher_arr[j, 0]
        if abs(t_lower - candidate_ts) <= time_diff_max:
            indices_lower.append(i)
            indices_higher.append(j)

    # Return indices in the order corresponding to the original input tuples.
    if lower_label == 'tuple1':
        return np.array(indices_lower), np.array(indices_higher)
    else:
        return np.array(indices_higher), np.array(indices_lower)