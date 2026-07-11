import numpy as np

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

def cubic_spline(start_pos=None, end_pos=None, start_derivative=None, end_derivative=None):

    if start_pos is None or end_pos is None:
        raise ValueError("Start and end positions must be provided.")
    if start_derivative is None:
        raise ValueError("Start derivative must be provided.")
    if end_derivative is None:
        raise ValueError("End derivative must be provided.")
    
    p0 = start_pos
    p1 = end_pos
    r0 = start_derivative / np.linalg.norm(start_derivative)
    r1 = end_derivative / np.linalg.norm(end_derivative)
    
    p0x, p0y, p0z = p0[0], p0[1], p0[2]
    p1x, p1y, p1z = p1[0], p1[1], p1[2]
    r0x, r0y, r0z = r0[0], r0[1], r0[2]
    r1x, r1y, r1z = r1[0], r1[1], r1[2]
    
    cubic_spline_inverse_matrix = np.array([[2, -2, 1, 1],
                                            [-3, 3, -2, -1],
                                            [0, 0, 1, 0],
                                            [1, 0, 0, 0]])

    
    x_coefficients = cubic_spline_inverse_matrix @ np.array([p0x, p1x, r0x, r1x])
    y_coefficients = cubic_spline_inverse_matrix @ np.array([p0y, p1y, r0y, r1y])
    z_coefficients = cubic_spline_inverse_matrix @ np.array([p0z, p1z, r0z, r1z])
    
    return x_coefficients, y_coefficients, z_coefficients

def three_point_cubic_spline(start_pos=None, start_derivative=None, mid0_pos=None, mid1_pos=None, end_derivative=None, end_pos=None):
    
    if start_pos is None or end_pos is None:
        raise ValueError("Start and end positions must be provided.")
    if mid0_pos is None and start_derivative is None:
        raise ValueError("Either mid0 position or start derivative must be provided.")
    if mid1_pos is None and end_derivative is None:
        raise ValueError("Either mid1 position or end derivative must be provided.")
    
    p0 = start_pos
    p1 = end_pos    
    if mid0_pos is None:
        mid0_pos = start_pos + start_derivative
    if mid1_pos is None:
        mid1_pos = end_pos + end_derivative
    r0 = (mid0_pos - start_pos) / np.linalg.norm(mid0_pos - start_pos)
    r1 = (end_pos - mid1_pos) / np.linalg.norm(end_pos - mid1_pos)
    
    p0x, p0y, p0z = p0[0], p0[1], p0[2]
    p1x, p1y, p1z = p1[0], p1[1], p1[2]
    r0x, r0y, r0z = r0[0], r0[1], r0[2]
    r1x, r1y, r1z = r1[0], r1[1], r1[2]
    
    x_constraints = np.array([p0x, p1x, r0x, r1x])
    y_constraints = np.array([p0y, p1y, r0y, r1y])
    z_constraints = np.array([p0z, p1z, r0z, r1z])
    
    cubic_spline_inverse_matrix = np.array([[2, -2, 1, 1],
                                            [-3, 3, -2, -1],
                                            [0, 0, 1, 0],
                                            [1, 0, 0, 0]])

    
    x_coefficients = cubic_spline_inverse_matrix @ x_constraints
    y_coefficients = cubic_spline_inverse_matrix @ y_constraints
    z_coefficients = cubic_spline_inverse_matrix @ z_constraints
    
    return x_coefficients, y_coefficients, z_coefficients

def angular_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi