import numpy as np
from scipy.spatial.transform import Rotation as R    

def rot_x_matrix(theta):
    """
    Create a rotation matrix for a rotation about the X-axis.

    Parameters:
        theta (float): Rotation angle in radians.

    Returns:
        Rotation matrix as a np array of shape (3, 3).
    """
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[1, 0, 0],
                                 [0, c, -s],
                                 [0, s, c]])
    return rotation_matrix

def rot_y_matrix(theta):
    """
    Create a rotation matrix for a rotation about the Y-axis.

    Parameters:
        theta (float): Rotation angle in radians.

    Returns:
        Rotation matrix as a np array of shape (3, 3).
    """
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, 0, s],
                                 [0, 1, 0],
                                 [-s, 0, c]])
    return rotation_matrix

def rot_z_matrix(theta):
    """
    Create a rotation matrix for a rotation about the Z-axis.

    Parameters:
        theta (float): Rotation angle in radians.

    Returns:
        Rotation matrix as a np array of shape (3, 3).
    """
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s, 0],
                                 [s, c, 0],
                                 [0, 0, 1]])
    return rotation_matrix

def rot_x_homogeneous_matrix(theta):
    """
    Create a homogeneous transformation matrix for a rotation about the X-axis.

    Parameters:
        theta (float): Rotation angle in radians.

    Returns:
        Homogeneous transformation matrix as a np array of shape (4, 4).
    """
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[1, 0, 0],
                                 [0, c, -s],
                                 [0, s, c]])
    
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    
    return homogeneous_matrix

def rot_y_homogeneous_matrix(theta):
    """
    Create a homogeneous transformation matrix for a rotation about the Y-axis.

    Parameters:
        theta (float): Rotation angle in radians.

    Returns:
        Homogeneous transformation matrix as a np array of shape (4, 4).
    """
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, 0, s],
                                 [0, 1, 0],
                                 [-s, 0, c]])
    
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    
    return homogeneous_matrix

def rot_z_homogeneous_matrix(theta):
    """
    Create a homogeneous transformation matrix for a rotation about the Z-axis.

    Parameters:
        theta (float): Rotation angle in radians.

    Returns:
        Homogeneous transformation matrix as a np array of shape (4, 4).
    """
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s, 0],
                                 [s, c, 0],
                                 [0, 0, 1]])
    
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    
    return homogeneous_matrix

def translation_homogeneous_matrix(translation_vector):
    """
    Create a homogeneous transformation matrix for a translation.

    Parameters:
        translation_vector (array-like): Translation vector in the form (x, y, z).

    Returns:
        Homogeneous transformation matrix as a np array of shape (4, 4).
    """
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, 3] = translation_vector
    
    return homogeneous_matrix


def quaternion_to_rotation_matrix(quaternion):
    """
    Converts a quaternion into a rotation matrix.

    Parameters:
        quaternion (array-like): Quaternion in the form (x, y, z, w).

    Returns:
        Rotation matrix as a np array of shape (3, 3).
    """
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix

def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Converts a rotation matrix into a quaternion.

    Parameters:
        rotation_matrix (array-like): Rotation matrix of shape (3, 3).

    Returns:
        Quaternion as a np array in the form (x, y, z, w).
    """
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    return quaternion

def rotation_matrix_to_homogeneous_matrix(rotation_matrix, translation=np.array([0.0, 0.0, 0.0])):
    """
    Converts a rotation matrix and translation vector into a homogeneous transformation matrix.

    Parameters:
        rotation_matrix (array-like): Rotation matrix of shape (3, 3).
        translation (array-like): Translation vector in the form (x, y, z)
    Returns:
        Homogeneous transformation matrix np array of shape (4, 4).
    """
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation
    return homogeneous_matrix

def rotation_matrix_to_rotvec(rotation_matrix):
    """
    Converts a rotation matrix into a rotation vector.

    Parameters:
        rotation_matrix (array-like): Rotation matrix of shape (3, 3).

    Returns:
        Rotation vector as a np array of shape (3,).
    """
    rotation = R.from_matrix(rotation_matrix)
    rotvec = rotation.as_rotvec()
    return rotvec

def quaternion_to_homogeneous_matrix(quaternion, translation=np.array([0.0, 0.0, 0.0])):
    """
    Converts a quaternion and translation vector into a homogeneous transformation matrix.

    Parameters:
        quaternion (array-like): Quaternion in the form (x, y, z, w).
        translation (array-like): Translation vector in the form (x, y, z)

    Returns:
        Homogeneous transformation matrix np array of shape (4, 4).
    """
    # Convert quaternion to rotation matrix
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()

    # Create the homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation

    return homogeneous_matrix

def rotvec_to_quaternion(rotvec):
    """
    Converts a rotation vector into a quaternion.

    Parameters:
        rotvec (array-like): Rotation vector of shape (3,).

    Returns:
        Quaternion as a np array in the form (x, y, z, w).
    """
    rotation = R.from_rotvec(rotvec)
    quaternion = rotation.as_quat()
    return quaternion

def quaternion_to_rotvec(quaternion):
    """
    Converts a quaternion into a rotation vector.

    Parameters:
        quaternion (array-like): Quaternion in the form (x, y, z, w).

    Returns:
        Rotation vector as a np array of shape (3,).
    """
    rotation = R.from_quat(quaternion)
    rotvec = rotation.as_rotvec()
    return rotvec

def transform_to_matrix(transform):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    
    translation_vector = np.array([translation.x, translation.y, translation.z])
    rotation_quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
    
    rotation_matrix = quaternion_to_rotation_matrix(rotation_quaternion)
    
    homogeneous_matrix = rotation_matrix_to_homogeneous_matrix(rotation_matrix, translation_vector)
    
    return homogeneous_matrix

def reverse_homogeneous_matrix(matrix):
    new_matrix = np.eye(4)
    new_matrix[0:3, 0:3] = np.transpose(matrix[0:3, 0:3])
    new_matrix[0:3, 3] = -1 * np.transpose(matrix[0:3, 0:3]) @ matrix[0:3, 3]
    return new_matrix

def tf_transform_to_homogeneous_matrix(tf_transform):
    """
    Converts a ROS2 Transform message to a homogeneous transformation matrix.

    Parameters:
        tf_transform (geometry_msgs.msg.Transform): ROS2 Transform message.

    Returns:
        Homogeneous transformation matrix as a np array of shape (4, 4).
    """
    translation = tf_transform.translation
    rotation = tf_transform.rotation

    translation_vector = np.array([translation.x, translation.y, translation.z])
    rotation_quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w])

    rotation_matrix = quaternion_to_rotation_matrix(rotation_quaternion)
    homogeneous_matrix = rotation_matrix_to_homogeneous_matrix(rotation_matrix, translation_vector)

    return homogeneous_matrix

def angle_between_vectors(v1, v2):
    """
    Calculate the angle between two vectors in radians.

    Parameters:
        v1 (array-like): First vector.
        v2 (array-like): Second vector.

    Returns:
        Angle between the vectors in radians.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)