import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np


def get_theta_matrix(theta: float) -> np.array:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def get_alpha_matrix(alpha: float) -> np.array:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1],
        ]
    )


def get_d_matrix(d: float) -> np.array:
    d_mat = np.eye(4)
    d_mat[2, 3] = d
    return d_mat


def get_a_matrix(a: float) -> np.array:
    a_mat = np.eye(4)
    a_mat[0, 3] = a
    return a_mat


def get_transformation_matrix(theta, d, a, alpha) -> np.array:
    return (
        get_theta_matrix(theta)
        @ get_d_matrix(d)
        @ get_a_matrix(a)
        @ get_alpha_matrix(alpha)
    )


def get_transformation_matrix_with_zero_rotation(x, y, z) -> np.array:
    transformation_mat = np.eye(4)
    transformation_mat[0, 3] = x
    transformation_mat[1, 3] = y
    transformation_mat[2, 3] = z
    return transformation_mat


def axis_angle_and_position_to_transformation_matrix(axis_angle_vector, positions):
    euler_angles = [
        axis_angle_vector[0] * axis_angle_vector[3],
        axis_angle_vector[1] * axis_angle_vector[3],
        axis_angle_vector[2] * axis_angle_vector[3],
    ]
    rot_mat = R.from_rotvec(euler_angles).as_matrix()

    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[0, 3] = positions[0]
    transform[1, 3] = positions[1]
    transform[2, 3] = positions[2]

    return transform
