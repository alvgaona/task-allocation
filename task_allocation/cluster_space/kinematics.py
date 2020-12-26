import numpy as np
import scipy.linalg


def compute_forward_pose(r):
    x1, y1, yaw1, x2, y2, yaw2 = r
    
    a = x1 - x2
    b = y1 - y2
    
    x_c = (x1 + x2) / 2
    y_c = (y1 + y2) / 2
    d = (1 / 2) * np.sqrt(a ** 2 + b ** 2)
    
    yaw_c = np.arctan2(a, b)
    
    phi1 = yaw1 - yaw_c
    phi2 = yaw2 - yaw_c
    
    return x_c, y_c, d, yaw_c, phi1, phi2,


def compute_inverse_pose(c):
    x_c, y_c, d, yaw_c, phi1, phi2 = c
    
    x1 = x_c + d * np.sin(yaw_c)
    x2 = x_c - d * np.sin(yaw_c)
    
    y1 = y_c + d * np.cos(yaw_c)
    y2 = y_c - d * np.cos(yaw_c)
    
    yaw1 = yaw_c + phi1
    yaw2 = yaw_c + phi2
    
    return x1, y1, yaw1, x2, y2, yaw2


def compute_jacobian_matrix(robots_pose):
    r"""Compute jacobian matrix based on the forward kinematics relationship
    
    Arguments:
        robots_pose: robots position vector defined as (x1, y1, yaw1, x2, y2, yaw2)
    Returns:
        J: jacobian matrix
    """
    
    x1, y1, yaw1, x2, y2, yaw2 = robots_pose
    
    a = x1 - x2
    b = y1 - y2
    
    A = 2 * np.sqrt(a ** 2 + b ** 2)
    B = a ** 2 + b ** 2
    
    J = np.zeros((6, 6))
    
    J[0, :] = np.asarray([1 / 2, 1 / 2, 0, 0, 0, 0])
    J[1, :] = np.asarray([0, 1 / 2, 0, 0, 1 / 2, 0])
    J[2, :] = (1 / A) * np.asarray([x1 - x2, y1 - y2, 0, -x1 + x2, -y1 + y2, 0])
    J[3, :] = (1 / B) * np.asarray([-y1 + y2, x1 - x2, 0, y1 - y2, -x1 + x2, 0])
    J[4, :] = (1 / B) * np.asarray([y1 - y2, -x1 + x2, 1, -y1 + y2, x1 - x2, 0])
    J[5, :] = (1 / B) * np.asarray([y1 - y2, -x1 + x2, 0, -y1 + y2, x1 - x2, 1])
    
    return J


def compute_inverse_jacobian_matrix(cluster_pose):
    r"""Compute jacobian matrix based on the inverse kinematics relationship

    Arguments:
        cluster_pose: cluster position vector defined as (x_c, y_c, d, yaw_c, phi1, phi2)
    Returns:
        Jinv: jacobian matrix
    """
    
    x_c, y_c, d, yaw_c, phi1, phi2 = cluster_pose
    
    Jinv = np.zeros((6, 6))
    
    Jinv[0, :] = np.asarray([1, 0, np.sin(yaw_c), d * np.cos(yaw_c), 0, 0])
    Jinv[1, :] = np.asarray([0, 1, np.cos(yaw_c), -d * np.sin(yaw_c), 0, 0])
    Jinv[2, :] = np.asarray([1, 0, -np.sin(yaw_c), -d * np.cos(yaw_c), 0, 0])
    Jinv[3, :] = np.asarray([0, 1, -np.cos(yaw_c), d * np.sin(yaw_c), 0, 0])
    Jinv[4, :] = np.asarray([0, 0, 0, 1, 1, 0])
    Jinv[5, :] = np.asarray([0, 0, 0, 1, 0, 1])
    
    return Jinv


def compute_dot_jacobian_matrix(r, rdot):
    r"""Compute derivative of the jacobian matrix (forward kinematics relationship) with respect to time t

        Arguments:
            r: position vector of the robots defined as (x1, x2, yaw1, x1, x2, yaw2)
            rdot: velocities vector of the robots defined as (x1d, x2d, yaw2d, x1d, x2d, yaw2d)
        Returns:
            Jdot: jacobian matrix derivative with respect to time t for a given robot positions and velocities
        """
    
    x1, y1, _, x2, y2, _ = r
    x1d, y1d, _, x2d, y2d, _ = rdot
    
    B = (x1 - x2) ** 2 + (y1 - y2) ** 2
    
    Jdot = np.zeros((6, 6))
    
    Jdot[2, :] = (1 / 2) * np.power(B, 3 / 2) * np.asarray([
        (y1 - y2) * ((y1 - y2) * (x1d - x2d) - (x1 - x2) * (y1d - y2d)),
        (x1 - x2) * (-(y1 - y2) * (x1d - x2d) + (x1 - x2) * (y1d - y2d)),
        0,
        (y1 - y2) * (-(y1 - y2) * (x1d - x2d) + (x1 - x2) * (y1d - y2d)),
        -(x1 - x2) * (-(y1 - y2) * (x1d - x2d) + (x1 - x2) * (y1d - y2d)),
        0
    ])
    
    Jdot[3, :] = (1 / np.power(B, 2)) * np.asarray([
        2 * (x1 - x2) * (y1 - y2) * x1d - 2 * (x1 - x2) * (y1 - y2) * x2d - (x1 - x2 + y1 - y2) * (x1 - x2 - y1 + y2) *
        (y1d - y2d),
        B * (x1d - x2d) - (x1 - x2) * (2 * (x1 - x2) * (x1d - x2d) + 2 * (y1 - y2) * (y1d - y2d)),
        0,
        -2 * (x1 - x2) * (y1 - y2) * x1d + 2 * (x1 - x2) * (y1 - y2) * x2d + (x1 - x2 + y1 - y2) * (x1 - x2 - y1 + y2) *
        (y1d - y2d),
        B * (-x1d + x2d) - (-x1 + x2) * (2 * (x1 - x2) * (x1d - x2d) + 2 * (y1 - y2) * (y1d - y2d)),
        0
    ])
    
    Jdot[4, :] = (1 / np.power(B, 2)) * np.asarray([
        -2 * (x1 - x2) * (y1 - y2) * x1d + 2 * (x1 - x2) * (y1 - y2) * x2d + (x1 - x2 + y1 - y2) * (x1 - x2 - y1 + y2) *
        (y1d - y2d),
        B * (-x1d + x2d) - (-x1 + x2) * (2 * (x1 - x2) * (x1d - x2d) + 2 * (y1 - y2) * (y1d - y2d)),
        0,
        2 * (x1 - x2) * (y1 - y2) * x1d - 2 * (x1 - x2) * (y1 - y2) * x2d - (x1 - x2 + y1 - y2) * (x1 - x2 - y1 + y2) *
        (y1d - y2d),
        B * (x1d - x2d) - (x1 - x2) * (2 * (x1 - x2) * (x1d - x2d) + 2 * (y1 - y2) * (y1d - y2d)),
        0
    ])
    
    Jdot[5, :] = (1 / np.power(B, 2)) * np.asarray([
        -2 * (x1 - x2) * (y1 - y2) * x1d + 2 * (x1 - x2) * (y1 - y2) * x2d + (x1 - x2 + y1 - y2) * (x1 - x2 - y1 + y2) *
        (y1d - y2d),
        B * (-x1d + x2d) - (-x1 + x2) * (2 * (x1 - x2) * (x1d - x2d) + 2 * (y1 - y2) * (y1d - y2d)),
        0,
        2 * (x1 - x2) * (y1 - y2) * x1d - 2 * (x1 - x2) * (y1 - y2) * x2d - (x1 - x2 + y1 - y2) * (x1 - x2 - y1 + y2) *
        (y1d - y2d),
        B * (x1d - x2d) - (x1 - x2) * (2 * (x1 - x2) * (x1d - x2d) + 2 * (y1 - y2) * (y1d - y2d)),
        0
    ])
    
    return Jdot


def robot_cluster(m, Izz, bx, by, btheta):
    Ai = np.asmatrix([
        [m, 0, 0],
        [0, m, 0],
        [0, 0, Izz]
    ])
    
    Bi = np.asarray([
        [bx, 0, 0],
        [0, by, 0],
        [0, 0, btheta]
    ])
    
    A = scipy.linalg.block_diag(Ai, Ai)
    B = scipy.linalg.block_diag(Bi, Bi)
    
    return A, B


def cluster_state_space(m, Izz, bx, by, btheta):
    Ai = np.asmatrix([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, -bx / m, 0, 0],
        [0, 0, 0, 0, -by / m, 0],
        [0, 0, 0, 0, 0, -btheta / Izz]
    ])
    
    Bi = np.asmatrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1 / m, 0, 0],
        [0, 1 / m, 0],
        [0, 0, 1 / Izz]
    ])
    
    A = scipy.linalg.block_diag(Ai, Ai)
    B = scipy.linalg.block_diag(Bi, Bi)
    
    C = np.eye(12)
    D = np.zeros((12, 6))
    
    return A, B, C, D
