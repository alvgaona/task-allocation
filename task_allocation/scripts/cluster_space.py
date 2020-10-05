#!/usr/local/bin/python3

r""" Cluster Space Formulation based on "Dynamic Control of Mobile Multirobot Systems:
    The Cluster Space Formulation", I. Mas et al.

    This example shows how to allocate tasks on a meta-agent comprised of N agents or robots.
    In this particular example N equals 3.
   
    Each robot is considered holonomic (moves in every direction) and planar.
    It can move in x and y directions, and rotate around the z axis (described by its yaw angle).
   
    The position vector of a robot i is described as:
   
    .. math::
        r_i = \left[x_i, y_i, \theta_i\right]^T
        
        i = 1, 2, 3
        
    Whereas, the cluster is described as:
   
    .. math::
        c_i = \left[x_c, y_c, \theta_c, \phi_1, \phi_2, \phi_3, p, q, \beta\right]^T

   The dynamics of the cluster is represented in Eq. (10).
"""

import cvxpy as cp
import numpy as np

from numpy.linalg import pinv

np.set_printoptions(suppress=True)

if __name__ == '__main__':
    N = 3  # Number of robots
    M = 3  # Number of tasks
    DOF = 3  # Number of Degrees of Freedom (DOF)
    
    A1 = np.diag([10, 10, 1])
    b1 = np.diag([1, 1, 1])
    A2 = np.diag([5, 5, 1])
    b2 = np.diag([5, 5, 5])
    A3 = np.diag([1, 1, 1])
    b3 = np.diag([10, 10, 10])
    
    A = np.diag([A1, A2, A3])
    b = np.vstack((b1, b2, b3))
    
    r = np.ones((N*DOF, 1))