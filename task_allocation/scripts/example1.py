#!/usr/local/bin/python3

r""" Example 1 from "Adaptive Task Allocation for Heterogeneous Multi-Robot Teams with
    Evolving and Unknown Robot Capabilities.", Yousef Emam et al.
    
    It consists of a fleet of two heterogeneous robots asked to execute 2 tasks effectively.
    Each robot has its own specialization matrix defined.
    
    Specialization matrices
    
    .. math::
       S_1 = diag([1, 0])
       S_2 = diag([1, 1])
    
    Another parameter defined is the Global Task Specification defining the fraction of robots
    desired to execute each task.
    
    Global Task Specification
    
    .. math::
       \pi^* = [\frac{1}{2} \frac{1}{2}]
"""

import cvxpy as cp
import numpy as np

from numpy.linalg import pinv

np.set_printoptions(suppress=True)

if __name__ == '__main__':
    M = 2  # Number of Tasks
    N = 2  # Number of Robots

    # Specialization matrices
    S = [
        np.diag([1, 0]),
        np.diag([1, 1]),
    ]

    P = np.hstack((
        np.matmul(S[0], pinv(S[0])),
        np.matmul(S[1], pinv(S[1])),
    ))

    # Global Task Specification
    global_task_specification = np.asarray([[0.5], [0.5]])

    # Parameters
    C = 1
    L = 5
    KAPPA = 1 / 10
    DELTA_MAX = 10

    # Control variables
    u = cp.Variable((N, 2))

    # Slack variables
    delta = cp.Variable((N, 2))
    alpha = cp.Variable((N * M, 1))

    # Objective
    objective = C * cp.norm2(global_task_specification - 1 / N * P @ alpha) ** 2 + \
        L * (cp.norm2(u[0, :]) ** 2 + cp.quad_form(delta[0, :], S[0]) +
             cp.norm2(u[1, :]) ** 2 + cp.quad_form(delta[1, :], S[1]))

    # Constraints
    constraints = [
        np.ones((1, M)) @ alpha[0:2] == 1.0,  # (14d)
        np.ones((1, M)) @ alpha[2:5] == 1.0,  # (14d)
        alpha <= 1,  # (14f)
        alpha >= 0  # (14f)
    ]

    obj = cp.Minimize(objective)
    prob = cp.Problem(obj, constraints)

    prob.solve()

    print("Multi-robot system task priorities: \n", alpha.value)
    print("Heterogeneous Task Specification: \n", (1 / N * P @ alpha).value)

    print("Optimal control input: \n", u.value)
    print("Slack variables: \n", delta.value)
