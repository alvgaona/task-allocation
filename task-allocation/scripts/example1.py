#!/usr/local/bin/python3

import cvxpy as cp
import numpy as np

from numpy.linalg import pinv

np.set_printoptions(suppress=True)

if __name__ == '__main__':
    """
    This script implements Example 1 from "Adaptive Task Allocation for Heterogeneous Multi-Robot Teams
    with Evolving and Unknown Robot Capabilities.", Yousef Emam et al
    
    It consists of a fleet of two heterogeneous robots asked to execute 2 tasks effectively.
    The specialization matrices for each robot are:
    
    S1 = diag([1,0])
    S2 = diag([1,1])
    
    Also, the desired global task specification is:
    
    pi* = [1/2 1/2]
    
    Meaning that only one robot is needed to execute each task.
    
    It's worth mentioning that even though the slack variables and control input variables are defined
    it will not impact the task allocation variable for the robots since there are no constraints related to the
    slack and control variables.
    
    After running the optimization problem the heterogeneous task specification will match the desired one.
    """
    
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
    
    # CVXPY parameters
    C = 1  # Scaling constant
    L = 5  # Scaling constant
    
    # Prioritization parameters
    kappa = 1 / 10
    delta_max = 10
    Pr = np.asarray([[-kappa, 1, 0], [0, -kappa]])  # Prioritization Matrix
    
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
