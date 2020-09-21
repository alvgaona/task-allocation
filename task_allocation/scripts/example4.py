#!/usr/local/bin/python3

r""" Example 4 from "An Optimal Task Allocation Strategy", G. Notomista et al.

    Given 3 goals and 3 tasks, each of which consists of reaching 1 specific goal,
    a robot is asked to perform those 3 tasks at the same time. However, this is
    impossible for the robot to do, so the algorithm makes the robot stay at equal
    distances from each goal.
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


def plot_robot_trajectory(x, p):
    plt.plot(x[:, 0], x[:, 1], linestyle='dashed')
    plt.plot(x[-1, 0], x[-1, 1], marker='>')
    
    plt.plot(p[0, 0], p[0, 1], marker='o')
    plt.text(p[0, 0] + 0.05, p[0, 1] + 0.05, '$p_1$', fontsize=12)
    plt.plot(p[1, 0], p[1, 1], marker='o')
    plt.text(p[1, 0] + 0.05, p[1, 1] - 0.05, '$p_2$', fontsize=12)
    plt.plot(p[2, 0], p[2, 1], marker='o')
    plt.text(p[2, 0] - 0.1, p[2, 1] - 0.1, '$p_3$', fontsize=12)
    
    plt.title('Single-integrator Robot Trajectory')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.legend(['Trajectory'])
    plt.show()


def task_allocation(x, p):
    M = 3  # Number of Tasks
    
    # Parameters
    L = 1
    DELTA_MAX = 3.5
    
    u = cp.Variable(q)
    alpha = cp.Variable(M)
    delta = cp.Variable(M)
    
    objective = cp.sum_squares(u) + L * cp.sum_squares(delta)
    
    # Constraints
    constraints = [
        -2 * (x - p[0, :]).T @ u >= cp.sum_squares((x - p[0, :])) - delta[0],
        -2 * (x - p[1, :]).T @ u >= cp.sum_squares((x - p[1, :])) - delta[1],
        -2 * (x - p[2, :]).T @ u >= cp.sum_squares((x - p[2, :])) - delta[2],
        cp.norm_inf(delta) <= DELTA_MAX,
    ]
    
    obj = cp.Minimize(objective)
    prob = cp.Problem(obj, constraints)
    
    prob.solve()
    
    return u, alpha, delta


def run(p):
    T = 50  # Number of time steps
    
    x = np.zeros((T, d))
    
    for t in range(T):
        u, alpha, delta = task_allocation(x[t, :], p)
        
        if t <= T - 2:
            x[t + 1, :] = x[t, :] + u.value
    
    return x


if __name__ == '__main__':
    d = 2  # State vector dimension
    q = 2  # Control vector dimension
    
    goals = np.asarray([[2, 2], [1, 2], [2, 1]])
    
    x = run(goals)
    
    plot_robot_trajectory(x, goals)
