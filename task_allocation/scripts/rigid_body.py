#!/usr/local/bin/python3

import scipy.signal

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import pinv

np.set_printoptions(suppress=True)

d = 3  # State vector dimension
q = 3  # Control vector dimension


def plot_state(x, t):
    plt.figure()
    plt.plot(t, x[:, 0], linestyle='dashed')
    plt.plot(t, x[:, 1], linestyle='dashed')
    plt.plot(t, x[:, 2], linestyle='dashed')

    plt.title('Planar Robot Position')
    plt.xlabel('Timestamp')
    plt.ylabel('Position')
    plt.legend(['Position X', 'Position Y', 'Orientation'])
    plt.show()


def plot_robot_trajectory(x, p):
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], linestyle='dashed')
    plt.plot(x[-1, 0], x[-1, 1], marker='>')

    plt.plot(p[0, 0], p[0, 1], marker='o')
    plt.text(p[0, 0] + 0.05, p[0, 1] + 0.05, '$p_1$', fontsize=12)
    plt.plot(p[1, 0], p[1, 1], marker='o')
    plt.text(p[1, 0] + 0.05, p[1, 1] - 0.05, '$p_2$', fontsize=12)
    plt.plot(p[2, 0], p[2, 1], marker='o')
    plt.text(p[2, 0] - 0.1, p[2, 1] - 0.1, '$p_3$', fontsize=12)

    plt.title('Planar Robot Trajectory')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.legend(['Trajectory'])
    plt.show()


def task_allocation(x, p):
    M = 3  # Number of Tasks
    N = 1  # Number of Robots

    # Specialization matrices
    S = [
        np.diag([1, 1, 1]),
    ]

    P = np.hstack((
        np.matmul(S[0], pinv(S[0])),
    ))

    # Global Task Specification
    global_task_specification = np.asarray([[0.5], [0.5], [0.5]])

    # Parameters
    C = 1
    L = 1
    DELTA_MAX = 10
    KAPPA = 10

    u = cp.Variable(q)
    alpha = cp.Variable(M)
    delta = cp.Variable(M)

    objective = C * cp.sum_squares(global_task_specification - cp.reshape(1 / N * P @ alpha, (M, 1))) + cp.sum_squares(
        u) + L * cp.sum_squares(delta)

    # Constraints
    constraints = [
        -2 * (x - p[0, :]).T @ u >= cp.log(cp.sum_squares((x - p[0, :]))) - delta[0],
        -2 * (x - p[1, :]).T @ u >= cp.log(cp.sum_squares((x - p[1, :]))) - delta[1],
        -2 * (x - p[2, :]).T @ u >= cp.log(cp.sum_squares((x - p[2, :]))) - delta[2],
        cp.norm_inf(delta) <= DELTA_MAX,
        np.ones((1, M)) @ alpha == 1.0,
        delta[2] <= KAPPA * (delta[0] - DELTA_MAX * (1 - alpha[0])),
        delta[2] <= KAPPA * (delta[1] - DELTA_MAX * (1 - alpha[1])),
        alpha <= 1,
        alpha >= 0,
    ]

    obj = cp.Minimize(objective)
    prob = cp.Problem(obj, constraints)

    prob.solve()

    return u, alpha, delta


def main():
    p = np.asarray([[1, 1, np.pi / 2], [2, 2, np.pi / 4], [0, 2, np.pi / 6]])
    
    T = 1000
    
    timesteps, dt = np.linspace(0, 100, T, retstep=True)

    Izz = 1
    m = 10
    bx = 5
    by = 5
    btheta = 1

    A = np.asmatrix([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, -bx / m, 0, 0],
        [0, 0, 0, 0, -by / m, 0],
        [0, 0, 0, 0, 0, -btheta / Izz]
    ])

    B = np.asmatrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1 / m, 0, 0],
        [0, 1 / m, 0],
        [0, 0, 1 / Izz]
    ])

    C = np.eye(6)
    D = np.zeros((6, 3))

    sys = scipy.signal.cont2discrete((A, B, C, D), dt, method='foh')

    Ad = sys[0]
    Bd = sys[1]

    r = np.zeros((T, 6))

    for it, _ in enumerate(timesteps):
        x = r[it, 0:3]

        u, alpha, delta = task_allocation(x, p)

        if it <= T - 2:
            r[it + 1, :] = Ad.dot(r[it, :]) + Bd.dot(u.value)

    plot_robot_trajectory(r, p)
    plot_state(r, timesteps)


if __name__ == '__main__':
    main()
