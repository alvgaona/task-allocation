#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from task_allocation.cluster_space.kinematics import robot_cluster, cluster_state_space, compute_jacobian_matrix, \
    compute_inverse_jacobian_matrix, compute_forward_pose, compute_dot_jacobian_matrix
from task_allocation.cluster_space.parameters import DOF
from task_allocation.cluster_space.optimization import compute_task_allocation

np.set_printoptions(suppress=True)


def plot_cluster_state(x):
    plt.figure()
    plt.plot(x[:, 0], linestyle='dashed')
    plt.plot(x[:, 1], linestyle='dashed')
    plt.plot(x[:, 2], linestyle='dashed')
    plt.plot(x[:, 3], linestyle='dashed')
    plt.plot(x[:, 4], linestyle='dashed')
    plt.plot(x[:, 5], linestyle='dashed')
    
    plt.title('Cluster Position')
    plt.xlabel('Timestamp')
    plt.ylabel('Position')
    plt.legend(['Position X', 'Position Y', 'Orientation', 'Distance', 'Angle1', 'Angle2'])
    plt.show()


def plot_robot_state(x):
    plt.figure()
    plt.plot(x[:, 0], linestyle='dashed')
    plt.plot(x[:, 1], linestyle='dashed')
    plt.plot(x[:, 2], linestyle='dashed')
    
    plt.title('Robot Position')
    plt.xlabel('Timestamp')
    plt.ylabel('Position')
    plt.legend(['Position X', 'Position Y', 'Orientation'])
    plt.show()


def plot_robots_trajectory(c, r, p):
    plt.figure()
    plt.plot(c[:, 0], c[:, 1], linestyle='dashed', color='blue')
    plt.plot(r[:, 0], r[:, 1], linestyle='dashed', color='magenta')
    plt.plot(r[:, 3], r[:, 4], linestyle='dashed', color='red')
    
    plt.plot(p[0, 0], p[0, 1], marker='o')
    plt.text(p[0, 0] + 0.05, p[0, 1] + 0.05, '$p_1$', fontsize=12)
    plt.plot(p[1, 0], p[1, 1], marker='o')
    plt.text(p[1, 0] + 0.05, p[1, 1] - 0.05, '$p_2$', fontsize=12)
    plt.plot(p[2, 0], p[2, 1], marker='o')
    plt.text(p[2, 0] - 0.1, p[2, 1] - 0.1, '$p_3$', fontsize=12)
    
    # Markers
    plt.plot(c[-1, 0], c[-1, 1], marker='>')
    plt.plot(r[-1, 0], r[-1, 1], marker='>')
    plt.plot(r[-1, 3], r[-1, 4], marker='>')
    
    plt.title('Planar Robot Trajectory')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.legend(['Cluster Trajectory', 'Robot 1 Trajectory', 'Robot 2 Trajectory'])
    plt.show()


def main():
    timesteps: int = 100  # Number of time steps

    Izz = 1
    m = 10
    bx = 5
    by = 5
    btheta = 1

    goals = np.asarray([[2, 1.6, 1.57], [1, 1.8, 1.57], [0, 1.5, 1.57]])

    A, B = robot_cluster(m, Izz, bx, by, btheta)
    state_space = cluster_state_space(m, Izz, bx, by, btheta)

    r = np.zeros((timesteps, 2 * DOF))
    rdot = np.zeros((timesteps, 2 * DOF))

    # Set initial pose of the robots
    r[0, :] = np.asarray([5, 1, 0, 15, 1, 0])

    c = np.zeros((timesteps, 2 * DOF))
    cdot = np.zeros((timesteps, 2 * DOF))

    # Set initial pose of the cluster
    c[0, :] = np.asarray([10, 1, 0, 5, 1, 0])

    x = np.zeros((timesteps, 12))

    for t in range(timesteps):
        u, alpha, delta = compute_task_allocation(c[t, 0:3], goals)

        control = np.concatenate((u.value, [0, 0, 0]))

        if t <= timesteps - 2:
            J = compute_jacobian_matrix(r[t, :])
            delta = np.transpose(np.linalg.inv(J)).dot(A).dot(np.linalg.inv(J))
            upsilon = np.transpose(np.linalg.inv(J)).dot(B).dot(np.linalg.inv(J))

            Jinv = compute_inverse_jacobian_matrix(c[t, :])
            Jdot = compute_dot_jacobian_matrix(r[t, :], rdot[t, :])
            mu = upsilon.dot(cdot[t, :]) - delta.dot(Jdot).dot(Jinv).dot(cdot[t, :])

            # FIX: Remove hardcoded value. Keep in mind control is way too big.
            F = delta.dot(0.1 * control) + mu

            gamma = np.transpose(J).dot(F)

            x[t, 0:3] = r[t, 0:3]
            x[t, 3:6] = rdot[t, 0:3]
            x[t, 6:9] = r[t, 3:6]
            x[t, 9:12] = rdot[t, 3:6]

            sys = cont2discrete(state_space, 1, method='foh')

            Ad = sys[0]
            Bd = sys[1]

            x[t + 1, :] = Ad.dot(x[t, :]) + Bd.dot(gamma)

            r[t + 1, 0:3] = x[t + 1, 0:3]
            rdot[t + 1, 0:3] = x[t + 1, 3:6]
            r[t + 1, 3:6] = x[t + 1, 6:9]
            rdot[t + 1, 3:6] = x[t + 1, 9:12]

            c[t + 1, :] = compute_forward_pose(r[t + 1, :])

    plot_cluster_state(c)
    plot_robots_trajectory(c, r, goals)


if __name__ == '__main__':
    main()
