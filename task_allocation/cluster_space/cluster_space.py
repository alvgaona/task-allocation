#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from task_allocation.cluster_space.kinematics import robot_cluster, cluster_state_space, compute_jacobian_matrix, \
    compute_inverse_jacobian_matrix, compute_forward_pose, compute_dot_jacobian_matrix
from task_allocation.cluster_space.parameters import DOF
from task_allocation.cluster_space.optimization import compute_task_allocation
from task_allocation.cluster_space.parameters import num_tasks

np.set_printoptions(suppress=True)


def plot_cluster_state(t, x):
    plt.figure()
    plt.plot(t, x[:, 0], linestyle='dashed')
    plt.plot(t, x[:, 1], linestyle='dashed')
    plt.plot(t, x[:, 2], linestyle='dashed')
    plt.plot(t, x[:, 3], linestyle='dashed')
    plt.plot(t, x[:, 4], linestyle='dashed')
    plt.plot(t, x[:, 5], linestyle='dashed')
    
    plt.title('Cluster Position')
    plt.xlabel('Time [s]')
    plt.ylabel('State')
    plt.legend(['$x_c$', '$y_c$', '$\\theta_c$', 'd', '$\phi_1$', '$\phi_2$'])
    plt.show()


def plot_robot_state(t, x):
    plt.figure()
    plt.plot(t, x[:, 0], linestyle='dashed')
    plt.plot(t, x[:, 1], linestyle='dashed')
    plt.plot(t, x[:, 2], linestyle='dashed')
    
    plt.title('Robot Position')
    plt.xlabel('Time [s]')
    plt.ylabel('State')
    plt.legend(['$x_i$', '$y_i$', '$\\theta_i$'])
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
    plt.xlabel('$x_c / x_i$')
    plt.ylabel('$y_c / y_i$')
    plt.legend(['Cluster position', '$Robot_1$ position', '$Robot_2$ position'])
    plt.show()
    
    
def plot_task_effectiveness(t, delta):
    plt.figure()
    plt.plot(t, delta[:, 0], linestyle='dashed')
    plt.plot(t, delta[:, 1], linestyle='dashed')
    plt.plot(t, delta[:, 2], linestyle='dashed')

    plt.title('Task effectiveness')
    plt.xlabel('Time [s]')
    plt.ylabel('$\delta_i$')
    plt.legend(['$\delta_1$', '$\delta_2$', '$\delta_3$'])


def main():
    T = 1000
    
    timesteps, dt = np.linspace(0, 100, T, retstep=True)
    
    print(dt)

    Izz = 1
    m = 10
    bx = 5
    by = 5
    btheta = 1
    
    goals = np.asarray([[10, 10, 3 * np.pi / 2], [15, 10, 3 * np.pi / 2], [10, 15, 3 * np.pi / 2]])

    A, B = robot_cluster(m, Izz, bx, by, btheta)
    state_space = cluster_state_space(m, Izz, bx, by, btheta)

    r = np.zeros((T, 2 * DOF))
    rdot = np.zeros((T, 2 * DOF))

    # Set initial pose of the robots
    r[0, :] = np.asarray([5, 1, 0, 15, 1, 0])

    c = np.zeros((T, 2 * DOF))
    cdot = np.zeros((T, 2 * DOF))

    # Set initial pose of the cluster
    c[0, :] = np.asarray([10, 1, 0, 5, 1, 0])

    x = np.zeros((T, 12))
    slack_variables = np.zeros((T, num_tasks))
    
    for idx, t in enumerate(timesteps):
        u, alpha, delta = compute_task_allocation(c[idx, 0:3], goals)
        control = np.concatenate((u.value, [0, 0, 0]))
        
        slack_variables[idx, :] = delta.value
        
        print("Timestep:", t, "-", "u1:", u.value[0], "u2:", u.value[1], "u3:", u.value[2])
        
        if idx <= T - 2:
            J = compute_jacobian_matrix(r[idx, :])
            delta = np.transpose(np.linalg.inv(J)).dot(A).dot(np.linalg.inv(J))
            upsilon = np.transpose(np.linalg.inv(J)).dot(B).dot(np.linalg.inv(J))

            Jinv = compute_inverse_jacobian_matrix(c[idx, :])
            Jdot = compute_dot_jacobian_matrix(r[idx, :], rdot[idx, :])
            mu = upsilon.dot(cdot[idx, :]) - delta.dot(Jdot).dot(Jinv).dot(cdot[idx, :])

            F = delta.dot(control) + mu

            gamma = np.transpose(J).dot(F)

            x[idx, 0:3] = r[idx, 0:3]
            x[idx, 3:6] = rdot[idx, 0:3]
            x[idx, 6:9] = r[idx, 3:6]
            x[idx, 9:12] = rdot[idx, 3:6]

            sys = cont2discrete(state_space, dt, method='foh')

            Ad = sys[0]
            Bd = sys[1]
            
            x[idx + 1, :] = Ad.dot(x[idx, :]) + Bd.dot(gamma)

            r[idx + 1, 0:3] = x[idx + 1, 0:3]
            rdot[idx + 1, 0:3] = x[idx + 1, 3:6]
            r[idx + 1, 3:6] = x[idx + 1, 6:9]
            rdot[idx + 1, 3:6] = x[idx + 1, 9:12]

            c[idx + 1, :] = compute_forward_pose(r[idx + 1, :])
        
        print("Timestep:", t, "-", "xc:", c[idx,0], "yc:", c[idx,1], "yawc:", c[idx,2], "d:", c[idx,3], "phi1:", c[idx,4], "phi2:", c[idx,5])
    
    plot_task_effectiveness(timesteps, slack_variables)
    plot_cluster_state(timesteps, c)
    plot_robots_trajectory(c, r, goals)


if __name__ == '__main__':
    main()
