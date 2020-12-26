#!/usr/local/bin/python3

import numpy as np
import logging
from scipy.signal import cont2discrete
from task_allocation.cluster_space.kinematics import robot_cluster, cluster_state_space, compute_jacobian_matrix, \
    compute_inverse_jacobian_matrix, compute_forward_pose, compute_dot_jacobian_matrix
from task_allocation.cluster_space.parameters import DOF
from task_allocation.cluster_space.optimization import compute_task_allocation
from task_allocation.cluster_space.parameters import num_tasks, control_dim
from task_allocation.cluster_space.visual import plot_cluster_state, plot_robots_trajectory, plot_slack_variables, \
    plot_control_input

np.set_printoptions(suppress=True)

FORMAT = '%(levelname)s %(asctime)s %(filename)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="%Y-%m-%dT%H:%M:%S%z")
logger = logging.getLogger(__name__)


def main():
    T = 1000
    timesteps, dt = np.linspace(0, 100, T, retstep=True)
    
    logger.info("Used timestep: {} s".format(dt))
    
    Izz = 41
    m = 150
    bx = 100
    by = 400
    btheta = 25
    
    goals = np.asarray(
        [[10, 10, 1],
         [15, 10, 1],
         [10, 15, 1]]
    )
    
    A, B = robot_cluster(m, Izz, bx, by, btheta)
    state_space = cluster_state_space(m, Izz, bx, by, btheta)
    
    r = np.zeros((T, 2 * DOF))
    rdot = np.zeros((T, 2 * DOF))
    
    # Set initial pose of the robots
    r[0, :] = np.asarray([5, 1, 0, 15, 1, 0])
    
    c = np.zeros((T, 2 * DOF))
    cdot = np.zeros((T, 2 * DOF))
    
    # Set initial pose of the cluster
    c[0, :] = np.asarray([10, 1, 0, 5, 0, 0])
    
    x = np.zeros((T, 12))
    slack_variables = np.zeros((T, num_tasks))
    control_input = np.zeros((T, control_dim))
    control = np.zeros(6)
    
    for idx, t in enumerate(timesteps):
        optimization_state = np.append([], c[idx, 0:2])
        optimization_state = np.append(optimization_state, c[idx, 3])
        
        u, alpha, delta = compute_task_allocation(optimization_state, goals)
        
        control[0] = u.value[0]
        control[1] = u.value[1]
        control[2] = 0
        control[3] = u.value[2]
        control[4] = 0
        control[5] = 0
        
        slack_variables[idx, :] = delta.value
        control_input[idx, :] = np.concatenate((control[0:2], [control[3]]))
        
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
            
    plot_control_input(timesteps, control_input)
    plot_slack_variables(timesteps, slack_variables)
    plot_cluster_state(timesteps, c)
    plot_robots_trajectory(c, r, goals)


if __name__ == '__main__':
    main()
