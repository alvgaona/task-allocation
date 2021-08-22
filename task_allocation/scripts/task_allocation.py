#!/usr/local/bin/python3

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import pinv

np.set_printoptions(suppress=True)

M = 3  # Number of Tasks
N = 1  # Number of Robots


def plot_optimization_variables(t, alpha, delta):
    legend = []
    
    plt.figure()
    for i in range(alpha.shape[1]):
        plt.plot(t, alpha[:, i], linestyle='dashed')
        legend.append('$\\alpha_%s$' % i)
    plt.title("Task priorities")
    plt.xlabel("Time [s]")
    plt.ylabel("$\\alpha$")
    plt.legend(legend)
    plt.show()
    
    legend = []
    plt.figure()
    for i in range(delta.shape[1]):
        plt.plot(t, delta[:, i], linestyle='dashed')
        legend.append('$\delta_%s$' % i)
    plt.title("Slack variables")
    plt.xlabel("Time [s]")
    plt.ylabel("$\delta")
    plt.legend(legend)
    plt.show()
    

def task_allocation():
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
    KAPPA = 1
    
    alpha = cp.Variable(M)
    delta = cp.Variable(M)
    
    objective = C * cp.sum_squares(global_task_specification - cp.reshape(1 / N * P @ alpha, (M, 1))) + L * cp.sum_squares(delta)
    
    # Constraints
    constraints = [
        cp.norm_inf(delta) <= DELTA_MAX,
        np.ones((1, M)) @ alpha == 1.0,
        delta[1] <= KAPPA * (delta[2] - DELTA_MAX * (1 - alpha[2])),
        delta[1] <= KAPPA * (delta[0] - DELTA_MAX * (1 - alpha[0])),
        alpha <= 1,
        alpha >= 0,
    ]
    
    obj = cp.Minimize(objective)
    prob = cp.Problem(obj, constraints)
    
    prob.solve()
    
    return alpha, delta


def main():
    T = 1000
    t, dt = np.linspace(0, 100, T, retstep=True)
    
    task_priorities = np.zeros((T, M))
    slack_variables = np.zeros((T, M))
    
    for idx, _ in enumerate(t):
        alpha, delta = task_allocation()
        
        print(alpha.value)
        print(delta.value)
        
        task_priorities[idx, :] = alpha.value
        slack_variables[idx, :] = delta.value

    plot_optimization_variables(t, task_priorities, slack_variables)


if __name__ == '__main__':
    main()
