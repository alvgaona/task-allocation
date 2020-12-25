import numpy as np
import cvxpy as cp

from scipy.linalg import pinv
from task_allocation.cluster_space.parameters import control_dim, num_tasks, num_robots


def compute_task_allocation(x, p):
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
    
    u = cp.Variable(control_dim)
    alpha = cp.Variable(num_tasks)
    delta = cp.Variable(num_tasks)
    
    objective = C * cp.sum_squares(global_task_specification - cp.reshape(1 / num_robots * P @ alpha, (num_tasks, 1))) + \
                cp.sum_squares(u) + L * cp.sum_squares(delta)
    
    # Constraints
    constraints = [
        -2 * (x - p[0, :]).T @ u >= cp.log(cp.sum_squares((x - p[0, :]))) - delta[0],
        -2 * (x - p[1, :]).T @ u >= cp.log(cp.sum_squares((x - p[1, :]))) - delta[1],
        -2 * (x - p[2, :]).T @ u >= cp.log(cp.sum_squares((x - p[2, :]))) - delta[2],
        cp.norm_inf(delta) <= DELTA_MAX,
        np.ones((1, num_tasks)) @ alpha == 1.0,
        delta[1] <= KAPPA * (delta[0] - DELTA_MAX * (1 - alpha[0])),
        delta[1] <= KAPPA * (delta[2] - DELTA_MAX * (1 - alpha[2])),
        alpha <= 1,
        alpha >= 0,
        u <= 0.2,
        u >= -0.2
    ]
    
    obj = cp.Minimize(objective)
    prob = cp.Problem(obj, constraints)
    
    prob.solve(solver='OSQP', verbose=False)
    
    return u, alpha, delta
