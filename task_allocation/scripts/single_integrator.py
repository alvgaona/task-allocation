#!/usr/local/bin/python3

r""" Single-Integrator State Space Representation

    .. math::
        \dot{x} = u, x \in \mathbb{R}^{2}, u \in \mathbb{R}^2
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import StateSpace, lsim

np.set_printoptions(suppress=True)

if __name__ == '__main__':
    A = np.zeros((2, 2))
    B = np.identity(2)
    C = np.identity(2)
    D = np.zeros((2, 2))

    sys = StateSpace(A, B, C, D)

    T = 50  # Number of time steps

    step_function = np.ones((T, 2))

    linear_function = np.hstack([
        -0.59421658 * np.linspace(0, T - 1, T).reshape(-1, 1),
        -0.59421658 * np.linspace(0, T - 1, T).reshape(-1, 1)
    ])

    t, y1, _ = lsim(sys, step_function, range(T))
    _, y2, _ = lsim(sys, linear_function, range(T))
    
    plt.plot(t, y1[:, 0])
    plt.plot(t, y2[:, 0])
    plt.show()
