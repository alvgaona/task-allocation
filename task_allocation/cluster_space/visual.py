import matplotlib.pyplot as plt


def plot_control_input(t, u):
    plt.figure()
    for i in range(u.shape[1]):
        plt.plot(t, u[:, i], linestyle='dashed')
    
    plt.title('Control input')
    plt.xlabel('Time [s]')
    plt.ylabel('$u_i$')
    plt.legend(['$u_0$', '$u_1$', '$u_2$'])
    plt.show()


def plot_cluster_state(t, x):
    plt.figure()
    plt.plot(t, x[:, 0], linestyle='dashed')
    plt.plot(t, x[:, 1], linestyle='dashed')
    plt.plot(t, x[:, 2], linestyle='dashed')
    plt.plot(t, x[:, 3], linestyle='dashed')
    plt.plot(t, x[:, 4], linestyle='dashed')
    plt.plot(t, x[:, 5], linestyle='dashed')
    
    plt.title('Cluster State-Space Variables')
    plt.xlabel('Time [s]')
    plt.ylabel('State')
    plt.legend(['$x_c$', '$y_c$', 'd', '$\\theta_c$', '$\phi_1$', '$\phi_2$'])
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


def plot_slack_variables(t, delta):
    plt.figure()
    plt.plot(t, delta[:, 0], linestyle='dashed')
    plt.plot(t, delta[:, 1], linestyle='dashed')
    plt.plot(t, delta[:, 2], linestyle='dashed')
    
    plt.title('Task effectiveness')
    plt.xlabel('Time [s]')
    plt.ylabel('$\delta_i$')
    plt.legend(['$\delta_1$', '$\delta_2$', '$\delta_3$'])
