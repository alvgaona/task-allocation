import numpy as np
import scipy.linalg

from numpy.testing import *


from ..kinematics import compute_forward_pose, compute_inverse_pose, compute_dot_jacobian_matrix, \
    compute_inverse_jacobian_matrix, compute_jacobian_matrix, robot_cluster, cluster_state_space


class TestKinematics(TestCase):
    
    def test_forward_pose(self):
        robots_pose = np.asarray([3.5, -2.2, 1.0, 0, 3.4, 33.3])
        
        cluster_pose = compute_forward_pose(robots_pose)
        
        assert_array_almost_equal(cluster_pose, np.asarray([1.75, 0.6, 3.3018, 2.5829, -1.5829, 30.7170]), decimal=4)
    
    def test_inverse_pose(self):
        cluster_pose = np.asarray([1.8, 0.6, 3.3, 2.6, -1.6, 30.7])
        
        robots_pose = compute_inverse_pose(cluster_pose)
        
        assert_array_almost_equal(robots_pose, np.asarray([3.5012, -2.2277, 1.0, 0.0988, -2.2277, 33.3]), decimal=4)
    
    def test_jacobian_matrix(self):
        robots_pose = np.asarray([3.5, -2.2, 1.0, 0, 3.4, 33.3])
        
        J = compute_jacobian_matrix(robots_pose)
        
        expected_J = np.asarray([
            [0.5, 0.5, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0.5, 0],
            [0.26499947, -0.42399915, 0, -0.26499947, 0.42399915, 0],
            [0.12841091, 0.08025682, 0, -0.12841091, -0.08025682, 0],
            [-0.12841091, -0.08025682, 0.02293052, 0.12841091, 0.08025682, 0],
            [-0.12841091, -0.08025682, 0, 0.12841091, 0.08025682, 0.02293052]
        ])
        
        assert_array_almost_equal(J, expected_J)
    
    def test_inverse_jacobian_matrix(self):
        cluster_pose = np.asarray([1.8, 0.6, 3.3, 2.6, -1.6, 30.7])
        
        Jinv = compute_inverse_jacobian_matrix(cluster_pose)
        
        expected_Jinv = np.asarray([
            [1, 0, 0.51550137, -2.82773289, 0, 0],
            [0, 1, -0.85688875, -1.70115453, 0, 0],
            [1, 0, -0.51550137, 2.82773289, 0, 0],
            [0, 1, 0.85688875, 1.70115453, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 1]
        ])
        
        assert_array_almost_equal(Jinv, expected_Jinv)
    
    def test_dot_jacobian_matrix(self):
        robots_pose = np.asarray([3.5, -2.2, 1.0, 0, 3.4, 33.3])
        robots_velocities = np.asarray([-1.2, 2, 1.0, 2.2, -3.4, -1.0])
        
        Jdot = compute_dot_jacobian_matrix(robots_pose, robots_velocities)
        
        expected_Jdot = np.asarray([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [-0.00136115, -0.00085072, 0, 0.00136115, 0.00085072, 0],
            [0.1243401, 0.0771393, 0, -0.1243401, -0.0771393, 0],
            [-0.1243401, -0.0771393, 0, 0.1243401, 0.0771393, 0],
            [-0.1243401, -0.0771393, 0, 0.1243401, 0.0771393, 0]
        ])
        
        assert_array_almost_equal(Jdot, expected_Jdot)
    
    def test_cluster_state_space(self):
        Izz = 1
        m = 10
        bx = 5
        by = 5
        btheta = 1
        
        A, B, C, D = cluster_state_space(m, Izz, bx, by, btheta)

        Ai = np.asmatrix([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, -0.5, 0, 0],
            [0, 0, 0, 0, -0.5, 0],
            [0, 0, 0, 0, 0, -1]
        ])
        
        Bi = np.asmatrix([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1 / 10, 0, 0],
            [0, 1 / 10, 0],
            [0, 0, 1]
        ])

        assert_array_almost_equal(A, scipy.linalg.block_diag(Ai, Ai))
        assert_array_almost_equal(B, scipy.linalg.block_diag(Bi, Bi))
        assert_array_almost_equal(C, np.eye(12))
        assert_array_almost_equal(D, np.zeros((12, 6)))

    def test_robot_cluster(self):
        Izz = 1
        m = 10
        bx = 5
        by = 5
        btheta = 1
    
        A, B = robot_cluster(m, Izz, bx, by, btheta)

        Ai = np.asmatrix([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 1]
        ])

        Bi = np.asarray([
            [5, 0, 0],
            [0, 5, 0],
            [0, 0, 1]
        ])

        assert_array_almost_equal(A, scipy.linalg.block_diag(Ai, Ai))
        assert_array_almost_equal(B, scipy.linalg.block_diag(Bi, Bi))
