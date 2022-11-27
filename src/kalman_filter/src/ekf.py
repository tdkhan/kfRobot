import numpy as np
from math import *

class ExtendedKalmanFilter:
    def __init__(self, _s, _P, _t, p_dev, l_dev, r_dev):
        # Load noise scaling values
        self.p_dev = p_dev
        self.a_dev = l_dev

        # Dimension of state, lidar and radar measurements
        self.s_dim = 3
        self.z_a_dim = 2

        # Load in the initial state vector, uncertainty and time
        self.s_k_k = _s
        self.P_k_k = _P
        self.t = _t

    # State transition matrix, 3 dimensional identity matrix
    def F(self, dt):
        # Start with a six dimensional identity matrix
        F = np.eye(self.s_dim)

        return F

    # Control to state matrix, maps (x_dot_local, yaw_dot) to (x_glob, y,glob, yaw)
    def B(self, s_k, dt):
        B = np.array([[cos(s_k[2]), 0],
                      [sin(s_k[2]), 0],
                      [0, dt]])

        return B

    # Process noise for near-constant velocity model
    def Q(self, dt):
        sigma = self.p_dev * np.eye(self.s_dim)
        Q = self.F(dt) @ sigma @ self.F(dt).T

        return Q

    # State to measurement matrix H for Aruco Markers, function of current state (s_k)
    def H_a(self, s_k):
        # Euclidean transformation, rotation and then translation
        E = np.array([[cos(s_k[2]), sin(s_k[2]), (-s_k[0]*cos(s_k[2]) - s_k[1]*sin(s_k[2]))/s_k[2]],
                      [-sin(s_k[2]), cos(s_k[2]), (s_k[0]*sin(s_k[2]) - s_k[1]*sin(s_k[2]))/s_k[2]],
                      [0, 0, 1/s_k[2]]])
                      # For this matrix to work, there must be an additional row which equates to one 
                      # E[2,2] and s_k[2] both must be one
                      # The 1/yaw in the third column allows there to be a third element 
                      # in the state vector which is then cancelled out

        # Matrix which removes the extra element in the output vector which is always one
        L = np.array([[1, 0, 0],
                      [0, 1, 0]])

        return L @ E

    # Aruco Marker Measurement Noise, function of distance from measurement to robot
    def R_a(self, z_k, s_k):
        # Distance in local frame to observation
        r = np.linalg.norm(z_k)

        M = np.array([[0.1*r, 0],
                      [0, 0.1*pi*r]])

        return self.H_a(s_k) @ M @ self.H_a(s_k)

    # H function multiplexer
    def H(self, s_k, sensor):
        if sensor=="a":
            return self.H_a(s_k)
        elif sensor=="i":
            return self.H_i(s_k)
        else:
            print("Sensor not found")

    # R function sltiplexor
    def R(self, z_k, s_k, sensor):
        if sensor=="a":
            return self.R_a(z_k, s_k)
        elif sensor=="i":
            return self.R_i(s_k)
        else:
            print("Sensor not found")
        
    # Prediction step of KF, prior distribution
    def predict(self, t):
        dt = t - self.t
        self.s_k_k_min = self.F(dt) @ self.s_k_k
        self.P_k_k_min = self.F(dt) @ self.P_k_k @ self.F(dt).T + self.Q(dt)

    # Update step of KF, get posterior distribution, function of measurement and sensor
    def update(self, z, s):
        S_k = self.H(self.s_k_k_min, s) @ self.P_k_k_min @ self.H(self.s_k_k_min, s).T + self.R(self.s_k_k_min, s)
        K_k = self.P_k_k_min @ self.H(self.s_k_k_min, s).T @ np.linalg.inv(S_k)

        z_bar = z - self.H(self.s_k_k_min, s) @ self.s_k_k_min

        self.s_k_k = self.s_k_k_min + K_k @ z_bar
        self.P_k_k = self.P_k_k_min - K_k @ self.H(self.s_k_k_min, s) @ self.P_k_k_min

    def getEst(self):
        return self.s_k_k, self.P_k_k
