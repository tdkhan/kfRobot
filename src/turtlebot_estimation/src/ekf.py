import numpy as np
from math import *

class ExtendedKalmanFilter:
    def __init__(self, _s, _P, _t, p_dev):
        # Load noise scaling values
        self.p_dev = np.array(p_dev)

        # Dimension of state, lidar and radar measurements
        self.s_dim = 3
        self.z_a_dim = 3

        # Load in the initial state vector, uncertainty and time
        self.s_k_k_min = _s
        self.P_k_k_min = _P
        self.t = _t

    # State transition matrix, 3 dimensional identity matrix
    def F(self, dt):
        # Start with a six dimensional identity matrix
        F = np.eye(self.s_dim)

        return F

    # Control to state matrix, maps (x_dot_local, yaw_dot) to (x_glob, y_glob, yaw)
    def B(self, s_k, dt):
        B = np.array([[cos(s_k[2])*dt, 0],
                      [sin(s_k[2])*dt, 0],
                      [0, dt]])

        return B

    # Process noise for near-constant velocity model
    def Q(self, dt):
        # Square the deviation vector to get variance then put on diagonal of Q
        sigma = np.diag(self.p_dev**2)
        Q = self.F(dt) @ sigma @ self.F(dt).T

        return Q

    # State to measurement matrix H for Aruco Markers, 3 dimensional identity matrix
    def H_a(self):
        H = np.eye(self.z_a_dim)

        return H

    # Aruco Marker Measurement Noise, function of distance from measurement to robot
    def R_a(self, s):
        # Measurement noise is calculated within aurco_pose_node
        # R matrix contained within s is simply returned
        return s

    # H function multiplexer
    def H(self, s_k, sensor):
        # If an array is passed, H_a for aruco
        if type(sensor) == np.ndarray:
            return self.H_a()
        elif sensor=="i":
            return self.H_i(s_k)
        else:
            print("Sensor not found")

    # R function multiplexor
    def R(self, s_k, sensor):
        # If an array is passed, R_a for aruco
        if type(sensor) == np.ndarray:
            return self.R_a(sensor)
        elif sensor=="i":
            return self.R_i(s_k)
        else:
            print("Sensor not found")
        
    # Prediction step of KF, prior distribution
    def predict(self, t, u_k):
        duration = t - self.t
        dt = duration.nsecs * 10**-9

        self.s_k_k_min = self.F(dt) @ self.s_k_k_min + self.B(self.s_k_k_min, dt) @ u_k
        self.P_k_k_min = self.F(dt) @ self.P_k_k_min @ self.F(dt).T + self.Q(dt)
        self.t = t

    # Update step of KF, get posterior distribution, function of measurement and sensor
    def update(self, z, s):
        S_k = self.H(self.s_k_k_min, s) @ self.P_k_k_min @ self.H(self.s_k_k_min, s).T + self.R(self.s_k_k_min, s)
        K_k = self.P_k_k_min @ self.H(self.s_k_k_min, s).T @ np.linalg.inv(S_k)

        z_bar = z - self.H(self.s_k_k_min, s) @ self.s_k_k_min

        self.s_k_k = self.s_k_k_min + K_k @ z_bar
        self.P_k_k = self.P_k_k_min - K_k @ self.H(self.s_k_k_min, s) @ self.P_k_k_min

        self.s_k_k_min = self.s_k_k
        self.P_k_k_min = self.P_k_k

    def getPriorEst(self):
        return self.s_k_k_min, self.P_k_k_min

    def getEst(self):
        return self.s_k_k, self.P_k_k
