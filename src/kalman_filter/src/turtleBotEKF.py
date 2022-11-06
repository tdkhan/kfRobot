import rospy
import numpy as np
from math import *

# Very WIP class for filtering pose of turtlebot Pose

# This is based off of the abstractKF class 
# but as there are a lot of new of functions needed I didn't think inheriting it would help

class TurtleBotFilter():
    def __init__(self, _x, _P):
        # Initialize current time and time difference
        self.t = rospy.Time.now()
        self.dt = rospy.Duration(0)

        # State space transition is identity, transition comes from control input
        self.F = np.eye(3)

        # Load in the initial state vector and uncertainty
        self.x = _x
        self.P = _P

        # No process noise for the location of a non-moving object
        process_unc = 0
        self.Q = np.diag(np.full(2, process_unc))
        # Q = process_unc * np.eye(2)

    # Control space to state space matrix is a function of yaw
    def B(self, yaw, dt):
        return np.array([[cos(yaw) * dt, 0],
                         [sin(yaw) * dt, 0],
                         [0, dt]])

    # State to IMU measurement matrix
    # z_imu is a vector of (x_accel, yaw, yaw_dot)
    def H_imu(self, dt):
        return np.array([[]])

    # Measurement noise is a linear function of distance (old function)
    def R(self):
        meas_unc = self.distance * 5 + 2
        return np.diag(np.full(2, meas_unc))

    # Update the distance (old function)
    def set_dynamic_variables(self, distance):
        self.distance = distance

    # Prediction step of KF, prior distribution
    def predict(self, u=False):
        # If we don't have a control input
        if not u:
            u = np.zeros(self.u_dim)

        self.x_k_k_min = self.F @ self.x_k_k + self.B(self.x_k_k[3]) @ u
        self.P_k_k_min = self.F @ self.P_k_k @ self.F.T + self.Q

        
    # Update step of KF, get posterior distribution
    def update(self,z):
        
        S_k = self.H @ self.P_k_k_min @ self.H.T + self.R()
        K_k = self.P_k_k_min @ self.H.T @ np.linalg.inv(S_k)

        z_bar = z - self.H @ self.x_k_k_min

        self.x_k_k = self.x_k_k_min + K_k @ z_bar
        self.P_k_k = self.P_k_k_min - K_k @ self.H @ self.P_k_k_min
    

    def getEst(self):
        return self.x_k_k, self.P_k_k
