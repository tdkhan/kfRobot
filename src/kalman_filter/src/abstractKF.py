import rospy
import numpy as np
from math import *

# The Abstract Kalman Filter Class contains a kalman filter implimentation which can be extended to any use case
# There is an example class inheriting the AbstractKF class at the bottom of the python script

# This is an abstract class (designed to be inherited by another class)
# I had previously used this for 2d position filtering
class AbstractKF:
    def __init__(self, _F, _H, _B, _x, _P, _Q):
        
        # State Transition Matrix
        self.F = _F
        # Observation Matrix
        self.H = _H
        
        # Control to State Transition Matrix
        # self.B = _B (Now a function of yaw)

        # Initial State Vector
        self.x_k_k = _x
        # Initial Uncertainty
        self.P_k_k = _P

        # Process Noise Uncertainty
        self.Q = _Q
        # Measurement Noise Uncertainty
        # self.R=_R (Now a function to allow for dynamic values)
        
        # Dimensions of State, Observation and Control Vectors
        self.x_dim = _x.shape[0]
        self.z_dim = _H.shape[0]
        self.u_dim = _B.shape[0]
        

    # Abstract method for the measurement noise
    def R(self):
        pass


    # Pass in new information that the R function relies on
    def set_dynamic_variables(self):
        pass


    # Prediction step of KF, prior distribution
    def predict(self, u=False):
        # If we don't have a control input
        if not u:
            u = np.zeros(self.u_dim)

        self.x_k_k_min = self.F @ self.x_k_k + self.B @ u
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


# This is what a class inheriting the 
class PositionFilter(AbstractKF):
    def __init__(self, _x, _P):
        # Identity for state tranistion and observation transition
        F = np.eye(2)
        H = np.eye(2)

        # Load in the initial state vector and uncertainty
        x = _x
        P = _P

        # No process noise for the location of a non-moving object
        process_unc = 0
        Q = np.diag(np.full(2, process_unc))
        # Q = process_unc * np.eye(2)

        AbstractKF.__init__(self, F, H, x, P, Q)

    # Measurement noise is a linear function of distance
    def R(self):
        meas_unc = self.distance * 5 + 2
        return np.diag(np.full(2, meas_unc))

    # Update the distance
    def set_dynamic_variables(self, distance):
        self.distance = distance

