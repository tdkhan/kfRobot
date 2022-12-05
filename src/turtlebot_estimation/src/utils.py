import numpy as np
from math import cos, sin, pi
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Quaternion

def getEuler(quat_msg):
    q = [quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w]
    return euler_from_quaternion(q)

def getQuaternion(roll, pitch, yaw):
    quaternion = quaternion_from_euler(roll, pitch, yaw)

    orientation = Quaternion()
    orientation.x = quaternion[0]
    orientation.y = quaternion[1]
    orientation.z = quaternion[2]
    orientation.w = quaternion[3]

    return orientation

def covarianceToMsg(P):
    # Make 36 dimensional row vector of 0.0
    cov_msg = 36 * [0.0]

    # (xx, xy), (yx, yy)
    cov_msg[:2], cov_msg[6:8] = P[0][:2], P[1][:2]

    # (xyaw, yyaw)
    cov_msg[5], cov_msg[11] = P[0][2], P[1][2]

    # (yawx, yawy)
    cov_msg[30:32] = P[2][:2]

    # (yawyaw)
    cov_msg[35] = P[2][2]

    return cov_msg

def msgToCovariance(cov_msg):
    # Make 36 dimensional row vector of 0.0
    P = np.zeros((3,3))

    # (xx, xy), (yx, yy)
    P[0][:2], P[1][:2] = cov_msg[:2], cov_msg[6:8] 

    # (xyaw, yyaw)
    P[0][2], P[1][2] = cov_msg[5], cov_msg[11]

    # (yawx, yawy)
    P[2][:2] = cov_msg[30:32]

    # (yawyaw)
    P[2][2] = cov_msg[35]

    return P

# Rotation Matrix
def R(rad):
    return np.array([[cos(rad), -sin(rad)],
                    [sin(rad), cos(rad)]])

# Rotation Matrix for 3d vector
def R3(rad):
    return np.array([[cos(rad), -sin(rad), 0],
                    [sin(rad), cos(rad), 0],
                    [0, 0, 1]])

def wrap_angle(rad):
    if rad > 2*pi:
        return rad - 2*pi
    elif rad < 0:
        return rad + 2*pi
    else:
        return rad