#!/usr/bin/env python3
import rospy
import numpy as np
from math import pi
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from ekf import ExtendedKalmanFilter

from geometry_msgs.msg import Twist, Quaternion
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

class EstimationNode:
    def __init__(self):
        # Track initialization
        self.initialized = False

        # Control inputs
        rospy.Subscriber("/cmd_vel", Twist, self.controlCb)

        # IMU data
        rospy.Subscriber("/imu", Imu, self.imuCb)

        # Ground truth location
        rospy.Subscriber("/odom", Odometry, self.odomCb)

        # Output estimated odometry (pose and twist) 
        # Odometry message contains a covariance display we can use
        self.est_odom_pub = rospy.Publisher("/odom_est", Odometry, queue_size=1)

        # Wait for first odometry (ground truth) callback before initializing filter
        self.got_ground_truth = False
        while not rospy.is_shutdown():
            if self.got_ground_truth:
                break
            rospy.sleep(0.2)

        # Initial state, covariance and time (Set to ground truth location for debugging)
        # s_init = np.array([-2.0, -0.5, 0.0])
        s_init = self.ground_truth_loc
        P_init = np.diag([0.5, 0.5, pi/6])
        t_init = rospy.Time.now()

        # Noise values (standard deviations)
        process_deviation = [0.15, 0.15, pi/24]
        aruco_deviation = 1

        # Create new EKF class
        self.ekf = ExtendedKalmanFilter(s_init, P_init, t_init, 
            process_deviation, aruco_deviation)

        # Wait for callbacks
        self.initialized = True
        rospy.spin()

    # Publish the estimate as odometry with coavariance
    def publishEstimate(self, gaussian):
        s_k_k, P_k_k = gaussian
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "odom"
        odom_msg.pose.pose.position.x = s_k_k[0]
        odom_msg.pose.pose.position.y = s_k_k[1]
        odom_msg.pose.pose.orientation = self.getQuaternion(0, 0, s_k_k[2])
        odom_msg.pose.covariance = self.getCovariance(P_k_k)

        s_string = str(np.round(s_k_k, decimals=2))
        P_string = str(np.round(np.diag(P_k_k), decimals=2))
        rospy.loginfo_throttle(1, "Publishing s:" + s_string + " ~:" + P_string)

        self.est_odom_pub.publish(odom_msg)

    # Run kalman predict step with new control input
    def controlCb(self, msg):
        # Wait until we are initialized
        if self.initialized:
            # Format msg into control vector
            u_k = np.array([msg.linear.x, msg.angular.z])

            self.ekf.predict(rospy.Time.now(), u_k)

            # For initial pipeline setup work with prior estimate
            self.publishEstimate(self.ekf.getPriorEst())

    def imuCb(self, msg):
        pass

    def odomCb(self, msg):
        pos = msg.pose.pose.position
        yaw = self.getEuler(msg.pose.pose.orientation)[2]
        self.ground_truth_loc = np.array([pos.x, pos.y, yaw])

        self.got_ground_truth = True

    def getEuler(self, quat_msg):
        q = [quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w]
        return euler_from_quaternion(q)

    def getQuaternion(self, roll, pitch, yaw):
        quaternion = quaternion_from_euler(roll, pitch, yaw)

        orientation = Quaternion()
        orientation.x = quaternion[0]
        orientation.y = quaternion[1]
        orientation.z = quaternion[2]
        orientation.w = quaternion[3]

        return orientation

    def getCovariance(self, P_k_k):
        # Make 36 dimensional row vector of 0.0
        cov = 36 * [0.0]

        # (xx, xy), (yx, yy)
        cov[:2], cov[6:8] = P_k_k[0][:2], P_k_k[1][:2]

        # (xyaw, yyaw)
        cov[5], cov[11] = P_k_k[0][2], P_k_k[1][2]

        # (yawx, yawy)
        cov[30:32] = P_k_k[2][:2]

        # (yawyaw)
        cov[35] = P_k_k[2][2]

        return cov

if __name__ == "__main__":
    rospy.init_node("estimation_node")
    est_node = EstimationNode()
