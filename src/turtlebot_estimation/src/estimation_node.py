#!/usr/bin/env python3
import rospy
import numpy as np
from math import pi
import random
from ekf import ExtendedKalmanFilter
from utils import getEuler, getQuaternion, covarianceToMsg, msgToCovariance

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

class EstimationNode:
    def __init__(self):
        # Track initialization
        self.initialized = False

        # Control inputs
        rospy.Subscriber("/cmd_vel", Twist, self.controlCb)

        # Pose measurements from Aruco Pose Node
        rospy.Subscriber("/aruco_measurements", PoseWithCovarianceStamped, self.arucoCb)

        # IMU data
        rospy.Subscriber("/imu", Imu, self.imuCb)

        # Ground truth location
        rospy.Subscriber("/odom", Odometry, self.odomCb)

        # Output estimated odometry (pose and twist) 
        # Odometry message contains a covariance display we can use
        self.est_odom_pub = rospy.Publisher("/odom_est", Odometry, queue_size=1)

        # Timer for publishing estimate at 10 Hz
        self.est_pub_timer = rospy.Timer(rospy.Duration(0.1), self.publishEstimateTimerCb)

        # Run prediction step asynchronously once we get the first command
        self.got_cmd = None
        self.predict_timer = rospy.Timer(rospy.Duration(0.1), self.predictTimerCb)

        # Wait for first odometry (ground truth) callback before initializing filter
        self.got_ground_truth = False
        while not rospy.is_shutdown():
            if self.got_ground_truth:
                break
            rospy.sleep(0.2)

        # Initial state, covariance and time
        s_init = np.array([random.uniform(-30, 30),
                           random.uniform(-30, 30), 
                           random.uniform(0, 2*pi)])
        P_init = np.diag([0.5, 0.5, pi/6])
        t_init = rospy.Time.now()

        # Set initial state to ground truth location for debugging
        # s_init = self.ground_truth_loc

        # Noise values (standard deviations of x, y, yaw)
        process_deviation = [0.15, 0.15, pi/24]

        # Create new EKF class
        self.ekf = ExtendedKalmanFilter(s_init, P_init, t_init, 
            process_deviation)

        # Wait for callbacks
        self.initialized = True
        rospy.spin()

    # Publish the estimate as odometry with coavariance
    def publishEstimateTimerCb(self, event):
        # Wait until we are initialized
        if self.initialized:
            s_k_k, P_k_k = self.ekf.getPriorEst()
            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = "odom"
            odom_msg.pose.pose.position.x = s_k_k[0]
            odom_msg.pose.pose.position.y = s_k_k[1]
            odom_msg.pose.pose.orientation = getQuaternion(0, 0, s_k_k[2])
            odom_msg.pose.covariance = covarianceToMsg(P_k_k)

            s_string = str(np.round(s_k_k, decimals=2))
            P_string = str(np.round(np.diag(P_k_k), decimals=2))
            # rospy.loginfo_throttle(1, "Publishing s:" + s_string + " ~:" + P_string)

            self.est_odom_pub.publish(odom_msg)

    # Run prediction step at 10 hz to account for slow command rate from teleop node
    def predictTimerCb(self, event):
        if self.initialized and self.got_cmd:
            self.ekf.predict(rospy.Time.now(), self.u_k)

    # Run kalman predict step with new control input
    def controlCb(self, msg):
        # Wait until we are initialized
        if self.initialized:
            # Format msg into control vector
            self.u_k = np.array([msg.linear.x, msg.angular.z])
            self.got_cmd = True
            
    def imuCb(self, msg):
        pass

    def arucoCb(self, msg):
        # Wait until we are initialized
        if self.initialized:
            pos = msg.pose.pose.position
            yaw = getEuler(msg.pose.pose.orientation)[2]

            z = np.array([pos.x, pos.y, yaw])
            cov_vec = msg.pose.covariance
            R = msgToCovariance(cov_vec)

            self.ekf.update(z, R)

    def odomCb(self, msg):
        pos = msg.pose.pose.position
        yaw = getEuler(msg.pose.pose.orientation)[2]
        self.ground_truth_loc = np.array([pos.x, pos.y, yaw])

        self.got_ground_truth = True

if __name__ == "__main__":
    rospy.init_node("estimation_node")
    est_node = EstimationNode()
