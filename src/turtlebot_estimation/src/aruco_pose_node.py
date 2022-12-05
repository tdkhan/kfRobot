#!/usr/bin/env python3
import rospy
import numpy as np
from math import pi, cos, sin, atan2
from utils import getEuler, getQuaternion, covarianceToMsg, R3, R, wrap_angle

from fiducial_msgs.msg import FiducialTransformArray
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped

class ArucoToPoseNode:
    def __init__(self):
        
        # Standard deviation scaling values for aruco detection pose (distance, radial, yaw)
        self.aruco_dev = np.array([0.01, 0.07, 0.0005])
        self.min_aruco_range = 0.1
        self.max_aruco_range = 10.0

        # Input fiducial transforms
        rospy.Subscriber("/fiducial_transforms", FiducialTransformArray, self.arucoCb)

        # Output pose array for visualization and pose with covariance for the measurement to the kalman filter
        self.poses_viz_pub = rospy.Publisher("/aruco_poses", PoseArray, queue_size=1)
        self.pose_cov_pub = rospy.Publisher("/aruco_measurements", PoseWithCovarianceStamped, queue_size=1)

        # Dictionary of ground truth locations of Aruco Markers
        self.aruco_gt_dict = {0: (-9.0, -6.6027, 0.5, 3e-06, -0, 0),
                              1: (-9.0, -2.43358, 0.5, 3e-06, 0, 4e-06),
                              2: (-9.0, 0.600883, 0.5, 3e-06, 0, 4e-06),
                              3: (-4.58408, 6.7, 0.5, 3e-06, -0, -1.56993),
                              4: (-1.31296, -8.47, 0.5, 3e-06, -0, 1.56996),
                              5: (-5.49648, -8.47, 0.5, -3e-06, -0, 1.56997),
                              6: (1.0501, -5.60745, 0.5, 3e-06, -0, 3.13998),
                              7: (-2.45, 4.66213, 0.5, 3e-06, 0, 4e-06),
                              8: (1.58425, 6.7, 0.5, 3e-06, -0, -1.56994),
                              9: (6.02877, 6.7, 0.5, 3e-06, -0, -1.56994),
                              10: (9.68324, 2.85356, 0.5, -3e-06, -0, 3.13999),
                              11: (9.68324, -1.38058, 0.5, -3e-06, -0, 3.13999),
                              12: (7.08134, -2.855, 0.5, 3e-06, -0, 1.56997),
                              13: (5.23, 0.869286, 0.5, 3e-06, 0, 3.13999),
                              14: (4.23567, -8.47, 0.5, 3e-06, -0, 1.56996),
                              15: (9.68324, -5.61203, 0.499993, -6e-06, -0, 3.14),
                              16: (1.20916, -6.0305, 0.5, -3e-06, -0, 4e-06),
                              17: (6.14084, -3.05, 0.5, 3e-06, -0, -1.56994),
                              18: (5.72798, 1.38, 0.5, -3e-06, -0, 1.56997),
                              19: (-9, 6,.03583, 0.5, -3e-06, -0, 4e-06),
                              20: (-2.65, 3.73864, 0.5, 3e-06, 0, 3.13997)
                              }

        # Wait for callbacks
        rospy.spin()

    def arucoCb(self, msg):
        timestamp = msg.header.stamp
        
        # Camera to robot frame transformation
        CR = np.array([[cos(pi/2), sin(pi/2), 0.076],
                      [-sin(pi/2), cos(pi/2), 0.0],
                      [0, 0, 1]])

        poses = []
        filtered_count = 0
        max = 0

        for t in msg.transforms:
            pos_ar_gl = self.aruco_gt_dict[t.fiducial_id]           # Position of aruco marker in global frame
            pos_ar_cam = t.transform.translation                    # Position of aruco marker in camera frame
            yaw_ar = getEuler(t.transform.rotation)[1]              # Yaw of aruco marker (from pitch in camera frame)
            yaw_r = wrap_angle(pi + pos_ar_gl[5] + yaw_ar)          # Yaw of robot

            vec_a_g = pos_ar_gl[0:2]                                # Vector from global origin to aruco marker

            # Transform camera frame vector to robot frame
            vec_l = CR @ np.array([pos_ar_cam.x, pos_ar_cam.z, 1])  # Vector from robot to aruco marker in robot local frame
            
            vec_g = R(yaw_r) @ vec_l[:2]                            # Vector from robot to aruco marker in global frame

            vec_r = vec_a_g - vec_g                                 # Vector from global origin to robot

            pose = (vec_r[0], vec_r[1], yaw_r)

            dist =np.linalg.norm(vec_g)

            # Only take measurements that are within distance threshold
            if dist < self.max_aruco_range:
                self.publishPoseMeasurement(timestamp, pose, vec_g, yaw_ar, yaw_r)
            else:
                filtered_count+=1
                pose = (vec_r[0], vec_r[1], yaw_r + pi)

            # Track max distance for print statement
            if dist > max:
                max = dist

            poses.append(pose)
        
        if filtered_count > 0:
            rospy.loginfo_throttle(0.5, "Filtered " + str(filtered_count) + " poses at " + str(max) + "m max")

        self.publishPosesVisual(timestamp, poses)

    def publishPoseMeasurement(self, timestamp, pose, vec_m, yaw_ar, yaw_r):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = "odom"

        pose_msg.pose.pose.position.x = pose[0]
        pose_msg.pose.pose.position.y = pose[1]
        pose_msg.pose.pose.orientation = getQuaternion(0, 0, pose[2])

        cov = self.getCovariance(vec_m, yaw_ar, yaw_r)
        pose_msg.pose.covariance = covarianceToMsg(cov)

        self.pose_cov_pub.publish(pose_msg)

    def getCovariance(self, vec_m, yaw_ar, yaw_r):
        dev = self.aruco_dev
        r = max(np.linalg.norm(vec_m), self.min_aruco_range)
        theta = atan2(vec_m[1], vec_m[0])
        
        length_var = dev[0]*r**2
        radial_var = dev[1]*pi*r**2
        yaw_var = dev[2]*pi*r**2
        M = np.array([[length_var, 0, 0],
                      [0, radial_var, 0],
                      [0, 0, yaw_var]])

        return R3(theta) @ M @ R3(theta).T

    def publishPosesVisual(self, timestamp, poses):
        poses_msg = PoseArray()
        poses_msg.header.stamp = timestamp
        poses_msg.header.frame_id = "odom"
        
        for pose in poses:
            pose_msg = Pose()
            pose_msg.position.x = pose[0]
            pose_msg.position.y = pose[1]
            pose_msg.orientation = getQuaternion(0, 0, pose[2])

            poses_msg.poses.append(pose_msg)

        self.poses_viz_pub.publish(poses_msg)

if __name__ == "__main__":
    rospy.init_node("aruco_pose_node")
    aruco_pose_node = ArucoToPoseNode()
