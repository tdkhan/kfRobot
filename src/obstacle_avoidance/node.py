#!/usr/bin/env python3
import rospy
import numpy as np
from math import pi

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# Very WIP obstacle avoidance node, general idea is as follows:

# Take in lidar laser scan
# Check if we are too close to any of the points
# If not drive in a straight line at a constant rate
# If we are too close:
# Loop through lidar scan and find segments of which all points are above a threshold range
# Find the segment which is furthest away
# Turn towards the desired segment using open loop time based control
# Drive straight until we get to close to another object

class ObstacleAvoidance:
    def __init__(self):
        # Command message output
        self.cmd_msg = Twist()

        # Rate to publish commands
        self.cmd_rate = 10 # Hz

        # Rate at which robot moves
        self.forward_rate = 0.5 # m/s
        self.turn_rate = 0.5 # rad/s

        # Minimum range where we begin to turn away
        self.min_range = 2.0 # meters

        # Minimum and maximum angle to check for obstacles in
        self.min_a = 2*pi - pi/3
        self.max_a = pi/3

        # Range which is considered open to go towards
        self.open_range = 1 # meters

        # Minimum open arc length to be considered navigable
        self.open_arc_length = 10 # deg

        # Subscribe to laser scan, publish to command velocity topic
        rospy.Subscriber("/scan", LaserScan, self.scanCb)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # Publish cmd_vel messages to turtlebot at fixed rate
        self.cmd_timer = rospy.Timer(rospy.Duration(1/self.cmd_rate), self.cmdPubTimerCb)

        # Begin to move forward
        self.cmd_msg.linear.x = self.forward_rate
        self.turning = False

        # Wait for callbacks and allow timers to run
        rospy.spin()


    # Function which receives scan from Lidar
    def scanCb(self, msg):
        # Save angle increment parameter
        self.angle_increment = msg.angle_increment

        # Create a vector of angles to reference ranges with
        ranges = msg.ranges
        bearings = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))

        min_i = np.argmin(ranges)
        # If a lidar return is below the threshold (too close)
        if (bearings[min_i] > self.min_a or bearings[min_i] < self.max_a) \
            and ranges[min_i] < self.min_range and not self.turning:
            rospy.loginfo_throttle(1, "Object at " + str(bearings[min_i]*180/pi) + 
                " degrees, " + str(ranges[min_i]) + "m away")
            open_bearing = self.findOpenBearing(ranges, bearings)
            self.turnTowards(open_bearing)
                
    # Find the bearing of the farthest arc, all ranges in arc must be above open_range parameter
    def findOpenBearing(self, ranges, bearings):
        segments = []
        start_index = None
        arc_length = 0

        # Loop through the row vectors
        for i in range(len(ranges)):
            # If we get a point that is too close
            if ranges[i] < self.open_range:
                # Reset counters
                arc_length = 0
                start_index = None

            # If we reach our arc length threshold
            elif arc_length >= self.open_arc_length:
                # Add center row vector to list of segments
                mid_index = int((i - start_index) / 2 + start_index)
                segments.append((ranges[mid_index], bearings[mid_index]))
                # Reset counters
                arc_length = 0
                start_index = None

            # If we are starting a new segment
            elif not start_index:
                start_index = i
                arc_length += ranges[i] * self.angle_increment

            # Otherwise just add the arc length to the counter
            else:
                arc_length += ranges[i] * self.angle_increment

        # Filter in only sectors we want to turn towards
        filt_segs = []
        for i, seg in enumerate(segments):
            # Only append sectors which are not in front of robot within angle parameters
            if self.max_a < seg[1] < self.min_a:
                filt_segs.append(seg)

        # Find the bearing segment with the maximum range
        max_range, max_bearing = filt_segs[np.argmax(filt_segs[0])]

        # Return the bearing of the maximum range segment
        return max_bearing

    # Turn the robot towards a bearing
    def turnTowards(self, bearing):
        self.turning = True
        if bearing < pi:
            turn_time = bearing / self.turn_rate
            self.cmd(left=True)
            rospy.loginfo("Turning left " + str(bearing*180/pi) + " degrees over " + str(turn_time) + " seconds")

        elif bearing >= pi:
            turn_time = 2*pi - bearing / self.turn_rate
            self.cmd(right=True)
            rospy.loginfo("Turning right to " + str((2*pi - bearing)*180/pi) + " over " + str(turn_time) + " seconds")
        else:
            rospy.logerr("Got weird bearing angle: " + str(bearing))
            return
        
        rospy.Timer(rospy.Duration(turn_time), self.turnTimerCb, oneshot=True)

    # Once we have turned for the desired length of time (timer length)
    # Begin going straight again
    def turnTimerCb(self, event):
        self.cmd(forward=True)
        self.turning = False

    # Publish cmd_vel messages to turtlebot at fixed rate
    def cmdPubTimerCb(self, event):
        self.cmd_pub.publish(self.cmd_msg)
        # pass

    # Function for changing commands to robot
    def cmd(self, forward=False, left=False, right=False):
        if forward:
            self.cmd_msg.linear.x = self.forward_rate
            self.cmd_msg.angular.z = 0.0
        
        elif left:
            self.cmd_msg.linear.x = 0.0
            self.cmd_msg.angular.z = self.turn_rate
        
        elif right:
            self.cmd_msg.linear.x = 0.0
            self.cmd_msg.angular.z = -self.turn_rate

if __name__ == "__main__":
    rospy.init_node("obstacle_avoidance_node")
    obs_avoid = ObstacleAvoidance()
