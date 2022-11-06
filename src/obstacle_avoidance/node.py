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
        rospy.init_node("obstacle_avoidance_node")

        # Command message output
        self.cmd_msg = Twist


        ###### Parameters
        # Rate at which robot turns
        self.turn_rate = 0.2 # rad/s

        # Minimum range where we begin to turn away
        self.min_range = 0.5 # meters

        # Range which is considered open to go towards
        self.open_range = 1 # meters

        # Minimum open arc length to be considered navigable
        self.open_arc_length = 10 # deg


        # Subscribe to laser scan, publish to command velocity topic
        rospy.Subscriber("/scan:", LaserScan, self.scanCb)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist)


    # Function which receives scan from Lidar
    def scan_cb(self, msg):
        # Save angle increment parameter
        self.angle_increment = msg.angle_increment

        # Create a vector of angles to reference ranges with
        ranges = msg.ranges
        bearings = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))

        # Pack them together for ease of reference
        scan = np.array([ranges, bearings]).T

        # Loop through scan
        for range in scan:
            # If a lidar return is below the threshold
            if range < self.min_range:
                open_bearing = self.find_open_bearing(scan)
                self.turn_towards(open_bearing)
                
    # Find the bearing of the farthest arc, all ranges in arc must be above open_range parameter
    def find_open_bearing(self, scan):
        segments = np.array([])
        start_index = None
        arc_length = 0

        # Loop through the row vectors
        for i, range, bearing in enumerate(scan):
            # If we get a point that is too close
            if range < self.open_range:
                # Reset counters
                arc_length = 0
                start_index = None

            # If we reach our arc length threshold
            elif arc_length >= self.open_arc_length:
                # Add center row vector to list of segments
                mid_index = (i - start_index) / 2 + start_index
                np.append(segments, scan[i])
                # Reset counters
                arc_length = 0
                start_index = None

            # If we are starting a new segment
            elif not start_index:
                start_index = i
                arc_length += range * self.angle_increment

            # Otherwise just add the arc length to the counter
            else:
                arc_length += range * self.angle_increment

        # Find the maximum range segment
        max = segments[np.argmax(segments, 0)]

        # Return the bearing of the maximum range segment
        return max[1]

    
    def turn_towards(self, bearing):
        if bearing > pi:
            bearing -= pi

        turn_time = bearing / self.turn_rate


    # Function for sending commands to robot
    def cmd(self, vel, yaw_rate):
        self.cmd_msg.linear.x = vel
        self.cmd_msg.angular.z = yaw_rate

        self.cmd_pub.publish(self.cmd_msg)

