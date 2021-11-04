#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import pdb

#ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class reactive_follow_gap:

    def __init__(self):
        self.lidar_sub = rospy.Subscriber("scan", LaserScan, self.lidar_callback)
        self.drive_pub = rospy.Publisher("drive", AckermannDriveStamped, queue_size=10)
        self.processed_lidar_pub = rospy.Publisher("proc_scan", LaserScan, queue_size=10)
    

    def preprocess_lidar(self, ranges, data: LaserScan):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """

        # Truncate FOV
        min_id = round((-math.pi/2 - data.angle_min) / data.angle_increment)
        max_id = round((math.pi/2 - data.angle_min) / data.angle_increment)

        angle_min = data.angle_min + min_id*data.angle_increment
        angle_max = data.angle_min + max_id*data.angle_increment

        ranges = list(ranges)[min_id:max_id]
        data.angle_min = angle_min
        data.angle_max = angle_max
        data.ranges = ranges

        # De-noise the data
        for i in range(len(ranges)):
            if ranges[i] < data.range_min:
                ranges[i] = data.range_min
            elif ranges[i] > data.range_max:
                ranges[i] = data.range_max

        # proc_ranges = list()

        # for i in range(len(ranges) - 5):
        #     proc_ranges.append(sum(ranges[i:i+5])/5)

        return ranges


    def set_bubble_around_index(self, ranges, idx, data: LaserScan):

        RADIUS = 0.5 #m

        angular_radius = math.atan(RADIUS / max(ranges[idx], 0.05))
        index_radius = round(angular_radius / data.angle_increment)

        for i in range(max(0, idx-index_radius), min(len(ranges), idx+index_radius)):
            ranges[i] = 0


    def find_max_gap(self, free_space_ranges, data: LaserScan):
        """ Return the start index & end index of the max gap in free_space_ranges
        """

        MAX_INDEX = len(free_space_ranges) 

        start_i = 0
        end_i = start_i

        max_gap_start_i = start_i
        max_gap_end_i = start_i


        while end_i < MAX_INDEX:
            while end_i < MAX_INDEX and free_space_ranges[end_i] != 0:
                end_i += 1
            
            if end_i - start_i > max_gap_end_i - max_gap_start_i:
                max_gap_start_i = start_i
                max_gap_end_i = end_i

            end_i += 1
            start_i = end_i

        return max_gap_start_i, max_gap_end_i
    

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        
        gap = ranges[start_i: end_i+1]
        furthest_dist, idx = max([(val, id) for id, val in enumerate(gap)])

        idx2 = idx

        while gap[idx2] == gap[idx]:
            idx2 -= 1

        idx = (idx + idx2) // 2

        return idx + start_i
    
    def lidar_callback(self, data : LaserScan):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """

        proc_ranges = self.preprocess_lidar(data.ranges, data)

        #Find closest point to LiDAR
        closest_dist, idx = min((val, idx) for (idx, val) in enumerate(proc_ranges))

        #Eliminate all points inside 'bubble' (set them to zero) 
        self.set_bubble_around_index(proc_ranges, idx, data)

        #Find max length gap 
        start_i, end_i = self.find_max_gap(proc_ranges, data)

        #Find the best point in the gap 
        best_point = self.find_best_point(start_i, end_i, proc_ranges)

        #Publish Drive message

        # angle to best point
        angle = data.angle_min + best_point * data.angle_increment

        # Display processed ranges on RVIZ
        proc = LaserScan()
        proc.range_min = data.range_min
        proc.range_max = data.range_max
        proc.header = data.header
        proc.angle_increment = data.angle_increment
        proc.angle_max = data.angle_max
        proc.angle_min = data.angle_min
        proc.ranges = proc_ranges

        self.processed_lidar_pub.publish(proc)

        # Source: Lab 3
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"

        # Use a PID with error=angle to determine steering angle

        drive_msg.drive.steering_angle = 0.5 * angle

        if abs(math.degrees(angle)) < 10:
            velocity = 2
        elif abs(math.degrees(angle)) < 20:
            velocity = 1
        else:
            velocity = 1

        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)



def main(args):
    rospy.init_node("FollowGap_node", anonymous=True)
    rfgs = reactive_follow_gap()
    rospy.sleep(0.1)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)