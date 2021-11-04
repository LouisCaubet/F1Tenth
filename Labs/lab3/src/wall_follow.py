#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np

#ROS Imports
import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

#PID CONTROL PARAMS
kp = 1.5
kd = 0.05
ki = 0.25
servo_offset = 0.0
prev_error = 0.0 
error = 0.0
integral = 0.0
prev_time = 0
current_time = 0

#WALL FOLLOW PARAMS
ANGLE_RANGE = 270 # Hokuyo 10LX has 270 degrees scan
DESIRED_ANGLE = math.pi/4
LOOKAHEAD_DISTANCE = 0.9
DESIRED_DISTANCE = 1
VELOCITY = 2.00 # meters per second
CAR_LENGTH = 0.50 # Traxxas Rally is 20 inches or 0.5 meters

class WallFollow:
    """ Implement Wall Following on the car
    """
    def __init__(self):
        #Topics & Subs, Pubs
        global prev_time, current_time
        prev_time = rospy.Time.now().to_sec()
        current_time = prev_time

        self.lidar_sub = rospy.Subscriber("scan", LaserScan, callback=self.lidar_callback)
        self.drive_pub = rospy.Publisher("drive", AckermannDriveStamped, queue_size=10)

    def getRange(self, data: LaserScan, angle: float):
        # data: single message from topic /scan
        # angle IN RADIANS: between -45 to 225 degrees, where 0 degrees is directly to the right
        # Outputs length in meters to object with angle in lidar scan field of view
        # make sure to take care of nans etc.

        def findFirstAcceptableMeasure(start_angle):
            index = math.floor((start_angle - data.angle_min) / data.angle_increment)
            start_angle = data.angle_min + data.angle_increment*index
            dist = data.ranges[index]

            return dist, start_angle

        dist_a, angle_a = findFirstAcceptableMeasure(angle)
        dist_b, angle_b = findFirstAcceptableMeasure(math.pi/2)

        theta = angle_b - angle_a

        alpha = math.atan((dist_a*math.cos(theta) - dist_b)/(dist_a * math.sin(theta)))
        D = dist_b * math.cos(alpha)
        Dnext = D + LOOKAHEAD_DISTANCE * math.sin(alpha)

        return Dnext

    def pid_control(self, error, velocity):
        global integral
        global prev_error
        global kp, ki, kd
        global prev_time, current_time

        dt = current_time - prev_time

        integral += error*dt
        derivative = (error - prev_error)/dt
        angle = (kp * error + ki * integral + kd * derivative)    

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = angle

        if abs(math.degrees(angle)) < 10:
            velocity = 1.5
        elif abs(math.degrees(angle)) < 20:
            velocity = 1
        else:
            velocity = 0.5

        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)

    def followLeft(self, data, leftDist):
        #Follow left wall as per the algorithm 
        return leftDist - DESIRED_DISTANCE

    def lidar_callback(self, data: LaserScan):
        global prev_time, current_time, prev_error, error
        prev_time = current_time
        current_time = rospy.Time.now().to_sec()

        distanceToWall = self.getRange(data, DESIRED_ANGLE)

        prev_error = error
        error = self.followLeft(data, distanceToWall)
        self.pid_control(error, VELOCITY)

def main(args):
    rospy.init_node("WallFollow_node", anonymous=True)
    rospy.sleep(2)
    wf = WallFollow()
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)