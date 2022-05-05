#!/usr/bin/env python
import sys
import math

#ROS Imports
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

# Imports for dataset generation
import pickle
import os
import time

#PID CONTROL PARAMS
kp = 1
kd = 0
ki = 0
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

FOLLOW_RIGHT = True
EXPORT_DATASET = True
IS_SIMULATOR = True

class WallFollow:
    """ Implement Wall Following on the car
    """
    def __init__(self, dataset_generator):
        #Topics & Subs, Pubs
        global prev_time, current_time
        prev_time = rospy.Time.now().to_sec()
        current_time = prev_time

        self.lidar_sub = rospy.Subscriber("scan", LaserScan, callback=self.lidar_callback)
        if IS_SIMULATOR:
            self.drive_pub = rospy.Publisher("drive", AckermannDriveStamped, queue_size=10)
        else:
            self.drive_pub = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=10)

        self.active = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/active", String, queue_size=10)

        self.dsgen = dataset_generator

        self.skip = 1
        self.total_time = 0
            

    def getRange(self, data, angle):
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

    def pid_control(self, error, velocity, data):
        global integral
        global prev_error
        global kp
        global ki
        global kd
        global prev_time, current_time

        dt = current_time - prev_time

        integral += error*dt
        derivative = (error - prev_error)/dt
        angle = (kp * error + ki * integral + kd * derivative)

        # 0.34 is the max steering angle
        angle = min(angle, 0.34)
        angle = max(angle, -0.34)

        if FOLLOW_RIGHT:
            angle *= -1

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = angle

        if abs(math.degrees(angle)) < 10:
            velocity = 3
        elif abs(math.degrees(angle)) < 20:
            velocity = 2
        else:
            velocity = 1

        drive_msg.drive.speed = velocity
        self.active.publish("Navigation")
        self.drive_pub.publish(drive_msg)

        if EXPORT_DATASET:
            self.dsgen.data.append([data.ranges, [drive_msg.drive.steering_angle, drive_msg.drive.speed]])

    def computeError(self, distance):
        #Follow wall as per the algorithm 
        return distance - DESIRED_DISTANCE

    def lidar_callback(self, data):

        if FOLLOW_RIGHT:
            data.ranges = list(reversed(data.ranges))

        start = time.time()
        self.skip+=1
        global prev_time, current_time, prev_error, error
        prev_time = current_time
        current_time = rospy.Time.now().to_sec()

        distanceToWall = self.getRange(data, DESIRED_ANGLE)

        prev_error = error
        error = self.computeError(distanceToWall)
        self.pid_control(error, VELOCITY, data)
        end = time.time()
        self.total_time += end-start


class DatasetGenerator:

    def __init__(self):
        self.data = []

    def save(self):
        with open("/home/louis/catkin_ws/dataset.pickle", "wb+") as file:
            rospy.loginfo(os.path.realpath(file.name))
            pickle.dump(self.data, file)

def main(args):
    rospy.init_node("WallFollow_node", anonymous=True)
    rospy.sleep(2)
    dsgen = DatasetGenerator()
    wf = WallFollow(dsgen)
    rospy.spin()
    if EXPORT_DATASET:
        dsgen.save()

if __name__=='__main__':
	main(sys.argv)