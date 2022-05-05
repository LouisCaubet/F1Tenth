#!/usr/bin/env python3
import sys

# ROS Imports
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped

# Imports for dataset generation
import pickle
import os
import time

import torch
from PIL import Image
from math import sin, cos, radians

from config_loader import *
from lidar_preprocessing import restrict_lidar_fov, convert_lidar_to_image, preprocess_image
from network import Network

IS_SIMULATOR = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class NeuralNetworkNode:

    def __init__(self):
        self.model = Network()
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(os.path.join(
            os.path.dirname(__file__), DEFAULT_PTH_FILE), map_location=device))
        self.model.eval()

        self.lidar_sub = rospy.Subscriber("scan", LaserScan, callback=self.lidar_callback)
        self.tracked_pose = rospy.Subscriber("tracked_pose", PoseStamped, callback=self.tracking_pose_callback)
        if IS_SIMULATOR:
            self.drive_pub = rospy.Publisher("drive", AckermannDriveStamped, queue_size=10)
        else:
            self.drive_pub = rospy.Publisher(
                "/vesc/high_level/ackermann_cmd_mux/input/nav_0", AckermannDriveStamped, queue_size=10)

        self.active = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/active", String, queue_size=10)

        self.step = 0
        self.time = time.time()

        self.steering_angle = 0
        self.velocity = 0

        # 0: Starting
        # 1: Left start area, run NN
        # 2: First lap complete, switch to TEB
        self.race_step = 0
        self.starting_pose = None

    def lidar_callback(self, data: LaserScan):

        if self.race_step == 2:
            return

        self.step += 1

        if self.step % 80 != 0:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = rospy.Time.now()
            drive_msg.header.frame_id = "laser"
            drive_msg.drive.steering_angle = self.steering_angle
            drive_msg.drive.speed = self.velocity
            self.active.publish("Navigation")
            self.drive_pub.publish(drive_msg)
            return

        start = time.time()

        ranges = data.ranges
        if IS_SIMULATOR:
            ranges = restrict_lidar_fov(ranges)

        input = convert_lidar_to_image(ranges)
        input = preprocess_image(input)
        input = input.to(device)

        output = self.model(input[None, ...])
        steering_angle = output[:, 0].detach().item()
        velocity = output[:, 1].detach().item()

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = velocity
        self.active.publish("Navigation")
        self.drive_pub.publish(drive_msg)
        print(steering_angle, velocity)

        self.steering_angle = steering_angle
        self.velocity = velocity

        exectime = time.time() - start
        rospy.loginfo_throttle(0.5, "Execution time:" + str(exectime) + "- Framerate: " + str(1/exectime) + " fps")
        rospy.loginfo_throttle(0.5, "Time between callbacks: " + str((time.time() - self.time) / self.step))

    def tracking_pose_callback(self, data: PoseStamped):
        if self.starting_pose == None:
            self.starting_pose = data.pose.position
            return

        position = data.pose.position

        # Compute distance to current point
        d_squared = (position.x - self.starting_pose.x)**2 + \
                    (position.y - self.starting_pose.y)**2 + (position.z - self.starting_pose.z)**2

        if d_squared < 0.2:
            if self.race_step == 1:
                # Start TEB
                self.race_step += 1
                os.popen("rosrun map_server map_saver -f gridmap --occ 65 --free 20")
                os.popen("roslaunch teb_local_planner_tutorials navigation.launch &")
                os.popen("rosrun teb_local_planner_tutorials cmd_vel_to_ackermann_drive.py")
        else:
            if self.race_step == 0:
                self.race_step = 1


def main(args):
    rospy.init_node("NeuralNetwork_node", anonymous=True)
    rospy.sleep(2)
    NeuralNetworkNode()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
