#!/usr/bin/env python
import rospy

from math import cos, degrees
import pdb

from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

class Safety(object):

    DECELERATION = 8.26 # (m/s²)

    emergency_brake = False
    speed = 0

    """
    The class that handles emergency braking.
    """
    def __init__(self):
        """
        One publisher should publish to the /brake topic with a AckermannDriveStamped brake message.

        One publisher should publish to the /brake_bool topic with a Bool message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0
        # TODO: create ROS subscribers and publishers.

        rospy.Subscriber("scan", LaserScan, callback=self.scan_callback)
        rospy.Subscriber("odom", Odometry, callback=self.odom_callback)

        self.brake = rospy.Publisher("brake", AckermannDriveStamped, queue_size=10)
        self.brake_bool = rospy.Publisher("brake_bool", Bool, queue_size=10)


    def odom_callback(self, odom_msg: Odometry):
        # update current speed
        v = odom_msg.twist.twist.linear
        self.speed = v.x

    def scan_callback(self, scan_msg: LaserScan):
        # calculate TTC (Time To Collision)
        # Supposes no moving obstacles
        theta = scan_msg.angle_min
        for r in scan_msg.ranges:

            # Compute TTC
            dr = self.speed * cos(theta)
            if dr > 0:
                ttc = r / dr
                if ttc <= 2 * abs(2 / self.DECELERATION):
                    rospy.loginfo("AEB triggered! \n By point @" + str(theta) + "°, d=" + str(r) + "m, dr=" + str(dr) + "\n ttc=" + str(ttc) + "\n speed=" + str(self.speed))
                    drive = AckermannDriveStamped()
                    drive.drive.speed = 0
                    self.brake.publish(drive)
                    self.brake_bool.publish(Bool(True))
                    self.emergency_brake = True
                    break

            # Increase theta
            theta += scan_msg.angle_increment


        # If no risk of collision is found, publish False to brake_bool
        if not self.emergency_brake:
            self.brake_bool.publish(Bool(False))
        self.emergency_brake = False



def main():
    rospy.init_node('safety_node')
    sn = Safety()
    rospy.spin()


if __name__ == '__main__':
    main()