#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class LaserScanPublisher:
    def __init__(self):
        self.sub = rospy.Subscriber('line_marker', Marker)
        self.pub = rospy.Publisher('laser_scan', LaserScan, queue_size=10)
        rospy.init_node('laser_scan_publisher', anonymous=True)

    def publish_laser_scan(self, X1, Y1, X2, Y2):
        # Create a LaserScan message
        scan = LaserScan()
        scan.header.frame_id = "zed_base_link"  # Change to your specific frame
        scan.header.stamp = rospy.Time.now()
        
        scan.angle_min = -1.57  # -90 degrees
        scan.angle_max = 1.57   # 90 degrees
        scan.angle_increment = 3.14 / 180  # Angular resolution of 1 degree
        scan.time_increment = 0  # Not necessary for static scans
        scan.range_min = 0.0
        scan.range_max = 100.0  # Max range of LaserScan

        # Compute distances to each point and populate scan ranges
        # Assuming the points are on a line perpendicular to the direction of scan
        ranges = []
        # Calculate distance using Pythagorean theorem
        distance1 = np.sqrt(X1**2 + Y1**2)
        distance2 = np.sqrt(X2**2 + Y2**2)
        ranges.extend([distance1, distance2])  # Extend this list based on your sensor setup
        
        # Fill the ranges list to complete the 180 degrees
        num_measurements = int((scan.angle_max - scan.angle_min) / scan.angle_increment) + 1
        full_ranges = [float('inf')] * num_measurements
        # Place measurements at the correct angles (approximation)
        mid_index = num_measurements // 2
        full_ranges[mid_index - 1] = distance1
        full_ranges[mid_index] = distance2

        scan.ranges = full_ranges

        # Publish the scan
        self.pub.publish(scan)

# Usage
if __name__ == '__main__':
    scan_publisher = LaserScanPublisher()
    rospy.sleep(1)  # Wait for the publisher to establish connection with the master
    scan_publisher.publish_laser_scan(1.0, 2.0, 3.0, 4.0)  # Example distances
