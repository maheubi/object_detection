#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from math import cos, sin, pi 

def publish_circle_marker():
    rospy.init_node('circle_marker_publisher', anonymous=True)
    marker_pub = rospy.Publisher('circle_marker', Marker, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz
    
    while not rospy.is_shutdown():
        # Create Marker messages for each circle
        markers = []

        # Define circle parameters
        circle_params = [(1.0, (1.0, 0.0, 0.0)),   # Radius 0.5 meter, red
                         (2.0, (1.0, 1.0, 0.0))]   # Radius 2.5 meters, yellow


        for index, (start_radius, color) in enumerate(circle_params):
            # Create Marker message
            marker = Marker()
            marker.header.frame_id = "base_link"  # Assuming base link frame
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2 * start_radius  # Diameter of the circle
            marker.scale.y = 2 * start_radius  # Diameter of the circle
            marker.scale.z = 0.01  # Thickness of the cylinder
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 0.5  # Semi-transparent
            marker.ns = "circle_" + str(index)  # Unique namespace for each circle

            markers.append(marker)

        # Publish all markers
        for marker in markers:
            marker_pub.publish(marker)

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_circle_marker()
    except rospy.ROSInterruptException:
        pass

