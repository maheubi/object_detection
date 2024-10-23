#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PointStamped
from dynamic_reconfigure.client import Client

last_received_time = None
dwa_client = None

def callback(data):
    global last_received_time
    last_received_time = rospy.get_time()
    rospy.loginfo("Received min distance: %f", data.point.x)
    velocity_command(data)

def default_velocity(event=None):
    global last_received_time
    if last_received_time is not None and rospy.get_time() - last_received_time > 5:
        rospy.loginfo("No depth values received for more than 5 seconds. Setting default velocity.")
        dwa_client.update_configuration({'max_vel_trans': 1.0})

def listener():
    global dwa_client
    rospy.init_node('distance_listener', anonymous=True)
    dwa_client = Client('/move_base/DWAPlannerROS', timeout=30)
    rospy.Subscriber("/min_distance", PointStamped, callback)
    rospy.Timer(rospy.Duration(1), default_velocity)  # Check every 5 seconds
    rospy.spin()
    
def velocity_command(data):
    x = data.point.x
    if x > 3:
        dwa_client.update_configuration({'max_vel_trans': 0.2, 'max_vel_x': 0.2})
        rospy.loginfo("new Velocity : %f", 0.6)
    elif 1 < x < 3:
        dwa_client.update_configuration({'max_vel_trans': 0.1, 'max_vel_x': 0.1})
        rospy.loginfo("new Velocity : %f", 0.3)
    elif x < 1.5:
        dwa_client.update_configuration({'max_vel_trans': 0.005, 'max_vel_x': 0.005})
        rospy.loginfo("new Velocity : %f", 0.1)
    else:
        dwa_client.update_configuration({'max_vel_trans': 0.4})    

if __name__ == '__main__':
    listener()
