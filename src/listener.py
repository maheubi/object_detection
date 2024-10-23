import rospy
from geometry_msgs.msg import PointStamped

def callback(data):
    rospy.loginfo("Received average distance: %f", data.point.x)

def listener():
    rospy.init_node('distance_listener', anonymous=True)
    rospy.Subscriber("average_distance", PointStamped, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()