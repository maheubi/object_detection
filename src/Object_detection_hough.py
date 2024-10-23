#!/usr/bin/env python3
# Import ROS libraries and messages
import rospy
import sys
import os
import torch  # type: ignore
import cProfile
import timeit
import datetime
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ultralytics import YOLO  # type: ignore
import tensorrt as trt  # type: ignore
import numpy as np
from geometry_msgs.msg import PointStamped
import message_filters
import math
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan

class ObjectDetector:

    def __init__(self, model):
        self.bridge = CvBridge()
        self.model = model
        self.Z1_old = 0
        self.Z2_old = 0
        # Subscribe to the image and depth topics
        self.image_sub = rospy.Subscriber("/zed/zed_nodelet/left/image_rect_color"\
                                          ,Image, self.image_callback, queue_size=1, buff_size=2**24,tcp_nodelay=False)
        self.depth_sub = rospy.Subscriber("/zed/zed_nodelet/depth/depth_registered"\
                                          ,Image, self.depth_callback, queue_size=1, buff_size=2**24,tcp_nodelay=False)
        self.info_sub = rospy.Subscriber("/zed/zed_nodelet/left/camera_info"\
                                          ,CameraInfo, self.info_callback, queue_size=1)

        # Publisher for the plotted image
        self.image_pub = rospy.Publisher("image_plotted", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("depth_plotted", Image, queue_size=1)
        self.min_distance_pub = rospy.Publisher("min_distance",
                                                PointStamped,
                                                queue_size=1)
        self.marker_pub = rospy.Publisher("line_marker", Marker, queue_size=1)
        self.scan_pub = rospy.Publisher('scan1', LaserScan, queue_size=1)
        self.point_pub = rospy.Publisher('points', Marker, queue_size=1)
      

    def image_callback(self, data):
        start_time = time.time()

        try:

            #convert image to numpy
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            print(cv_image.shape)
            cv_image = cv2.resize(cv_image, (640, 384))
            #perform model inference
            results = self.model(cv_image,
                                 device='0',
                                 imgsz=640,
                                 conf=0.78,
                                 half=True)

            if results[0].masks is not None:
                #plot images with detections
                image_plotted = results[0].plot()
                
                #Perform depth calculation for depth_image
                self.calculate_depth(image_plotted, cv_image, results)

            return cv_image
        except CvBridgeError as e:
            print(e)

        end_time = time.time()
        loop_time = end_time - start_time
        #print("One loop of hough took {:.2f} seconds.".format(loop_time))

    def info_callback(self, data):

        self.P = data.K

    def depth_callback(self, data):

        start_time = time.time()
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                data, desired_encoding="32FC1")
        except CvBridgeError as e:
            print(e)

        end_time = time.time()
        elapsed_time = end_time - start_time
        # print("Depth callback took {:.2f} seconds.".format(elapsed_time))

    def mean_depth(self, image, cv_image, x1, y1, x2, y2):
        h, w = image.shape
        results = []

        # List to handle multiple points
        points = [(x1, y1), (x2, y2)]

        for (x, y) in points:
            if x >= 0 and x < w and y >= 0 and y < h:
                # Define top-left and bottom-right coordinates for the rectangle
                top_left = (x - 10, y - 21)
                bottom_right = (x + 10, y + 21)

                # Draw the rectangle on the depth image and the visual image
                cv2.rectangle(cv_image, top_left, bottom_right, (0, 255, 0), 2)
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

                # Extract the 11x11 patch around (x, y)
                patch = image[y - 21:y + 21, x - 10:x + 10]
                #print("Patch around ({}, {}):".format(x, y), patch)

                # Exclude 0, infinite, and NaN values
                valid_depths = patch[np.where((patch > 0)
                                              & (np.isfinite(patch))
                                              & ~np.isnan(patch))]

                # Calculate the mean depth if there are any valid depths remaining
                if valid_depths.size > 0:
                    smallest_values = np.sort(valid_depths)[:10]
                    print("smalles values", smallest_values)
                    # Then, calculate the mean of these five values
                    mean_val = np.mean(smallest_values)
                    results.append(mean_val)
                    mean_val = round(mean_val, 2)
                    print("mean_value", mean_val)
                else:
                    results.append(np.nan)
            else:
                results.append(np.nan)

        # Display the images with patches
        #cv2.imshow("Depth Image with Patches", image)
        #cv2.imshow("CV Image with Patches", cv_image)
        #cv2.waitKey(1)
        #cv2.destroyAllWindows()

        return results

    def calculate_depth(self, image_plotted, cv_image, results):
        start_time = time.time()

        #extract mask and bounding boxes of
        masks = results[0].masks.data
        boxes = results[0].boxes.data
        clss = boxes[:, 5]
        indices = torch.where(clss == 0)
        masks = masks[indices]

        mask = torch.any(masks, dim=0).int() * 255
        mask_uint8 = mask.cpu().numpy().astype(np.uint8)

        # Erode the mask to shrink it
        kernel_size = 4  # Determines how much you want to erode the mask
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)

        min_depth = float('inf')
        f_x = self.P[0]*640/960
        f_y = self.P[4]*384/540
        c_x = self.P[2]*640/960
        c_y = self.P[5]*384/540
        #T_x = self.P[7]
        #T_y = self.P[11]
        #print("Matrix", self.P)

        #cv2.imshow("masks", mask_uint8)
        #cv2.waitKey(1)
        #cv2.destroyAllWindows()
        if mask is None:
            return

        if mask is not None:

            # Apply binary mask to the original image
            #print("type people_mask",type(mask_uint8))
            #print("cv_image", cv_image.shape)
            #print("mask shape" , mask_uint8.shape)
            masked_image = cv2.bitwise_and(
                cv_image, cv_image,
                mask=mask_uint8)  #what does this do?????????
            
            #print("size depth_image",self.depth_image.shape)
            image = cv2.resize(self.depth_image, (640, 384))
            #cv2.imshow("masks", image)
            #cv2.waitKey(1)
            image = cv2.bitwise_and(
                image, image,
                mask=mask_uint8)  #what does this do?????????
            #print("size depth image after resize", image.shape)
            #cv2.imshow("newmask", image)
            #cv2.waitKey(1)


            depth_values = []
            coordinates_with_depth = []
            coordinates = cv2.findNonZero(mask_uint8)
            depth_values = image[coordinates[:, 0][:, 1], coordinates[:, 0][:, 0]]

            # Find valid (finite and not NaN) depth values and their corresponding coordinates
            valid_indices = np.isfinite(depth_values) & (
                depth_values < min_depth) & ~np.isnan(depth_values)
            valid_depth_values = depth_values[valid_indices]
            valid_coordinates = coordinates[valid_indices, 0, :]

            if len(valid_depth_values) > 0:
                min_index = np.argmin(valid_depth_values)
                min_depth = valid_depth_values[min_index]
                x_min, y_min = valid_coordinates[min_index]
                coordinates_with_depth.append((x_min, y_min, min_depth))
                print("minimum Depth", min_depth)
            min_distance_msg = PointStamped()
            min_distance_msg.header.stamp = rospy.Time.now()
            min_distance_msg.header.frame_id = "zed_left_camera_frame"  # Modify the frame_id as needed
            min_distance_msg.point.x = min_depth
            min_distance_msg.point.y = 0.0  # You can set other coordinates if needed
            min_distance_msg.point.z = 0.0
            self.min_distance_pub.publish(min_distance_msg)

            # Apply Canny edge detection and Hough Line Transformation
            gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_masked_image, 1, 20)
            #cv2.imshow("edges", edges)
            #cv2.waitKey(1)

            lines = cv2.HoughLinesP(edges,
                                    1,
                                    np.pi / 180,
                                    10,
                                    minLineLength=10,
                                    maxLineGap=200)

            #cv2.imshow("lines", lines)
            #cv2.waitKey(1)
            if lines is not None:
                # Find the longest line
                max_length = 0
                longest_line = None
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if length > max_length:
                        max_length = length
                        longest_line = (x1, y1, x2, y2)

                # Draw the longest line on the image
                x1, y1, x2, y2 = longest_line
                print("points", x1, y1, x2, y2)
                #image = cv2.resize(self.depth_image, (640,384))

                #Z1 = image[y1,x1]
                #Z2 = image[y2,x2]
                # Calculate the mean depths around each point
                X1, X2 = self.mean_depth(image, cv_image, x1, y1, x2, y2)
                #Z3_2, Z2_2 = self.mean_depth(image,cv_image,x3,y3,x2,y2)

                #print("Mean depth around (x1, y1):", Z1)
                #print("Mean depth around (x2, y2):", Z2)
                #print("mean depth z3_1", Z3_1, "meand depth z3_2", Z3_2)

                if X2 < X1:
                    Depth_min = X2
                else:
                    Depth_min = X1


                if np.isfinite(X1) and ~np.isnan(X1) and  Depth_min/min_depth <= 1.1\
                  and np.isfinite(X2) and ~np.isnan(X2) and X2/X1 <=1.4: #abs(self.X2_old/X2) < 1.1 and abs(self.X1_old/X1) < 1.1
                    print("X1 equals", X1, "X2 equals", X2)
                    Y1 = -((x1 - c_x) * X1) / f_x
                    Z1 = -((y1 - c_y) * X1) / f_y
                    Y2 = -((x2 - c_x) * X2) / f_x
                    Z2 = -((y2 - c_y) * X2) / f_y
                    Y1 = np.round(Y1, 1)
                    Z1 = np.round(Z1, 1)
                    Y2 = np.round(Y2, 1)
                    Z2 = np.round(Z2, 1)
                    X1 = np.round(X1, 1)
                    X2 = np.round(X2, 1)
                    #X3 = ((x3-c_x)*Z3_2)/f_x
                    #Y3 = ((y3-c_y)*Z3_2)/f_y
                    print("wolrd coodrinates = ", X1, Y1, Z1)
                    print("world Coordinates = ", X2, Y2, Z2)
                    self.publish_line_marker(X1, Y1, X2, Y2, Z1, Z2)
                    self.X1_old = X1
                    self.X2_old = X2
                else:
                    print("Z1 is NAN")

                cv2.line(image_plotted, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(image_plotted, (x1, y1),
                           radius=5,
                           color=(0, 255, 0),
                           thickness=-1)
                cv2.circle(image_plotted, (x2, y2),
                           radius=5,
                           color=(0, 255, 0),
                           thickness=-1)

                # Display the image with the longest line
                #cv2.imshow("Image with Longest Line and Start Point",  image_plotted)
                #cv2.waitKey(100)
                #cv2.destroyAllWindows()
                image = np.copy(image_plotted)
                ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
                self.image_pub.publish(ros_image)

            else:
                rospy.logwarn("No lines found in the Hough transformation.")
                # Display the image with the longest line

            #else:
            #  print("No contours found in the mask")
        #end_time = time.time()
        #elapsed_time = end_time - start_time
        return

        #print("Calculate depth function took {:.2f} seconds.".format(elapsed_time))
    def interpolate_points(self, x1, y1, x2, y2, num_points=200):
        """ Linearly interpolate num_points between two points (x1, y1) and (x2, y2) """
        points = []
        for i in range(num_points + 1):
            alpha = i / num_points
            x = x1 + (x2 - x1) * alpha
            y = y1 + (y2 - y1) * alpha
            points.append((x, y))
        return points

    def extend_line(self, x1, y1, x2, y2, extension_length=0.5):

        # Calculate the vector components and its length
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)

        # Normalize the vector
        dx /= length
        dy /= length

        # Extend the line on both ends
        x1_extended = x1 - dx * extension_length
        y1_extended = y1 - dy * extension_length
        x2_extended = x2 + dx * extension_length
        y2_extended = y2 + dy * extension_length

        return x1_extended, x2_extended, y1_extended, y2_extended

    def publish_line_marker(self, X1, Y1, X2, Y2, Z1, Z2):

        x1, x2, y1, y2 = self.extend_line(X1, Y1, X2, Y2)

        # Create a Marker message
        marker = Marker()
        marker.header.frame_id = "zed_left_camera_frame"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width

        # Set the color (RGBA) of the line
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Create two points for the line
        point1 = Point()
        point1.x = x1
        point1.y = y1
        point1.z = 0  # Depth value

        point2 = Point()
        point2.x = x2
        point2.y = y2
        point2.z = 0  # Depth value

        point = Marker()
        point.header.frame_id = "zed_left_camera_frame"
        point.header.stamp = rospy.Time.now()
        point.ns = "points"
        point.id = 0
        point.type = marker.POINTS
        point.action = marker.ADD
        point.pose.orientation.w = 1.0

        # Set the scale of the marker
        point.scale.x = 0.2  # size of points
        point.scale.y = 0.2
        point.scale.z = 0.2

        # Points are blue
        point.color.a = 1.0  # Alpha must be non-zero
        point.color.b = 1.0

        # Create two points for the line
        point3 = Point()
        point3.x = X1
        point3.y = Y1
        point3.z = 0  # Depth value

        point4 = Point()
        point4.x = X2
        point4.y = Y2
        point4.z = 0  # Depth value
        # Add the points to the marker
        marker.points.append(point1)
        marker.points.append(point2)
        point.points.append(point3)
        point.points.append(point4)

        # Publish the marker
        self.marker_pub.publish(marker)
        self.point_pub.publish(point)
        line_points = self.interpolate_points(
            x1, y1, x2, y2)  # Adjust number of points as needed

        scan = LaserScan()
        scan.header.frame_id = "zed_left_camera_frame"
        scan.header.stamp = rospy.Time.now()
        scan.angle_min = -1.57  # -90 degrees
        scan.angle_max = 1.57  # 90 degrees
        scan.angle_increment = 3.14 / 180  # 1 degree in radians
        scan.range_min = 0.4
        scan.range_max = 10.0  # example max range
        scan.ranges = [float('inf')] * 360  # initialize all ranges to inf

        def calculate_range_and_angle(x, y):
            return math.sqrt(x**2 + y**2), math.atan2(y, x)

            # Populate the LaserScan message

        for x, y in line_points:
            range, angle = calculate_range_and_angle(x, y)
            if -1.57 <= angle <= 1.57:
                index = int((angle - scan.angle_min) / scan.angle_increment)
                if 0 <= index < len(scan.ranges):
                    scan.ranges[index] = min(
                        scan.ranges[index],
                        range)  # ensure the closest range is set

        # Publish the LaserScan
        self.scan_pub.publish(scan)


def main():
    model = YOLO(
        '/home/user/catkin_ws/src/object_detection/src/models/best_run27_n.pt',
        'detect')
    #model = YOLO('/home/user/catkin_ws/src/object_detection/src/yolov8n-seg.engine','detect')
    #model = YOLO('yolov8n-seg.pt')
    #labels = open("labels.txt" , "w")
    rospy.init_node('Object_Detector', anonymous=True)
    ic = ObjectDetector(model)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
