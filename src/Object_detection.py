#!/usr/bin/env python3
# Import ROS libraries and messages 
"""import rospy
import sys
import os 
import torch # type: ignore
import cProfile
import timeit
import datetime
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ultralytics import YOLO  # type: ignore
import tensorrt as trt # type: ignore
import numpy as np
from geometry_msgs.msg import PointStamped
import message_filters


class ObjectDetector:
  def __init__(self, model):    
    self.bridge = CvBridge()
    self.model = model

   # Subscribe to the image and depth topics
    self.image_sub = rospy.Subscriber("/zed/zed_nodelet/left/image_rect_gray"\
                                      ,Image, self.image_callback, queue_size=1, buff_size=2**24,tcp_nodelay=True) 
   
    self.depth_sub = rospy.Subscriber("/zed/zed_nodelet/depth/depth_registered"\
                                      ,Image, self.depth_callback, queue_size=1, buff_size=2**24,tcp_nodelay=True)
    
    # Publisher for the plotted image
    self.image_pub = rospy.Publisher("image_plotted", Image, queue_size=1)
    self.depth_pub = rospy.Publisher("depth_plotted", Image, queue_size=1)
    self.min_distance_pub = rospy.Publisher("min_distance", PointStamped, queue_size=1)

  def image_callback(self,data):  
    start_time = time.time()
    try:

      #convert image to numpy
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
      #perform model inference
      results = self.model(cv_image, device ='0' , imgsz=640, conf=0.1, half=True)
          

      if results[0].masks is not None:
        #plot images with detections
        image_plotted = results[0].plot()

      
        #Perform depth calculation for depth_image
        self.calculate_depth(image_plotted,results)
      else: 
        image_plotted=cv_image

      ros_image = self.bridge.cv2_to_imgmsg(image_plotted, "bgr8")
      #execution_time = time.time() - start_time
      #remaining_time = max(0, 0.145 - execution_time)
      #time.sleep(remaining_time)
      self.image_pub.publish(ros_image)

    except CvBridgeError as e:
      print(e)
    
    #end_time = time.time()
    #loop_time = end_time - start_time
    #print("One loop of image_callback took {:.2f} seconds.".format(loop_time))

  def depth_callback(self, data):

    start_time = time.time()
    try:
      self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")  
    except CvBridgeError as e:
        print(e)

    #end_time = time.time()
    #elapsed_time = end_time - start_time
    # print("Depth callback took {:.2f} seconds.".format(elapsed_time))
  
  def calculate_depth(self, image_plotted, results):
    start_time = time.time()
     
    #convert image to numpy
    masks = results[0].masks.data
    boxes = results[0].boxes.data
    clss = boxes[:,5]
    people_indices = torch.where(clss == 0)
    #scale for visualizing results
    people_masks = masks[people_indices]
    people_mask = torch.any(people_masks, dim=0).int() * 255
    people_mask_uint8 = people_mask.cpu().numpy().astype(np.uint8)
    min_depth = float('inf')    

    #cv2.imshow("masks", people_mask_uint8)
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()
    if people_mask is None:
      #end_time = time.time()
      #elapsed_time = end_time - start_time
      #print("Calculate depth function took {:.2f} seconds.".format(elapsed_time))
      return 0
      
    if people_mask is not None:

      image = cv2.resize(self.depth_image, (640,384))

      depth_values = []
      coordinates_with_depth = []
      coordinates = cv2.findNonZero(people_mask_uint8)
      depth_values = image[coordinates[:, 0, 1], coordinates[:, 0, 0]]

       # Find valid (finite and not NaN) depth values and their corresponding coordinates
      valid_indices = np.isfinite(depth_values) & (depth_values < min_depth) & ~np.isnan(depth_values)
      valid_depth_values = depth_values[valid_indices]
      valid_coordinates = coordinates[valid_indices, 0, :]

      if len(valid_depth_values) > 0:
        min_index = np.argmin(valid_depth_values)
        min_depth = valid_depth_values[min_index]
        x_min, y_min = valid_coordinates[min_index]
        coordinates_with_depth.append((x_min, y_min, min_depth))
  
      # Draw a circle at the coordinates with minimal depth value
      radius = 10  # Adjust as needed
      color = (0, 255, 0)  # Green color, adjust as needed
      thickness = 3  # Adjust as neededs

      image_plotted = cv2.resize(image_plotted,(640,384))
      image_plotted = cv2.circle(image_plotted, (x_min, y_min), radius, color, thickness)
      # Add text to the image indicating the minimum depth value
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 1  # smaller font size
      font_thickness = 2  # thicker font
      font_color = (0, 255, 0)  
      min_depth_text = "{:.3f}".format(min_depth)
      text_size = cv2.getTextSize(min_depth_text, font, font_scale, font_thickness)[0]
      text_position = (10, image.shape[0] - 10)  # bottom left corner
      image_plotted = cv2.putText(image_plotted, min_depth_text, text_position, font, font_scale, font_color, font_thickness)
      mage_plotted = cv2.putText(image_plotted, f'Min Depth: {min_depth}', (x_min, y_min), font, font_scale, font_color)
      cv2.imshow("point2",image_plotted)
      cv2.waitKey(1)
      min_distance_msg = PointStamped()
      min_distance_msg.header.stamp = rospy.Time.now()
      min_distance_msg.header.frame_id = "zed_base_link"# Modify the frame_id as needed
      min_distance_msg.point.x = min_depth
      min_distance_msg.point.y = 0.0  # You can set other coordinates if needed
      min_distance_msg.point.z = 0.0
      self.min_distance_pub.publish(min_distance_msg)

    else:
      min_distance_msg = PointStamped()
      min_distance_msg.header.stamp = rospy.Time.now()
      min_distance_msg.header.frame_id = "zed_base_link"# Modify the frame_id as needed
      min_distance_msg.point.x = min_depth
      min_distance_msg.point.y = 0.0  # You can set other coordinates if needed
      min_distance_msg.point.z = 0.0
      self.min_distance_pub.publish(min_distance_msg)  

      #else:
      #  print("No contours found in the mask")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Calculate depth function took {:.2f} seconds.".format(elapsed_time))
     
    



def main():
  #model = YOLO('/home/user/catkin_ws/src/object_detection/src/yolov8n-seg.engine','detect')
  #model = YOLO('/home/user/catkin_ws/src/object_detection/src/best.pt')
  #labels = open("labels.txt" , "w")
  model = YOLO('yolov8n-seg.pt')
  rospy.init_node('Object_Detector', anonymous=True)
  ic = ObjectDetector(model)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")


if __name__ == '__main__':
  main()  
  #cProfile.run('main()', filename='profile_results.txt')
  #print(timeit.timeit("main()", setup="from __main__ import main", number=1))
"""


import rospy
import sys
import os 
import torch # type: ignore
import cProfile
import timeit
import datetime
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ultralytics import YOLO  # type: ignore
import tensorrt as trt # type: ignore
import numpy as np
from geometry_msgs.msg import PointStamped
import message_filters


class ObjectDetector:
  def __init__(self, model):    
    self.bridge = CvBridge()
    self.model = model

   # Subscribe to the image and depth topics
    self.image_sub = rospy.Subscriber("/zed/zed_nodelet/left/image_rect_color"\
                                      ,Image, self.image_callback, queue_size=1, buff_size=2**24,tcp_nodelay=True) 
   
    self.depth_sub = rospy.Subscriber("/zed/zed_nodelet/depth/depth_registered"\
                                      ,Image, self.depth_callback, queue_size=1, buff_size=2**24,tcp_nodelay=True)
    
    # Publisher for the plotted image
    self.image_pub = rospy.Publisher("image_plotted", Image, queue_size=1)
    self.depth_pub = rospy.Publisher("depth_plotted", Image, queue_size=1)
    self.min_distance_pub = rospy.Publisher("min_distance", PointStamped, queue_size=1)

  def image_callback(self,data):  
    start_time = time.time()
    try:

      #convert image to numpy
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
      #perform model inference
      results = self.model(cv_image, device ='0' , imgsz=640, conf=0.1, half=True)
          

      if results[0].masks is not None:
        #plot images with detections
        image_plotted = results[0].plot()

      
        #Perform depth calculation for depth_image
        self.calculate_depth(image_plotted,results)
      else: 
        image_plotted=cv_image

      ros_image = self.bridge.cv2_to_imgmsg(image_plotted, "bgr8")
      #execution_time = time.time() - start_time
      #remaining_time = max(0, 0.145 - execution_time)
      #time.sleep(remaining_time)
      self.image_pub.publish(ros_image)

    except CvBridgeError as e:
      print(e)
    
    #end_time = time.time()
    #loop_time = end_time - start_time
    #print("One loop of image_callback took {:.2f} seconds.".format(loop_time))

  def depth_callback(self, data):

    start_time = time.time()
    try:
      self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")  
    except CvBridgeError as e:
        print(e)

    #end_time = time.time()
    #elapsed_time = end_time - start_time
    # print("Depth callback took {:.2f} seconds.".format(elapsed_time))
  
  def calculate_depth(self, image_plotted, results):
    start_time = time.time()
     
    #convert image to numpy
    masks = results[0].masks.data
    boxes = results[0].boxes.data
    clss = boxes[:,5]
    
    people_indices = torch.where(clss == 0)#56)
    #scale for visualizing results
    people_masks = masks[people_indices]
    people_mask = torch.any(people_masks, dim=0).int() * 255
    people_mask_uint8 = people_mask.cpu().numpy().astype(np.uint8)
    min_depth = float('inf')    

    #cv2.imshow("masks", people_mask_uint8)
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()
    if people_mask is None:
      #end_time = time.time()
      #elapsed_time = end_time - start_time
      #print("Calculate depth function took {:.2f} seconds.".format(elapsed_time))
      return 0
      
    if people_mask is not None:

      image = cv2.resize(self.depth_image, (640,384))

      depth_values = []
      coordinates_with_depth = []
      coordinates = cv2.findNonZero(people_mask_uint8)
      depth_values = image[coordinates[:, 0, 1], coordinates[:, 0, 0]]

       # Find valid (finite and not NaN) depth values and their corresponding coordinates
      valid_indices = np.isfinite(depth_values) & (depth_values < min_depth) & ~np.isnan(depth_values)
      valid_depth_values = depth_values[valid_indices]
      valid_coordinates = coordinates[valid_indices, 0, :]

      if len(valid_depth_values) > 0:
        min_index = np.argmin(valid_depth_values)
        min_depth = valid_depth_values[min_index]
        x_min, y_min = valid_coordinates[min_index]
        coordinates_with_depth.append((x_min, y_min, min_depth))
  
      # Draw a circle at the coordinates with minimal depth value
      radius = 10  # Adjust as needed
      color = (0, 255, 0)  # Green color, adjust as needed
      thickness = 3  # Adjust as neededs

      image_plotted = cv2.resize(image_plotted,(640,384))
      image_plotted = cv2.circle(image_plotted, (x_min, y_min), radius, color, thickness)
      # Add text to the image indicating the minimum depth value
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 1  # smaller font size
      font_thickness = 2  # thicker font
      font_color = (0, 255, 0)  
      min_depth_text = "{:.3f}".format(min_depth)
      text_size = cv2.getTextSize(min_depth_text, font, font_scale, font_thickness)[0]
      text_position = (10, image.shape[0] - 10)  # bottom left corner
      image_plotted = cv2.putText(image_plotted, min_depth_text, text_position, font, font_scale, font_color, font_thickness)
      mage_plotted = cv2.putText(image_plotted, f'Min Depth: {min_depth}', (x_min, y_min), font, font_scale, font_color)
      #cv2.imshow("point2",image_plotted)
      #cv2.waitKey(1)
      min_distance_msg = PointStamped()
      min_distance_msg.header.stamp = rospy.Time.now()
      min_distance_msg.header.frame_id = "zed_base_link"# Modify the frame_id as needed
      min_distance_msg.point.x = min_depth
      min_distance_msg.point.y = 0.0  # You can set other coordinates if needed
      min_distance_msg.point.z = 0.0
      self.min_distance_pub.publish(min_distance_msg)

    else:
      min_distance_msg = PointStamped()
      min_distance_msg.header.stamp = rospy.Time.now()
      min_distance_msg.header.frame_id = "zed_base_link"# Modify the frame_id as needed
      min_distance_msg.point.x = min_depth
      min_distance_msg.point.y = 0.0  # You can set other coordinates if needed
      min_distance_msg.point.z = 0.0
      self.min_distance_pub.publish(min_distance_msg)  

      #else:
      #  print("No contours found in the mask")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Calculate depth function took {:.2f} seconds.".format(elapsed_time))
     
    



def main():
  model = YOLO('/home/user/catkin_ws/src/object_detection/src/models/best.pt','detect')
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
  #cProfile.run('main()', filename='profile_results.txt')
  #print(timeit.timeit("main()", setup="from __main__ import main", number=1))