#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import math
from ultralytics import YOLO
torch.cuda.empty_cache()
class Houghline:
    def __init__(self):
        rospy.init_node('yolo_segmentation_node', anonymous=True)
        self.bridge = CvBridge()

        # Load YOLO model
        self.model = YOLO("/home/taheuber/runs/segment/train4/weights/best.pt")

        # Setup subscriber for image topic
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # Resize image
            image_resized = cv2.resize(cv_image, (640, 640))

            # Perform YOLO model inference
            results = self.model(image_resized, device='0', imgsz=640, conf=0.6, half=True)
            masks = results[0].masks.data

            if masks is not None:
                # Combine segmentation masks into a single mask
                track_masks = torch.any(masks, dim=0).int() * 255
                track_masks_uint8 = track_masks.cpu().numpy().astype(np.uint8)

                # Apply binary mask to the original image
                masked_image = cv2.bitwise_and(image_resized, image_resized, mask=track_masks_uint8)

                # Apply Canny edge detection
                gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_masked_image, 250, 300)

                # Hough Line Transformation
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=10)

                if lines is not None:
                    # Find the longest line
                    max_length = 0
                    longest_line = None
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if length > max_length:
                            max_length = length
                            longest_line = (x1, y1, x2, y2)

                    # Draw the longest line on the image
                    x1, y1, x2, y2 = longest_line
                    angle = math.atan2(y2 - y1, x2 - x1)
                    line_image = np.copy(image_resized)
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(line_image, (x2, y2), radius=5, color=(0, 255, 0), thickness=-1)

                    # Display the image with the longest line
                    cv2.imshow("Image with Longest Line and Start Point", line_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    rospy.logwarn("No lines found in the Hough transformation.")
            else:
                rospy.logwarn("No masks found in the segmentation results.")

        except CvBridgeError as e:
            rospy.logerr(e)

def main():
    yolo_segmentation_node = Houghline()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

if __name__ == '__main__':
    main()
