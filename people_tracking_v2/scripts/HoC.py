#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from people_tracking_v2.msg import DetectionArray  # Assuming you have a custom message type for YOLO detections
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class HoCNode:
    def __init__(self):
        rospy.init_node('hoc_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.segmented_image_sub = rospy.Subscriber('/segmented_images', Image, self.segmented_image_callback)
        self.detection_sub = rospy.Subscriber('/yolo_detections', DetectionArray, self.detection_callback)
        
        self.segmented_images = []
        
        rospy.spin()
        
    def segmented_image_callback(self, msg):
        try:
            segmented_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.segmented_images.append(segmented_image)
        except CvBridgeError as e:
            rospy.logerr(e)
        
    def detection_callback(self, msg):
        if not self.segmented_images:
            return
        
        for i, (detection, segmented_image) in enumerate(zip(msg.detections, self.segmented_images)):
            hoc = self.compute_hoc(segmented_image)
            rospy.loginfo(f'HoC for detection #{i + 1}: {hoc}')
        
        # Clear the segmented images list after processing
        self.segmented_images = []
            
    def compute_hoc(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
        
if __name__ == '__main__':
    try:
        HoCNode()
    except rospy.ROSInterruptException:
        pass
