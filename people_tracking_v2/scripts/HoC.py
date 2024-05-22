#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from people_tracking_v2.msg import DetectionArray  # Assuming you have a custom message type for YOLO detections
from cv_bridge import CvBridge
import cv2
import numpy as np

class HoCNode:
    def __init__(self):
        rospy.init_node('hoc_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/Webcam/image_raw', Image, self.image_callback)
        self.detection_sub = rospy.Subscriber('/yolo_detections', DetectionArray, self.detection_callback)
        
        self.current_image = None
        
        rospy.spin()
        
    def image_callback(self, msg):
        self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
    def detection_callback(self, msg):
        if self.current_image is None:
            return
        
        for i, detection in enumerate(msg.detections):
            x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
            roi = self.current_image[int(y1):int(y2), int(x1):int(x2)]
            hoc = self.compute_hoc(roi)
            rospy.loginfo(f'HoC for detection #{i + 1}: {hoc}')
            
    def compute_hoc(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
        
if __name__ == '__main__':
    try:
        HoCNode()
    except rospy.ROSInterruptException:
        pass
