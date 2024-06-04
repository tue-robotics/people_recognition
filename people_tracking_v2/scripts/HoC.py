#!/usr/bin/env python

import rospy
from people_tracking_v2.msg import SegmentedImages, HoCVector, HoCVectorArray  # Custom message for batch segmented images and HoC vectors
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class HoCNode:
    def __init__(self, initialize_node=True):
        if initialize_node:
            rospy.init_node('hoc_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.segmented_images_sub = rospy.Subscriber('/segmented_images', SegmentedImages, self.segmented_images_callback)
        
        # Publisher for HoC vectors
        self.hoc_vector_pub = rospy.Publisher('/hoc_vectors', HoCVectorArray, queue_size=10)
        
        if initialize_node:
            rospy.spin()
        
    def segmented_images_callback(self, msg):
        # rospy.loginfo(f"First segmented image received at: {rospy.Time.now()}")  # Log first message timestamp
        # rospy.loginfo(f"Received batch of {len(msg.images)} segmented images")
        
        hoc_vectors = HoCVectorArray()
        hoc_vectors.header.stamp = msg.header.stamp  # Use the same timestamp as the incoming message
        
        for i, segmented_image_msg in enumerate(msg.images):
            try:
                segmented_image = self.bridge.imgmsg_to_cv2(segmented_image_msg, "bgr8")
                hoc_hue, hoc_sat = self.compute_hoc(segmented_image)
                rospy.loginfo(f'Computed HoC for segmented image #{i}')

                # Extract the ID from the incoming message
                detection_id = msg.ids[i]
                rospy.loginfo(f"Received Detection ID: {detection_id} for segmented image #{i}")

                # Create HoCVector message
                hoc_vector = HoCVector()
                hoc_vector.id = detection_id
                hoc_vector.hue_vector = hoc_hue.tolist()
                hoc_vector.sat_vector = hoc_sat.tolist()
                hoc_vectors.vectors.append(hoc_vector)
            except CvBridgeError as e:
                rospy.logerr(f"Failed to convert segmented image: {e}") 
            except IndexError as e:
                rospy.logerr(f"IndexError: {e}. This might happen if there are more segmented images than detections.")
        
        # Publish the HoC vectors as a batch
        self.hoc_vector_pub.publish(hoc_vectors)
        
    def compute_hoc(self, segmented_image):
        # Convert to HSV
        hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
        
        # Create a mask to ignore black pixels
        mask = cv2.inRange(hsv, (0, 0, 1), (180, 255, 255))
        
        # Use the same number of bins for Hue and Saturation
        bins = 256
        
        # Compute histogram for Hue and Saturation using the mask
        hist_hue = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
        hist_sat = cv2.calcHist([hsv], [1], mask, [bins], [0, 256])
        
        cv2.normalize(hist_hue, hist_hue)
        cv2.normalize(hist_sat, hist_sat)
        
        # Flatten the histograms
        return hist_hue.flatten(), hist_sat.flatten()

if __name__ == '__main__':
    try:
        HoCNode()
    except rospy.ROSInterruptException:
        pass
