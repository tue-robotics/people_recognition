#!/usr/bin/env python

import rospy
from people_tracking_v2.msg import SegmentedImages, HoCVector  # Custom message for batch segmented images and HoC vectors
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
        self.hoc_vector_pub = rospy.Publisher('/hoc_vectors', HoCVector, queue_size=10)
        
        if initialize_node:
            rospy.spin()
        
    def segmented_images_callback(self, msg):
        rospy.loginfo(f"Received batch of {len(msg.images)} segmented images")
        for i, segmented_image_msg in enumerate(msg.images):
            try:
                segmented_image = self.bridge.imgmsg_to_cv2(segmented_image_msg, "bgr8")
                hoc_hue, hoc_sat = self.compute_hoc(segmented_image)
                rospy.loginfo(f'Computed HoC for segmented image #{i + 1}')

                # Extract the ID from the incoming message
                detection_id = msg.ids[i]
                rospy.loginfo(f"Received Detection ID: {detection_id} for segmented image #{i + 1}")

                # Publish the HoC vectors with the detection ID
                self.publish_hoc_vectors(hoc_hue, hoc_sat, detection_id)
            except CvBridgeError as e:
                rospy.logerr(f"Failed to convert segmented image: {e}") 
            except IndexError as e:
                rospy.logerr(f"IndexError: {e}. This might happen if there are more segmented images than detections.")
        
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
    
    def publish_hoc_vectors(self, hue_vector, sat_vector, detection_id):
        """Publish the computed HoC vectors."""
        hoc_msg = HoCVector()
        hoc_msg.header.stamp = rospy.Time.now()
        hoc_msg.id = detection_id
        hoc_msg.hue_vector = hue_vector.tolist()
        hoc_msg.sat_vector = sat_vector.tolist()
        self.hoc_vector_pub.publish(hoc_msg)

if __name__ == '__main__':
    try:
        HoCNode()
    except rospy.ROSInterruptException:
        pass
