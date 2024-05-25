#!/usr/bin/env python

import rospy
from people_tracking_v2.msg import SegmentedImages  # Custom message for batch segmented images
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os

class HoCNode:
    def __init__(self):
        rospy.init_node('hoc_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.segmented_images_sub = rospy.Subscriber('/segmented_images', SegmentedImages, self.segmented_images_callback)
        
        # Ensure the directory exists
        self.hoc_data_dir = os.path.expanduser('~/hoc_data')
        if not os.path.exists(self.hoc_data_dir):
            os.makedirs(self.hoc_data_dir)
            rospy.loginfo(f"Created directory: {self.hoc_data_dir}")
        else:
            rospy.loginfo(f"Using existing directory: {self.hoc_data_dir}")
        
        rospy.spin()
        
    def segmented_images_callback(self, msg):
        rospy.loginfo(f"Received batch of {len(msg.images)} segmented images")
        for i, segmented_image_msg in enumerate(msg.images):
            try:
                segmented_image = self.bridge.imgmsg_to_cv2(segmented_image_msg, "bgr8")
                hoc_hue, hoc_sat = self.compute_hoc(segmented_image)
                rospy.loginfo(f'HoC for segmented image #{i + 1} (Hue): {hoc_hue}')
                rospy.loginfo(f'HoC for segmented image #{i + 1} (Saturation): {hoc_sat}')
                
                # Save the HoC data
                hue_save_path = os.path.join(self.hoc_data_dir, f'hoc_hue_detection_{i + 1}.npy')
                sat_save_path = os.path.join(self.hoc_data_dir, f'hoc_sat_detection_{i + 1}.npy')
                np.save(hue_save_path, hoc_hue)
                np.save(sat_save_path, hoc_sat)
                
                # Print statements to verify file saving
                rospy.loginfo(f'Saved Hue HoC to {hue_save_path}')
                rospy.loginfo(f'Saved Sat HoC to {sat_save_path}')
            except CvBridgeError as e:
                rospy.logerr(f"Failed to convert segmented image: {e}")
        
    def compute_hoc(self, segmented_image):
        hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for Hue (180 bins) and Saturation (256 bins)
        hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        cv2.normalize(hist_hue, hist_hue)
        cv2.normalize(hist_sat, hist_sat)
        
        # Flatten the histograms
        return hist_hue.flatten(), hist_sat.flatten()
        
if __name__ == '__main__':
    try:
        HoCNode()
    except rospy.ROSInterruptException:
        pass
