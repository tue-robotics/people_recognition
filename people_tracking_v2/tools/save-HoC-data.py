#!/usr/bin/env python

import rospy
import numpy as np
from people_tracking_v2.msg import SegmentedImages  # Custom message for batch segmented images
from cv_bridge import CvBridge, CvBridgeError
import os
import sys

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from HoC import HoCNode  # Import the HoCNode class

class SaveHoCDataNode:
    def __init__(self):
        rospy.init_node('save_hoc_data_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.segmented_images_sub = rospy.Subscriber('/segmented_images', SegmentedImages, self.segmented_images_callback)
        
        # File to save HoC data
        self.hoc_data_file = os.path.expanduser('~/hoc_data/hoc_data.npz')
        
        # Instantiate the HoCNode for HoC computation without initializing the node again
        self.hoc_node = HoCNode(initialize_node=False)
        
        rospy.spin()
        
    def segmented_images_callback(self, msg):
        rospy.loginfo(f"Received batch of {len(msg.images)} segmented images")
        all_hue_histograms = []
        all_sat_histograms = []
        for i, segmented_image_msg in enumerate(msg.images):
            try:
                segmented_image = self.bridge.imgmsg_to_cv2(segmented_image_msg, "bgr8")
                hoc_hue, hoc_sat = self.hoc_node.compute_hoc(segmented_image)
                all_hue_histograms.append(hoc_hue)
                all_sat_histograms.append(hoc_sat)
            except CvBridgeError as e:
                rospy.logerr(f"Failed to convert segmented image: {e}")
        
        # Save all histograms in a single .npz file
        np.savez(self.hoc_data_file, hue=all_hue_histograms, sat=all_sat_histograms)
        rospy.loginfo(f'Saved HoC data to {self.hoc_data_file}')

if __name__ == '__main__':
    try:
        SaveHoCDataNode()
    except rospy.ROSInterruptException:
        pass
