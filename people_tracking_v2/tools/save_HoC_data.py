#!/usr/bin/env python

import rospy
import numpy as np
from people_tracking_v2.msg import HoCVectorArray  # Custom message for HoC vectors
import os

class SaveHoCDataNode:
    def __init__(self):
        rospy.init_node('save_hoc_data_node', anonymous=True)

        # Subscriber for HoC vectors
        self.hoc_vector_sub = rospy.Subscriber('/hoc_vectors', HoCVectorArray, self.hoc_vector_callback)

        # File to save HoC data
        self.hoc_data_file = os.path.expanduser('~/hoc_data/hoc_data.npz')

        rospy.spin()

    def hoc_vector_callback(self, msg):
        if not msg.vectors:
            rospy.loginfo("No HoC vectors received, skipping saving.")
            return
        
        rospy.loginfo(f"Received {len(msg.vectors)} HoC vectors")
        
        # Store only the first HoC vector
        hoc_vector = msg.vectors[0]
        hoc_hue = np.array(hoc_vector.hue_vector)
        hoc_sat = np.array(hoc_vector.sat_vector)

        # Save the first histograms in a single .npz file
        np.savez(self.hoc_data_file, hue=hoc_hue, sat=hoc_sat)
        rospy.loginfo(f'Saved HoC data to {self.hoc_data_file}')

if __name__ == '__main__':
    try:
        SaveHoCDataNode()
    except rospy.ROSInterruptException:
        pass
