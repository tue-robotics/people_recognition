#!/usr/bin/env python

import rospy
import numpy as np
from people_tracking_v2.msg import HoCVector  # Import the custom message type
from std_msgs.msg import String
import os

class CompareHoCDataNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('compare_hoc_data', anonymous=True)
        
        # Subscriber to HoC vectors
        self.subscriber_hoc = rospy.Subscriber('/hoc_vectors', HoCVector, self.hoc_callback)
        
        # Publisher for debug information or status updates
        self.publisher_debug = rospy.Publisher('/comparison/debug', String, queue_size=10)
        
        # Load saved HoC data
        self.hoc_data_file = os.path.expanduser('~/hoc_data/hoc_data.npz')
        self.load_hoc_data()
        
        rospy.spin()
    
    def load_hoc_data(self):
        """Load the saved HoC data from the .npz file."""
        if os.path.exists(self.hoc_data_file):
            data = np.load(self.hoc_data_file)
            self.saved_hue = data['hue'][0]
            self.saved_sat = data['sat'][0]
            rospy.loginfo(f"Loaded HoC data from {self.hoc_data_file}")
        else:
            rospy.logerr(f"HoC data file {self.hoc_data_file} not found")
            self.saved_hue = None
            self.saved_sat = None

    def hoc_callback(self, msg):
        """Callback function to handle new HoC detections."""
        if self.saved_hue is None or self.saved_sat is None:
            rospy.logerr("No saved HoC data available for comparison")
            return

        hue_vector = msg.hue_vector
        sat_vector = msg.sat_vector
        distance_score = self.compute_distance_score(hue_vector, sat_vector)
        rospy.loginfo(f"Distance score for detection: {distance_score:.2f}")
        self.publish_debug_info(distance_score)
    
    def compute_distance_score(self, hue_vector, sat_vector):
        """Compute the distance score between the current detection and saved data."""
        hue_vector = np.array(hue_vector)
        sat_vector = np.array(sat_vector)
        
        hue_distance = self.compute_distance(hue_vector, self.saved_hue)
        sat_distance = self.compute_distance(sat_vector, self.saved_sat)
        
        return (hue_distance + sat_distance) / 2
    
    def compute_distance(self, vector1, vector2):
        """Compute the Euclidean distance between two vectors."""
        return np.linalg.norm(vector1 - vector2)
    
    def publish_debug_info(self, distance_score):
        """Publish debug information about the current comparison."""
        debug_msg = String()
        debug_msg.data = f"Distance score: {distance_score:.2f}"
        self.publisher_debug.publish(debug_msg)

if __name__ == '__main__':
    try:
        CompareHoCDataNode()
    except rospy.ROSInterruptException:
        pass
