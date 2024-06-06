#!/usr/bin/env python

import rospy
import numpy as np
import message_filters
from people_tracking_v2.msg import HoCVectorArray, ComparisonScoresArray, ComparisonScores  # Import the custom message types
from std_msgs.msg import String
import os
import csv

class HoCComparisonNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('hoc_comparison_node', anonymous=True)
        
        # Subscriber for HoC vectors
        hoc_sub = message_filters.Subscriber('/hoc_vectors', HoCVectorArray)
        
        # Synchronize the subscriber using message_filters
        ts = message_filters.ApproximateTimeSynchronizer([hoc_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.hoc_callback)
        
        # Publisher for comparison scores
        self.comparison_pub = rospy.Publisher('/comparison/scores_array', ComparisonScoresArray, queue_size=10)
        # Publisher for operator identification results
        self.operator_pub = rospy.Publisher('/operator_identification', String, queue_size=10)
        
        # Load saved HoC data
        self.hoc_data_file = os.path.expanduser('~/hoc_data/hoc_data.npz')
        self.load_hoc_data()
        
        # Threshold for determining operator
        self.operator_threshold = 0.6

        # File to save operator detection IDs and scores
        self.operator_log_file = os.path.expanduser('~/operator_detections.csv')
        self.init_operator_log_file()
        
        rospy.spin()
    
    def load_hoc_data(self):
        """Load the saved HoC data from the .npz file (HoC)."""
        if os.path.exists(self.hoc_data_file):
            data = np.load(self.hoc_data_file)
            self.saved_hue = data['hue']
            self.saved_sat = data['sat']
            rospy.loginfo(f"Loaded HoC data from {self.hoc_data_file}")
        else:
            rospy.logerr(f"HoC data file {self.hoc_data_file} not found")
            self.saved_hue = None
            self.saved_sat = None

    def init_operator_log_file(self):
        """Initialize the operator log file."""
        with open(self.operator_log_file, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['Frame Timestamp', 'Operator Detection ID', 'HoC Distance Score'])
        rospy.loginfo(f"Initialized operator log file at {self.operator_log_file}")

    def hoc_callback(self, hoc_array):
        """Callback function to handle HoC data."""
        rospy.loginfo("hoc_callback invoked")

        if self.saved_hue is None or self.saved_sat is None:
            rospy.logerr("No saved HoC data available for comparison")
            return

        if not hoc_array.vectors:
            rospy.logerr("Received empty HoC array")
            return

        comparison_scores_array = ComparisonScoresArray()
        comparison_scores_array.header.stamp = hoc_array.header.stamp

        best_score = float('inf')
        best_detection_id = None

        for hoc_msg in hoc_array.vectors:
            rospy.loginfo(f"Processing Detection ID {hoc_msg.id}")

            # Compare HoC data
            hue_vector = hoc_msg.hue_vector
            sat_vector = hoc_msg.sat_vector
            hoc_distance_score = self.compute_hoc_distance_score(hue_vector, sat_vector)
            rospy.loginfo(f"Detection ID {hoc_msg.id}: HoC Distance score: {hoc_distance_score:.2f}")

            # Check if this is the best (smallest distance) operator candidate
            if hoc_distance_score < self.operator_threshold and hoc_distance_score < best_score:
                best_score = hoc_distance_score
                best_detection_id = hoc_msg.id

            # Create and append ComparisonScores message
            comparison_scores_msg = ComparisonScores()
            comparison_scores_msg.header.stamp = hoc_msg.header.stamp  # Use the timestamp from the HoC message
            comparison_scores_msg.header.frame_id = hoc_msg.header.frame_id
            comparison_scores_msg.id = hoc_msg.id
            comparison_scores_msg.hoc_distance_score = hoc_distance_score
            comparison_scores_msg.pose_distance_score = 0.0  # Set to 0.0 since it's not used
            comparison_scores_array.scores.append(comparison_scores_msg)

        # Determine and save the operator
        if best_detection_id is not None:
            operator_status = f"Operator (Distance score: {best_score:.2f})"
            self.save_operator_detection(hoc_array.header.stamp, best_detection_id, best_score)
        else:
            operator_status = "None"
            self.save_operator_detection(hoc_array.header.stamp, "None", None)

        # Publish the operator identification result
        operator_msg = String()
        operator_msg.data = f"Detection ID {best_detection_id}: {operator_status}"
        self.operator_pub.publish(operator_msg)

        # Publish the comparison scores as a batch
        self.comparison_pub.publish(comparison_scores_array)

    def save_operator_detection(self, timestamp, detection_id, score):
        """Save the operator detection ID and score for each frame."""
        with open(self.operator_log_file, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, detection_id, score])
        rospy.loginfo(f"Saved operator detection ID {detection_id} with score {score} at timestamp {timestamp}")

    def compute_hoc_distance_score(self, hue_vector, sat_vector):
        """Compute the distance score between the current detection and saved data (HoC)."""
        hue_vector = np.array(hue_vector)
        sat_vector = np.array(sat_vector)
        
        hue_distance = self.compute_distance(hue_vector, self.saved_hue)
        sat_distance = self.compute_distance(sat_vector, self.saved_sat)
        
        return (hue_distance + sat_distance)
    
    def compute_distance(self, vector1, vector2):
        """Compute the Euclidean distance between two vectors (General)."""
        return np.linalg.norm(vector1 - vector2)

if __name__ == '__main__':
    try:
        HoCComparisonNode()
    except rospy.ROSInterruptException:
        pass
