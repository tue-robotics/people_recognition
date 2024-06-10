#!/usr/bin/env python

import rospy
import numpy as np
import message_filters
from people_tracking_v2.msg import HoCVectorArray, BodySizeArray, ComparisonScoresArray, ComparisonScores  # Import the custom message types
from std_msgs.msg import String
import os

class ComparisonNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('comparison_node', anonymous=True)
        
        # Synchronize the subscribers using message_filters
        hoc_sub = message_filters.Subscriber('/hoc_vectors', HoCVectorArray)
        pose_sub = message_filters.Subscriber('/pose_distances', BodySizeArray)
        
        ts = message_filters.ApproximateTimeSynchronizer([hoc_sub, pose_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.sync_callback)
        
        # Publisher for comparison scores
        self.comparison_pub = rospy.Publisher('/comparison/scores_array', ComparisonScoresArray, queue_size=10)
        
        # Load saved HoC and Pose data
        self.hoc_data_file = os.path.expanduser('~/hoc_data/hoc_data.npz')
        self.pose_data_file = os.path.expanduser('~/pose_data/pose_data.npz')
        self.load_hoc_data()
        self.load_pose_data()
        
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

    def load_pose_data(self):
        """Load the saved Pose data from the .npz file (Pose)."""
        if os.path.exists(self.pose_data_file):
            data = np.load(self.pose_data_file)
            self.saved_pose_data = {
                'head_feet_distance': data['head_feet_distance']
            }
            rospy.loginfo(f"Loaded Pose data from {self.pose_data_file}")
        else:
            rospy.logerr(f"Pose data file {self.pose_data_file} not found")
            self.saved_pose_data = None

    def sync_callback(self, hoc_array, pose_array):
        """Callback function to handle synchronized HoC and pose data."""
        rospy.loginfo("sync_callback invoked")

        if self.saved_hue is None or self.saved_sat is None:
            rospy.logerr("No saved HoC data available for comparison")
            return

        if not hoc_array.vectors or not pose_array.distances:
            rospy.logerr("Received empty HoC or Pose array")
            return

        comparison_scores_array = ComparisonScoresArray()
        comparison_scores_array.header.stamp = hoc_array.header.stamp

        for hoc_msg, pose_msg in zip(hoc_array.vectors, pose_array.distances):
            rospy.loginfo(f"Processing Detection ID {hoc_msg.id}")

            # Compare HoC data
            hue_vector = hoc_msg.hue_vector
            sat_vector = hoc_msg.sat_vector
            hoc_distance_score = self.compute_hoc_distance_score(hue_vector, sat_vector)
            rospy.loginfo(f"Detection ID {hoc_msg.id}: HoC Distance score: {hoc_distance_score:.2f}")

            # Compare pose data
            head_feet_distance = pose_msg.head_feet_distance
            head_feet_saved = np.mean(self.saved_pose_data['head_feet_distance'])

            distance_score = self.compute_distance(head_feet_distance, head_feet_saved)
            rospy.loginfo(f"Detection ID {pose_msg.id}: Pose Distance score: {distance_score:.2f}")

            # Create and append ComparisonScores message
            comparison_scores_msg = ComparisonScores()
            comparison_scores_msg.header.stamp = hoc_msg.header.stamp  # Use the timestamp from the HoC message
            comparison_scores_msg.header.frame_id = hoc_msg.header.frame_id
            comparison_scores_msg.id = hoc_msg.id
            comparison_scores_msg.hoc_distance_score = hoc_distance_score
            comparison_scores_msg.pose_distance_score = distance_score  # Save head-feet distance as pose_distance_score

            # Log the scores
            rospy.loginfo(f"Publishing scores - Detection ID {comparison_scores_msg.id}: HoC Distance score: {comparison_scores_msg.hoc_distance_score:.2f}, Pose Distance score: {comparison_scores_msg.pose_distance_score:.2f}")

            comparison_scores_array.scores.append(comparison_scores_msg)

        # Publish the comparison scores as a batch
        self.comparison_pub.publish(comparison_scores_array)

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
    
    def publish_debug_info(self, hoc_distance_score, pose_distance_score, detection_id):
        """Publish debug information about the current comparison (General)."""
        debug_msg = String()
        debug_msg.data = f"Detection ID {detection_id}: HoC Distance score: {hoc_distance_score:.2f}, Pose Distance score: {pose_distance_score:.2f}"
        self.publisher_debug.publish(debug_msg)

if __name__ == '__main__':
    try:
        ComparisonNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
